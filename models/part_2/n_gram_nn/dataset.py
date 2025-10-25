import torch
import nltk
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from collections import Counter
from nltk.tokenize import word_tokenize
from datasets import load_dataset

from logger import divider_logger, info_logger

nltk.download('punkt')

#### SPECIAL TOKENS
UNKNOWN_TOKEN = "<UNK>"
START_TOKEN = "<STRT>"

#### LOADING DATASETS FOR TRAINING AND VALIDATION
def load_datasets(column, language=None, val_split=0.1):
    # Load datasets for training and evaluation
    dataset = load_dataset("coastalcph/tydi_xor_rc")
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()

    # Datasets
    if language is None:
        train_set = list(df_train[column])
        test_set = list(df_val[column])
    else:
        train_set = list(df_train[df_train['lang'] == language][column])
        test_set = list(df_val[df_val['lang'] == language][column])
    
    train_array = np.array(train_set)

    val_mask = np.array([random.random() <= val_split for _ in range(len(train_array))])
    val_set = train_array[val_mask].tolist()
    train_set = train_array[~val_mask].tolist()
    
    return train_set, val_set, test_set
    
#### PREPROCESSING OF THE SENTENCES INTO TOKENS
def preprocess_text(sentence):
    tokens = word_tokenize(sentence)
    return tokens

#### TOKENIZE TRAINING, EVALUATION AND TEST DATASETS
def tokenize_datasets(train, get_context=True):
    tokens = [preprocess_text(sentence) for sentence in train]    
    if get_context:
        return tokens, int(sum(len(sentence_tokens) for sentence_tokens in tokens) / len(tokens))
    return tokens

#### GENERATE THE VOCABULARY AND WORD_TO_IDX FOR THE TRAINING DATA
def vocab_generation(tokenized_corpus, freq_threshold=0.01, replace_threshold=0.1, replace_freq=True):
    word_corpus = [word for sentence in tokenized_corpus for word in sentence]
    word_counts = Counter(word_corpus) # Get frequency of words
    
    # Unique words in the entire corpus
    vocab = set(word_corpus)
    # Add unknown word token
    vocab.add(UNKNOWN_TOKEN)
    vocab.add(START_TOKEN)
    
    # Create word to index mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Get top % of infrequent words
    num_freq_words = max(1, int(len(word_counts) * freq_threshold))
    word_freq = sorted(word_counts.keys(), key=(lambda word: word_counts[word]), reverse=replace_freq)
    word_freq = word_freq[:num_freq_words]
    info_logger(f"Top {freq_threshold*100:.2f}% frequent words ({len(word_freq)}) out of {len(vocab)} total words")
    
    # Recalculate replacement amount according to word frequency
    total = len(word_corpus)
    freq_occurrences = sum(word_counts[word] for word in word_freq)
    true_replace_fraction = (replace_threshold * total) / freq_occurrences
    info_logger(f"The fragment to replace is {true_replace_fraction} of the {len(word_freq)} frequent words ({freq_occurrences} out of {total})")
    
    return vocab, word_to_idx, word_freq, true_replace_fraction

#### REPLACE EITHER THE MOST OR LEAST FREQUENT WORDS IN A PART OF THE SENTENCES
def replace_freq_word_to_unknown(vocab, freq_words, token_sentences, replace_threshold=0.5):
    # For more efficient lookups
    freq_words = set(freq_words)
    
    replaced = 0
    total = 0
    freq_token_sentences = []
    for sentence in token_sentences:
        sentence = np.array([word if word in vocab else UNKNOWN_TOKEN for word in sentence ])
        freq_words_mask = np.array([word in freq_words for word in sentence])
        replace_words_mask = np.array([freq_words_mask[i] and random.random() < replace_threshold for i in range(len(sentence))])
        
        # Replace the words with unknown token
        sentence[replace_words_mask] = UNKNOWN_TOKEN
        
        replaced += np.sum(replace_words_mask)
        total += len(sentence)
        
        freq_token_sentences.append(sentence.tolist())
        
    info_logger(f"Replaced {replaced} / {total} tokens ({replaced/total*100:.2f}%)")
    
    return freq_token_sentences

# CALCULATE MAX CONTEXT WINDOW
def calculate_max_context_window(language):
    if language == "en":
        train_set, _, _ = load_datasets("context")
    else:
        train_set, _, _ = load_datasets("question", language)
    
    _, max_context_window = tokenize_datasets(train_set)
    return max_context_window

# SENTENCE DATASET CLASS
class SentenceDataset(Dataset):
    def __init__(self, tokenized_sentences, word_to_idx, context_window):
        self.tokenized_sentences = tokenized_sentences
        self.word_to_idx = word_to_idx
        self.context_window = context_window
        
        self.context_center_pairs = []
        for sentence in tokenized_sentences:
            idx_sentence = [word_to_idx.get(word, word_to_idx[UNKNOWN_TOKEN]) for word in sentence]
            for target_pos in range(len(idx_sentence)):
                context_init = max(target_pos - context_window, 0)
                context = idx_sentence[context_init:target_pos]
                while len(context) < context_window:
                    context = [word_to_idx[START_TOKEN]] + context
                target_word = idx_sentence[target_pos]
                self.context_center_pairs.append((context, target_word))

    def __len__(self):
        return len(self.context_center_pairs)

    def __getitem__(self, idx):
        context, center = self.context_center_pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(center, dtype=torch.long)

#### PREPARE THE TRAIN, VALIDATION AND EVALUATION LOADERS
def prepare_dataset_loaders(language, freq_threshold, replace_freq, batch_size, replace_fraction, context_window=None):
    if language == "en":
        train_set, val_set, test_set = load_datasets("context")
    else:
        train_set, val_set, test_set = load_datasets("question", language)
    
    if context_window is None:
        train_sentence_tokens, context_window = tokenize_datasets(train_set)
    else:
        train_sentence_tokens = tokenize_datasets(train_set, False)
    val_sentence_tokens = tokenize_datasets(val_set, False)
    test_sentence_tokens = tokenize_datasets(test_set, False)
    
    info_logger(f"Train dataset contains {len(train_sentence_tokens)} samples")
    info_logger(f"Validation dataset contains {len(val_sentence_tokens)} samples")
    info_logger(f"Test dataset contains {len(test_sentence_tokens)} samples")
    divider_logger()

    vocab, word_to_idx, freq_words, true_replace_fraction = vocab_generation(train_sentence_tokens, freq_threshold, replace_fraction, replace_freq)
    
    replaced_train_sentence_tokens = replace_freq_word_to_unknown(vocab, freq_words, train_sentence_tokens, true_replace_fraction)
    train_dataset = SentenceDataset(replaced_train_sentence_tokens, word_to_idx, context_window)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    replaced_val_sentence_tokens = replace_freq_word_to_unknown(vocab, freq_words, val_sentence_tokens, true_replace_fraction)
    val_dataset = SentenceDataset(replaced_val_sentence_tokens, word_to_idx, context_window)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    replaced_test_sentence_tokens = replace_freq_word_to_unknown(vocab, freq_words, test_sentence_tokens, true_replace_fraction)
    test_dataset = SentenceDataset(replaced_test_sentence_tokens, word_to_idx, context_window)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    divider_logger()
    info_logger(f"==== Example Sentence ====")
    info_logger(f"Original:  {train_set[0]}")
    info_logger(f"Tokenized: {train_sentence_tokens[0]}")
    info_logger(f"Replaced:  {replaced_train_sentence_tokens[0]}")
    info_logger(f"Window:    {train_dataset[context_window-1]}")
    
    return train_loader, val_loader, test_loader, vocab