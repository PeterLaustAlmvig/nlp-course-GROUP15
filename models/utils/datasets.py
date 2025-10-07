import torch

import unicodedata as ud

from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from collections import Counter

class WordDataset(Dataset):
    """Sentence Dataset for training models on predicting the likelihood of an entire piece of text,
    the pieces are tokenized using the English word_tokenize meaning that they are split mainly using spaces.
    
    It is possible to specify a pre-generated vocab in case the model has been trained to only allow certain words,
    otherwise if vocab is None then it will generate the vocab based on the supplied sentences in sentence_data.
    
    The sentences are encoded based on each words presence in the word_to_idx/vocab, and words that are not present will
    be set to -1, the same system is applied when creating the target tensor giving each word a score of either 1 or -1.
    """
        
    def __init__(self, sentence_data: list, vocab=None, word_to_idx=None, idx_to_word=None, add_special_tokens=True):
        # Tokenize sentences
        self.sentences_tokens = [_tokenize_sentence(s) for s in sentence_data]

        # Generate vocab if not provided
        if vocab is None:
            self.vocab, self.word_to_idx, self.idx_to_word = _generate_vocab_keying(
                self.sentences_tokens, add_special_tokens=add_special_tokens
            )
        else:
            self.vocab = vocab
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word

        # Set the ids for the padding, unknown and end-of-sentence tokens
        self.pad_idx = self.word_to_idx.get("<PAD>", None)
        self.unk_idx = self.word_to_idx.get("<UNK>", None)
        self.eos_idx = self.word_to_idx.get("<EOS>", None)

    def __len__(self):
        return len(self.sentences_tokens)
    
    def __getitem__(self, idx):
        tokens = self.sentences_tokens[idx]
        
        # Create input sequence by replacing each word with its id
        # Unknown words are replaced with a special unknown token,
        # as we cannot ensure that all words in evaluation samples exist in validation sets
        encoded = [self.word_to_idx.get(word, self.unk_idx) for word in tokens]
        input_seq = torch.tensor(encoded, dtype=torch.long)
        
        # Create target sequence which is shifted by one, as to predict the next words in a sequence
        target_seq = torch.tensor(encoded[1:] + [self.eos_idx], dtype=torch.long)
        
        return input_seq, target_seq
    
def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lens = [len(seq) for seq in inputs]

    # Pad sequences so they have the same length across samples
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return padded_inputs, padded_targets, input_lens
    
def _tokenize_sentence(sentence):
    # Remove all punctuation characters, keeping in mind that arabic is written from right to left
    sentence = ''.join([char for char in sentence if not ud.category(char).startswith('P')])
    # Tokenize the question into words
    words = word_tokenize(sentence)
    return words

def _generate_vocab_keying(sentences_tokens, add_special_tokens=True, min_freq=1):
    # Create vocab of distinct words using frequency counter and sort for reproducibility
    words = [word for sentence in sentences_tokens for word in sentence]
    word_freq = Counter(words)
    vocab = [word for word, freq in word_freq.items() if freq >= min_freq]
    vocab = sorted(vocab)

    # Add the special tokens for padding, unknown words, and end-of-sentence
    if add_special_tokens:
        specials = ["<PAD>", "<UNK>", "<EOS>"]
        vocab = specials + vocab

    # Create lookup dicts for tensors
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    return vocab, word_to_idx, idx_to_word