import numpy as np

from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset
from collections import Counter
from transformers import (
    AutoTokenizer
)

from logger import divider_logger, info_logger

QUESTION_KEY = "question"
CONTEXT_KEY = "context"
ANSWERABILITY_KEY = "answerable"
ANSWER_KEY = "answer"
SEQUENCE_KEY = "answer_start"
LANGUAGE_KEY = "lang"

def correct_labels(context, answer, answerable):
    context = str(context).lower().strip()
    answer = str(answer).lower().strip()
    return (answer in context and answerable) or (answer not in context and not answerable)
    
def load_datasets(language=None, val_split=0.1):
    # Load datasets for training and evaluation
    dataset = load_dataset("coastalcph/tydi_xor_rc")
    if language is None:
        dataset = dataset.filter(lambda x: x[LANGUAGE_KEY] in ["ko", "ar", "te"])
    else:
        dataset = dataset.filter(lambda x: x[LANGUAGE_KEY] == language)
        
    mislabelled_dataset = dataset.filter(lambda x: not correct_labels(x[CONTEXT_KEY], x[ANSWER_KEY], x[ANSWERABILITY_KEY]))
    info_logger(f"Found {len(mislabelled_dataset)} mislabelled instances")
    for split_name, split_dataset in mislabelled_dataset.items():
        split_dataset.to_csv(f"roberta_labeller_results/mislabels_{split_name}.csv")
    
    dataset = dataset.filter(lambda x: correct_labels(x[CONTEXT_KEY], x[ANSWER_KEY], x[ANSWERABILITY_KEY]))
    dataset = dataset.cast_column(ANSWERABILITY_KEY, ClassLabel(num_classes=2, names=[False, True]))
    
    split_set = dataset["train"].train_test_split(test_size=val_split, seed=42, stratify_by_column=ANSWERABILITY_KEY)
    train_set, val_set = split_set["train"], split_set["test"]

    test_set = dataset["validation"]
    
    train_set = train_set.map(lambda x: {ANSWERABILITY_KEY: bool(x[ANSWERABILITY_KEY])})
    val_set = val_set.map(lambda x: {ANSWERABILITY_KEY: bool(x[ANSWERABILITY_KEY])})
    test_set = test_set.map(lambda x: {ANSWERABILITY_KEY: bool(x[ANSWERABILITY_KEY])})
    
    return train_set, val_set, test_set

def balance_dataset(dataset, column=ANSWERABILITY_KEY, oversample_ratio=1.0, undersample_ratio=1.0):
    labels = dataset[column]
    label_count = Counter(labels)
    majority_class = 0 if label_count[0] > label_count[1] else 1
    minority_class = 1 - majority_class
    
    majority_set = dataset.filter(lambda x: x[column] == majority_class)
    minority_set = dataset.filter(lambda x: x[column] == minority_class)
    total_majority = len(majority_set)
    total_minority = len(minority_set)
    
    # Oversample minority class
    if oversample_ratio > 0.0:
        samples_to_add = int(total_majority * oversample_ratio) - total_minority
        if samples_to_add > 0:
            indices = np.random.choice(total_minority, size=samples_to_add)
            oversampled = minority_set.select(indices)
            minority_set = concatenate_datasets([minority_set, oversampled])
        
    # Undersample majority class
    if undersample_ratio > 0.0:
        samples_to_keep = int(total_majority * undersample_ratio)
        if samples_to_keep < total_majority:
            majority_set = majority_set.shuffle().select(range(samples_to_keep))
    
    balanced_set = concatenate_datasets([majority_set, minority_set])
    return balanced_set.shuffle()

def preprocess(samples, tokenizer, max_input_length=512):
    inputs = tokenizer(
        text=samples[QUESTION_KEY],
        text_pair=samples[CONTEXT_KEY],
        truncation="only_second",
        max_length=max_input_length,
        return_offsets_mapping=True,
        padding=False,
    )
    
    offset_mappings = inputs.pop("offset_mapping")
    answers = samples[ANSWER_KEY]
    answer_start = samples[SEQUENCE_KEY]
    answerability = samples[ANSWERABILITY_KEY]
    
    labels = []
    for idx, token_offsets in enumerate(offset_mappings):
        answer = answers[idx]
        start_char = answer_start[idx]
        end_char = start_char + len(answer)
        sequence_idxs = inputs.sequence_ids(idx)
        seq_len = len(sequence_idxs)
        seq_label = [-100] * seq_len
        
        if answerability[idx]:
            context_start = next(seq_idx for seq_idx in range(seq_len) if sequence_idxs[seq_idx] == 1)
            context_end = next(seq_idx for seq_idx in reversed(range(seq_len)) if sequence_idxs[seq_idx] == 1)
            context_offset = token_offsets[context_start:context_end+1]
            
            for token_idx, (token_start, token_end) in enumerate(context_offset, start=context_start):
                if token_start < end_char and token_end > start_char:
                    seq_label[token_idx] = 1
                else:
                    seq_label[token_idx] = 0
        labels.append(seq_label)
    inputs["labels"] = labels
    
    return inputs

def prepare_datasets(model_name, oversample_ratio=1.0, undersample_ratio=1.0, language=None):
    train_set, val_set, test_set = load_datasets(language)
    train_splits = Counter(train_set[ANSWERABILITY_KEY])
    info_logger(f"Original Train dataset class split: {train_splits}, with a total of {len(train_set)} samples")
    
    train_set = balance_dataset(train_set, ANSWERABILITY_KEY, oversample_ratio, undersample_ratio)
    
    train_splits = Counter(train_set[ANSWERABILITY_KEY])
    info_logger(f"Sampled Train dataset class split: {train_splits}, with a total of {len(train_set)} samples")
    val_splits = Counter(val_set[ANSWERABILITY_KEY])
    info_logger(f"Validation dataset class split: {val_splits}, with a total of {len(val_set)} samples")
    test_splits = Counter(test_set[ANSWERABILITY_KEY])
    info_logger(f"Test dataset class split: {test_splits}, with a total of {len(test_set)} samples")
    divider_logger()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        info_logger("No pad token detected. Setting pad token to eos token.")
        
    train_set = train_set.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    val_set = val_set.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    
    if language is None:
        test_ko = test_set.filter(lambda x: x[LANGUAGE_KEY] == "ko")
        test_ar = test_set.filter(lambda x: x[LANGUAGE_KEY] == "ar")
        test_te = test_set.filter(lambda x: x[LANGUAGE_KEY] == "te")
        
        test_sets = {
            "ko": test_ko.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True),
            "ar": test_ar.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True),
            "te": test_te.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True)
        }
    else:
        test_sets = {language: test_set.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True)}
    
    return train_set, val_set, test_sets, tokenizer