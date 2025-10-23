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
LANGUAGE_KEY = "lang"

#### LOADING DATASETS FOR TRAINING AND VALIDATION
def load_datasets(language=None, val_split=0.1):
    # Load datasets for training and evaluation
    dataset = load_dataset("coastalcph/tydi_xor_rc")
    if language is None:
        dataset = dataset.filter(lambda x: x["lang"] in ["ko", "ar", "te"])
    else:
        dataset = dataset.filter(lambda x: x["lang"] == language)
    
    dataset = dataset.cast_column(ANSWERABILITY_KEY, ClassLabel(num_classes=2, names=[False, True]))
    dataset = dataset.rename_column(ANSWERABILITY_KEY, "label")
    
    split_set = dataset["train"].train_test_split(test_size=val_split, seed=42, stratify_by_column="label")
    train_set, val_set = split_set["train"], split_set["test"]
    
    test_set = dataset["validation"]
    
    return train_set, val_set, test_set

def balance_dataset(dataset, column="label", oversample_ratio=1.0, undersample_ratio=1.0):
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

def preprocess(samples, tokenizer, max_seq_length=128, pad_to_max_length=True):
    return tokenizer(
        samples[QUESTION_KEY],
        samples[CONTEXT_KEY],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if pad_to_max_length else False,
    )

def prepare_datasets(model_name, oversample_ratio=1.0, undersample_ratio=1.0, language=None):
    train_set, val_set, test_set = load_datasets(language)
    train_splits = Counter(train_set["label"])
    info_logger(f"Original Train dataset class split: {train_splits}, with a total of {len(train_set)} samples")
    
    train_set = balance_dataset(train_set, "label", oversample_ratio, undersample_ratio)
    
    train_splits = Counter(train_set["label"])
    info_logger(f"Sampled Train dataset class split: {train_splits}, with a total of {len(train_set)} samples")
    val_splits = Counter(val_set["label"])
    info_logger(f"Validation dataset class split: {val_splits}, with a total of {len(val_set)} samples")
    test_splits = Counter(test_set["label"])
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