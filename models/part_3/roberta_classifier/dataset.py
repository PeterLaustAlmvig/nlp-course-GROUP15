import torch
import nltk
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
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
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()

    # Datasets
    df_train = df_train[df_train[LANGUAGE_KEY] == language][[QUESTION_KEY, CONTEXT_KEY, ANSWERABILITY_KEY]]
    test_set = df_val[df_val[LANGUAGE_KEY] == language][[QUESTION_KEY, CONTEXT_KEY, ANSWERABILITY_KEY]]
    
    dataset = df_train.train_test_split(test_size=val_split, seed=42, stratify_by_column=ANSWERABILITY_KEY)
    train_set, val_set = dataset["train"], dataset["test"]
    
    rename_map = {ANSWERABILITY_KEY: "label"}
    train_set.rename(columns=rename_map, inplace=True)
    val_set.rename(columns=rename_map, inplace=True)
    test_set.rename(columns=rename_map, inplace=True)
    
    return train_set, val_set, test_set

def preprocess(samples, tokenizer, max_seq_length=128, pad_to_max_length=True):
    return tokenizer(
        samples[QUESTION_KEY],
        samples[CONTEXT_KEY],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length" if pad_to_max_length else False,
    )

def prepare_datasets(language, model_name):
    train_set, val_set, test_set = load_datasets(language)
    
    info_logger(f"Train dataset contains {len(train_set)} samples")
    info_logger(f"Validation dataset contains {len(val_set)} samples")
    info_logger(f"Test dataset contains {len(test_set)} samples")
    divider_logger()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        info_logger("No pad token detected. Setting pad token to eos token.")
        
    train_set = train_set.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    val_set = val_set.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    test_set = test_set.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, batched=True)
        
    return train_set, val_set, test_set, tokenizer