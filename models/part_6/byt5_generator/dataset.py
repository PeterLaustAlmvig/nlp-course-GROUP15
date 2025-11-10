import pandas as pd

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer
)

from logger import divider_logger, info_logger

QUESTION_KEY = "question"
CONTEXT_KEY = "context"
ANSWER_KEY = "answer_inlang"
LANGUAGE_KEY = "lang"

def correct_labels(context, answer, answerable):
    context = str(context).lower().strip()
    answer = str(answer).lower().strip()
    return (answer in context and answerable) or (answer not in context and not answerable)
    
def load_datasets(val_split=0.1):
    dataset = load_dataset("coastalcph/tydi_xor_rc")
    dataset = dataset.filter(lambda x: x[LANGUAGE_KEY] in ["te"] and x[ANSWER_KEY] is not None)
    
    mislabelled_dataset = dataset.filter(lambda x: not correct_labels(x[CONTEXT_KEY], x["answer"], x["answerable"]))
    info_logger(f"Found {len(mislabelled_dataset)} mislabelled instances")
    for split_name, split_dataset in mislabelled_dataset.items():
        split_dataset.to_csv(f"byt5_training_results/mislabels_{split_name}.csv")
    
    dataset = dataset.filter(lambda x: correct_labels(x[CONTEXT_KEY], x["answer"], x["answerable"]))
    
    split_set = dataset["train"].train_test_split(test_size=val_split, seed=42)
    train_set, val_set = split_set["train"], split_set["test"]
    
    test_set = Dataset.from_json("test.json")
    
    return train_set, val_set, test_set

def preprocess(samples, tokenizer, max_input_length=512):
    inputs = tokenizer(
        text=samples[QUESTION_KEY],
        text_pair=samples[CONTEXT_KEY],
        truncation=True,
        max_length=max_input_length,
        padding="max_length",
    )
    
    answers = tokenizer(
        samples[ANSWER_KEY],
        truncation=True,
        max_length=max_input_length,
        padding="max_length",
    )
    inputs["label"] = answers["input_ids"]
    return inputs

def prepare_datasets(model_name):
    train_set, val_set, test_set = load_datasets()
    
    info_logger(f"Train dataset has total of {len(train_set)} samples")
    info_logger(f"Validation dataset has total of {len(val_set)} samples")
    info_logger(f"Test dataset dataset has total of {len(test_set)} samples")
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