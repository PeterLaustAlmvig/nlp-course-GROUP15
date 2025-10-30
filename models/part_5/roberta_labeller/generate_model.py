import json
import os
import torch
import sys
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForTokenClassification,
)

from dataset import prepare_datasets, concatenate_datasets
from train import train_binary, evaluate_binary
from finetuning import save_results
from seeding import enforce_reproducibility

import pandas as pd
from tabulate import tabulate

def get_balanced_top_configs(csv_path, top_k=3):
    df = pd.read_csv(csv_path)

    # Compute a "balance score" that rewards configs good on all accuracies
    # Using geometric mean instead of arithmetic mean to penalize imbalance
    df["balance_score"] = (
        (df["accuracy"] * df["true_accuracy"] * df["false_accuracy"]) ** (1/3)
    )

    # Sort and pick the best ones
    top_configs = df.sort_values("balance_score", ascending=False).head(top_k)

    print("=== Top Balanced Sampling Configurations ===")
    print(tabulate(
        top_configs[[
            "over_sampling", "under_sampling",
            "accuracy", "true_accuracy", "false_accuracy",
            "loss", "balance_score"
        ]],
        headers="keys", tablefmt="fancy_grid", showindex=False
    ))
    
    best_config = top_configs.head(1).iloc[0]
    return (best_config["over_sampling"], best_config["under_sampling"])

if __name__ == "__main__":
    results_dir = "roberta_labeller_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_name = "FacebookAI/xlm-roberta-base"
    oversample_ratio, undersample_ratio = get_balanced_top_configs(f"{results_dir}/tuning_metrics.csv")
    enforce_reproducibility(42)

    # ==== PREPARE DATASETS ====
    train_set, val_set, test_sets, tokenizer = prepare_datasets(model_name, oversample_ratio, undersample_ratio, None)

    # ==== MODEL ====
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
    )

    # ==== TRAIN MODEL ====
    model, tokenizer, epoch_history = train_binary(model, train_set, val_set, tokenizer, 3, results_dir)

    # ==== EVALUATE MODEL ====
    eval_results = evaluate_binary(model, tokenizer, test_sets["ko"], "ko")
    save_results(results_dir, epoch_history, eval_results, "ko")
    
    eval_results = evaluate_binary(model, tokenizer, test_sets["ar"], "ar")
    save_results(results_dir, epoch_history, eval_results, "ar")
    
    eval_results = evaluate_binary(model, tokenizer, test_sets["te"], "te")
    save_results(results_dir, epoch_history, eval_results, "te")
    
    overall_test_set = concatenate_datasets(list(test_sets.values()))
    eval_results = evaluate_binary(model, tokenizer, overall_test_set)
    save_results(results_dir, epoch_history, eval_results)