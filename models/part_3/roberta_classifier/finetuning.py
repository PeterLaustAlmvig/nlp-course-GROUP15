import json
import os
import torch
import argparse
import logging
import sys
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
)

from dataset import prepare_datasets
from train import train_binary, evaluate_binary
from visualisation import plot_two_curves
from logger import divider_logger, info_logger

def save_results(result_dir, language, epoch_logs, test_logs):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    epoch_file = os.path.join(result_dir, f"{language}_train_metrics_per_epoch.csv")
    epoch_df = pd.DataFrame(epoch_logs)
    epoch_df.to_csv(epoch_file)
    info_logger(f"Saved per-epoch metrics at: {epoch_file}")
    
    plot_two_curves(epoch_df["loss"].to_list(), epoch_df["accuracy"].to_list(), save_path=f"{result_dir}/{language}_training_plot.pdf")

    test_file = os.path.join(result_dir, f"{language}_test_evaluation_metrics.json")
    test_df = pd.DataFrame(test_logs)
    test_df.to_json(test_file)
    info_logger(f"Saved test metrics at: {test_file}")
    divider_logger()
    
def save_model(model, tokenizer, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    info_logger(f"Model and tokenizer saved at {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning of model")
    parser.add_argument("--language", type=str, required=True, help="Language code to run tuning on, e.g., 'en', 'te', 'ar', or 'ko'")
    parser.add_argument("--model", type=str, required=True, help="Model to finetune")
    args = parser.parse_args()
    language = args.language
    model_name = args.model
    
    results_dir = "roberta_classifier"

    # ==== PREPARE DATASETS ====
    train_set, val_set, test_set, tokenizer = prepare_datasets()
    divider_logger()
    
    # ==== MODEL ====
    info_logger(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # binary classification
    )
    divider_logger()
    
    # ==== TRAIN MODEL ====
    epoch_history = train_binary(model, train_set, val_set, tokenizer, 10, results_dir)
    
    # ==== EVALUATE MODEL ====
    eval_results = evaluate_binary(model, tokenizer, test_set)
    
    # ==== SAVE RESULTS & MODEL ====
    save_results(results_dir, language, epoch_history, eval_results)
    save_model(model, tokenizer, f"{results_dir}/{language}_finetuned")    