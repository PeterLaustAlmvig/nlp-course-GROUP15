import itertools
import os
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
)

from dataset import prepare_datasets
from train import train_binary, evaluate_binary
from visualisation import plot_two_curves
from logger import divider_logger, info_logger
from seeding import enforce_reproducibility

def save_results(result_dir, language, epoch_logs, test_logs):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    epoch_file = os.path.join(result_dir, f"{language}_train_metrics_per_epoch.csv")
    epoch_df = pd.DataFrame(epoch_logs)
    epoch_df.to_csv(epoch_file)
    info_logger(f"Saved per-epoch metrics at: {epoch_file}")
    
    plot_two_curves(epoch_df["eval_loss"].to_list(), epoch_df["eval_accuracy"].to_list(), save_path=f"{result_dir}/{language}_training_plot.pdf")

    test_file = os.path.join(result_dir, f"{language}_test_evaluation_metrics.csv")
    test_df = pd.DataFrame(test_logs)
    test_df.to_json(test_file)
    info_logger(f"Saved test metrics at: {test_file}")
    divider_logger()
    
def save_tuning_results(result_dir, language, test_logs):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    epoch_file = os.path.join(result_dir, f"{language}_tuning_metrics.csv")
    epoch_df = pd.DataFrame(test_logs)
    epoch_df.to_csv(epoch_file)
    info_logger(f"Saved tuning metrics at: {epoch_file}")
    divider_logger()
    
def save_model(model, tokenizer, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    info_logger(f"Model and tokenizer saved at {model_dir}")
    
def compute_metrics_tuning(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = labels.astype(int)

    accuracy = (preds == labels).mean()

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    true_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "true_accuracy": true_accuracy,
        "false_accuracy": false_accuracy
    }

def sampling_tuning(language, model_name, output_dir):
    # -------------------------
    # Hyperparameter options
    # -------------------------
    over_sampling_options = [0.0, 0.25, 0.5, 1.0]
    under_sampling_options = [0.0, 0.25, 0.5, 1.0]
    parameters = {
        "over_sampling": [],
        "under_sampling": [],
        "accuracy": [],
        "true_accuracy": [],
        "false_accuracy": [],
        "loss": []
    }
    # -------------------------
    # Loop over all combinations
    # -------------------------
    for over_sampling_ratio, under_sampling_ratio in itertools.product(
            over_sampling_options,
            under_sampling_options):
        
        enforce_reproducibility(42)
        
        info_logger(f"\n=== Running combination ===")
        info_logger(f"Over Sampling Ratio: {over_sampling_ratio}, Under Sampling Ratio: {under_sampling_ratio}")
        divider_logger()

        # ==== PREPARE DATASETS ====
        train_set, val_set, test_sets, tokenizer = prepare_datasets(model_name, over_sampling_ratio, under_sampling_ratio, language)

        # ==== MODEL ====
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )

        # ==== TRAIN MODEL ====
        model, tokenizer, _ = train_binary(model, train_set, val_set, tokenizer, 1, output_dir)

        # ==== EVALUATE MODEL ====
        eval_results = evaluate_binary(model, tokenizer, test_sets[language], compute_metrics_tuning)
        
        parameters["over_sampling"].append(over_sampling_ratio)
        parameters["under_sampling"].append(under_sampling_ratio)
        parameters["accuracy"].append(eval_results["eval_accuracy"])
        parameters["true_accuracy"].append(eval_results["eval_true_accuracy"])
        parameters["false_accuracy"].append(eval_results["eval_false_accuracy"])
        parameters["loss"].append(eval_results["eval_loss"])
        divider_logger()
        divider_logger()
    return parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning of model")
    parser.add_argument("--language", type=str, required=True, help="Language code to run tuning on, e.g., 'en', 'te', 'ar', or 'ko'")
    args = parser.parse_args()
    language = args.language
    model_name = "FacebookAI/xlm-roberta-base"
    
    results_dir = f"roberta_classifier_results_{language}"
    
    results = sampling_tuning(language, model_name, results_dir)
    
    save_tuning_results(results_dir, language, results)