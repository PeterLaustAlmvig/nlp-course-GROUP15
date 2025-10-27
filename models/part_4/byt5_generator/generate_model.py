import os
import pandas as pd

from transformers import AutoModelForSeq2SeqLM
from logger import divider_logger, info_logger
from seeding import enforce_reproducibility
from dataset import prepare_datasets
from train import train_seq2seq, evaluate_seq2seq

def save_results(result_dir, step_logs, test_logs):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save per-step logs for visualization
    step_file = os.path.join(result_dir, f"train_metrics_per_step.csv")
    pd.DataFrame(step_logs).to_csv(step_file, index=False)
    info_logger(f"Saved training step logs at: {step_file}")

    # Save test evaluation logs
    test_file = os.path.join(result_dir, f"test_evaluation_metrics.json")
    pd.DataFrame(test_logs).to_json(test_file)
    info_logger(f"Saved test evaluation metrics at: {test_file}")
    divider_logger()

def save_model(model, tokenizer, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    info_logger(f"Model and tokenizer saved at {model_dir}")

if __name__ == "__main__":
    model_name = "google/byt5-base"
    output_dir = f"byt5_training_results"
    epochs = 20

    enforce_reproducibility(42)

    # ==== PREPARE DATASETS ====
    train_set, val_set, test_set, tokenizer = prepare_datasets(model_name)
    
    # ==== MODEL ====
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # ==== TRAIN MODEL ====
    model, tokenizer, step_logs = train_seq2seq(
        model, train_set, val_set, tokenizer, epochs, output_dir
    )
    
    # ==== EVALUATE MODEL ====
    test_logs = evaluate_seq2seq(model, tokenizer, test_set, output_dir)
    
    # ==== SAVE RESULTS ====
    save_results(output_dir, step_logs, test_logs)
    
    # ==== SAVE MODEL ====
    save_model(model, tokenizer, output_dir)
