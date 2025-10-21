import json
import os
import torch
import argparse
import logging
import sys
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
import evaluate

from logger import divider_logger, info_logger

# ==== METRICS ====
metric_acc = evaluate.load("accuracy")
metric_prec = evaluate.load("precision")
metric_rec = evaluate.load("recall")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "precision": metric_prec.compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall": metric_rec.compute(predictions=preds, references=labels, average="binary")["recall"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }

def train_binary(model, train_set, val_set, tokenizer, epochs, output_dir):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        do_train=True,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    info_logger(f"Starting training of model (answerability classification).")
    trainer.train()
    divider_logger()
    info_logger("Training completed.")

    # Extract only epoch-level logs with loss and accuracy
    epoch_logs = []
    for log in trainer.state.log_history:
        if "epoch" in log:
            epoch_entry = {"epoch": log["epoch"]}
            if "loss" in log:
                epoch_entry["loss"] = log["loss"]
            if "eval_accuracy" in log:
                epoch_entry["accuracy"] = log["eval_accuracy"]
            info_logger(f"Epoch {epoch_entry['epoch']:.1f} - Loss: {epoch_entry.get('loss', 'N/A'):.4f}, Accuracy: {epoch_entry.get('accuracy', 'N/A'):.4f}")
            epoch_logs.append(epoch_entry)
            
    return model, tokenizer, epoch_logs

def evaluate_binary(model, tokenizer, test_set):
    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Trainer only for evaluation
    trainer = Trainer(
        model=model,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    info_logger("Running evaluation...")
    results = trainer.evaluate()

    # Log metrics nicely
    info_logger("Evaluation Results:")
    for k, v in results.items():
        if k.startswith("eval_"):
            info_logger(f"{k}: {v:.4f}")

    return results