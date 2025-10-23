import evaluate
import torch
import math

import numpy as np
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from transformers import (
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)

from logger import divider_logger, info_logger

# ==== METRICS ====
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")
metric_bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred, tokenizer, max_target_length=128):
    input_ids, labels = eval_pred

    # Decode labels
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Generate predictions
    outputs = model.generate(
        input_ids=torch.tensor(input_ids).to(model.device),
        max_length=max_target_length
    )
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Compute metrics
    bleu = metric_bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    rouge = metric_rouge.compute(predictions=preds, references=refs)["rougeL"]
    bertscore = metric_bertscore.compute(predictions=preds, references=refs, lang="te")["f1"].mean()

    return {
        "bleu": bleu,
        "rougeL": rouge,
        "bertscore": bertscore
    }


def train_seq2seq(model, train_set, val_set, tokenizer, epochs, output_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    no_steps_pr_eval = max(1, ((len(train_set) / 8) * epochs) // 10)
    print(no_steps_pr_eval)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=epochs,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=no_steps_pr_eval,
        eval_strategy="steps",
        eval_steps=no_steps_pr_eval,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,  # Important for seq2seq metrics
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )

    info_logger("Starting seq2seq training for generative QA.")
    trainer.train()
    divider_logger()
    info_logger("Training completed.")

    return model, tokenizer

    info_logger("Starting training of model (answerability classification).")
    trainer.train()
    divider_logger()
    info_logger("Training completed.")

    # Extract per-step logs for visualization
    step_logs = []
    for log in trainer.state.log_history:
        # Save logged eval steps
        if "eval_loss" in log:
            step_logs.append(log)
    
    return model, tokenizer, step_logs


def evaluate_seq2seq(model, tokenizer, test_set):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        predict_with_generate=True
    )

    info_logger("Running evaluation...")
    results = trainer.evaluate()

    info_logger("Evaluation Results:")
    for k, v in results.items():
        if k.startswith("eval_"):
            info_logger(f"{k}: {v}")

    return results