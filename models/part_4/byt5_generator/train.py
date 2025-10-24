import evaluate
import torch
import math

import numpy as np
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from transformers import (
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)

from logger import divider_logger, info_logger

# ==== METRICS ====
metric_bleu = evaluate.load("bleu")
metric_rouge = evaluate.load("rouge")
metric_bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    # Replace -100 with pad_token_id so tokenizer can decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    refs  = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Remove any examples where reference is empty
    filtered = [(p, r) for p, r in zip(preds, refs) if r.strip() != ""]
    if len(filtered) == 0:
        # avoid ZeroDivisionError
        return {"bleu": 0.0, "rougeL": 0.0, "bertscore_f1": 0.0}

    preds, refs = zip(*filtered)

    # BLEU expects a list of strings for predictions, list of list of strings for references
    bleu = metric_bleu.compute(
        predictions=list(preds),
        references=[[r] for r in refs]
    )["bleu"]

    rouge_l = metric_rouge.compute(predictions=list(preds), references=list(refs))["rougeL"]

    bertscore_f1 = np.mean(metric_bertscore.compute(
        predictions=list(preds),
        references=list(refs),
        lang="te"
    )["f1"])

    return {
        "bleu": bleu,
        "rougeL": rouge_l,
        "bertscore_f1": bertscore_f1
    }


def train_seq2seq(model, train_set, val_set, tokenizer, epochs, output_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    no_steps_pr_eval = max(1, ((len(train_set) / 2) * epochs) // 10)

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )

    info_logger("Starting seq2seq training for generative QA.")
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


def evaluate_seq2seq(model, tokenizer, test_set, output_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    eval_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=2,
        predict_with_generate=True,  
        do_train=False,
        do_eval=True,
        report_to=[]                    
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        eval_dataset=test_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )

    info_logger("Running evaluation...")
    results = trainer.evaluate()

    info_logger("Evaluation Results:")
    for k, v in results.items():
        if k.startswith("eval_"):
            info_logger(f"{k}: {v}")

    return results