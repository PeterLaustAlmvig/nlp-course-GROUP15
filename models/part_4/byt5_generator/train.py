import evaluate
import torch
import math

import numpy as np
import torch.nn as nn

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

    # If predictions are token IDs, decode them
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # sometimes returned as (logits, …)

    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 with pad_token_id before decoding labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ---------------------------
    # Character-level BLEU
    # ---------------------------
    preds_char = [list(p) for p in preds]              # list of chars
    refs_char = [[list(r)] for r in refs]              # wrap each ref in a list
    bleu = metric_bleu.compute(predictions=preds_char, references=refs_char)["bleu"]

    # ---------------------------
    # Character-level ROUGE-L
    # ---------------------------
    # ROUGE expects "tokens" as space-separated strings, so join chars
    preds_rouge = [' '.join(list(p)) for p in preds]
    refs_rouge = [' '.join(list(r)) for r in refs]
    rouge_l = metric_rouge.compute(predictions=preds_rouge, references=refs_rouge)["rougeL"]

    # ---------------------------
    # BERTScore (works at text level)
    # ---------------------------
    bertscore = metric_bertscore.compute(predictions=preds, references=refs, lang="te")
    bert_f1 = np.mean(bertscore["f1"])

    return {
        "bleu": bleu,
        "rougeL": rouge_l,
        "bertscore_f1": bert_f1
    }


def train_seq2seq(model, train_set, val_set, tokenizer, epochs, output_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    no_steps_pr_eval = max(1, ((len(train_set) / 2) * epochs) // 20)

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