import evaluate
import torch
import math

import numpy as np
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from transformers import (
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments
)

from logger import divider_logger, info_logger

# ==== METRICS ====
metric_acc = evaluate.load("accuracy")
metric_prec = evaluate.load("precision")
metric_rec = evaluate.load("recall")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Accuracy
    accuracy = metric_acc.compute(predictions=preds, references=labels)["accuracy"]

    # Plain precision, recall, F1 (no averaging argument)
    precision = metric_prec.compute(predictions=preds, references=labels)["precision"]
    recall = metric_rec.compute(predictions=preds, references=labels)["recall"]
    f1 = metric_f1.compute(predictions=preds, references=labels)["f1"]

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_flat = cm.flatten().tolist()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm_flat
    }
    
class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if hasattr(self, "train_dataset") and "label" in self.train_dataset.column_names:
            labels = np.array(self.train_dataset["label"])
            counts = np.bincount(labels)
            
            self.class_weights = 1.0 / counts
            self.class_weights = self.class_weights / self.class_weights.sum()
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)
        else:
            raise ValueError("WeightedTrainer requires a train_dataset to compute class weights")
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_binary(model, train_set, val_set, tokenizer, epochs, output_dir):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    no_steps_pr_eval = max(1, ((len(train_set) / 8) * epochs) // 10)
    print(no_steps_pr_eval)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=no_steps_pr_eval,
        eval_strategy="steps",
        eval_steps=no_steps_pr_eval,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        do_train=True,
        do_eval=True
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

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


def evaluate_binary(model, tokenizer, test_set, compute_metrics_fn=compute_metrics):
    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Trainer only for evaluation
    trainer = Trainer(
        model=model,
        eval_dataset=test_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    info_logger("Running evaluation...")
    results = trainer.evaluate()

    # Log metrics nicely
    info_logger("Evaluation Results:")
    for k, v in results.items():
        if k.startswith("eval_"):
            info_logger(f"{k}: {v}")

    return results