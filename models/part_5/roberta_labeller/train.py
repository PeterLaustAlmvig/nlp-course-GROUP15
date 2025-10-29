import numpy as np

from sklearn.metrics import confusion_matrix
from transformers import (
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments
)

from logger import divider_logger, info_logger

def confusion_matrix_manual(preds, labels):
    TP = FP = TN = FN = 0

    for true_seq, pred_seq in zip(labels, preds):
        true_seq = list(true_seq)
        pred_seq = list(pred_seq)
        true_has_answer = any(label == 1 for label in true_seq)
        pred_has_answer = any(label == 1 for label in pred_seq)

        if true_has_answer and pred_has_answer:
            if true_seq == pred_seq:
                TP += 1
            else:
                FN += 1  # mismatched spans count as wrong
        elif true_has_answer and not pred_has_answer:
            FN += 1
        elif not true_has_answer and pred_has_answer:
            FP += 1
        else:
            TN += 1

    return TP, FN, TN, FP

def compute_metrics(eval_pred, answerable_flags):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=2)
    
    pred_answerable = [True if 1 in seq else False for seq in preds]

    # Mask out ignored tokens (-100)
    mask = labels != -100
    preds = [p[m] for p, m in zip(preds, mask)]
    labels = [l[m] for l, m in zip(labels, mask)]
    
    TP, FN, TN, FP = confusion_matrix_manual(preds, labels)
    
    total = TP + FN + TN + FP
    accuracy = (TP + TN) / total if total > 0 else 0.0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Answerable True False Scores
    cm = confusion_matrix(answerable_flags, pred_answerable, labels=[True, False])
    
    # Accuracy per answerable class
    # Accuracy = TP / (TP + FN) for each class
    acc_true = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    acc_false = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    cm = cm.flatten().tolist()

    return {
        "accuracy": accuracy,
        "f1": f1,
        "accuracy_answerable_true": acc_true,
        "accuracy_answerable_false": acc_false,
        "cm": cm
    }


def train_binary(model, train_set, val_set, tokenizer, epochs, output_dir):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    no_steps_pr_eval = max(1, ((len(train_set) / 8) * epochs) // 10)
    
    answerable_flags = np.array(val_set["answerable"], dtype=bool)

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
        metric_for_best_model="f1",
        greater_is_better=True,
        do_train=True,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, answerable_flags),
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


def evaluate_binary(model, tokenizer, test_set):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    answerable_flags = np.array(test_set["answerable"], dtype=bool)

    # Trainer only for evaluation
    trainer = Trainer(
        model=model,
        eval_dataset=test_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, answerable_flags),
    )

    info_logger("Running evaluation...")
    results = trainer.evaluate()

    # Log metrics nicely
    info_logger("Evaluation Results:")
    for k, v in results.items():
        if k.startswith("eval_"):
            info_logger(f"{k}: {v}")

    return results