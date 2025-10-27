import evaluate
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from logger import divider_logger, info_logger

# ==========================================
# Metric + Model Setup
# ==========================================
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
metric_rouge = evaluate.load("rouge")
metric_bertscore = evaluate.load("bertscore")
metric_bleu = evaluate.load("bleu")

# ===============================
# F1 Score (character-level)
# ===============================
def char_level_f1(pred, ref):
    pred_chars = set(pred.strip())
    ref_chars = set(ref.strip())
    if not pred_chars or not ref_chars:
        return 0.0
    tp = len(pred_chars & ref_chars)
    precision = tp / len(pred_chars)
    recall = tp / len(ref_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ==========================================
# Compute Metrics Function
# ==========================================
def compute_metrics(eval_pred, tokenizer, answerable_flags):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    if isinstance(predictions, torch.Tensor):
        predictions = torch.argmax(predictions, dim=-1)
        
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    answerable_flags = np.array(answerable_flags, dtype=bool)

    if len(answerable_flags) != len(preds):
        raise ValueError(f"Length mismatch: {len(answerable_flags)} flags vs {len(preds)} predictions")

    # ===============================
    # F1 Score
    # ===============================
    f1_scores = np.array([char_level_f1(p, r) for p, r in zip(preds, refs)])
    f1_overall = float(np.mean(f1_scores))
    f1_answerable = float(np.mean(f1_scores[answerable_flags])) if np.any(answerable_flags) else np.nan
    f1_unanswerable = float(np.mean(f1_scores[~answerable_flags])) if np.any(~answerable_flags) else np.nan

    # ===============================
    # BERTScore
    # ===============================
    bert_scores = metric_bertscore.compute(predictions=preds, references=refs, lang="te")
    bert_f1 = np.array(bert_scores["f1"])

    bert_overall = float(np.mean(bert_f1))
    bert_answerable = float(np.mean(bert_f1[answerable_flags])) if np.any(answerable_flags) else np.nan
    bert_unanswerable = float(np.mean(bert_f1[~answerable_flags])) if np.any(~answerable_flags) else np.nan

    # ===============================
    # Semantic Similarity
    # ===============================
    pred_embeds = embed_model.encode(preds, convert_to_tensor=True)
    ref_embeds = embed_model.encode(refs, convert_to_tensor=True)
    cosine_scores = util.cos_sim(pred_embeds, ref_embeds).diagonal().cpu().numpy()

    semantic_overall = float(np.mean(cosine_scores))
    semantic_answerable = float(np.mean(cosine_scores[answerable_flags])) if np.any(answerable_flags) else np.nan
    semantic_unanswerable = float(np.mean(cosine_scores[~answerable_flags])) if np.any(~answerable_flags) else np.nan

    # ===============================
    # Return all metrics
    # ===============================
    return {
        # F1 Score
        "f1_overall": f1_overall,
        "f1_answerable": f1_answerable,
        "f1_unanswerable": f1_unanswerable,

        # BERTScore
        "bertscore_overall": bert_overall,
        "bertscore_answerable": bert_answerable,
        "bertscore_unanswerable": bert_unanswerable,

        # Semantic Similarity
        "semantic_overall": semantic_overall,
        "semantic_answerable": semantic_answerable,
        "semantic_unanswerable": semantic_unanswerable,
    }


# ==========================================
# Training Function
# ==========================================
def train_seq2seq(model, train_set, val_set, tokenizer, epochs, output_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    no_steps_pr_eval = max(1, ((len(train_set) / 2) * epochs) // 20)

    # Extract the answerability flags from validation set
    answerable_flags = np.array(val_set["answerable"], dtype=bool)

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
        metric_for_best_model="loss",
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        report_to=[]
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, answerable_flags)
    )

    info_logger("Starting seq2seq training for generative QA.")
    trainer.train()
    divider_logger()
    info_logger("Training completed.")

    # Extract per-step logs for visualization
    step_logs = [log for log in trainer.state.log_history if "eval_loss" in log]

    return model, tokenizer, step_logs


# ==========================================
# Evaluation Function
# ==========================================
def evaluate_seq2seq(model, tokenizer, test_set, output_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    answerable_flags = np.array(test_set["answerable"], dtype=bool)

    eval_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
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
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, answerable_flags)
    )

    info_logger("Running evaluation...")
    results = trainer.evaluate()

    info_logger("Evaluation Results:")
    actual_results = {}
    for k, v in results.items():
        if k.startswith("eval_"):
            info_logger(f"{k}: {v}")
            actual_results[k] = v

    return actual_results
