# # Split train and validation sets into new dataframes for ar, ko and te based on the lang column
# # Test sets
# df_train_ar = df_train[df_train['lang'] == 'ar']
# df_train_ko = df_train[df_train['lang'] == 'ko']
# df_train_te = df_train[df_train['lang'] == 'te']

# # Validation sets
# df_val_ar = df_val[df_val['lang'] == 'ar']
# df_val_ko = df_val[df_val['lang'] == 'ko']
# df_val_te = df_val[df_val['lang'] == 'te']


import numpy as np, re
#from datasets import load_dataset

# Use existing df_val if present; else load from datasets
# try:
#     df_val  # noqa: F821f
# except NameError:
#     ds = load_dataset("coastalcph/tydi_xor_rc")
#     df_val = ds["validation"].to_pandas()

def char_ngrams(s, n=3):
    s = re.sub(r"\s+", " ", s).lower().strip()
    if not s: return set()
    if len(s) < n: return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def overlap_score(q, c, n=3):
    qg = char_ngrams(q, n)
    if not qg: return 0.0
    cg = char_ngrams(c, n)
    if not cg: return 0.0
    return len(qg & cg) / len(qg)

N = 3
THRESH = 0.03
langs = ["ar", "ko", "te"]

for lg in langs:
    sub = df_val[df_val["lang"] == lg]
    preds = [overlap_score(q, ctx, n=N) >= THRESH for q, ctx in zip(sub["question"], sub["context"])]
    gold = sub["answertable"].astype(bool).tolist()
    acc = np.mean([p == g for p, g in zip(preds, gold)])
    tp = sum(1 for p, g in zip(preds, gold) if p and g)
    tn = sum(1 for p, g in zip(preds, gold) if (not p) and (not g))
    fp = sum(1 for p, g in zip(preds, gold) if p and (not g))
    fn = sum(1 for p, g in zip(preds, gold) if (not p) and g)
    print(f"Language={lg}  n={N}  threshold={THRESH:.4f}  "
          f"accuracy={acc:.4f}  support={len(sub)}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")

