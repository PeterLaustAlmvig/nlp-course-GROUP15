import pandas as pd
from tabulate import tabulate

def get_balanced_top_configs(csv_path, top_k=3):
    df = pd.read_csv(csv_path)

    # Normalize accuracy columns to [0, 1] for fair comparison
    for col in ["accuracy", "true_accuracy", "false_accuracy"]:
        df[col + "_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Compute a "balance score" that rewards configs good on all accuracies
    # Using geometric mean instead of arithmetic mean to penalize imbalance
    df["balance_score"] = (
        (df["accuracy_norm"] * df["true_accuracy_norm"] * df["false_accuracy_norm"]) ** (1/3)
    )

    # Sort and pick the best ones
    top_configs = df.sort_values("balance_score", ascending=False).head(top_k)

    print("=== Top Balanced Sampling Configurations ===")
    print(tabulate(
        top_configs[[
            "over_sampling", "under_sampling",
            "accuracy", "true_accuracy", "false_accuracy",
            "loss", "balance_score"
        ]],
        headers="keys", tablefmt="fancy_grid", showindex=False
    ))
    
    best_config = top_configs.head(1).iloc[0]
    return (best_config["over_sampling"], best_config["under_sampling"])