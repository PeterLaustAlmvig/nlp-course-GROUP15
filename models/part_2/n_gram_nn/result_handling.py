import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tabulate import tabulate

from dataset import prepare_dataset_loaders, calculate_max_context_window
from model import SentenceModel
from train import *
from seeding import enforce_reproducibility
from visualisation import *
from tuning import calculate_possible_windows

REPLACE_FREQ_KEY = "replace_type"
TOP_FRACTION_KEY = "top_fraction"
EMBEDDING_DIM_KEY = "embedding_dim"
HIDDEN_DIM_KEY = "hidden_dim"
CONTEXT_WINDOW_NORM_KEY = "context_window_norm"
CONTEXT_WINDOW_KEY = "context_window"
REPLACE_FRAC_KEY = "replace_frac"
LOSS_KEY = "loss"

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    info_logger("CUDA detected")
    
def best_configuration(result_files, result_folder):
    dfs = []
    for lang, file in result_files.items():
        df = pd.read_csv(f"{result_folder}/{file}")
        df["language"] = lang
        df = normalize_context_windows(df)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    param_cols = ["replace_type", "top_fraction", "embedding_dim", "hidden_dim", "context_window_norm", "replace_frac"]

    # Pivot to have each language's loss as a separate column
    pivot_df = combined.pivot_table(
        index=param_cols,
        columns="language",
        values="loss"
    ).reset_index()

    # Compute mean and std across the three languages (row-wise)
    pivot_df["mean_loss"] = pivot_df[["ko", "ar", "te"]].mean(axis=1)
    pivot_df["std_loss"] = pivot_df[["ko", "ar", "te"]].std(axis=1)

    # Keep only configurations that have loss for all languages
    pivot_df = pivot_df.dropna(subset=["ko", "ar", "te"])
    return pivot_df, dfs
    
def normalize_context_windows(df):
    normalized = df.copy()
    
    normalized['context_window_norm'] = None
    
    languages = normalized['language'].unique()

    for lang in languages:
        lang_mask = normalized['language'] == lang
        windows = normalized.loc[lang_mask, 'context_window']
        
        max_window = windows.max()
        # exclude 1 when looking for min
        min_window = windows[windows > 1].min()
        
        normalized.loc[lang_mask, 'context_window_norm'] = windows.apply(
            lambda x: 'max' if x == max_window else ('min' if x == min_window else None)
        )

    normalized = normalized.drop(columns=['context_window'])
    normalized = normalized.dropna(subset=['context_window_norm']).reset_index(drop=True)
    
    return normalized


def concat_languages(result_files, result_folder):
    dfs = []
    for lang, file in result_files.items():
        df = pd.read_csv(f"{result_folder}/{file}")
        df["language"] = lang
        df = normalize_context_windows(df)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    param_cols = ["replace_type", "top_fraction", "embedding_dim", "hidden_dim", "context_window_norm", "replace_frac"]

    # Pivot to have each language's loss as a separate column
    pivot_df = combined.pivot_table(
        index=param_cols,
        columns="language",
        values="loss"
    ).reset_index()

    # Compute mean and std across the three languages (row-wise)
    pivot_df["mean_loss"] = pivot_df[["ko", "ar", "te"]].mean(axis=1)
    pivot_df["std_loss"] = pivot_df[["ko", "ar", "te"]].std(axis=1)

    # Keep only configurations that have loss for all languages
    pivot_df = pivot_df.dropna(subset=["ko", "ar", "te"])
    return pivot_df, dfs

def calculate_best_configuration(result_folder, language="cross"):
    if language == "en":
        result_file = "en_tuning_results.csv"
        best_configs = pd.read_csv(f"{result_folder}/{result_file}")
        best_configs = best_configs.sort_values("loss").head(5)
    else:
        result_files = {
            "ko": "ko_tuning_results.csv",
            "ar": "ar_tuning_results.csv",
            "te": "te_tuning_results.csv"
        }

        best_configs, language_dfs = best_configuration(result_files, result_folder)

        # Sort by mean_loss
        best_configs = best_configs.sort_values("mean_loss").head(5)
        
    best_configs.to_csv(f"{result_folder}/{language}_top_5_configs.csv")

    # Print the best results
    print("=== Top 5 Language Configurations ===")
    print(tabulate(best_configs, headers='keys', tablefmt='fancy_grid', showindex=False))

    best_config = best_configs.head(1).iloc[0]  # This gives a Series instead of a dict

    embed_dim = best_config[EMBEDDING_DIM_KEY]
    hidden_dim = best_config[HIDDEN_DIM_KEY]
    top_frac = best_config[TOP_FRACTION_KEY]
    replace_type = best_config[REPLACE_FREQ_KEY]
    replace_fraction = best_config[REPLACE_FRAC_KEY]
    context_window_norm = best_config[CONTEXT_WINDOW_KEY if language == "en" else CONTEXT_WINDOW_NORM_KEY]

    print("\n=== Best Model Configuration ===")
    print(f"Embedding dimension       : {embed_dim}")
    print(f"Hidden dimension          : {hidden_dim}")
    print(f"Top fraction              : {top_frac}")
    print(f"Replacement type          : {'Frequent Words' if replace_type else 'Infrequent words'}")
    print(f"Replacement fraction      : {replace_fraction}")
    print(f"Context window normalization: {context_window_norm}")
    print("="*35 + "\n")
    
    return embed_dim, hidden_dim, top_frac, replace_type, replace_fraction, context_window_norm

def get_context_window(language, context_window_norm):
    context_window = calculate_max_context_window(language)
    if context_window_norm == "min":
        context_windows = calculate_possible_windows(context_window)
        context_window = min(window for window in context_windows if window > 1)
    return context_window

def generate_model(language, result_folder):
    embed_dim, hidden_dim, top_frac, replace_type, replace_fraction, context_window_norm = calculate_best_configuration(result_folder, language)

    enforce_reproducibility(42)

    model_name = f"model_{language}.model"
    context_window = get_context_window(language, context_window_norm) if language != "en" else context_window_norm
    batch_size = 16
    epochs = 10

    train_loader, val_loader, test_loader, vocab = prepare_dataset_loaders(language, top_frac, replace_type, batch_size, replace_fraction, context_window)

    model = SentenceModel(len(vocab), embed_dim, hidden_dim, context_window).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.NLLLoss()

    val_losses, val_accs, val_pp, val_topk = train(model, device, train_loader, val_loader, optimizer, criterion, epochs)

    plot_two_curves(val_losses, val_accs, save_path=f"{result_folder}/losses_{language}.pdf")
    plot_one_curve(val_pp, "Validation Perplexity", "Perplexity", save_path=f"{result_folder}/perplexity_{language}.pdf")
    plot_one_curve(val_topk, "Validation Top k Accuracy", "Accuracy", save_path=f"{result_folder}/accuracy_{language}.pdf")

    torch.save(model.state_dict(), f"{result_folder}/{model_name}")

    _, _, _, _ = evaluate(model, device, test_loader, criterion, top_k=5)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run n-gram NN hyperparameter tuning for a language")
    parser.add_argument(
        "--language", 
        type=str, 
        required=True,
        help="Language code to run tuning on, e.g., 'en', 'te', 'ar', or 'ko'"
    )
    args = parser.parse_args()
    
    language = args.language
    result_folder = "n_gram_nn_results"
    
    generate_model(language, result_folder)
    