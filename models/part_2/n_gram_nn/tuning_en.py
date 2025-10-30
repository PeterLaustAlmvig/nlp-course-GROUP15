import itertools
import math
import os
import argparse

import torch.nn as nn
import torch.optim as optim
import pandas as pd

from dataset import prepare_dataset_loaders, calculate_max_context_window
from model import SentenceModel
from train import *
from seeding import enforce_reproducibility
from logger import divider_logger, info_logger

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    info_logger("CUDA detected")

def calculate_possible_windows(windows):
    last_window = windows[-1]
    if last_window <= 2:
        return windows
    
    new_window = math.floor(last_window / 2)
    return calculate_possible_windows(windows + [new_window])
    
def save_parameters_to_csv(parameters, output_folder, filename="experiment_results.csv"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    df = pd.DataFrame(parameters)
    df.to_csv(f"{output_folder}/{filename}", index=False)
    info_logger(f"Saved {len(df)} experiment results to '{filename}'")

def parameter_tuning(language, embedding_dim_options, hidden_dim_options, top_fraction_options, replace_freq_options, replace_fraction_options, context_window_options):
    # -------------------------
    # Hyperparameter options
    # -------------------------
    batch_size = 64
    epochs = 5
    
    parameters = {
        "replace_type": [],
        "top_fraction": [],
        "embedding_dim": [],
        "hidden_dim": [],
        "context_window": [],
        "replace_frac": [],
        "loss": []
    }
    # -------------------------
    # Loop over all combinations
    # -------------------------
    for embed_dim, hidden_dim, top_frac, replace_type, context_window, replace_fraction in itertools.product(
            embedding_dim_options,
            hidden_dim_options,
            top_fraction_options,
            replace_freq_options,
            context_window_options,
            replace_fraction_options):

        enforce_reproducibility(42)
        
        info_logger(f"\n=== Running combination ===")
        info_logger(f"Embed: {embed_dim}, Hidden: {hidden_dim}, Top fraction: {top_frac}, "
            f"Replace: {replace_type}, Context: {context_window}")
        divider_logger()

        # -------------------------
        # Step 1: Create dataloaders
        # -------------------------
        train_loader, val_dataloader, _, vocab = prepare_dataset_loaders(language, top_frac, replace_type, batch_size, replace_fraction, context_window)
        divider_logger()
        # -------------------------
        # Step 4: Create model
        # -------------------------
        model = SentenceModel(len(vocab), embed_dim, hidden_dim, context_window).to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        criterion = nn.NLLLoss()

        # -------------------------
        # Step 5: Train model (few epochs for hyperparameter estimate)
        # -------------------------
        val_losses, _, _, _ = train(model, device, train_loader, val_dataloader, optimizer, criterion, epochs)
        
        # -------------------------
        # Step 6: Save current parameters and loss
        # -------------------------
        parameters["replace_type"].append(replace_type)
        parameters["top_fraction"].append(top_frac)
        parameters["embedding_dim"].append(embed_dim)
        parameters["hidden_dim"].append(hidden_dim)
        parameters["context_window"].append(context_window)
        parameters["replace_frac"].append(replace_fraction)
        parameters["loss"].append(val_losses[-1])
        
        print(f"Current combination results:")
        print(f"{'Replace':<12} | {'TopFrac':<8} | {'EmbedDim':<8} | {'HiddenDim':<9} | {'ContextWin':<11} | {'ReplaceFrac':<12} | {'Loss':<8}")
        print("-"*80)
        print(f"{replace_type!s:<12} | {top_frac:<8} | {embed_dim:<8} | {hidden_dim:<9} | {context_window:<11} | {replace_fraction:<12} | {val_losses[-1]:<8.4f}")

        divider_logger()
        divider_logger()
    
    return parameters
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run n-gram NN hyperparameter tuning for a language")
    parser.add_argument("--language", type=str, required=True, 
                        help="Language code to run tuning on, e.g., 'en', 'te', 'ar', or 'ko'")
    parser.add_argument("--embedding_dims", type=str, default="64,128,256",
                        help="Comma-separated list of embedding dimensions")
    parser.add_argument("--hidden_dims", type=str, default="64,128,256",
                        help="Comma-separated list of hidden dimensions")
    parser.add_argument("--top_fractions", type=str, default="0.01,0.05,0.1",
                        help="Comma-separated list of top fraction options")
    parser.add_argument("--replace_types", type=str, default="True,False",
                        help="Comma-separated list of replace frequency options")
    parser.add_argument("--replace_fractions", type=str, default="0.01,0.05,0.1",
                        help="Comma-separated list of replace fractions")
    parser.add_argument("--context_windows", type=int, default="auto",
                        help="context window to use")
    args = parser.parse_args()
    
    language = args.language
    embedding_dim_options = [int(x) for x in args.embedding_dims.split(",")]
    hidden_dim_options = [int(x) for x in args.hidden_dims.split(",")]
    top_fraction_options = [float(x) for x in args.top_fractions.split(",")]
    replace_freq_options = [x.lower() == "true" for x in args.replace_types.split(",")]
    replace_fraction_options = [float(x) for x in args.replace_fractions.split(",")]
    context_window_options = [int(args.context_windows)]
    
    # Run parameter tuning for the selected language
    parameters = parameter_tuning(language, embedding_dim_options, hidden_dim_options, top_fraction_options, replace_freq_options, replace_fraction_options, context_window_options)
    
    # Save results
    folder = "n_gram_nn_results"
    filename = f"{language}_tuning_results.csv"
    save_parameters_to_csv(parameters, folder, filename)