import math
import os
import random
import argparse
import sys

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import *

from utils.datasets import *
from utils.models import *
from utils.train_eval import *
from utils.visualisation import *

import logging

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def hyperparameter_tuning(train_dataset, train_dataloader, best_results_path):
    embedding_dims = [16, 32, 64, 128]
    hidden_sizes = [32, 64, 128, 256]
    num_layerss = [1, 5, 10]
    drop_outs = [0.0, 0.25, 0.5]
    lr = 0.01
    max_epochs = 500

    last_best_acc = float("-inf")
    all_best_accs = []

    for embedding_dim in embedding_dims:
        for hidden_size in hidden_sizes:
            for num_layers in num_layerss:
                for drop_out in drop_outs:
                    model = LSTMModel(device, embedding_dim, hidden_size, len(train_dataset.vocab), num_layers, drop_out)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = SGD(model.parameters(), lr=lr)

                    train_losses, accuracies = sentence_train(device, epochs=max_epochs, model=model, dataloader=train_dataloader, 
                                                    optimizer=optimizer, criterion=criterion, padding_token_idx=train_dataset.pad_idx, logger=logger, print_interval=max_epochs+1)

                    best_acc = max(accuracies)
                    all_best_accs.append({'embedding_dim': embedding_dim, 'hidden_size': hidden_size, 'num_layers': num_layers, 'drop_out': drop_out, 'best_acc': best_acc})
                    logger.info(f"[INFO] Trained with embedding_dim={embedding_dim}, hidden_size={hidden_size}, num_layers={num_layers}, drop_out={drop_out}, and got best_acc={best_acc:.4f} after {len(accuracies)} epochs")
                    if best_acc > last_best_acc:
                        last_best_acc = best_acc
                        logger.info(f"[SUCCESS] New best setting found with embedding_dim={embedding_dim}, hidden_size={hidden_size}, num_layers={num_layers}, drop_out={drop_out}, best_acc={best_acc:.4f}")
                        plot_loss_and_acc(train_losses, accuracies, title=f"Train Metrics (emb={embedding_dim}, hid={hidden_size}, layers={num_layers}, drop={drop_out})", 
                                    save_path=f"lstm_results/ko_models/{embedding_dim}_{hidden_size}_{num_layers}_{drop_out}_train_metrics.png")

    all_best_accs = pd.DataFrame(all_best_accs)
    all_best_accs.to_csv(best_results_path, index=False)
    logger.info(f"[INFO] Hyperparameter tuning completed. Results saved to '{best_results_path}'.")

    best_row = all_best_accs.loc[all_best_accs['best_acc'].idxmax()]
    embedding_dim = best_row['embedding_dim']
    hidden_size   = best_row['hidden_size']
    num_layers    = best_row['num_layers']
    drop_out      = best_row['drop_out']
    best_acc      = best_row['best_acc']

    logger.info("\nThe best hyperparameters seem to be:")
    logger.info(f"    embedding_dim: {embedding_dim}")
    logger.info(f"    hidden_size:   {hidden_size}")
    logger.info(f"    num_layers:    {num_layers}")
    logger.info(f"    drop_out:      {drop_out}")
    logger.info(f"    best_acc:      {best_acc}")

    return embedding_dim, hidden_size, num_layers, drop_out

def find_hyperparameters_lstm(result_folder, train_dataset, train_dataloader):
    embedding_dim, hidden_size, num_layers, drop_out = hyperparameter_tuning(train_dataset, train_dataloader, f"{result_folder}/lstm_hyperparameter_tuning_results.csv")
    return embedding_dim, hidden_size, num_layers, drop_out

def train_hyper_model(embedding_dim, hidden_size, num_layers, drop_out, df_train_ko):
    batch_size = 32
    train_dataset = WordDataset(df_train_ko["question"])
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    result_folder = "lstm_results/ko_models/"

    model = LSTMModel(device, embedding_dim, hidden_size, len(train_dataset.vocab), num_layers, drop_out)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    train_losses, accuracies = sentence_train(device, epochs=100, model=model, dataloader=train_dataloader, 
                                            optimizer=optimizer, criterion=criterion, print_interval=10)
    plot_loss_and_acc(train_losses, accuracies, title=f"Train Metrics (emb={embedding_dim}, hid={hidden_size}, layers={num_layers}, drop={drop_out})", 
                    save_path=f"{result_folder}/{embedding_dim}_{hidden_size}_{num_layers}_{drop_out}_final_train_metrics.png")
    torch.save(model.state_dict(), f"{result_folder}/ko_lstm_model.pth")
    logger.info("[INFO] Model training complete and saved.")

def validate_hyper_model(model, val_dataloader):
    val_accuracy = sentence_validate(device, model, val_dataloader)
    logger.info(f"[INFO] Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning.")
    parser.add_argument("--language", type=str, default="ko", help="Language to create model for: 'ko', 'te', or 'ar'")
    parser.add_argument("--column", type=str, default="question", help="Column in the dataset to use for training (default: 'question')")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits = {'train': 'train.parquet', 'validation': 'validation.parquet'}
    df_train = pd.read_parquet("hf://datasets/coastalcph/tydi_xor_rc/" + splits["train"], engine='fastparquet')
    df_val = pd.read_parquet("hf://datasets/coastalcph/tydi_xor_rc/" + splits["validation"], engine='fastparquet')
    logger.info(f"[INFO] Train size: {len(df_train)}, Validation size: {len(df_val)}")
    
    if args.language in ['ko', 'te', 'ar']:
        df_train = df_train[df_train['lang'] == args.language]
        df_val = df_val[df_val['lang'] == args.language]

    batch_size = 32
    train_dataset = WordDataset(df_train[args.column])
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    val_dataset = WordDataset(df_val[args.column], vocab=train_dataset.vocab, word_to_idx=train_dataset.word_to_idx, idx_to_word=train_dataset.idx_to_word)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    
    result_folder = f"lstm_results/{args.language}_models/"
    os.makedirs(result_folder, exist_ok=True)
    
    embedding_dim, hidden_size, num_layers, drop_out = find_hyperparameters_lstm(result_folder, train_dataset, train_dataloader)
    train_hyper_model(embedding_dim, hidden_size, num_layers, drop_out, df_train)
    validate_hyper_model(torch.load(f"{result_folder}/{args.language}_lstm_model.pth"), val_dataloader)
