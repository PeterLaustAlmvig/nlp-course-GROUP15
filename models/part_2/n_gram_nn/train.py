import tqdm
import torch
import math

from logger import divider_logger, info_logger

def train(model, device, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=10):
    model.to(device)
    
    val_losses, val_accs, val_pp, val_topk = [], [], [], []
    last_perplexity = math.inf
    
    for _ in tqdm.tqdm(range(num_epochs)):
        # ---- Training phase (silent) ----
        model.train()
        for context, target in train_dataloader:
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            log_probs = model(context)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()

        # ---- Validation phase ----
        avg_loss, perplexity, accuracy, topk_accuracy = evaluate(model, device, val_dataloader, criterion, print_result=False)
        val_losses.append(avg_loss)
        val_accs.append(accuracy)
        val_pp.append(perplexity)
        val_topk.append(topk_accuracy)
        if perplexity > last_perplexity:
            return val_losses, val_accs, val_pp, val_topk
        else:
            last_perplexity = perplexity
            
    return val_losses, val_accs, val_pp, val_topk

# Evaluation function
def evaluate(model, device, dataloader, criterion, top_k=5, print_result=True):
    model.eval()
    total_loss, total_correct, total_examples = 0, 0, 0
    total_topk_correct = 0

    with torch.no_grad():
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            log_probs = model(context)  # [batch, vocab_size]
            loss = criterion(log_probs, target)

            total_loss += loss.item() * context.size(0)
            total_examples += target.size(0)

            # top-1 accuracy
            preds = torch.argmax(log_probs, dim=1)
            total_correct += (preds == target).sum().item()

            # top-k accuracy
            topk_preds = torch.topk(log_probs, k=top_k, dim=1).indices
            topk_correct = (topk_preds == target.unsqueeze(1)).any(dim=1).sum().item()
            total_topk_correct += topk_correct

    avg_loss = total_loss / total_examples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_examples
    topk_accuracy = total_topk_correct / total_examples

    if print_result:
        info_logger(
            f"Eval Loss: {avg_loss:.4f} | "
            f"Perplexity: {perplexity:.2f} | "
            f"Accuracy: {accuracy*100:.2f}% | "
            f"Top-{top_k} Accuracy: {topk_accuracy*100:.2f}%"
        )
    
    return avg_loss, perplexity, accuracy, topk_accuracy