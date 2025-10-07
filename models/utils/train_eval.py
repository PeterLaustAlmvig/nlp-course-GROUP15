import torch

# Training function for the probability of a sentence
def sentence_train(device, epochs, model, dataloader, optimizer, criterion, padding_token_idx, print_interval=1):
    model = model.to(device)
    model.train()
    losses = []
    accuracies = []

    since_better_model = 3 # Stop early if no improvement after three passes
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_sentence_acc = 0.0
        total_sentences = 0

        for input_token, target_token, _ in dataloader:
            input_token, target_token = input_token.to(device), target_token.to(device)
            optimizer.zero_grad()

            output, _ = model(input_token)
            loss = criterion(output.transpose(1, 2), target_token)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = output.argmax(dim=2)
            mask = (target_token != padding_token_idx).float() # Remove padding from accuracy calculation
            word_matches = (preds == target_token).float() * mask
            sentence_acc_batch = word_matches.sum(dim=1) / mask.sum(dim=1)

            total_sentence_acc += sentence_acc_batch.sum().item()
            total_sentences += input_token.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_sentence_acc / total_sentences
        
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        
        if avg_acc < min(accuracies, default=float('inf')) and avg_acc > 0.5:
            since_better_model = 3  # model was better than current best
        elif avg_acc > 0.5:
            since_better_model -= 1 # model was worse than current best
        
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            
        if since_better_model == 0:
            return losses, accuracies

    return losses, accuracies

# Validation function for the probability of a sentence
def sentence_validate(device, model, dataloader):
    model = model.to(device)
    model.eval()
    total_sentences = 0
    correct_sentences = 0

    with torch.no_grad():
        for input_token, target_token, _ in dataloader:  # assumes collate_fn returns lengths
            input_token, target_token = input_token.to(device), target_token.to(device)
            output, _ = model(input_token)  # (batch, seq_len, vocab_size)

            preds = output.argmax(dim=2)  # (batch, seq_len)
            correct_sentences += (preds == target_token).all(dim=1).sentence_correct.sum().item()
            total_sentences += input_token.size(0)

    accuracy = correct_sentences / total_sentences

    print(f"Validation Accuracy: {accuracy*100:.2f}%")

    return accuracy
