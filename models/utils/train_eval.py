import torch

# Training function for the probability of a sentence
def train(device, epochs, model, dataloader, optimizer, criterion, print_interval=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_token, target_token in dataloader:
            input_token, target_token = input_token.to(device), target_token.to(device)
            optimizer.zero_grad()
            output, _ = model(input_token)
            loss = criterion(output, target_token.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Validation function for the probability of a sentence
def validate(device, model, dataloader, criterion):
    model.eval()
    total_loss = 0
    predict = 0
    with torch.no_grad():
        for input_token, target_token in dataloader:
            input_token, target_token = input_token.to(device), target_token.to(device)
            output, _ = model(input_token)
            loss = criterion(output, target_token)
            total_loss += loss.item()
            predict += (output.argmax(1) == target_token).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = predict / len(dataloader)
    print(f"Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}")
    return avg_loss

def _sentence_probability(word_probabilities:list):
    pass