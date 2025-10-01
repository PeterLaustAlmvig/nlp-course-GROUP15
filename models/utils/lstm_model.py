import torch
from torch.nn import nn

class LSTMModel(nn.Module):
    def __init__(self, device, input_size, hidden_size, vocab_size, no_lstm_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=no_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        h0, c0 = (torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device))
        x = self.embedding(x)
        x, (h0, c0) = self.lstm(x, (h0, c0))
        x = x.contiguous().view(-1, self.hidden_size)
        x = self.fc(x[:, -1, :])
        self.fc_weights = x
        return x