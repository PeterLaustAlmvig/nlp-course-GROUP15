import torch
from torch import nn
    
class LSTMModel(nn.Module):
    def __init__(self, device, input_size, hidden_size, vocab_size, num_layers=1, drop_out=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.embedding = nn.Embedding(vocab_size, input_size)

        # Apply dropout between layers if more than one
        if num_layers > 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=drop_out, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        else:
            h0, c0 = hidden

        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)  # (batch, seq_len, vocab_size)
        return out, (hn, cn)