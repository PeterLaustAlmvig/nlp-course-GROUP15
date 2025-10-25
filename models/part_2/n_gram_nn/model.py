import torch.nn as nn
import torch.nn.functional as F

# Feed forward network for sentence prediction
class SentenceModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(SentenceModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        embeds = self.embeddings(input_ids)
        embeds = embeds.view(embeds.size(0), -1)
        hidden = F.relu(self.fc1(embeds))
        hidden = self.drop(hidden)
        out = self.fc2(hidden)
        out = F.log_softmax(out, dim=1)
        return out