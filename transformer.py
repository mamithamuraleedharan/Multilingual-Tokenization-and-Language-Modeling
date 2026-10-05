import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self,vocab_size,hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim= hidden_dim)

    def forward(self,x):
        embedded_result = self.embedding(x)
        return embedded_result


