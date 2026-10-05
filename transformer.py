import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self,vocab_size,hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim= hidden_dim)
        self.pos_embedding = nn.Embedding(num_embeddings = 128 , embedding_dim = hidden_dim )

    def forward(self,x):
        embedded_result = self.embedding(x)
        positions = torch.arange(len(x))
        pos_embedding_result = self.pos_embedding(positions)
        combined_result =  embedded_result + pos_embedding_result
       
        return combined_result
        

class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim,hidden_dim)
        self.key = nn.Linear(hidden_dim,hidden_dim)
        self.value = nn.Linear(hidden_dim,hidden_dim)

    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return q,k,v

