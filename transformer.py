import torch
from torch import nn
import math

class Transformer(nn.Module):
    def __init__(self,vocab_size,hidden_dim,seq_len,num_heads):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim= hidden_dim)
        self.pos_embedding = nn.Embedding(num_embeddings = seq_len , embedding_dim = hidden_dim )
        self.attention = Attention(hidden_dim=hidden_dim, num_heads=num_heads)

    def forward(self,x):
        tok_emb_result = self.embedding(x)
        positions = torch.arange(len(x))
        pos_emb_result = self.pos_embedding(positions)
        combined_result =  tok_emb_result + pos_emb_result
        attention_result = self.attention(combined_result)
        return attention_result
        

class Attention(nn.Module):
    def __init__(self,hidden_dim,num_heads):
        super().__init__()
        #self.hidden_dim = hidden_dim
        #self.query = nn.Linear(hidden_dim,hidden_dim)
        #self.key = nn.Linear(hidden_dim,hidden_dim)
        #self.value = nn.Linear(hidden_dim,hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self,x):
        #q = self.query(x)
        #k = self.key(x)
        #v = self.value(x)

        # implementing the equation of scores
        
        #scores = torch.matmul(q,k.transpose(-2,-1)) 
        #scores = scores / math.sqrt(self.hidden_dim)
        #weight = torch.softmax(scores,dim = -1)
        #output = torch.matmul(weight,v)
        seq_len = len(x)
        casual_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        #print(casual_mask)
        output,weights = self.attention(x,x,x,attn_mask = casual_mask)       
        return output

class FeedForward(nn.Module):
    def __init__(self,hidden_dim,forward_dim):
        super().__init__()
        self.linear1 =nn.Linear(hidden_dim,forward_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(forward_dim,hidden_dim)


    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

