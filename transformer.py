import torch
from torch import nn


class TransformerLayer(nn.Module):
    def __init__(self,hidden_dim,num_heads,forward_dim,dropout):
        super().__init__()
        self.attention = Attention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.feedforward = FeedForward(hidden_dim = hidden_dim,forward_dim = forward_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):        
        x = x+self.dropout(self.attention(self.norm1(x)))
        x = x+self.dropout(self.feedforward(self.norm2(x)))
        return x        


class Transformer(nn.Module):
    def __init__(self,vocab_size,hidden_dim,seq_len,num_heads,forward_dim,num_layers,dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim= hidden_dim)
        self.pos_embedding = nn.Embedding(num_embeddings = seq_len , embedding_dim = hidden_dim )
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, forward_dim=forward_dim,dropout=dropout)
            for _ in range(num_layers)
            ])
        self.outputnorm = nn.LayerNorm(hidden_dim)
        self.outputlayer = nn.Linear(hidden_dim,vocab_size)
        self.emb_dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        tok_emb_result = self.embedding(x)
        positions = torch.arange(len(x))
        pos_emb_result = self.pos_embedding(positions)
        combined_result =  tok_emb_result + pos_emb_result
        combined_result = self.emb_dropout(combined_result)
        for layer in self.layers:
            combined_result = layer(combined_result)
        norm_result = self.outputnorm(combined_result)
        final_result = self.outputlayer(norm_result)
        return final_result
        

class Attention(nn.Module):
    def __init__(self,hidden_dim,num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self,x):
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

import torch
from torch import nn


class TransformerLayer(nn.Module):
    def __init__(self,hidden_dim,num_heads,forward_dim,dropout):
        super().__init__()
        self.attention = Attention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.feedforward = FeedForward(hidden_dim = hidden_dim,forward_dim = forward_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):        
        x = x+self.dropout(self.attention(self.norm1(x)))
        x = x+self.dropout(self.feedforward(self.norm2(x)))
        return x        


class Transformer(nn.Module):
    def __init__(self,vocab_size,hidden_dim,seq_len,num_heads,forward_dim,num_layers,dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim= hidden_dim)
        self.pos_embedding = nn.Embedding(num_embeddings = seq_len , embedding_dim = hidden_dim )
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, forward_dim=forward_dim,dropout=dropout)
            for _ in range(num_layers)
            ])
        self.outputnorm = nn.LayerNorm(hidden_dim)
        self.outputlayer = nn.Linear(hidden_dim,vocab_size)
        self.emb_dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        tok_emb_result = self.embedding(x)
        positions = torch.arange(len(x))
        pos_emb_result = self.pos_embedding(positions)
        combined_result =  tok_emb_result + pos_emb_result
        combined_result = self.emb_dropout(combined_result)
        for layer in self.layers:
            combined_result = layer(combined_result)
        norm_result = self.outputnorm(combined_result)
        final_result = self.outputlayer(norm_result)
        return final_result
        

class Attention(nn.Module):
    def __init__(self,hidden_dim,num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self,x):
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

