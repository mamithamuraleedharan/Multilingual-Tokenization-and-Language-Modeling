from pathlib import Path
from char_tokenizer import build_char_vocab
from bpe_tokenizer import bpe_load_tokenizer
from data_loader import load_id,chunking,loading
import math
import torch
from torch import nn
from transformer import Transformer
import time
import json
import argparse

data_directory = Path(__file__).parent/"Data"
# data_directory = Path("/srv/data/lt2326-h26/a1")
file_names = ["en.txt","tr.txt","zh.txt"]
chunk_size = 128
BPE_directory = Path(__file__).parent / "bpe_data"
BPE_files = {"small_bpe":"small_tokenizer.json","large_bpe":"large_tokenizer.json"}

def get_data(tokenizer_name,data_dir,batch_size):
    train_dir = Path(data_dir)/"train"
    valid_dir = Path(data_dir)/"valid"

    if tokenizer_name == "char":
        tokenizer,id_ch_dict = build_char_vocab(file_names,train_dir)
        tokenizer_type = "char"
        vocab_size = len(tokenizer)

    else:
        tokenizer = bpe_load_tokenizer(str(BPE_directory/ BPE_files[tokenizer_name]))
        tokenizer_type = "bpe"
        vocab_size = tokenizer.get_vocab_size()


    train_ids = load_id(train_dir,file_names,tokenizer_type,tokenizer)
    valid_ids = load_id(valid_dir,file_names,tokenizer_type,tokenizer)

    train_x,train_y = chunking(train_ids,chunk_size)
    valid_x,valid_y = chunking(valid_ids,chunk_size)

    train_loader = loading(train_x,train_y,batch_size,True)
    valid_loader = loading(valid_x,valid_y,batch_size,False)

    return train_loader,valid_loader,vocab_size

def evaluate(model,loader,loss_fn,device,vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x) 
            loss = loss_fn(output.reshape(-1,vocab_size),y.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss/len(loader)
    return avg_loss

def train_one_epoch(model, loader, loss_fn, optimizer, device, vocab_size,max_batches=None, printforevery=200):
    model.train()
    total_loss = 0
    total_tokens = 0
    progress_loss = 0
    
    for i,(x,y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad() # Zero your gradients for every batch!
        output = model(x) # Make predictions for this batch
        loss = loss_fn(output.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward() # Compute the loss and its gradients
        optimizer.step()  # Adjust learning weights
        
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        progress_loss += loss.item()
        if (i + 1) % printforevery == 0:
            print(f" batch {i + 1}/{len(loader)} loss: {progress_loss/printforevery:.3f}")
            progress_loss = 0.0
    return total_loss/total_tokens  
        


def train(model,train_loader,valid_loader,loss_fn,optimizer,device, vocab_size,epochs,name):
    train_losses = []
    val_losses = []

    for epoch in range(1,epochs+1):
        train_loss = train_one_epoch(model,train_loader,loss_fn, optimizer,device,vocab_size)
        val_loss = evaluate(model,valid_loader,loss_fn,device, vocab_size)
        print(f"epoch {epoch}/{epochs} | train loss {train_loss:.3f} | valid loss {val_loss:.3f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(model.state_dict(),f"runs/{name}_model.pt")
    with open(f"runs/{name}_history.json","w") as f:
        json.dump({"train_loss": train_losses,"val_loss": val_losses},f)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",choices=["char","small_bpe","large_bpe"])
    parser.add_argument("--epochs",type = int,default = 20 )
    args = parser.parse_args()   
    
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader,valid_loader,vocab_size = get_data(args.tokenizer,data_directory,32)
    model = Transformer(vocab_size=vocab_size,hidden_dim=256,seq_len=chunk_size,num_heads=4,forward_dim=1024, num_layers=2,dropout=0.1).to(device)
   
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)

    print("untrained valid loss:",evaluate(model,valid_loader,loss_fn,device,vocab_size),"| random loss:", math.log(vocab_size))

    train(model,train_loader,valid_loader,loss_fn,optimizer,device,vocab_size,args.epochs,args.tokenizer)

if __name__ == "__main__":
    main()

    






