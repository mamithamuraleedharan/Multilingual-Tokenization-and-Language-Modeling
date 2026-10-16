from pathlib import Path
from char_tokenizer import build_char_vocab
from bpe_tokenizer import bpe_load_tokenizer
from data_loader import load_id,chunking,loading
import math
import torch
from torch import nn
from transformer import Transformer
import argparse

data_directory = Path(__file__).parent/"Data"
# data_directory = Path("/srv/data/lt2326-h26/a1")
file_names = ["en.txt","tr.txt","zh.txt"]
chunk_size = 128
BPE_directory = Path(__file__).parent / "bpe_data"
BPE_files = {"small_bpe":"small_tokenizer.json","large_bpe":"large_tokenizer.json"}

def get_tokenizer(tokenizer_name,data_dir):

    if tokenizer_name == "char":
        char_to_id, id_to_char = build_char_vocab(file_names,Path(data_dir)/"train")
        return char_to_id,"char",len(char_to_id)
    else:
        bpe_tokenizer = bpe_load_tokenizer(str(BPE_directory / BPE_files[tokenizer_name]))
        return bpe_tokenizer,"bpe",bpe_tokenizer.get_vocab_size()

def count_chars(path):
    total = 0
    with open(path,"r",encoding="utf-8") as file:
        for line in file:
            total = total + len(line.strip())
    return total        


def bpc_score(model,tokenizer_type,tokenizer,data_dir,lang_file, device,vocab_size):
    test_dir = Path(data_dir)/"test"
    ids = load_id(test_dir,[lang_file],tokenizer_type,tokenizer)
    chars = count_chars(test_dir / lang_file)

    inputs,targets = chunking(ids,chunk_size)
    loader = loading(inputs,targets,32,False)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    total_tokens = 0
    
    model.eval()  
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x) 
            loss = loss_fn(output.reshape(-1,vocab_size),y.reshape(-1))
            total_loss += loss.item()
            total_tokens+= y.numel()
    loss_per_token = total_loss/total_tokens
    tokens_per_char = len(ids)/chars
    bpc = loss_per_token * tokens_per_char/math.log(2)
    return loss_per_token,tokens_per_char,bpc

#####################################################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",choices=["char","small_bpe","large_bpe"],required=True)
    args = parser.parse_args()   
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer, tokenizer_type, vocab_size = get_tokenizer(args.tokenizer,data_directory)

    model = Transformer(vocab_size=vocab_size,hidden_dim=256,seq_len=chunk_size,num_heads=4,forward_dim=1024,num_layers=2,dropout=0.1).to(device)
    model.load_state_dict(torch.load(f"runs/{args.tokenizer}_model.pt", map_location=device))

    bpc_per_language = []
    for lang_file in file_names:
        loss_per_token,tokens_per_char,bpc = bpc_score(model, tokenizer_type,tokenizer,data_directory,lang_file,device,vocab_size)
        bpc_per_language.append(bpc)
        print(f"{args.tokenizer} {lang_file}: loss/token {loss_per_token:.3f}| "f"tokens/char {tokens_per_char:.3f} | BPC {bpc:.3f}")

    overall_bpc = sum(bpc_per_language)/len(bpc_per_language)
    print(f"{args.tokenizer} OVERALL BPC : {overall_bpc:.3f}")
        

if __name__ == "__main__":
    main()

    