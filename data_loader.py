from pathlib import Path
from char_tokenizer import char_encode
import torch

def load_id(folder,file_names,tokenizer_type,tokenizer):
    id_list = []
    for file_name in file_names:
        with open(Path(folder)/file_name,"r",encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if tokenizer_type == "char":
                    ids = char_encode(line,tokenizer)
                elif tokenizer_type == "bpe":
                    ids = tokenizer.encode(line).ids
                else:
                    raise ValueError("Unknown tokenizer type")
                id_list.extend(ids)
    return id_list

def chunking(ids,chunk_size):
    n = (len(ids)-1)//chunk_size
    ids = torch.tensor(ids)
    inputs = ids[0:n*chunk_size].view(n,chunk_size)
    targets = ids[1:n*chunk_size+1].view(n,chunk_size)
    return inputs,targets