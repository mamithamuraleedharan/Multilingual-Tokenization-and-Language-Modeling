from pathlib import Path
from char_tokenizer import build_char_vocab
from bpe_tokenizer import bpe_load_tokenizer
from data_loader import load_id,chunking,loading

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
    valid_loader = loading(valid_x,valid_y,batch_size,True)

    return train_loader,valid_loader,vocab_size


if __name__ == "__main__":
    train_loader,valid_loader,vocab_size = get_data("small_bpe",data_directory,32)
    print(vocab_size,len(train_loader),len(valid_loader))





