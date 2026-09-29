from pathlib import Path

def extract_char(filename):
    char_set = set()
    with open(filename,"r",encoding="utf-8") as file:
        for line in file:
           #print(line)
            char_set.update(line)
    return char_set

def build_char_vocab(file_names,file_path):
    vocabulary = set()   
    for file_name in file_names:
        char = extract_char(file_path/file_name)
        vocabulary.update(char)
    sorted_vocab = sorted(vocabulary)

    
    ch_id_dict = {}
    ch_id_dict["UNK"] = 0
    for i,c in enumerate(sorted_vocab,start = 1):
        ch_id_dict[c] = i
    id_ch_dict = dict(zip(ch_id_dict.values(),ch_id_dict.keys()))

    return ch_id_dict,id_ch_dict


# main

file_names = ["en.txt","tr.txt","zh.txt"]
file_path = Path("Data/train/")
ch_id_dict,id_ch_dict = build_char_vocab(file_names,file_path)
