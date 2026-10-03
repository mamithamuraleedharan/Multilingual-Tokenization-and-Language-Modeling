# imports

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

def bpe_train_tokenizer(file_name, vocab_size, save_path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"],vocab_size = vocab_size)
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train([file_name], trainer)
    tokenizer.save(save_path)
    return tokenizer
    
def bpe_load_tokenizer(save_path):
    return Tokenizer.from_file(save_path)


#main


file_name = "Data/tokenizer/balanced.txt"
small_save_path = "bpe_data/small_tokenizer.json"
large_save_path = "bpe_data/large_tokenizer.json"
small_vocab_size = 2000
large_vocab_size = 10000

#small_bpe_tokenizer = bpe_train_tokenizer(file_name,small_vocab_size,small_save_path)
#large_bpe_tokenizer = bpe_train_tokenizer(file_name,large_vocab_size,large_save_path)

small_bpe_tokenizer = bpe_load_tokenizer(small_save_path)
large_bpe_tokenizer = bpe_load_tokenizer(large_save_path)

sentence = "Unhappiness"

small_output = small_bpe_tokenizer.encode(sentence)
large_output = large_bpe_tokenizer.encode(sentence)

print("Small BPE:")
print(small_output.tokens)
print(small_output.ids)

print("Large BPE:")
print(large_output.tokens)
print(large_output.ids)