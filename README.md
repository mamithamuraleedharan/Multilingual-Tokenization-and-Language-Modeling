# Multilingual-Tokenization-and-Language-Modeling

This project investigates how different tokenization methods affects language modelling for English, Turkish and Chinese. A small decoder only transformer model is used for each tokenizer.

Three tokenizers are compared:
- Character-level tokenizer (9,444 tokens)
- Small Byte-Level BPE (2,000 tokens)
- Large Byte-Level BPE (10,000 tokens)

## Data

Data from Leipzig Corpora Collection is used

The dataset contains three splits:
- `train/` : Tokenizer and language-model training
- `valid/` : Validation and tokenizer analysis
- `test/` : Final model evaluation

The shared BPE tokenizers are trained using `tokenizer/balanced.txt`.
The complete dataset was used without subsampling.

## Requirements

- Python
- PyTorch
- Hugging Face Tokenizers
- Matplotlib

## Project Structure

| File / Directory | Description |
|---|---|
| `char_tokenizer.py` | Builds the character-level tokenizer |
| `bpe_tokenizer.py` | Trains and builds the BPE tokenizers |
| `transformer.py` | Defines the Transformer language model |
| `data_loader.py` | Loads and prepares the data |
| `train.py` | Trains the three language models |
| `evaluate.py` | Evaluates model performance |
| `Multilingual_Tokenization_Language_Modeling.ipynb` | Runs experiments and analyzes results |
| `bpe_data/` | Stores the saved BPE tokenizers |
| `runs/` | Stores training and evaluation results |

## Model Architecture

All three models use the same Transformer architecture:

- Transformer layers: 2
- Hidden dimension: 256
- Attention heads: 4
- Feed-forward dimension: 1024
- Context length: 128
- Dropout: 0.1

## How to run
Run the commands from the project directory on `mltgpu`.

### 1. Tokenizers
The character vocabulary is built from the training data using `char_tokenizer.py`.
The saved BPE tokenizers are available at:

- `bpe_data/small_tokenizer.json`
- `bpe_data/large_tokenizer.json`

The tokenizer training functions are implemented in `bpe_tokenizer.py`. The existing main section loads the saved tokenizers rather than retraining them.

### 2. Train the Models

```bash
python train.py --tokenizer char --epochs 20
python train.py --tokenizer small_bpe --epochs 20
python train.py --tokenizer large_bpe --epochs 20
```

Training saves model weights and loss histories in the `runs/` directory.

### 3. Evaluate the Models

After the trained model weights are available, run:

```bash
python evaluate.py --tokenizer char
python evaluate.py --tokenizer small_bpe
python evaluate.py --tokenizer large_bpe
```

### 4. Reproduce the Analysis

Open `Multilingual_Tokenization_Language_Modeling.ipynb` in Jupyter.

The notebook contains all the analysis performed for this assignment.


## Results

The Large BPE model achieved the lowest overall BPC (3.619). However, the character-level model achieved the best BPC for Chinese (6.529).

The results suggest that tokenization affects languages differently, and a larger shared BPE vocabulary does not necessarily improve performance for every language.


