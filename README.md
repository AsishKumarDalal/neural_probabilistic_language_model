# Neural Probabilistic Language Model

A from-scratch implementation of the classic **Neural Probabilistic Language Model (NPLM)** from Bengio et al. (2003), *"A Neural Probabilistic Language Model"* (JMLR). The model learns to predict the next word given a fixed window of previous words, using a feed-forward network with word embeddings.

## How it works

Given a window of `N` previous tokens, the model:

1. Looks up a dense **embedding** for each token (`nn.Embedding`)
2. Concatenates the embeddings into one flat vector
3. Passes it through a **tanh hidden layer**
4. Projects to a score per vocabulary word
5. Optionally adds a **direct connection** from the concatenated embeddings to the output (Bengio's original design found this helped)
6. Applies **softmax** to get a probability distribution over the next token

The model is trained with cross-entropy loss: maximize the probability of the actual next token in the training corpus.

```
[emb t-9] [emb t-8] ... [emb t-1]
        \      |      /
         concatenated vector
               |
          [tanh hidden layer]
               |
          [output layer + direct connections]
               |
            softmax
               |
      P(next token | context)
```

## Repository layout

| File | Description |
| --- | --- |
| `model.py` | The `NeuralModel` class (embedding → hidden → output, optional direct connections) |
| `train.py` | Full training pipeline: data download, BPE tokenizer training, dataset build, training loop, evaluation, text generation |
| `tokenizer.json` | The trained byte-level BPE tokenizer (generated on first run) |
| `model.pt` | Saved model weights (generated after training) |

## Requirements

- Python 3.8+
- `torch`
- `tokenizers`
- `datasets` (v1)

Install:

```bash
pip install torch tokenizers datasets
```

## Dataset

**WikiText-2** (`wikitext-2-raw-v1`) — a ~2M-token benchmark corpus of Wikipedia articles, commonly used for language-modeling experiments. The script downloads the raw train and validation splits from Hugging Face via the `datasets` library on first run.

## Tokenizer

A **byte-level BPE** tokenizer is trained on the WikiText-2 train split:

- Vocabulary size: **20,000** (+ 4 special tokens: `<unk>`, `<pad>`, `<bos>`, `<eos>`)
- `min_frequency=2`
- Byte-level, so any text can be encoded, and tokenization happens at the byte level to avoid out-of-vocabulary characters

The tokenizer is saved to `tokenizer.json` and reused on later runs (training is skipped if the file exists).

## Model configuration

Defaults in `train.py`:

| Hyperparameter | Value |
| --- | --- |
| `EMB_SIZE` (embedding dim) | 128 |
| `CONTEXT` (window size) | 10 |
| `HIDDEN` (hidden units) | 256 |
| `BATCH_SIZE` | 256 |
| `LR` (Adam) | 3e-4 |
| `EPOCHS` | 3 |
| Direct connections | enabled |

## Training

```bash
# Full run (3 epochs on the whole corpus)
python train.py

# Quick smoke test (first 50k tokens per split, 1 epoch)
python train.py --limit 50000 --epochs 1
```

The training input is built by sliding a window of `CONTEXT` tokens over the tokenized corpus: every context window predicts the token that follows it.

During training, the script prints:

- Average training loss every 200 steps
- Training and validation loss at the end of each epoch

After the last epoch it saves `model.pt`, and prints a sample of generated text from a prompt.

### Output

```text
device=cuda vocab=20004 train_tokens=...
epoch 1 step 200: loss 6.3102
...
epoch 3 done: train_loss 5.7123 val_loss 5.8810
saved model.pt and tokenizer.json
sample: the war was fought ...
```

Loss values are in nats. The validation loss is the number you care about — lower is better, and it should trend down across epochs.

## Generating text

`train.py` includes a `generate()` function that samples greedily from the model's softmax distribution given a prompt (with temperature control):

```python
from train import generate
from tokenizers import ByteLevelBPETokenizer
from model import NeuralModel
import torch

tok = ByteLevelBPETokenizer.from_file("tokenizer.json")
model = NeuralModel(tok.get_vocab_size(), 128, 10, 256)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))

print(generate(model, tok, "the king", n_tokens=60, temperature=0.8))
```

## Notes

- With 20k output classes, the output layer dominates the parameter count. The direct-connection layer roughly doubles it.
- Training on CPU for 3 epochs over full WikiText-2 will take a while; a GPU (free Colab tier works) is recommended.
- The validation loss is slightly higher than the training loss, which is expected — the model memorizes the training corpus to some degree.
- Higher `CONTEXT` gives the model more information but grows the first linear layer's input size linearly (`CONTEXT × EMB_SIZE`).
