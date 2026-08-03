import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")

from model import NeuralModel

VOCAB_SIZE = 20000
EMB_SIZE = 128
CONTEXT = 10
HIDDEN = 256
HID_LAYERS = 1
ACTIVATION = "relu"
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 3
DATASET_MAP = {"2": "wikitext-2-raw-v1", "103": "wikitext-103-raw-v1"}
MODEL_FILE = "model.pt"
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]


class NPLMDataset(Dataset):
    def __init__(self, tokens, context):
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.context = context

    def __len__(self):
        return len(self.data) - self.context

    def __getitem__(self, i):
        return self.data[i : i + self.context], self.data[i + self.context]


def build_tokenizer(ds_key, ds_name):
    tokenizer_file = f"tokenizer{ds_key}.json"
    if os.path.exists(tokenizer_file):
        return Tokenizer.from_file(tokenizer_file), tokenizer_file
    print(f"downloading {ds_name} train split...")
    ds = load_dataset("wikitext", ds_name, split="train")
    corpus_file = f"wikitext{ds_key}_train.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(ds["text"]))
    tok = ByteLevelBPETokenizer()
    tok.train(
        [corpus_file],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tok.save(tokenizer_file)
    print(f"tokenizer trained: {tok.get_vocab_size()} tokens")
    return tok, tokenizer_file


def get_split(split, tok, limit, ds_name):
    ds = load_dataset("wikitext", ds_name, split=split)
    ids = []
    for text in ds["text"]:
        ids.extend(tok.encode(text).ids)
    if limit:
        ids = ids[:limit]
    return NPLMDataset(ids, CONTEXT)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x, logits=True)
        total_loss += loss_fn(logits, y).item() * len(y)
        total_acc += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, total_acc / n


@torch.no_grad()
def generate(model, tok, prompt, n_tokens=40, device="cpu", temperature=0.8):
    model.eval()
    pad_id = tok.token_to_id("<pad>")
    bos_id = tok.token_to_id("<bos>")
    ids = [bos_id] + tok.encode(prompt).ids
    for _ in range(n_tokens):
        ctx = ids[-CONTEXT:]
        if len(ctx) < CONTEXT:
            ctx = [pad_id] * (CONTEXT - len(ctx)) + ctx
        x = torch.tensor([ctx], device=device)
        probs = model(x)[0]
        probs = probs.pow(1.0 / temperature)
        probs = probs / probs.sum()
        ids.append(torch.multinomial(probs, 1).item())
    return tok.decode(ids)


def plot_metrics(step_losses, epoch_edges, train_losses, val_losses, train_accs, val_accs, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    stride = max(1, len(step_losses) // 2000)
    xs = list(range(0, len(step_losses), stride))
    ys = step_losses[::stride]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, lw=0.6, alpha=0.6)
    for e in epoch_edges[1:-1]:
        ax.axvline(e, color="gray", ls="--", alpha=0.7)
    ax.set_xlabel("batch step")
    ax.set_ylabel("loss")
    ax.set_title("training loss per batch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_loss.png"), dpi=150)
    plt.close(fig)

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_losses, "o-", label="train")
    ax1.plot(epochs, val_losses, "o-", label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("loss per epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, train_accs, "o-", label="train")
    ax2.plot(epochs, val_accs, "o-", label="validation")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("top-1 accuracy per epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_accuracy.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=0, help="max tokens per split for quick tests")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset", type=str, default="2", choices=["2", "103"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_name = DATASET_MAP[args.dataset]
    tok, tokenizer_file = build_tokenizer(args.dataset, ds_name)
    vocab_size = tok.get_vocab_size()

    train_dl = DataLoader(
        get_split("train", tok, args.limit, ds_name),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_dl = DataLoader(get_split("validation", tok, args.limit, ds_name), batch_size=BATCH_SIZE)

    model = NeuralModel(
        vocab_size,
        EMB_SIZE,
        CONTEXT,
        HIDDEN,
        hid_layers=HID_LAYERS,
        d_conn=True,
        activation=ACTIVATION,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(
        f"device={device} vocab={vocab_size} "
        f"train_tokens={len(train_dl.dataset)} val_tokens={len(val_dl.dataset)}"
    )

    step_losses = []
    epoch_edges = [0]
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, logits=True)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_acc += (logits.argmax(1) == y).sum().item()
            n += len(y)
            step_losses.append(loss.item())
            pbar.set_postfix(loss=f"{total_loss / n:.4f}", acc=f"{total_acc / n:.4f}")
        pbar.close()
        epoch_edges.append(len(step_losses))
        train_losses.append(total_loss / n)
        train_accs.append(total_acc / n)
        val_loss, val_acc = evaluate(model, val_dl, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(
            f"epoch {epoch} done: train_loss {train_losses[-1]:.4f} "
            f"val_loss {val_losses[-1]:.4f} train_acc {train_accs[-1]:.4f} "
            f"val_acc {val_accs[-1]:.4f}"
        )

    plot_metrics(step_losses, epoch_edges, train_losses, val_losses, train_accs, val_accs)
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"saved {MODEL_FILE} and {tokenizer_file}")
    print(f"saved plots in artifacts/ (train_loss.png, loss_accuracy.png)")
    print("sample:", generate(model, tok, "the", device=device))


if __name__ == "__main__":
    main()
