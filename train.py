import argparse
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from torch.utils.data import DataLoader, Dataset

from model import NeuralModel

VOCAB_SIZE = 20000
EMB_SIZE = 128
CONTEXT = 10
HIDDEN = 256
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 3
TOKENIZER_FILE = "tokenizer.json"
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


def build_tokenizer():
    if os.path.exists(TOKENIZER_FILE):
        return Tokenizer.from_file(TOKENIZER_FILE)
    print("downloading wikitext-2 train split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    with open("wikitext2_train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ds["text"]))
    tok = ByteLevelBPETokenizer()
    tok.train(
        ["wikitext2_train.txt"],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tok.save(TOKENIZER_FILE)
    print(f"tokenizer trained: {tok.get_vocab_size()} tokens")
    return tok


def get_split(split, tok, limit):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
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
    total, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x, logits=True), y)
        total += loss.item() * len(y)
        n += len(y)
    return total / n


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=0, help="max tokens per split for quick tests")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer()
    vocab_size = tok.get_vocab_size()

    train_dl = DataLoader(
        get_split("train", tok, args.limit),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_dl = DataLoader(get_split("validation", tok, args.limit), batch_size=BATCH_SIZE)

    model = NeuralModel(vocab_size, EMB_SIZE, CONTEXT, HIDDEN, d_conn=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(
        f"device={device} vocab={vocab_size} "
        f"train_tokens={len(train_dl.dataset)} val_tokens={len(val_dl.dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, logits=True), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
            pbar.set_postfix(loss=f"{total / n:.4f}")
        pbar.close()
        val_loss = evaluate(model, val_dl, device)
        print(f"epoch {epoch} done: train_loss {total / n:.4f} val_loss {val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"saved {MODEL_FILE} and {TOKENIZER_FILE}")
    print("sample:", generate(model, tok, "the", device=device))


if __name__ == "__main__":
    main()
