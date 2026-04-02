import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
import csv, os

parser = argparse.ArgumentParser()
parser.add_argument("--n_layers", type=int, required=True)
parser.add_argument("--d_model", type=int, required=True)
parser.add_argument("--n_heads", type=int, required=True)
args = parser.parse_args()

EVAL_AT = [2_000_000, 20_000_000, 200_000_000]
TARGET_TOKENS = EVAL_AT[-1]

def build_dataset(tokenizer_path, context_length, cache_path="dataset_cache.npy"):
    if os.path.exists(cache_path):
        print("Loading cached dataset...")
        data = np.load(cache_path, mmap_mode='r')
        split = int(0.9 * len(data))
        return data[:split], data[split:]

    print("Tokenizing dataset (first time only)...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = load_dataset("alexliap/tinystories-gr", split="train")

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    all_tokens = []
    for example in dataset:
        ids = tokenizer.encode(example["greek_translation"]).ids
        all_tokens.extend(ids + [eos_id])

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    n = len(all_tokens)
    n_chunks = n // (context_length + 1)
    all_tokens = all_tokens[:n_chunks * (context_length + 1)]
    data = all_tokens.reshape(n_chunks, context_length + 1)

    data = data[np.random.default_rng(42).permutation(len(data))]
    np.save(cache_path, data)

    split = int(0.9 * len(data))
    return data[:split], data[split:]

class TokenDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx, :-1], dtype=torch.long)
        y = torch.tensor(self.data[idx, 1:], dtype=torch.long)
        return x, y

def evaluate(model, val_loader, device, n_batches=50):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= n_batches: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / min(n_batches, len(val_loader))

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

train_data, val_data = build_dataset("greek_bpe_tokenizer.json", context_length=256)

train_loader = DataLoader(TokenDataset(train_data), batch_size=32, shuffle=False)
val_loader = DataLoader(TokenDataset(val_data), batch_size=32)

config = GPTConfig(n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

n_params = model.count_params()
tokens_per_step = 32 * 256
total_steps = TARGET_TOKENS // tokens_per_step
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=3e-5)

tokens_seen = 0
step = 0
training_curve = []
eval_checkpoints = sorted(EVAL_AT)
next_eval_idx = 0

print(f"Model: {n_params:,} params, training for {TARGET_TOKENS:,} tokens")
print(f"Will evaluate at: {[f'{t:,}' for t in eval_checkpoints]}")

os.makedirs("curves", exist_ok=True)
os.makedirs("models", exist_ok=True)

model.train()
while tokens_seen < TARGET_TOKENS:
    for x, y in train_loader:
        if tokens_seen >= TARGET_TOKENS:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        tokens_seen += x.numel()
        step += 1
        if step % 5 == 0:
            training_curve.append({"tokens_seen": tokens_seen, "loss": loss.item()})
            print(f"  tokens: {tokens_seen:,} / {TARGET_TOKENS:,} | loss: {loss.item():.4f}")

        # Evaluate and log at checkpoints
        if next_eval_idx < len(eval_checkpoints) and tokens_seen >= eval_checkpoints[next_eval_idx]:
            checkpoint_tokens = eval_checkpoints[next_eval_idx]
            val_loss = evaluate(model, val_loader, device)
            n_flops = 6 * n_params * checkpoint_tokens
            print(f"\n  === EVAL @ {checkpoint_tokens:,} tokens | val_loss={val_loss:.4f} ===\n")

            file_exists = os.path.exists("results.csv")
            with open("results.csv", "a") as f:
                writer = csv.DictWriter(f, fieldnames=["n_params", "n_tokens", "flops", "val_loss"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({"n_params": n_params, "n_tokens": checkpoint_tokens, "flops": n_flops, "val_loss": val_loss})

            model.train()
            next_eval_idx += 1

# Save training curve
curve_path = f"curves/curve_N{n_params}.csv"
with open(curve_path, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["tokens_seen", "loss"])
    writer.writeheader()
    writer.writerows(training_curve)

# Save final model
torch.save(model.state_dict(), f"models/model_N{n_params}.pt")
print(f"Saved model to models/model_N{n_params}.pt")
