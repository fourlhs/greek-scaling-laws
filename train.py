import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset
import argparse
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
from schedules import make_lr_multiplier
import csv, os

parser = argparse.ArgumentParser()
parser.add_argument("--n_layers", type=int, required=True)
parser.add_argument("--d_model", type=int, required=True)
parser.add_argument("--n_heads", type=int, required=True)
parser.add_argument("--n_tokens", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42,
                    help="Seeds torch, numpy and random. The published sweep predates "
                         "this flag and was run unseeded.")
parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                    help="'constant' reproduces the committed results.csv. 'cosine' is "
                         "the better recipe -- see the note below.")
parser.add_argument("--warmup_frac", type=float, default=0.01,
                    help="Fraction of total steps spent warming up (cosine only).")
parser.add_argument("--min_lr_frac", type=float, default=0.1,
                    help="Floor of the cosine decay, as a fraction of peak lr.")
args = parser.parse_args()

# Reproducibility. The original sweep seeded only the dataset shuffle, so model
# initialisation varied run to run and nothing was bitwise reproducible.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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

train_loader = DataLoader(TokenDataset(train_data), batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(TokenDataset(val_data), batch_size=args.batch_size)

config = GPTConfig(n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

n_params = model.count_params()
n_params_non_embed = model.count_params_non_embedding()
tokens_seen = 0
step = 0
target_tokens = args.n_tokens
training_curve = []

# Total steps is known up front, so the schedule can be length-matched to this
# run's token budget rather than to some shared wall-clock notion of "an epoch".
tokens_per_step = args.batch_size * config.context_length
total_steps = max(1, math.ceil(target_tokens / tokens_per_step))
warmup_steps = max(1, int(args.warmup_frac * total_steps)) if args.lr_schedule == "cosine" else 0


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    make_lr_multiplier(args.lr_schedule, warmup_steps, total_steps, args.min_lr_frac),
)

print(f"Model: {n_params:,} params total, {n_params_non_embed:,} non-embedding "
      f"({100 * (n_params - n_params_non_embed) / n_params:.1f}% embeddings), "
      f"training for {target_tokens:,} tokens")
print(f"Schedule: {args.lr_schedule}, lr={args.lr:g}, seed={args.seed}, "
      f"{total_steps:,} steps"
      + (f" ({warmup_steps:,} warmup)" if args.lr_schedule == "cosine" else ""))

model.train()
while tokens_seen < target_tokens:
    for x, y in train_loader:
        if tokens_seen >= target_tokens:
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
            training_curve.append({
                "tokens_seen": tokens_seen,
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
            })
            print(f"  tokens: {tokens_seen:,} / {target_tokens:,} | "
                  f"loss: {loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

val_loss = evaluate(model, val_loader, device)
# C = 6ND uses non-embedding N: embedding lookups are gathers, not matmuls, so
# counting them inflates the compute estimate (6x for the smallest model here).
n_flops = 6 * n_params_non_embed * args.n_tokens

print(f"val_loss={val_loss:.4f}")

# save training curve
curve_path = f"curves/curve_N{n_params}_D{args.n_tokens}.csv"
os.makedirs("curves", exist_ok=True)
with open(curve_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["tokens_seen", "loss", "lr"])
    writer.writeheader()
    writer.writerows(training_curve)

# save to results.csv
fieldnames = ["n_params", "n_params_non_embed", "n_tokens", "flops", "val_loss"]
file_exists = os.path.exists("results.csv")
with open("results.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow({
        "n_params": n_params,
        "n_params_non_embed": n_params_non_embed,
        "n_tokens": args.n_tokens,
        "flops": n_flops,
        "val_loss": val_loss,
    })
