"""Backfill non-embedding parameter counts into an existing results.csv.

The original sweep recorded N as *total* parameters and derived FLOPs from it.
Kaplan et al. define N as non-embedding parameters, and at this scale the
difference is not cosmetic — the smallest model is 84% embedding table.

Retraining is unnecessary to correct this: val_loss does not depend on how the
parameters were counted, and the parameter count is a deterministic function of
(n_layers, d_model, vocab_size, context_length). This script recovers each run's
config from its recorded total, recomputes N and C = 6ND, and rewrites the CSV.

The analytic formula is validated against every recorded total before anything
is written; a mismatch aborts rather than silently emitting wrong numbers.

    python scripts/recompute_params.py
"""

import csv
import os
import sys

VOCAB_SIZE = 16000
CONTEXT_LENGTH = 256

# (n_layers, d_model, n_heads) — the grid defined in run_all.sh
GRID = [
    (4, 64, 4),
    (4, 128, 4),
    (6, 256, 8),
    (8, 512, 8),
]


def param_counts(n_layers, d_model, vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH):
    """Return (total, non_embedding) parameters for the GPT in model.py.

    Per block: qkv 3d^2 + proj d^2 + mlp 8d^2 + two LayerNorms 4d = 12d^2 + 4d.
    Plus final LayerNorm 2d. The LM head is weight-tied to tok_emb, so it adds
    nothing. Embeddings are tok_emb (vocab*d) and pos_emb (context*d).
    """
    blocks = n_layers * (12 * d_model**2 + 4 * d_model)
    final_ln = 2 * d_model
    non_embedding = blocks + final_ln

    embeddings = vocab_size * d_model + context_length * d_model
    total = embeddings + non_embedding

    return total, non_embedding


def build_lookup():
    """Map recorded total-parameter count -> (config, non-embedding count)."""
    lookup = {}
    for n_layers, d_model, n_heads in GRID:
        total, non_embedding = param_counts(n_layers, d_model)
        lookup[total] = {
            "n_layers": n_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_params_non_embed": non_embedding,
        }
    return lookup


def main():
    path = "results.csv"
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run the sweep first (bash run_all.sh).")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit(f"ERROR: {path} is empty.")

    if "n_params_non_embed" in rows[0]:
        print(f"{path} already carries n_params_non_embed — nothing to do.")
        return

    lookup = build_lookup()

    # Validate before writing: every recorded total must match the formula.
    recorded = {int(r["n_params"]) for r in rows}
    unknown = recorded - set(lookup)
    if unknown:
        sys.exit(
            "ERROR: these recorded n_params values match no grid config: "
            f"{sorted(unknown)}\nExpected one of {sorted(lookup)}.\n"
            "The formula in param_counts() and the model in model.py have diverged, "
            "or results.csv came from a different grid."
        )

    print(f"{'d_model':>8} {'total N':>12} {'non-embed N':>12} {'embed %':>9}")
    for total in sorted(lookup):
        info = lookup[total]
        non_embed = info["n_params_non_embed"]
        pct = 100 * (total - non_embed) / total
        print(f"{info['d_model']:>8} {total:>12,} {non_embed:>12,} {pct:>8.1f}%")

    out = []
    for r in rows:
        total = int(r["n_params"])
        non_embed = lookup[total]["n_params_non_embed"]
        n_tokens = int(r["n_tokens"])
        out.append({
            "n_params": total,
            "n_params_non_embed": non_embed,
            "n_tokens": n_tokens,
            "flops": 6 * non_embed * n_tokens,
            "val_loss": float(r["val_loss"]),
        })

    backup = path + ".bak"
    os.replace(path, backup)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_params", "n_params_non_embed", "n_tokens", "flops", "val_loss"]
        )
        writer.writeheader()
        writer.writerows(out)

    print(f"\nRewrote {path} ({len(out)} rows); previous version at {backup}.")
    print("FLOPs now derived from non-embedding N.")


if __name__ == "__main__":
    main()
