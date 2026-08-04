"""Training curves: loss over tokens, one panel per token budget.

Curve files are named by *total* parameters (that is what train.py had when the
sweep ran), but everything else in this project reports non-embedding N, so the
filename count is mapped through the grid before labelling. Colour is keyed to
model size and kept consistent across panels.

The y axis is logarithmic and the uniform-random baseline is drawn in. Both are
necessary rather than decorative: every run starts far ABOVE random-guessing
loss, and on a linear axis that opening spike flattens the entire rest of the
curve into an unreadable line along the bottom.

    python scripts/plot_curves.py
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")  # never block run_all.sh on an interactive window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_params import build_lookup

VOCAB_SIZE = 16000
RANDOM_BASELINE = np.log(VOCAB_SIZE)  # 9.68 nats

TOKEN_PANELS = {5000000: 0, 20000000: 1, 80000000: 2}
TOKEN_LABELS = {5000000: "5M tokens", 20000000: "20M tokens", 80000000: "80M tokens"}

lookup = build_lookup()

curves = []
for path in glob.glob("curves/curve_N*_D*.csv"):
    name = os.path.basename(path).replace("curve_", "").replace(".csv", "")
    total_params_str, tokens_str = name.split("_")
    total_params = int(total_params_str[1:])
    n_tokens = int(tokens_str[1:])

    if total_params not in lookup:
        print(f"  skipping {path}: n_params={total_params} matches no grid config")
        continue

    curves.append({
        "path": path,
        "n_params": total_params,
        "n_params_non_embed": lookup[total_params]["n_params_non_embed"],
        "n_tokens": n_tokens,
    })

if not curves:
    raise SystemExit("No curve files found under curves/.")

# Sort numerically by model size. The previous version sorted filenames as
# strings, which put N33506304 before N8886784 and scrambled both the legend
# order and the colour assignment.
curves.sort(key=lambda c: (c["n_tokens"], c["n_params_non_embed"]))

sizes = sorted({c["n_params_non_embed"] for c in curves})
palette = ["steelblue", "tomato", "forestgreen", "purple"]
colors = dict(zip(sizes, palette))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Training curves — loss over tokens (log scale)", fontsize=14)

for c in curves:
    ax = axes[TOKEN_PANELS[c["n_tokens"]]]
    df = pd.read_csv(c["path"])
    ax.plot(df["tokens_seen"], df["loss"], lw=1.1, alpha=0.9,
            color=colors[c["n_params_non_embed"]],
            label=f"N={c['n_params_non_embed']/1e6:.2f}M")

for n_tokens, idx in TOKEN_PANELS.items():
    ax = axes[idx]
    ax.axhline(RANDOM_BASELINE, color="crimson", ls=":", lw=1.4, alpha=0.9)
    ax.set_title(TOKEN_LABELS[n_tokens])
    ax.set_xlabel("Tokens seen")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8, title="non-embedding N", title_fontsize=8)

axes[0].set_ylabel("Training loss (nats, log scale)")
axes[0].text(0.03, RANDOM_BASELINE * 1.15,
             f"uniform-random baseline ({RANDOM_BASELINE:.2f})",
             fontsize=8, color="crimson", transform=axes[0].get_yaxis_transform())

plt.tight_layout()
plt.savefig("plots/training_curves.png", dpi=150)
print("Saved to plots/training_curves.png")

worst = max(pd.read_csv(c["path"])["loss"].max() for c in curves)
print(f"Peak logged loss across all runs: {worst:.1f} nats "
      f"({worst / RANDOM_BASELINE:.1f}x the uniform-random baseline of "
      f"{RANDOM_BASELINE:.2f}) -- see the initialisation note in the README.")
