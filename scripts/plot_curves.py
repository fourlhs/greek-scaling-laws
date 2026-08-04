import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never block run_all.sh on an interactive window
import matplotlib.pyplot as plt
import glob
import os

curve_files = sorted(glob.glob("curves/curve_N*_D*.csv"))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Training Curves — Loss over Tokens", fontsize=14)

token_groups = {5000000: 0, 20000000: 1, 80000000: 2}
token_labels = {5000000: "5M tokens", 20000000: "20M tokens", 80000000: "80M tokens"}

for f in curve_files:
    name = os.path.basename(f).replace("curve_", "").replace(".csv", "")
    parts = name.split("_")
    n_params = int(parts[0][1:])
    n_tokens = int(parts[1][1:])

    df = pd.read_csv(f)
    ax = axes[token_groups[n_tokens]]
    ax.plot(df["tokens_seen"], df["loss"], label=f"N={n_params/1e6:.1f}M")

for n_tokens, idx in token_groups.items():
    axes[idx].set_title(token_labels[n_tokens])
    axes[idx].set_xlabel("Tokens Seen")
    axes[idx].set_ylabel("Loss")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("plots/training_curves.png", dpi=150)
print("Saved to plots/training_curves.png")
