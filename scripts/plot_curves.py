import pandas as pd
import matplotlib.pyplot as plt
import glob

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Training Curves — Loss over Tokens", fontsize=14)

# Single-run curve files (curve_N*.csv without _D)
curve_files = sorted(glob.glob("curves/curve_N*.csv"))
curve_files = [f for f in curve_files if "_D" not in f]

for f in curve_files:
    import os
    name = os.path.basename(f).replace("curve_", "").replace(".csv", "")
    n_params = int(name[1:])
    df = pd.read_csv(f)
    ax.plot(df["tokens_seen"], df["loss"], label=f"N={n_params/1e6:.1f}M", linewidth=1.5)

ax.set_xlabel("Tokens Seen")
ax.set_ylabel("Loss")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("plots/training_curves.png", dpi=150)
plt.show()
print("Saved to plots/training_curves.png")
