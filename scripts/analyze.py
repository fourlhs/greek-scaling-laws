import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

def log_fit(x, y):
    """Fit L = a * X^b in log-log space (linear regression on logs)."""
    log_x, log_y = np.log(x), np.log(y)
    b, log_a = np.polyfit(log_x, log_y, 1)
    return np.exp(log_a), b

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Scaling Laws — Greek LM (21 runs)", fontsize=14)

configs = [
    (0, "n_params", "Parameters (N)", "n_tokens", "D", "Loss vs Model Size"),
    (1, "n_tokens", "Training Tokens (D)", "n_params", "N", "Loss vs Training Tokens"),
    (2, "flops", "FLOPs (6·N·D)", None, None, "Loss vs FLOPs"),
]

for idx, xcol, xlabel, group_col, group_prefix, title in configs:
    ax = axes[idx]

    # Scatter points colored by the other axis (or plain for FLOPs)
    if group_col:
        for val, grp in df.groupby(group_col):
            ax.scatter(grp[xcol], grp["val_loss"],
                       label=f"{group_prefix}={val/1e6:.0f}M", zorder=5)
    else:
        ax.scatter(df[xcol], df["val_loss"], color="purple", zorder=5)

    # Power-law fit: L = a * X^b
    x, y = df[xcol].values.astype(float), df["val_loss"].values.astype(float)
    a, b = log_fit(x, y)
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    ax.plot(x_fit, a * x_fit ** b, "k--", alpha=0.7,
            label=f"fit: L = {a:.1f}·X^{{{b:.3f}}}")
    print(f"{xlabel}: a={a:.3f}, b={b:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Val Loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/scaling_laws.png", dpi=150)
plt.show()
print("\nSaved to plots/scaling_laws.png")
