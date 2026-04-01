import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("results.csv")

def power_law(x, a, b, c):
    return a * x ** b + c

def fit_and_plot(ax, x, y, xlabel, color, scatter=True):
    try:
        popt, _ = curve_fit(power_law, x, y, p0=[10, -0.3, 2], maxfev=10000)
        a, b, c = popt
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        ax.plot(x_fit, power_law(x_fit, *popt), color=color, linestyle="--", label=f"fit: {a:.2f}·x^{b:.2f} + {c:.2f}")
        print(f"{xlabel}: a={a:.3f}, b={b:.3f}, c={c:.3f}")
    except Exception as e:
        print(f"Fit failed for {xlabel}: {e}")
    if scatter:
        ax.scatter(x, y, color=color, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Val Loss")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Scaling Laws — Greek LM", fontsize=14)

# Loss vs N
for n_tokens, grp in df.groupby("n_tokens"):
    axes[0].scatter(grp["n_params"], grp["val_loss"], label=f"D={n_tokens/1e6:.0f}M")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Parameters (N)")
axes[0].set_ylabel("Val Loss")
axes[0].set_title("Loss vs Model Size")
axes[0].legend()
axes[0].grid(True, which="both", ls="--", alpha=0.4)

# Loss vs D
for n_params, grp in df.groupby("n_params"):
    axes[1].scatter(grp["n_tokens"], grp["val_loss"], label=f"N={n_params/1e6:.1f}M")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Training Tokens (D)")
axes[1].set_ylabel("Val Loss")
axes[1].set_title("Loss vs Training Tokens")
axes[1].legend()
axes[1].grid(True, which="both", ls="--", alpha=0.4)

# Loss vs FLOPs
fit_and_plot(axes[2], df["flops"].values, df["val_loss"].values, "FLOPs", "purple")
axes[2].set_title("Loss vs FLOPs")

# Loss vs N x D
fit_and_plot(axes[0], df["n_params"].values, df["val_loss"].values, "Parameters (N)", "purple", scatter=False)
fit_and_plot(axes[1], df["n_tokens"].values, df["val_loss"].values, "Training Tokens (D)", "purple", scatter=False)

plt.tight_layout()
plt.savefig("plots/scaling_laws.png", dpi=150)
plt.show()
print("Saved to plots/scaling_laws.png")

# Compute-optimal frontier: for each FLOP budget, find the (N, D) with lowest loss
optimal = df.loc[df.groupby("flops")["val_loss"].idxmin()]

def linear_log(x, a, b):
    return a * np.log10(x) + b

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Compute-Optimal Frontier", fontsize=14)

# N_opt vs FLOPs
axes[0].scatter(optimal["flops"], optimal["n_params"], color="steelblue", s=80, zorder=5)
try:
    popt, _ = curve_fit(linear_log, optimal["flops"].values, np.log10(optimal["n_params"].values))
    x_fit = np.logspace(np.log10(optimal["flops"].min()), np.log10(optimal["flops"].max()), 200)
    axes[0].plot(x_fit, 10**linear_log(x_fit, *popt), color="steelblue", linestyle="--", label=f"N ∝ C^{popt[0]:.2f}")
    print(f"N_opt vs FLOPs: exponent={popt[0]:.3f}")
except Exception as e:
    print(f"N_opt fit failed: {e}")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("FLOPs")
axes[0].set_ylabel("Optimal N (params)")
axes[0].set_title("Optimal Model Size vs Compute")
axes[0].legend()
axes[0].grid(True, which="both", ls="--", alpha=0.4)

# D_opt vs FLOPs
axes[1].scatter(optimal["flops"], optimal["n_tokens"], color="tomato", s=80, zorder=5)
try:
    popt, _ = curve_fit(linear_log, optimal["flops"].values, np.log10(optimal["n_tokens"].values))
    x_fit = np.logspace(np.log10(optimal["flops"].min()), np.log10(optimal["flops"].max()), 200)
    axes[1].plot(x_fit, 10**linear_log(x_fit, *popt), color="tomato", linestyle="--", label=f"D ∝ C^{popt[0]:.2f}")
    print(f"D_opt vs FLOPs: exponent={popt[0]:.3f}")
except Exception as e:
    print(f"D_opt fit failed: {e}")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("FLOPs")
axes[1].set_ylabel("Optimal D (tokens)")
axes[1].set_title("Optimal Data vs Compute")
axes[1].legend()
axes[1].grid(True, which="both", ls="--", alpha=0.4)

# Loss vs FLOPs for optimal points
axes[2].scatter(optimal["flops"], optimal["val_loss"], color="purple", s=80, zorder=5)
fit_and_plot(axes[2], optimal["flops"].values, optimal["val_loss"].values, "FLOPs", "purple", scatter=False)
axes[2].set_title("Optimal Loss vs Compute")
axes[2].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/compute_optimal_frontier.png", dpi=150)
plt.show()
print("Saved to plots/compute_optimal_frontier.png")
print("\nOptimal (N, D) per FLOP budget:")
print(optimal[["flops", "n_params", "n_tokens", "val_loss"]].to_string(index=False))
