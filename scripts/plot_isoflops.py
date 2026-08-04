"""Loss vs compute for all 12 (N, D) allocations.

The grid is factorial rather than iso-FLOP, but it happens to contain two pairs
of runs at near-identical compute (within 0.5%), which give a direct read on
allocation: same budget, different split between parameters and data. Both are
annotated on the plot.

N is non-embedding throughout — see scripts/recompute_params.py.

    python scripts/plot_isoflops.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results.csv")

if "n_params_non_embed" not in df.columns:
    raise SystemExit(
        "results.csv has no n_params_non_embed column.\n"
        "Run: python scripts/recompute_params.py"
    )

N_COL = "n_params_non_embed"

fig, ax = plt.subplots(figsize=(10, 6.5))

sizes = sorted(df[N_COL].unique())
palette = ["steelblue", "tomato", "forestgreen", "purple"]
colors = dict(zip(sizes, palette))
markers = {5000000: "o", 20000000: "s", 80000000: "D"}

# One line per model size: the effect of feeding it more data
for n_params, grp in df.groupby(N_COL):
    grp = grp.sort_values("flops")
    ax.plot(grp["flops"], grp["val_loss"], color=colors[n_params], alpha=0.55, lw=1.5,
            label=f"N={n_params/1e6:.2f}M")
    for _, row in grp.iterrows():
        ax.scatter(row["flops"], row["val_loss"], color=colors[n_params],
                   marker=markers[int(row["n_tokens"])], s=130, zorder=5,
                   edgecolor="white", linewidth=0.8)

# Lower envelope: best loss achieved at or below each compute level
env = df.sort_values("flops").copy()
env["best"] = env["val_loss"].cummin()
env = env[env["val_loss"] == env["best"]]
ax.plot(env["flops"], env["val_loss"], color="black", ls="--", lw=1.6, alpha=0.75,
        zorder=4, label="compute-optimal frontier")

# Matched-compute pairs: same budget, different (N, D) split
rows = df.sort_values("flops").to_dict("records")
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        a, b = rows[i], rows[j]
        if abs(a["flops"] - b["flops"]) / a["flops"] >= 0.05:
            continue
        win, lose = (a, b) if a["val_loss"] < b["val_loss"] else (b, a)
        ax.annotate(
            "", xy=(win["flops"], win["val_loss"]), xytext=(lose["flops"], lose["val_loss"]),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.6, alpha=0.85,
                            connectionstyle="arc3,rad=0.25"),
        )
        gap = lose["val_loss"] - win["val_loss"]
        ax.text(win["flops"] * 1.06, (win["val_loss"] + lose["val_loss"]) / 2,
                f"same compute\n{gap:.2f} nats better\nwith {win['n_tokens']/win['n_params_non_embed']:.0f} tok/param",
                fontsize=7.5, color="crimson", va="center")

for d, m in markers.items():
    ax.scatter([], [], color="gray", marker=m, s=90, label=f"D={d/1e6:.0f}M")

ax.axhline(np.log(16000), color="gray", ls=":", lw=1.2, alpha=0.8)
ax.text(df["flops"].min(), np.log(16000) * 1.01, "uniform-random baseline (9.68)",
        fontsize=7.5, color="gray")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("FLOPs  (C = 6ND, non-embedding N)", fontsize=11)
ax.set_ylabel("Validation loss (nats)", fontsize=11)
ax.set_title("Compute allocation: at matched budget, data beat parameters", fontsize=12.5)
ax.legend(fontsize=8.5, ncol=2, loc="lower left")
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/isoflops.png", dpi=150)
print("Saved plots/isoflops.png")
