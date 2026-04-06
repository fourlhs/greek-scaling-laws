import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

fig, ax = plt.subplots(figsize=(9, 6))

# Dynamic colors and markers for all model sizes and token counts
unique_params = sorted(df["n_params"].unique())
unique_tokens = sorted(df["n_tokens"].unique())

color_palette = ["steelblue", "tomato", "forestgreen", "purple", "darkorange"]
marker_list = ["o", "s", "D", "^", "v"]

colors = {n: color_palette[i % len(color_palette)] for i, n in enumerate(unique_params)}
markers = {d: marker_list[i % len(marker_list)] for i, d in enumerate(unique_tokens)}

for _, row in df.iterrows():
    n, d = int(row["n_params"]), int(row["n_tokens"])
    ax.scatter(row["flops"], row["val_loss"],
               color=colors[n], marker=markers[d], s=120, zorder=5)

# Connect points with same N (shows effect of more data for same model)
for n_params, grp in df.groupby("n_params"):
    grp = grp.sort_values("flops")
    ax.plot(grp["flops"], grp["val_loss"], color=colors[n_params], alpha=0.5,
            label=f"N={n_params/1e6:.1f}M")

# Legend for markers
for d, m in markers.items():
    ax.scatter([], [], color="gray", marker=m, s=80, label=f"D={d/1e6:.0f}M")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("FLOPs (6·N·D)", fontsize=12)
ax.set_ylabel("Val Loss", fontsize=12)
ax.set_title("Loss vs Compute: Different (N, D) Allocations", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plots/isoflops.png", dpi=150)
plt.show()
print("Saved to plots/isoflops.png")
