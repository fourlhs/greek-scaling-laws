"""Fit scaling laws to the sweep results and render log-log plots.

Three fits, in increasing order of how much they should be trusted:

1. Per-slice marginals. To measure how loss responds to N, D must be held
   fixed — so we fit one exponent per D-slice and report the spread, rather
   than pooling all 12 points and letting the D effect leak into the N
   exponent (which is what the first version of this script did).

2. Pooled L(C) = a*C^b + c against compute, with c >= 0 enforced. Cross-entropy
   has a non-negative floor; an unconstrained fit here returned c = -0.18,
   which is not a physically meaningful irreducible loss.

3. Joint Chinchilla surface L(N, D) = E + A/N^alpha + B/D^beta over all 12
   points at once. This is the principled estimator: it separates the two axes
   by construction instead of by slicing, and E is a real irreducible-loss term.

N is non-embedding throughout — see scripts/recompute_params.py.

    python scripts/analyze.py
"""

import json

import matplotlib
matplotlib.use("Agg")  # never block run_all.sh on an interactive window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Kaplan et al. (2020), table 1 / eqs 1.1-1.3
KAPLAN = {"N": -0.076, "D": -0.095, "C": -0.050}

VOCAB_SIZE = 16000
RANDOM_BASELINE = np.log(VOCAB_SIZE)  # loss of a uniform predictor: 9.68 nats

df = pd.read_csv("results.csv")

if "n_params_non_embed" not in df.columns:
    raise SystemExit(
        "results.csv has no n_params_non_embed column.\n"
        "Run: python scripts/recompute_params.py"
    )

N_COL = "n_params_non_embed"
report = {}


# --------------------------------------------------------------------- helpers
def loglog_slope(x, y):
    """Local power-law exponent: slope of log(loss) vs log(x), plus R^2.

    Two parameters against as few as three points, so it stays well-posed where
    a three-parameter a*x^b + c fit would not.
    """
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def power_law(x, a, b, c):
    return a * x**b + c


def fit_power_law(x, y):
    """a*x^b + c with c >= 0. Returns (params, stderr, r2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    popt, pcov = curve_fit(
        power_law, x, y,
        p0=[1e3, -0.15, 1.0],
        bounds=([0, -2.0, 0.0], [np.inf, 0.0, np.inf]),  # c >= 0: loss has a floor
        maxfev=200000,
    )
    pred = power_law(x, *popt)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return popt, np.sqrt(np.diag(pcov)), 1 - ss_res / ss_tot


def chinchilla(xy, E, A, alpha, B, beta):
    n, d = xy
    return E + A * n ** (-alpha) + B * d ** (-beta)


# ------------------------------------------------- 1. per-slice marginal fits
print("=" * 72)
print("PER-SLICE MARGINAL EXPONENTS  (the other axis held fixed)")
print("=" * 72)

for axis, group_col, x_col in [("N", "n_tokens", N_COL), ("D", N_COL, "n_tokens")]:
    slopes = []
    print(f"\nLoss vs {axis}:")
    for key, grp in df.groupby(group_col):
        grp = grp.sort_values(x_col)
        slope, _, r2 = loglog_slope(grp[x_col], grp["val_loss"])
        slopes.append(slope)
        label = f"D={key/1e6:.0f}M" if group_col == "n_tokens" else f"N={key/1e6:.2f}M"
        print(f"  {label:>12}  b = {slope:+.4f}   R^2 = {r2:.4f}   ({len(grp)} points)")

    mean, std = float(np.mean(slopes)), float(np.std(slopes))
    print(f"  {'mean':>12}  b = {mean:+.4f} +/- {std:.4f}    (Kaplan: {KAPLAN[axis]:+.3f})")
    report[f"marginal_{axis}"] = {
        "per_slice": [float(s) for s in slopes], "mean": mean, "std": std,
    }

# For contrast: the same N exponent computed the old way — pooled across all 12
# points while D also varies, and using total rather than non-embedding params.
pooled_old, _, _ = loglog_slope(df["n_params"], df["val_loss"])
pooled_new, _, _ = loglog_slope(df[N_COL], df["val_loss"])
print(f"\n  Pooled over all 12 points (D not held fixed):")
print(f"    total N        b = {pooled_old:+.4f}   <- the original reported figure")
print(f"    non-embed N    b = {pooled_new:+.4f}")
report["pooled_N_total"] = float(pooled_old)
report["pooled_N_non_embed"] = float(pooled_new)


# ------------------------------------------------------ 2. pooled compute fit
print("\n" + "=" * 72)
print("COMPUTE FIT   L(C) = a*C^b + c,  c >= 0")
print("=" * 72)

(a, b, c), (sa, sb, sc), r2 = fit_power_law(df["flops"], df["val_loss"])
print(f"  a = {a:.4g} +/- {sa:.2g}")
print(f"  b = {b:+.4f} +/- {sb:.4f}     (Kaplan: {KAPLAN['C']:+.3f})")
print(f"  c = {c:.4f} +/- {sc:.4f}     (irreducible loss, constrained >= 0)")
print(f"  R^2 = {r2:.4f}")
report["compute_fit"] = {"a": float(a), "b": float(b), "c": float(c), "r2": float(r2)}

slope_C, _, r2_C = loglog_slope(df["flops"], df["val_loss"])
print(f"  log-log slope (no offset term): b = {slope_C:+.4f}, R^2 = {r2_C:.4f}")
report["compute_slope"] = float(slope_C)


# -------------------------------------------------- 3. joint Chinchilla surface
print("\n" + "=" * 72)
print("JOINT FIT   L(N, D) = E + A/N^alpha + B/D^beta")
print("=" * 72)

xy = (df[N_COL].values.astype(float), df["n_tokens"].values.astype(float))
y = df["val_loss"].values.astype(float)

try:
    popt, pcov = curve_fit(
        chinchilla, xy, y,
        p0=[1.5, 400.0, 0.34, 400.0, 0.28],  # Chinchilla's own fitted values as seed
        bounds=([0, 0, 0, 0, 0], [10, 1e14, 2, 1e14, 2]),
        maxfev=500000,
    )
    E, A, alpha, B, beta = popt
    err = np.sqrt(np.diag(pcov))
    pred = chinchilla(xy, *popt)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    print(f"  E     = {E:.4f} +/- {err[0]:.4f}   (irreducible loss)")
    print(f"  A     = {A:.4g}")
    print(f"  alpha = {alpha:.4f} +/- {err[2]:.4f}   (N exponent)")
    print(f"  B     = {B:.4g}")
    print(f"  beta  = {beta:.4f} +/- {err[4]:.4f}   (D exponent)")
    print(f"  R^2   = {r2:.4f}   RMSE = {rmse:.4f} nats")
    print(f"\n  Chinchilla for reference: E=1.69, alpha=0.34, beta=0.28")
    print(f"  Implied compute-optimal scaling: N* ~ C^{beta/(alpha+beta):.3f}, "
          f"D* ~ C^{alpha/(alpha+beta):.3f}")
    print(f"  (Chinchilla: both ~ C^0.5; > 0.5 on D means data-hungrier than Chinchilla)")

    report["joint_fit"] = {
        "E": float(E), "A": float(A), "alpha": float(alpha),
        "B": float(B), "beta": float(beta), "r2": float(r2), "rmse": rmse,
        "N_exponent_of_C": float(beta / (alpha + beta)),
        "D_exponent_of_C": float(alpha / (alpha + beta)),
    }
except Exception as exc:  # keep the rest of the analysis usable
    print(f"  Joint fit failed: {exc}")
    report["joint_fit"] = None

print(f"\nUniform-random baseline over {VOCAB_SIZE} vocab: {RANDOM_BASELINE:.3f} nats")
print(f"Best run: {y.min():.3f}   Worst run: {y.max():.3f}")


# ----------------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Scaling Laws — Greek LM  (N = non-embedding parameters)", fontsize=14)

# Loss vs N, one fitted line per D-slice
for n_tokens, grp in df.groupby("n_tokens"):
    grp = grp.sort_values(N_COL)
    pts = axes[0].scatter(grp[N_COL], grp["val_loss"], zorder=5, s=55,
                          label=f"D={n_tokens/1e6:.0f}M")
    slope, intercept, _ = loglog_slope(grp[N_COL], grp["val_loss"])
    xs = np.logspace(np.log10(grp[N_COL].min()), np.log10(grp[N_COL].max()), 100)
    axes[0].plot(xs, np.exp(intercept) * xs**slope, "--", alpha=0.8,
                 color=pts.get_facecolor()[0], label=f"  b={slope:+.3f}")
axes[0].set_xlabel("Non-embedding parameters (N)")
axes[0].set_title("Loss vs Model Size")

# Loss vs D, one fitted line per N-slice
for n_params, grp in df.groupby(N_COL):
    grp = grp.sort_values("n_tokens")
    pts = axes[1].scatter(grp["n_tokens"], grp["val_loss"], zorder=5, s=55,
                          label=f"N={n_params/1e6:.2f}M")
    slope, intercept, _ = loglog_slope(grp["n_tokens"], grp["val_loss"])
    xs = np.logspace(np.log10(grp["n_tokens"].min()), np.log10(grp["n_tokens"].max()), 100)
    axes[1].plot(xs, np.exp(intercept) * xs**slope, "--", alpha=0.8,
                 color=pts.get_facecolor()[0], label=f"  b={slope:+.3f}")
axes[1].set_xlabel("Training tokens (D)")
axes[1].set_title("Loss vs Training Tokens")

# Loss vs compute, pooled fit
axes[2].scatter(df["flops"], df["val_loss"], color="purple", zorder=5, s=55)
xs = np.logspace(np.log10(df["flops"].min()), np.log10(df["flops"].max()), 200)
axes[2].plot(xs, power_law(xs, a, b, c), "--", color="purple",
             label=f"{a:.3g}·C^{b:.3f} + {c:.2f}")
axes[2].set_xlabel("FLOPs (C = 6ND)")
axes[2].set_title("Loss vs Compute")

for ax in axes:
    ax.axhline(RANDOM_BASELINE, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Validation loss (nats)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
axes[0].text(df[N_COL].min(), RANDOM_BASELINE * 1.02, "uniform-random baseline",
             fontsize=7, color="gray")

plt.tight_layout()
plt.savefig("plots/scaling_laws.png", dpi=150)
print("\nSaved plots/scaling_laws.png")

with open("scaling_fits.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved scaling_fits.json")
