---
title: "Deriving Scaling Laws from Scratch"
author: "Nikos Fourlis"
date: "April 2026"
geometry: margin=2.5cm
fontsize: 11pt
---

# Overview

This report presents an empirical derivation of scaling laws for a small Greek-language language model. Multiple models of varying sizes were trained on varying amounts of data, and the resulting validation losses were used to fit power-law relationships between model performance, parameter count, training tokens, and compute (FLOPs).

---

# Dataset

**Source:** `alexliap/tinystories-gr` — 2.14M Greek children's stories translated from the TinyStories corpus.

**Preprocessing:**

- Only the `greek_translation` column was used; English text was never seen by the model
- Text was tokenized and concatenated with `<|endoftext|>` separators
- Flattened token stream was chunked into sequences of length 257 (256 input + 1 target)
- Chunks were shuffled once with a fixed seed (`numpy.random.default_rng(42)`) and cached to disk
- 90/10 train/val split applied after shuffling

---

# Tokenizer

A BPE tokenizer was trained from scratch on the Greek corpus using the HuggingFace `tokenizers` library.

| Parameter | Value |
|---|---|
| Algorithm | Byte-level BPE |
| Vocabulary size | 16,000 |
| Special tokens | `<\|endoftext\|>`, `<\|pad\|>` |
| Pre-tokenizer | `ByteLevel` (preserves case and whitespace) |
| Decoder | `ByteLevel` |

**Justification:** Greek has rich morphology with many inflectional suffixes. A vocabulary of 16,000 subword tokens allows the tokenizer to capture common stems and suffixes without fragmenting too aggressively. Byte-level BPE was chosen to handle all Unicode characters natively without any unknown tokens.

---

# Model Architecture

A decoder-only transformer (GPT-style) was implemented in PyTorch.

| Component | Details |
|---|---|
| Attention | Causal multi-head self-attention (`scaled_dot_product_attention`) |
| Normalization | Pre-LayerNorm |
| FFN | Linear → GELU → Linear (4× expansion) |
| Positional encoding | Learned embeddings |
| Weight tying | Token embedding = LM head weights |

---

# Experiment Grid

Models were trained across 4 sizes and 3 token counts, giving 12 runs total.

| Model | Layers | d_model | Heads | Total params | Non-embedding N | Embedding share |
|---|---|---|---|---|---|---|
| Small | 4 | 64 | 4 | 1.24M | 0.20M | 84.0% |
| Medium | 4 | 128 | 4 | 2.87M | 0.79M | 72.5% |
| Large | 6 | 256 | 8 | 8.89M | 4.73M | 46.8% |
| XLarge | 8 | 512 | 8 | 33.51M | 25.18M | 24.8% |

**Training tokens:** 5M, 20M, 80M

**Other hyperparameters:** batch size 32, AdamW optimizer, lr = 3e-4, no scheduler.

**On the definition of N.** All fits below use **non-embedding** parameters, following Kaplan et al. (§2.1). With a 16,000-token vocabulary, the embedding table dominates the smaller models — 84% of all parameters at `d_model=64` — and its share falls monotonically as models grow. Counting embeddings would therefore make the N axis measure vocabulary size confounded with capacity, and would compress the true 127× capacity range into an apparent 27×. FLOPs are likewise computed as `C = 6·N_non-embedding·D`, since embedding lookups are gathers rather than matrix multiplies.

---

# Results

## Validation Loss Table

| N (non-emb) | D (tokens) | FLOPs | Tokens/param | Val Loss |
|---|---|---|---|---|
| 0.20M | 5M | 5.9e12 | 25 | 8.156 |
| 0.20M | 20M | 2.4e13 | 101 | 5.643 |
| 0.20M | 80M | 9.5e13 | 405 | 4.211 |
| 0.79M | 5M | 2.4e13 | 6 | 7.294 |
| 0.79M | 20M | 9.5e13 | 25 | 5.299 |
| 0.79M | 80M | 3.8e14 | 101 | 3.818 |
| 4.73M | 5M | 1.4e14 | 1 | 6.653 |
| 4.73M | 20M | 5.7e14 | 4 | 4.798 |
| 4.73M | 80M | 2.3e15 | 17 | 3.207 |
| 25.18M | 5M | 7.6e14 | 0.2 | 6.061 |
| 25.18M | 20M | 3.0e15 | 0.8 | 4.497 |
| 25.18M | 80M | 1.2e16 | 3 | 2.790 |

For reference, a uniform predictor over the 16,000-token vocabulary achieves a loss of ln(16000) = 9.68 nats. The weakest configuration improves on that by only 1.5 nats, which is relevant context for interpreting the low-token runs.

## Training Curves

![Training curves: loss over tokens for each model size](../plots/training_curves.png)

## Scaling Law Plots

![Scaling laws: loss vs N, D, and FLOPs on log-log axes](../plots/scaling_laws.png)

---

# Scaling Law Analysis

## Marginal Exponents

A marginal exponent requires the other axis to be held fixed. Fitting `L(N)` across all 12 runs while `D` also varies lets the data effect leak into the parameter exponent, so each exponent below is fit **one slice at a time** — the log-log slope of loss against one axis with the other held constant — and reported as a mean over slices with the spread across slices.

| Axis | Per-slice exponents | Mean | Kaplan et al. (2020) |
|---|---|---|---|
| Parameters (N) | -0.060, -0.048, -0.087 | **-0.065 ± 0.016** | -0.076 |
| Training tokens (D) | -0.238, -0.234, -0.263, -0.280 | **-0.254 ± 0.019** | -0.095 |

Individual slices fit their power laws tightly (R² between 0.98 and 0.9999), so the exponents are well estimated; the interpretive question is what they measure.

## Comparison with Literature

**Parameters.** The measured exponent of -0.065 ± 0.016 brackets Kaplan's -0.076. Given that this sweep spans roughly 1.3 decades of N against Kaplan's six, the agreement is closer than the scale difference would justify expecting, and should be read as encouraging rather than as a precise confirmation.

**Training tokens.** At -0.254, the data exponent is 2.7× steeper than Kaplan's -0.095, and this is a genuine deviation rather than a scale effect. The cause is identifiable: with a constant learning rate and no schedule, increasing D also increases the number of optimizer steps taken (611, 2,442 and 9,766 steps for the three budgets). Because no run reached convergence, the D axis conflates "how much data the model saw" with "how far into optimization training was stopped," and both contribute to the loss reduction. The exponent should be read as an upper bound on the true data exponent for this setup.

## Joint Fit

Fitting the Chinchilla parameterisation over all 12 points simultaneously separates the axes by construction rather than by slicing:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

| Parameter | Value | Chinchilla |
|---|---|---|
| E (irreducible loss) | 0.00 ± 1.67 | 1.69 |
| α (N exponent) | 0.253 ± 0.181 | 0.34 |
| β (D exponent) | 0.334 ± 0.127 | 0.28 |
| R² | 0.987 | — |
| RMSE | 0.179 nats | — |

The surface describes the data well, but the parameters are weakly identified and the error bars carry the real message. `E` collapses to its lower bound of zero because no run approached its loss floor, leaving no curvature from which to estimate an irreducible term. `α` and `β` overlap substantially, so while the point estimates imply a compute-optimal allocation of `N* ∝ C^0.57` and `D* ∝ C^0.43`, this is **not** statistically distinguishable from Chinchilla's balanced `C^0.5` on both axes. Five parameters against 12 points spanning 1.3 decades is enough to be informative and not enough to be conclusive.

---

# Compute-Optimal Allocation

The following plot shows all 12 runs on a single loss vs FLOPs chart. Each colour represents a model size and each marker shape a token count. Lines connect runs of the same model trained on increasing amounts of data.

![Loss vs Compute: Different (N, D) Allocations](../plots/isoflops.png)

## Matched-budget comparisons

The grid is factorial rather than iso-FLOP, but it contains two pairs of runs whose compute budgets coincide to within 0.5%. These give a direct read on allocation, holding compute fixed and varying only the split between parameters and data:

| Compute | Allocation A | Loss | Allocation B | Loss | Margin |
|---|---|---|---|---|---|
| 2.37e13 | N=0.20M, D=20M (101 tok/param) | **5.643** | N=0.79M, D=5M (6 tok/param) | 7.294 | 1.651 |
| 9.46e13 | N=0.20M, D=80M (405 tok/param) | **4.211** | N=0.79M, D=20M (25 tok/param) | 5.299 | 1.088 |

In both pairs the smaller model trained on more data wins, and by a wide margin. This is directionally consistent with Hoffmann et al.: the models in this grid are severely undertrained, with the largest configuration seeing just 0.2 tokens per parameter against Chinchilla's recommended ~20.

Two caveats keep this from being a clean confirmation. First, the winning allocations sit at 101 and 405 tokens per parameter — far past Chinchilla's optimum — which suggests the apparent value of extra data is inflated by the constant-learning-rate confound discussed above. Second, the pooled compute fit is poorly determined:

$$L(C) = a \cdot C^{b} + c, \qquad b = -0.148 \pm 0.218, \quad R^2 = 0.66$$

The uncertainty exceeds the estimate. This is a property of the experimental design, not a failure of fitting: a factorial grid places runs of equal compute at very different (N, D) splits and therefore very different losses, so compute alone is a poor predictor. Establishing a compute-optimal frontier properly requires iso-FLOP profiles — fixing a budget and sweeping the allocation to locate the minimum — which is the natural next experiment.

---

# Limitations

- **No learning rate schedule, and no run converged.** This is the dominant limitation. A constant lr=3e-4 was used with no warmup or decay, so increasing D also increases optimizer steps, and the D exponent measures optimization progress alongside data. The three budgets correspond to 611, 2,442 and 9,766 steps; the shortest runs end at 6–8 nats against a 9.68-nat random baseline, meaning they capture the initial transient rather than converged performance. A cosine schedule with warmup, length-matched per run, is the first correction to make.
- **Factorial rather than iso-FLOP grid.** Well suited to isolating the N and D axes cleanly, poorly suited to estimating a compute exponent or locating an allocation frontier (R² = 0.66 on the compute fit).
- **Untuned learning rate across a 127× parameter range.** The optimal learning rate shifts with model width, so a single value handicaps some sizes relative to others in an unknown direction.
- **Unseeded model initialisation.** The data shuffle is seeded (`default_rng(42)`), but `torch.manual_seed` is never called, so runs are not bitwise reproducible. With one run per grid cell there is no variance estimate; the uncertainties quoted here are spreads across slices, not across seeds.
- **Five parameters against 12 points.** The joint fit spans roughly 1.3 decades in each axis, leaving `E` unidentified and `α`, `β` with overlapping confidence intervals.
- **Tokenizer trained on the full corpus,** validation split included. The effect on measured loss is small, but it is a train/test contact point and is disclosed rather than dismissed.
- **Dataset quality:** The dataset is machine-translated Greek, so translation artifacts, a narrow register and a child-directed vocabulary all mean these exponents describe this corpus rather than the Greek language.
- **Memory footprint of dataset construction:** the token stream is accumulated in a Python list before conversion to `np.uint16`, which peaks at tens of gigabytes on the full corpus; streaming into a preallocated array would remove this.

# What I Would Do With More Compute

1. Warmup plus cosine decay scaled to each run's token budget, so that D measures data rather than optimizer progress — this alone should bring the D exponent substantially closer to the literature value.
2. Iso-FLOP profiles at three or four fixed budgets, sweeping the (N, D) split within each, to locate the compute-optimal frontier directly instead of inferring it from a factorial grid.
3. A short learning-rate sweep per model size, so the comparison across N is not confounded by a single untuned value.
4. Three seeds per configuration, to distinguish genuine curvature from run-to-run noise.
5. Extended token budgets for the larger models, so that at least the upper end of the grid approaches its loss floor and the fitted irreducible term `E` becomes identifiable.

---

# References

- Kaplan et al. (2020) — *Scaling Laws for Neural Language Models*
- Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models (Chinchilla)*
- Sardana & Frankle (2023) — *Beyond Chinchilla-Optimal*
