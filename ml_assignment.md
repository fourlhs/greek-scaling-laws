# Take-Home Assignment: Deriving Scaling Laws from Scratch
**Role:** Senior Machine Learning Engineer  

---

## Overview

Your task is to **empirically derive scaling laws** for a small Greek-language language model by running a structured set of pretraining experiments. You will train multiple models of varying sizes on varying amounts of data, collect validation loss measurements, and fit a scaling law that describes how model performance changes with compute, parameters, and data.

This is inspired by [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) and [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) — at a much smaller scale.

All experiments must be runnable on **CPU** or a free **Google Colab** instance (T4 GPU or CPU runtime).

---

## Dataset

Use the following HuggingFace dataset:

```
alexliap/tinystories-gr
```

The dataset contains **2.14M Greek children's stories** (translated from the original TinyStories corpus). You must use **only the `greek_translation` column** as your training corpus. Ignore `original_text` entirely — the model should never see any English text.

---

## Requirements

### 1. Tokenizer — Train from Scratch

You must train a **BPE tokenizer from scratch** on the Greek training corpus using the `tokenizers` library. Do **not** use a pretrained tokenizer (e.g. GPT-2 BPE, SentencePiece trained on English data).

Your tokenizer must:
- Be trained exclusively on the `greek_translation` column — no English text should influence the vocabulary
- Handle Greek Unicode correctly
- Have a vocabulary size you choose and justify (hint: Greek morphology is rich — consider what vocab size makes sense)
- Be saved and reused identically across all model runs

---

### 2. Model Architecture

You may implement the model using any of the following frameworks — choose what fits your environment:

- PyTorch Lightning
- HuggingFace transformers — you may use model classes (e.g. GPT2LMHeadModel with a custom config), but weights must still be randomly initialized
- MLX (recommended if you are on Apple Silicon / macOS) — use mlx-lm or implement directly with mlx.nn

---

### 3. Experiment Grid

Run a structured sweep across **two axes**:

| Axis | Suggested values |
|---|---|
| **Model size (N)** | ~ 1M, 3M, 10M, 30M parameters |
| **Training tokens (D)** | ~ 5M, 20M, 80M tokens |

This gives a grid of up to 12 runs. The goal is to observe the trend, not to fully converge. 

```
Important: The above sizes are examples, you can come up with ones that suit your system better.
```

For **each run**, record:

| Field | Description |
|---|---|
| `n_params` | Number of trainable parameters |
| `n_tokens` | Number of training tokens consumed |
| `flops` | Estimated as `~ 6 × N × D` |
| `val_loss` | Final validation loss |

Save results to a CSV or JSON file at the end of each run.

### 5. Scaling Law Analysis

After collecting all runs:

1. **Plot** validation loss vs. N, D, and FLOPs on a **log-log scale**
2. **Fit a power law** of the form `L(X) = a · X^b + c` to your data
3. **Identify the compute-optimal frontier**: for a fixed FLOP budget, what is the optimal (N, D) allocation according to your data?

---

### 6. Reproducibility

A shell script (e.g. run_all.sh) that installs dependencies and executes the full sweep in one command, alongside the relevant Python source files

---

## Deliverables

1. Runnable code — full pipeline from tokenizer training to scaling law plots, delivered as a shell script (`run_all.sh`) + Python source files for script-based submissions
2. PDF report covering the full methodology and results, including:

- Dataset preprocessing decisions (filtering strategy, train/val split)
- Tokenizer design choices (vocab size, normalization, special tokens)
- Model architecture and experiment grid (configs table)
- Training curves (loss over steps) for each run
- Scaling law plots on log-log axes (loss vs. N, D, and FLOPs)
- Fitted power-law exponents vs. literature values
- Compute-optimal (N, D) frontier derived from your data
- Discussion of limitations and what you would improve given more compute
3. **`results.csv`** — one row per run with columns: `n_params`, `n_tokens`, `flops`, `val_loss`

---

## Evaluation Criteria

| Area | What we look for |
|---|---|
| **Data pipeline** | Correct tokenizer training, sensible vocab size for Greek, clean preprocessing (*if any*) |
| **Experiment design** | Is the sweep well-structured to isolate the two scaling axes cleanly? |
| **ML fundamentals** | Correct training loop, valid loss measurement, no data leakage |
| **Scaling law analysis** | Correct power-law fitting and meaningful interpretation of exponents |
| **Critical thinking** | Does the candidate reason about deviations, dataset quirks, and limitations? |
| **Code quality** | Clean, modular, reproducible code |
| **Communication** | Clear README with insightful plots and honest discussion of trade-offs |

---

## Rules & Notes

- Do **not** fine-tune or adapt a pretrained model — all weights must be randomly initialized per run
- Do not share this assignment publicly

---

## Useful References

- Kaplan et al. (2020) — [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022) — [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)
- Sardana & Frankle (2023) — [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) *(bonus reading)*
- HuggingFace `tokenizers` — [BPE Trainer docs](https://huggingface.co/docs/tokenizers)

---
