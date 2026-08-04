#!/usr/bin/env bash
#
# Full pipeline: dependencies -> tokenizer -> 12-run sweep -> scaling law analysis.
# Reproduces every number in README.md and report/ from a clean clone.
#
#   bash run_all.sh              # full pipeline
#   SKIP_INSTALL=1 bash run_all.sh    # deps already installed
#   KEEP_RESULTS=1 bash run_all.sh    # append to an existing results.csv

set -euo pipefail

cd "$(dirname "$0")"

TOKENIZER="greek_bpe_tokenizer.json"
RESULTS="results.csv"

# 4 model sizes x 3 token counts = 12 runs
MODELS=(
    "4 64 4"    # ~1.2M params total / 0.20M non-embedding
    "4 128 4"   # ~2.9M params total / 0.79M non-embedding
    "6 256 8"   # ~8.9M params total / 4.73M non-embedding
    "8 512 8"   # ~33.5M params total / 25.2M non-embedding
)

TOKENS=(5000000 20000000 80000000)

# ---------------------------------------------------------------- dependencies
if [ -z "${SKIP_INSTALL:-}" ]; then
    echo "==> Installing dependencies"
    if command -v uv >/dev/null 2>&1; then
        uv sync
    else
        python -m pip install --quiet --upgrade pip
        python -m pip install --quiet -e .
    fi
else
    echo "==> Skipping dependency install (SKIP_INSTALL set)"
fi

# ------------------------------------------------------------------ tokenizer
# Not committed (it is a build artifact), so train it on first run. Trained once
# and reused byte-identically across all 12 runs, keeping D in a consistent unit.
if [ -f "$TOKENIZER" ]; then
    echo "==> Tokenizer already present: $TOKENIZER"
else
    echo "==> Training BPE tokenizer (once, ~10 min)"
    python train_tokenizer.py
fi

if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: $TOKENIZER missing after training step; aborting." >&2
    exit 1
fi

# -------------------------------------------------------------------- results
# train.py appends, so a stale results.csv would silently double the grid.
if [ -n "${KEEP_RESULTS:-}" ]; then
    echo "==> Appending to existing $RESULTS (KEEP_RESULTS set)"
elif [ -f "$RESULTS" ]; then
    echo "==> Archiving previous $RESULTS -> ${RESULTS}.bak"
    mv "$RESULTS" "${RESULTS}.bak"
fi

mkdir -p curves plots

# ---------------------------------------------------------------------- sweep
run=0
total=$(( ${#MODELS[@]} * ${#TOKENS[@]} ))

for model in "${MODELS[@]}"; do
    read -r n_layers d_model n_heads <<< "$model"
    for n_tokens in "${TOKENS[@]}"; do
        run=$(( run + 1 ))
        echo ""
        echo "==> Run ${run}/${total}: layers=${n_layers} d_model=${d_model} heads=${n_heads} tokens=${n_tokens}"
        python train.py \
            --n_layers "$n_layers" \
            --d_model "$d_model" \
            --n_heads "$n_heads" \
            --n_tokens "$n_tokens"
    done
done

# ------------------------------------------------------------------- analysis
echo ""
echo "==> Fitting scaling laws and rendering plots"
python scripts/analyze.py
python scripts/plot_curves.py
python scripts/plot_isoflops.py

echo ""
echo "All ${total} runs complete."
echo "  results -> $RESULTS"
echo "  curves  -> curves/"
echo "  plots   -> plots/"
