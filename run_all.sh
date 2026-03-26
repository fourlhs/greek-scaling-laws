#!/bin/bash
set -e

# 4 model sizes x 3 token counts = 12 runs
MODELS=(
    "4 64 4"    # 1.2M
    "4 128 4"   # 2.9M
    "6 256 8"   # 8.9M
    "8 512 8"   # 33.5M
)

TOKENS=(5000000 20000000 80000000)

for model in "${MODELS[@]}"; do
    read n_layers d_model n_heads <<< $model
    for n_tokens in "${TOKENS[@]}"; do
        echo "Running: layers=$n_layers d_model=$d_model tokens=$n_tokens"
        uv run train.py \
            --n_layers $n_layers \
            --d_model $d_model \
            --n_heads $n_heads \
            --n_tokens $n_tokens
    done
done

echo "All runs complete. Results in results.csv"