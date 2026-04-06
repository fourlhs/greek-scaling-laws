set -e

# 3 models, each trained to 200M tokens
# Evaluates at 2M, 20M, 200M checkpoints
MODELS=(
    "4 64 4"     # ~1M
    "6 256 8"    # ~10M
    "12 768 12"  # ~100M
)

for model in "${MODELS[@]}"; do
    read n_layers d_model n_heads <<< $model
    echo "Running: layers=$n_layers d_model=$d_model"
    uv run train.py \
        --n_layers $n_layers \
        --d_model $d_model \
        --n_heads $n_heads
done

echo "All runs complete. Results in results.csv"
