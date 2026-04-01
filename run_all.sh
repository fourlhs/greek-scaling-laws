set -e

# 3 model sizes x 3 token counts = 9 runs
MODELS=(
    "4 64 4"     # ~1M
    "6 256 8"    # ~10M
    "12 768 12"  # ~100M
)

TOKENS=(2000000 20000000 200000000)

for model in "${MODELS[@]}"; do
    read n_layers d_model n_heads <<< $model
    for n_tokens in "${TOKENS[@]}"; do
        echo "Running: layers=$n_layers d_model=$d_model tokens=$n_tokens"
        python train.py \
            --n_layers $n_layers \
            --d_model $d_model \
            --n_heads $n_heads \
            --n_tokens $n_tokens
    done
done

echo "All runs complete. Results in results.csv"