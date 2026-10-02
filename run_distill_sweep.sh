#!/bin/bash
# L9(3^4) Taguchi distillation sweep: alpha x matrix_lr x temperature x weight_decay
# Offline mode: reads pre-computed top-k=128 logprobs, no teacher model loaded
# 3 rounds of 3 GPUs, ~1.7h per run, ~5.1h total
# Usage: uv run bash run_distill_sweep.sh
set -e
cd "$(dirname "$0")"
mkdir -p out

run() {
    local gpu=$1 alpha=$2 mlr=$3 temp=$4 wd=$5
    local name="dsweep_a${alpha}_mlr${mlr}_t${temp}_wd${wd}"
    echo "[GPU $gpu] Starting $name"
    CUDA_VISIBLE_DEVICES=$gpu uv run python train.py config/distill_sweep_base.py \
        --alpha="$alpha" --matrix_lr="$mlr" --temperature="$temp" --weight_decay="$wd" \
        --out_dir="out/$name" --wandb_run_name="$name" \
        2>&1 | tee "out/${name}.log"
    echo "[GPU $gpu] Finished $name"
}

# L9 orthogonal array: each factor at each level appears exactly 3 times
echo "--- Round 1/3 ---"
run 0  0.5  0.01  1.0  0.005 &
sleep 5
run 1  0.5  0.02  2.0  0.01  &
sleep 5
run 2  0.5  0.04  4.0  0.05  &
wait

echo "--- Round 2/3 ---"
run 0  0.9  0.01  2.0  0.05  &
sleep 5
run 1  0.9  0.02  4.0  0.005 &
sleep 5
run 2  0.9  0.04  1.0  0.01  &
wait

echo "--- Round 3/3 ---"
run 0  1.0  0.01  4.0  0.01  &
sleep 5
run 1  1.0  0.02  1.0  0.05  &
sleep 5
run 2  1.0  0.04  2.0  0.005 &
wait

echo "=== Distill sweep complete ==="
