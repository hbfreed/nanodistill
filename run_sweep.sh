#!/bin/bash
# Muon + CCE hyperparameter sweep: matrix_lr x weight_decay
# Step 0: sanity check (1 run, GPU 0)
# Step 1: 3x3 grid (3 rounds of 3 GPUs)
# Usage: uv run bash run_sweep.sh
set -e
cd "$(dirname "$0")"
mkdir -p out

run() {
    local gpu=$1 mlr=$2 wd=$3
    local name="sweep_mlr${mlr}_wd${wd}"
    echo "[GPU $gpu] Starting $name"
    CUDA_VISIBLE_DEVICES=$gpu uv run python train.py config/sweep_base.py \
        --matrix_lr="$mlr" --weight_decay="$wd" \
        --out_dir="out/$name" --wandb_run_name="$name" \
        2>&1 | tee "out/${name}.log"
    echo "[GPU $gpu] Finished $name"
}

echo "--- Round 1/3 ---"
run 0 0.01 0.01 &
run 1 0.01 0.1  &
run 2 0.01 0.2  &
wait

echo "--- Round 2/3 ---"
run 0 0.02 0.01 &
run 1 0.02 0.1  &
run 2 0.02 0.2  &
wait

echo "--- Round 3/3 ---"
run 0 0.04 0.01 &
run 1 0.04 0.1  &
run 2 0.04 0.2  &
wait

echo "=== Sweep complete ==="
