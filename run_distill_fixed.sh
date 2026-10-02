#!/bin/bash
# Offline distillation on the REAL fineweb parquet logprobs with the tail-bucket sparse KL.
# 250M tokens each, same Muon settings as the CE sweep winner (val BPB 1.1164) for a direct comparison.
# GPU 0: pure KL (alpha=1.0).  GPU 1: alpha=0.9.
# Usage: uv run bash run_distill_fixed.sh
set -e
cd "$(dirname "$0")"
mkdir -p out

run() {
    local gpu=$1 alpha=$2
    local name="dfix_a${alpha}_mlr0.02_wd0.01"
    echo "[GPU $gpu] Starting $name"
    CUDA_VISIBLE_DEVICES=$gpu uv run python train.py config/distill_sweep_base.py \
        --alpha="$alpha" --out_dir="out/$name" --wandb_run_name="$name" \
        2>&1 | tee "out/${name}.log"
    echo "[GPU $gpu] Finished $name"
}

run 0 1.0 &
sleep 5
run 1 0.9 &
wait
echo "=== Fixed distill runs complete ==="
