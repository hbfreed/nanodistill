#!/bin/bash
# KL-focused learning-rate sweep (alpha=0.9, real parquet logprobs, tail-bucket sparse KL).
# 250M tokens per run, batch 12, compiled backbone. ~2.4 h per run, 3 rounds of 3 GPUs.
# Reference (same code): CE control 1.1027, alpha=0.9 base 1.1026, pure KL 1.1235 (val BPB).
#   Round 1: matrix_lr {0.02, 0.04, 0.06} at unembedding_lr 0.008, wd 0.01  (0.02 cell = replicate of base -> noise estimate)
#   Round 2: same matrix_lr grid at unembedding_lr 0.016
#   Round 3: wd 0.05 at mlr 0.04 (both ulr), and a long-stable-phase cell (cooldown 0.1) at base
# Usage: nohup uv run bash run_lr_sweep.sh > out/run_lr_sweep.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")"
mkdir -p out
export PYTHONUNBUFFERED=1

run() {
    local gpu=$1 mlr=$2 ulr=$3 wd=$4 cd=$5
    local name="lrsweep_mlr${mlr}_ulr${ulr}_wd${wd}_cd${cd}"
    echo "[$(date +%H:%M)] [GPU $gpu] Starting $name"
    CUDA_VISIBLE_DEVICES=$gpu uv run python train.py config/distill_sweep_base.py \
        --alpha=0.9 --matrix_lr="$mlr" --unembedding_lr="$ulr" --weight_decay="$wd" --cooldown_fraction="$cd" \
        --out_dir="out/$name" --wandb_run_name="$name" \
        > "out/${name}.log" 2>&1 || echo "[GPU $gpu] $name FAILED (see out/${name}.log)"
    echo "[$(date +%H:%M)] [GPU $gpu] Finished $name"
}

echo "--- Round 1/3 ---"
run 0 0.02 0.008 0.01 0.2 &
sleep 60
run 1 0.04 0.008 0.01 0.2 &
sleep 60
run 2 0.06 0.008 0.01 0.2 &
wait

echo "--- Round 2/3 ---"
run 0 0.02 0.016 0.01 0.2 &
sleep 60
run 1 0.04 0.016 0.01 0.2 &
sleep 60
run 2 0.06 0.016 0.01 0.2 &
wait

echo "--- Round 3/3 ---"
run 0 0.04 0.008 0.05 0.2 &
sleep 60
run 1 0.04 0.016 0.05 0.2 &
sleep 60
run 2 0.02 0.008 0.01 0.1 &
wait

echo "=== LR sweep complete ==="
for f in out/lrsweep_*.log; do echo "$(basename $f .log): $(grep -h '(final)' $f | tail -1)"; done
