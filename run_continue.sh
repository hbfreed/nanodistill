#!/bin/bash
# Continued pretraining: CE control vs distillation from 1B baseline
# 3 runs in parallel on 3 GPUs, ~3h each
# Usage: uv run bash run_continue.sh
set -e
cd "$(dirname "$0")"
mkdir -p out

LOGPROB_DIR="/media/henry/MoreFiles/teacher_logprobs_pl2048"

echo "--- CE control + distill alpha=0.5 + distill alpha=0.9 ---"

# GPU 0: CE continuation (control)
CUDA_VISIBLE_DEVICES=0 uv run python train.py config/continue_base.py \
    --mode="ce" \
    --out_dir="out/continue_ce" --wandb_run_name="continue-ce" \
    2>&1 | tee "out/continue_ce.log" &

sleep 5

# GPU 1: Distill alpha=0.5 (from-scratch sweep winner)
CUDA_VISIBLE_DEVICES=1 uv run python train.py config/continue_base.py \
    --mode="distill" --logprob_dir="$LOGPROB_DIR" \
    --alpha=0.5 --temperature=1.0 \
    --out_dir="out/continue_distill_a0.5" --wandb_run_name="continue-distill-a0.5" \
    2>&1 | tee "out/continue_distill_a0.5.log" &

sleep 5

# GPU 2: Distill alpha=0.9 (Peng et al. — should work with pretrained student)
CUDA_VISIBLE_DEVICES=2 uv run python train.py config/continue_base.py \
    --mode="distill" --logprob_dir="$LOGPROB_DIR" \
    --alpha=0.9 --temperature=1.0 \
    --out_dir="out/continue_distill_a0.9" --wandb_run_name="continue-distill-a0.9" \
    2>&1 | tee "out/continue_distill_a0.9.log" &

wait
echo "=== Continuation experiment complete ==="
