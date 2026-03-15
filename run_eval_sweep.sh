#!/bin/bash
# Eval-only pass for sweep runs missing final eval (rounds 1 & 2)
set -e
cd "$(dirname "$0")"

run_eval() {
    local gpu=$1 mlr=$2 wd=$3
    local name="sweep_mlr${mlr}_wd${wd}"
    echo "[GPU $gpu] Eval $name"
    CUDA_VISIBLE_DEVICES=$gpu uv run python train.py config/sweep_base.py \
        --matrix_lr="$mlr" --weight_decay="$wd" \
        --out_dir="out/$name" --wandb_run_name="${name}-eval" \
        --init_from=resume --eval_only=True \
        2>&1 | tee -a "out/${name}.log"
    echo "[GPU $gpu] Done $name"
}

echo "--- Round 1 evals ---"
run_eval 0 0.01 0.01 &
run_eval 1 0.01 0.1  &
run_eval 2 0.01 0.2  &
wait

echo "--- Round 2 evals ---"
run_eval 0 0.02 0.01 &
run_eval 1 0.02 0.1  &
run_eval 2 0.02 0.2  &
wait

echo "=== All evals done ==="
