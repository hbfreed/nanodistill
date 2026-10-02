#!/bin/bash
# Generate logprobs for all 3 OLMo-3 variants in parallel, one per GPU.

# GPU 0: base
CUDA_VISIBLE_DEVICES=0 uv run python generate_logprobs.py \
    --teacher allenai/Olmo-3-1025-7B \
    --data data/fineweb_edu/train.bin \
    --batch_size 8 \
    --max_chunks 488281 \
    --output /media/henry/MoreFiles/olmo3-base-logprobs &

sleep 10

# GPU 1: instruct
CUDA_VISIBLE_DEVICES=1 uv run python generate_logprobs.py \
    --teacher allenai/Olmo-3-7B-Instruct \
    --data data/fineweb_edu/train.bin \
    --batch_size 12 \
    --max_chunks 488281 \
    --output /media/henry/MoreFiles/olmo3-instruct-logprobs &

sleep 10

# GPU 2: think
CUDA_VISIBLE_DEVICES=2 uv run python generate_logprobs.py \
    --teacher allenai/Olmo-3-7B-Think \
    --data data/fineweb_edu/train.bin \
    --batch_size 14 \
    --max_chunks 488281 \
    --output /media/henry/MoreFiles/olmo3-think-logprobs &

wait
echo "All done!"
