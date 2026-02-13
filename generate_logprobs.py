"""Pre-compute teacher top-k logprobs for distillation.

Reads tokenized data from a .bin memmap file, runs a teacher model,
and saves (input_ids, topk_indices, topk_logprobs) as an HF Dataset.

Adapted from variable-flex-olmo/scripts/generate_logprobs.py.

Usage:
  # Single GPU:
  uv run python generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --output hbfreed/olmo7b-fineweb-logprobs

  # Multi-GPU data parallelism (each GPU gets a copy of the model):
  torchrun --standalone --nproc_per_node=4 generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --output hbfreed/olmo7b-fineweb-logprobs

  # Single-process model parallelism (large model split across GPUs):
  uv run python generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --device_map auto \
      --output hbfreed/olmo7b-fineweb-logprobs

  # Quantized teacher (int4):
  uv run python generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --quantize int4 \
      --output hbfreed/olmo7b-fineweb-logprobs-int4
"""

import argparse
import os

import numpy as np
import torch
from datasets import Dataset
from torch.distributed import destroy_process_group, init_process_group
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

torch.set_float32_matmul_precision("high")


@torch.compile
def extract_topk_logprobs(logits, k):
    """Extract top-k logprobs from raw logits efficiently.

    Computes log_softmax only over the top-k values using the logsumexp trick,
    avoiding a full-vocabulary log_softmax.
    """
    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_logprobs = topk_logits - logsumexp
    return topk_logprobs.half(), topk_indices.int()


def main():
    parser = argparse.ArgumentParser(description="Generate teacher logprobs for distillation")
    parser.add_argument("--teacher", type=str, required=True, help="HF model name for teacher")
    parser.add_argument("--data", type=str, required=True, help="Path to tokenized .bin file")
    parser.add_argument("--n_ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--top_k", type=int, default=128, help="Number of top logprobs to save")
    parser.add_argument("--output", type=str, required=True, help="HF Hub repo or local directory")
    parser.add_argument("--max_chunks", type=int, default=-1, help="Max chunks to process (-1 for all)")
    parser.add_argument("--quantize", type=str, default=None, choices=["int8", "int4"],
                        help="Quantize teacher model (int8 or int4 via bitsandbytes)")
    parser.add_argument("--device_map", type=str, default=None,
                        help="Device map for model loading (default: auto for single-process, local GPU for torchrun)")
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    use_compile = args.compile and not args.no_compile

    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda"

    is_main = rank == 0

    # Resolve device_map: torchrun = each rank gets its own GPU, single-process = auto
    if args.device_map is not None:
        device_map = args.device_map
    elif ddp:
        device_map = {"": device}  # whole model on this rank's GPU
    else:
        device_map = "auto"

    # Load tokenized data as memmap and chunk into sequences
    if is_main:
        print(f"Loading tokenized data from {args.data}...")
    data = np.memmap(args.data, dtype=np.uint32, mode="r")
    num_tokens = len(data)
    num_chunks = num_tokens // args.n_ctx
    if args.max_chunks > 0:
        num_chunks = min(num_chunks, args.max_chunks)
    if is_main:
        print(f"Total tokens: {num_tokens:,}, chunks of {args.n_ctx}: {num_chunks:,}")

    # Shard chunks across ranks
    all_chunk_indices = list(range(num_chunks))
    my_chunk_indices = all_chunk_indices[rank::world_size]
    if is_main:
        print(f"Rank {rank}/{world_size}: processing {len(my_chunk_indices)} chunks")

    # Build this rank's chunks tensor (on CPU, pinned for async transfer)
    chunks = torch.zeros(len(my_chunk_indices), args.n_ctx, dtype=torch.long)
    for i, chunk_idx in enumerate(my_chunk_indices):
        start = chunk_idx * args.n_ctx
        chunks[i] = torch.from_numpy(data[start : start + args.n_ctx].astype(np.int64))
    chunks = chunks.pin_memory()

    # Load teacher model
    quant_config = None
    if args.quantize == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        if is_main:
            print(f"Loading teacher model (int8): {args.teacher}...")
    elif args.quantize == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        if is_main:
            print(f"Loading teacher model (int4/nf4): {args.teacher}...")
    else:
        if is_main:
            print(f"Loading teacher model (bf16): {args.teacher}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        quantization_config=quant_config,
    )
    model.eval()

    if use_compile:
        if is_main:
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    # Arrow list columns use 32-bit offsets; cap rows per shard to stay under ~2GB
    per_row_topk = args.n_ctx * args.top_k
    max_rows_by_bytes = (2**31 - 1) // (per_row_topk * 4)  # int32 is tightest
    rows_per_shard = max(1, max_rows_by_bytes)

    rows = []
    shard_idx = 0
    num_batches = len(chunks) // args.batch_size

    # Determine if output is local or hub
    is_hub = "/" in args.output and not args.output.startswith(".")

    # Use rank-prefixed shard names to avoid collisions
    shard_prefix = f"rank{rank}_" if world_size > 1 else ""

    for batch_start in tqdm(
        range(0, num_batches * args.batch_size, args.batch_size),
        desc=f"[rank {rank}] Processing",
        disable=not is_main,
    ):
        batch = chunks[batch_start : batch_start + args.batch_size].to(
            model.device, non_blocking=True
        )

        with torch.inference_mode():
            logits = model(batch).logits  # [batch, seq, vocab]

        topk_logprobs, topk_indices = extract_topk_logprobs(logits, args.top_k)

        # Transfer to CPU
        batch_ids = chunks[batch_start : batch_start + args.batch_size].numpy()
        topk_indices_np = topk_indices.cpu().numpy()
        topk_logprobs_np = topk_logprobs.cpu().numpy()

        for i in range(len(batch_ids)):
            rows.append({
                "input_ids": batch_ids[i],
                "topk_indices": topk_indices_np[i],
                "topk_logprobs": topk_logprobs_np[i],
            })

        # Flush shard if needed
        if len(rows) >= rows_per_shard:
            shard = Dataset.from_list(rows)
            split_name = f"train_{shard_prefix}{shard_idx}"
            if is_hub:
                shard.push_to_hub(args.output, split=split_name)
            else:
                shard.save_to_disk(f"{args.output}/{split_name}")
            print(f"[rank {rank}] Saved shard {split_name} ({len(rows)} rows)")
            rows = []
            shard_idx += 1

    # Flush remaining rows
    if rows:
        shard = Dataset.from_list(rows)
        split_name = f"train_{shard_prefix}{shard_idx}"
        if is_hub:
            shard.push_to_hub(args.output, split=split_name)
        else:
            shard.save_to_disk(f"{args.output}/{split_name}")
        print(f"[rank {rank}] Saved final shard {split_name} ({len(rows)} rows)")

    if is_main:
        total_shards = (shard_idx + 1) * world_size
        print(f"Done! ~{total_shards} total shards across {world_size} rank(s)")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
