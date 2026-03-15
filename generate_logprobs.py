"""Pre-compute teacher top-k logprobs for distillation.

Reads tokenized data from a .bin memmap file, runs a teacher model,
and saves (input_ids, topk_indices, topk_logprobs) as parquet shards.

Adapted from variable-flex-olmo/scripts/generate_logprobs.py.

Usage:
  # Single GPU:
  uv run python generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --output /media/henry/MoreFiles/olmo3-base-logprobs

  # Multi-GPU data parallelism (each GPU gets a copy of the model):
  torchrun --standalone --nproc_per_node=4 generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --output /media/henry/MoreFiles/olmo3-base-logprobs

  # Single-process model parallelism (large model split across GPUs):
  uv run python generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --device_map auto \
      --output /media/henry/MoreFiles/olmo3-base-logprobs

  # Quantized teacher (int4):
  uv run python generate_logprobs.py \
      --teacher allenai/Olmo-3-1025-7B \
      --data data/fineweb_edu/train.bin \
      --batch_size 8 \
      --quantize int4 \
      --output /media/henry/MoreFiles/olmo3-base-logprobs-int4
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from datasets import Dataset
from torch.distributed import destroy_process_group, init_process_group
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

torch.set_float32_matmul_precision("high")


def extract_topk_logprobs(logits, k):
    """Extract top-k logprobs from raw logits efficiently.

    Computes log_softmax only over the top-k values using the logsumexp trick,
    avoiding a full-vocabulary log_softmax.
    """
    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_logprobs = topk_logits - logsumexp
    return topk_logprobs.half(), topk_indices.int()


def make_empty_rows():
    return {"input_ids": [], "topk_indices": [], "topk_logprobs": []}


def flush_shard(rows, output_path, shard_name, is_hub, rank):
    """Write accumulated rows as a parquet shard (or push to hub)."""
    shard = Dataset.from_dict(rows)
    if is_hub:
        shard.push_to_hub(output_path, split=shard_name)
    else:
        os.makedirs(output_path, exist_ok=True)
        shard.to_parquet(os.path.join(output_path, f"{shard_name}.parquet"))
    print(f"[rank {rank}] Saved {shard_name} ({len(rows['input_ids'])} rows)")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher logprobs for distillation")
    parser.add_argument("--teacher", type=str, required=True, help="HF model name for teacher")
    parser.add_argument("--data", type=str, required=True, help="Path to tokenized .bin file")
    parser.add_argument("--n_ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--top_k", type=int, default=128, help="Number of top logprobs to save")
    parser.add_argument("--output", type=str, required=True, help="HF Hub repo or local directory")
    parser.add_argument("--max_chunks", type=int, default=-1, help="Max chunks to process (-1 for all)")
    parser.add_argument("--rows_per_shard", type=int, default=2048, help="Rows per parquet shard")
    parser.add_argument("--quantize", type=str, default=None, choices=["int8", "int4"],
                        help="Quantize teacher model (int8 or int4 via bitsandbytes)")
    parser.add_argument("--device_map", type=str, default=None,
                        help="Device map for model loading (default: auto for single-process, local GPU for torchrun)")
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--max_pending_shards", type=int, default=2,
                        help="Max number of async shard writes queued before blocking")
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

    # Open memmap (no data copied to RAM)
    if is_main:
        print(f"Loading tokenized data from {args.data}...")
    data = np.memmap(args.data, dtype=np.uint32, mode="r")
    num_tokens = len(data)
    num_chunks = num_tokens // args.n_ctx
    if args.max_chunks > 0:
        num_chunks = min(num_chunks, args.max_chunks)
    if is_main:
        print(f"Total tokens: {num_tokens:,}, chunks of {args.n_ctx}: {num_chunks:,}")

    # Reshape to [num_chunks, n_ctx] so we can do vectorized row indexing.
    data_chunks = data[: num_chunks * args.n_ctx].reshape(num_chunks, args.n_ctx)

    # Shard chunks across ranks (just indices, no data)
    my_chunk_indices = np.arange(rank, num_chunks, world_size, dtype=np.int64)
    if is_main:
        print(f"Rank {rank}/{world_size}: processing {len(my_chunk_indices)} chunks")

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
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
        quantization_config=quant_config,
    )
    model.eval()
    model.config.use_cache = False

    def forward_and_extract(batch, k):
        logits = model(batch, use_cache=False).logits
        return extract_topk_logprobs(logits, k)

    if use_compile:
        if is_main:
            print("Compiling model with torch.compile...")
        forward_and_extract = torch.compile(forward_and_extract, fullgraph=True, mode="max-autotune")

    rows = make_empty_rows()
    shard_idx = 0
    num_batches = (len(my_chunk_indices) + args.batch_size - 1) // args.batch_size

    # Determine if output is local path or HF Hub repo (e.g. "user/repo-name")
    is_hub = "/" in args.output and not os.path.isabs(args.output) and not args.output.startswith(".")

    # Use rank-prefixed shard names to avoid collisions
    shard_prefix = f"rank{rank}_" if world_size > 1 else ""

    # Arrow list columns use 32-bit offsets and ~2GB buffers; cap rows/shard to avoid overflow.
    per_row_topk = args.n_ctx * args.top_k
    max_rows_by_elements = (2**31 - 1) // per_row_topk
    max_rows_by_bytes = (2**31 - 1) // (per_row_topk * 4)  # int32 topk_indices column is tightest
    rows_per_shard = max(1, min(args.rows_per_shard, max_rows_by_elements, max_rows_by_bytes))
    if is_main and rows_per_shard != args.rows_per_shard:
        print(f"Capping rows_per_shard from {args.rows_per_shard} to {rows_per_shard} for Arrow safety")

    if args.max_pending_shards < 1:
        raise ValueError("--max_pending_shards must be >= 1")

    input_device = model.device
    pin_memory = input_device.type == "cuda"
    cpu_buffers = [
        torch.empty((args.batch_size, args.n_ctx), dtype=torch.long, pin_memory=pin_memory),
        torch.empty((args.batch_size, args.n_ctx), dtype=torch.long, pin_memory=pin_memory),
    ]
    staged_input_ids = [None, None]
    staged_gpu_batches = [None, None]
    staged_batch_sizes = [0, 0]

    copy_stream = torch.cuda.Stream(device=input_device) if input_device.type == "cuda" else None
    ready_events = [torch.cuda.Event() for _ in range(2)] if copy_stream is not None else [None, None]

    def stage_batch(batch_idx, slot):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(my_chunk_indices))
        batch_chunk_indices = my_chunk_indices[start_idx:end_idx]
        batch_np = data_chunks[batch_chunk_indices]
        batch_size = end_idx - start_idx

        staged_input_ids[slot] = batch_np
        cpu_batch = cpu_buffers[slot]
        cpu_batch[:batch_size].copy_(torch.from_numpy(batch_np.astype(np.int64, copy=False)))
        if batch_size < args.batch_size:
            cpu_batch[batch_size:].zero_()  # pad to fixed shape to avoid torch.compile recompilation

        if copy_stream is None:
            staged_gpu_batches[slot] = cpu_batch.to(input_device)
        else:
            with torch.cuda.stream(copy_stream):
                staged_gpu_batches[slot] = cpu_batch.to(input_device, non_blocking=True)
                ready_events[slot].record(copy_stream)
        staged_batch_sizes[slot] = batch_size

    flush_executor = ThreadPoolExecutor(max_workers=1)
    pending_flushes = []

    def queue_flush(rows_to_flush, shard_name):
        future = flush_executor.submit(flush_shard, rows_to_flush, args.output, shard_name, is_hub, rank)
        pending_flushes.append(future)
        if len(pending_flushes) >= args.max_pending_shards:
            pending_flushes.pop(0).result()

    # Prime the pipeline (up to two staged batches).
    for init_batch_idx in range(min(2, num_batches)):
        stage_batch(init_batch_idx, init_batch_idx % 2)

    try:
        for batch_idx in tqdm(range(num_batches), desc=f"[rank {rank}] Processing", disable=not is_main):
            slot = batch_idx % 2
            if copy_stream is not None:
                torch.cuda.current_stream(device=input_device).wait_event(ready_events[slot])
            batch_gpu = staged_gpu_batches[slot]
            batch_size = staged_batch_sizes[slot]

            with torch.inference_mode():
                topk_logprobs, topk_indices = forward_and_extract(batch_gpu, args.top_k)

            # Save input_ids BEFORE staging the next batch (which overwrites staged_input_ids[slot]).
            batch_ids = staged_input_ids[slot].astype(np.uint32, copy=False)

            # Stage batch i+2 into the now-free slot to overlap copy with next compute.
            next_batch_idx = batch_idx + 2
            if next_batch_idx < num_batches:
                stage_batch(next_batch_idx, slot)

            # Transfer to CPU numpy in one step.
            topk_indices_np = topk_indices.cpu().numpy()
            topk_logprobs_np = topk_logprobs.cpu().numpy()

            rows["input_ids"].extend(batch_ids[:batch_size])
            rows["topk_indices"].extend(topk_indices_np[:batch_size])
            rows["topk_logprobs"].extend(topk_logprobs_np[:batch_size])

            # Flush shard if needed
            if len(rows["input_ids"]) >= rows_per_shard:
                shard_name = f"train_{shard_prefix}{shard_idx}"
                rows_to_flush = rows
                rows = make_empty_rows()
                queue_flush(rows_to_flush, shard_name)
                shard_idx += 1
    finally:
        # Flush remaining rows
        if rows["input_ids"]:
            shard_name = f"train_{shard_prefix}{shard_idx}"
            queue_flush(rows, shard_name)
            shard_idx += 1

        # Wait for all async writes to finish and surface any exceptions.
        while pending_flushes:
            pending_flushes.pop(0).result()
        flush_executor.shutdown(wait=True)

    if is_main:
        total_shards = shard_idx * world_size
        print(f"Done! ~{total_shards} total shards across {world_size} rank(s)")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
