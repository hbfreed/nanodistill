"""
nanoGPT-style training loop for "Does the Teacher Matter?" experiments.

Supports two modes:
  - "ce":      Cross-entropy baseline (student learns from raw data)
  - "distill": KL divergence distillation (student learns from teacher logprobs)

To run on a single GPU:
$ python train.py config/baseline_olmo.py

To run with DDP on 4 GPUs:
$ torchrun --standalone --nproc_per_node=4 train.py config/baseline_olmo.py
"""

import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import configure_optimizers, create_student

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # 'scratch' or 'resume'

# Training mode
mode = "ce"  # "ce" for cross-entropy baseline, "distill" for KL distillation
logprobs_path = ""  # HF dataset path for distillation (e.g., "hbfreed/olmo-fineweb-logprobs")
top_k = 128

# Data
dataset = "fineweb_edu"
gradient_accumulation_steps = 40
batch_size = 12
n_ctx = 2048

# Model
vocab_size = 100278  # MUST match tokenizer used for data prep

# Optimizer (nanoGPT defaults)
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# Logging
wandb_log = False
wandb_project = "nanodistill"
wandb_run_name = "baseline"

# System
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True
seed = 1337
backend = "nccl"
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP init
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * n_ctx
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"mode: {mode}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Data loading setup ---
data_dir = os.path.join("data", dataset)

# Load token_bytes for BPB evaluation
token_bytes_path = os.path.join(data_dir, "token_bytes.pt")
token_bytes = None
if os.path.exists(token_bytes_path):
    token_bytes = torch.load(token_bytes_path, weights_only=True).to(device)
    if master_process:
        print(f"Loaded token_bytes.pt (shape: {token_bytes.shape})")


def get_batch(split):
    """nanoGPT-style memmap random sampling. Returns (x, y) with pre-shifted targets."""
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint32, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint32, mode="r")
    ix = torch.randint(len(data) - n_ctx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + n_ctx]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + n_ctx]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# --- Distillation data loading ---
distill_train_loader = None
distill_train_iter = None

if mode == "distill":
    from datasets import concatenate_datasets, load_dataset as hf_load_dataset
    from liger_kernel.transformers import LigerKLDIVLoss
    from torch.utils.data import DataLoader, Dataset

    kld_loss = LigerKLDIVLoss(log_target=True)

    class LogprobsDataset(Dataset):
        def __init__(self, hf_dataset):
            self.data = hf_dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"]),
                "topk_indices": torch.tensor(item["topk_indices"]),
                "topk_logprobs": torch.tensor(item["topk_logprobs"]),
            }

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        topk_indices = torch.stack([b["topk_indices"] for b in batch])
        topk_logprobs = torch.stack([b["topk_logprobs"] for b in batch])
        return {
            "input_ids": input_ids,
            "teacher_indices": topk_indices,
            "teacher_logits": topk_logprobs,
        }

    if master_process:
        print(f"Loading distillation logprobs from {logprobs_path}...")

    # Load HF dataset (may have multiple splits/shards)
    ds = hf_load_dataset(logprobs_path)
    if isinstance(ds, dict):
        # Concatenate all splits
        splits = [ds[s] for s in sorted(ds.keys())]
        combined = concatenate_datasets(splits)
    else:
        combined = ds

    if master_process:
        print(f"Total distillation sequences: {len(combined)}")

    train_dataset = LogprobsDataset(combined)
    distill_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=collate_fn,
    )
    distill_train_iter = iter(distill_train_loader)


def get_distill_batch():
    """Get next batch from distillation dataloader, cycling if exhausted."""
    global distill_train_iter
    try:
        batch = next(distill_train_iter)
    except StopIteration:
        distill_train_iter = iter(distill_train_loader)
        batch = next(distill_train_iter)
    # Move to device
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["input_ids"] = batch["input_ids"].long()
    batch["teacher_indices"] = batch["teacher_indices"].long()
    return batch


# --- Model init ---
iter_num = 0
best_val_loss = 1e9

if init_from == "scratch":
    if master_process:
        print(f"Initializing new student model (vocab_size={vocab_size})")
    model = create_student(vocab_size)
elif init_from == "resume":
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_student(checkpoint["model_args"]["vocab_size"])
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.to(device)

# Print parameter counts
from model import count_parameters
if master_process:
    counts = count_parameters(model)
    print(f"Total params: {counts['total']:,}")
    print(f"Embedding params: {counts['embedding']:,}")
    print(f"Transformer core: {counts['transformer_core']:,}")

# GradScaler for float16
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

# Optimizer
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free memory

# Compile
if compile:
    if master_process:
        print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# --- Evaluation ---
@torch.no_grad()
def estimate_loss():
    """Estimate CE loss and BPB on train/val splits using memmap sampling."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        total_nats = 0.0
        total_bytes_count = 0
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()

            # BPB accumulation
            if token_bytes is not None:
                with ctx:
                    loss_2d = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), Y.view(-1), reduction="none"
                    )
                y_flat = Y.view(-1)
                num_bytes = token_bytes[y_flat]
                total_nats += (loss_2d * (num_bytes > 0).float()).sum().item()
                total_bytes_count += num_bytes.sum().item()

        out[split] = losses.mean().item()
        if token_bytes is not None and total_bytes_count > 0:
            out[f"{split}_bpb"] = total_nats / (math.log(2) * total_bytes_count)

    model.train()
    return out


# --- LR schedule ---
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# --- Wandb ---
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# --- Training loop ---
if mode == "ce":
    X, Y = get_batch("train")  # prefetch first batch
elif mode == "distill":
    distill_batch = get_distill_batch()

t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model

while True:
    # Set LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluate
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        ce_msg = f"step {iter_num}: train CE {losses['train']:.4f}, val CE {losses['val']:.4f}"
        if "val_bpb" in losses:
            ce_msg += f", train BPB {losses.get('train_bpb', 0):.4f}, val BPB {losses['val_bpb']:.4f}"
        print(ce_msg)

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/ce_loss": losses["train"],
                "val/ce_loss": losses["val"],
                "lr": lr,
            }
            if "train_bpb" in losses:
                log_dict["train/bpb"] = losses["train_bpb"]
            if "val_bpb" in losses:
                log_dict["val/bpb"] = losses["val_bpb"]
            wandb.log(log_dict)

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses["val"])
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": {"vocab_size": vocab_size},
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                # Also save HF-compatible format
                hf_dir = os.path.join(out_dir, "hf")
                raw_model.save_pretrained(hf_dir)
                print(f"saved HF model to {hf_dir}")

    if iter_num == 0 and eval_only:
        break

    # Forward/backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

        with ctx:
            if mode == "ce":
                logits = model(X).logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            elif mode == "distill":
                student_logprobs = F.log_softmax(model(distill_batch["input_ids"]).logits, dim=-1)
                student_at_topk = student_logprobs.gather(-1, distill_batch["teacher_indices"])
                teacher_logprobs = F.log_softmax(distill_batch["teacher_logits"].float(), dim=-1)
                loss = kld_loss(student_at_topk.float(), teacher_logprobs.float())

            loss = loss / gradient_accumulation_steps

        # Prefetch next batch
        if mode == "ce":
            X, Y = get_batch("train")
        elif mode == "distill":
            distill_batch = get_distill_batch()

        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}")
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": lossf, "train/time_ms": dt * 1000, "lr": lr})

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# Final checkpoint
if master_process:
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": {"vocab_size": vocab_size},
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    print(f"saving final checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    hf_dir = os.path.join(out_dir, "hf")
    raw_model.save_pretrained(hf_dir)
    print(f"saved final HF model to {hf_dir}")

if wandb_log and master_process:
    wandb.finish()

if ddp:
    destroy_process_group()
