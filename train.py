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
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from cut_cross_entropy import linear_cross_entropy
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from model import configure_muon_adamw, configure_optimizers, create_student
from sparse_kl import sparse_kl_loss

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
teacher_model = ""       # HF model name for online distillation
teacher_quantize = ""    # "" (bf16), "int8", or "int4"
temperature = 1.0        # KD temperature
alpha = 1.0  # KL weight in mixed loss: loss = alpha*KL + (1-alpha)*CE. Use 0.9 per Peng et al.
logprob_dir = ""  # dir of parquet shards from generate_logprobs.py (offline distill)

# Ablation knobs
optimizer_type = "adamw"  # "adamw" or "muon"
cce_impl = "liger"        # "liger" (Liger fused CE), "cce" (Apple CCE), or "cce_exact" (CCE with fp32 accumulators)
cce_filter_eps = None     # CCE filter_eps: None disables (safe for Muon), "high" for CCE default

# Muon-specific (only used when optimizer_type == "muon")
matrix_lr = 0.02
embedding_lr = 0.2
unembedding_lr = 0.008

# Data
dataset = "fineweb_edu"
batch_size = 12
n_ctx = 2048
total_tokens = 5_000_000_000  # total training tokens
tokens_per_step = 1_000_000   # tokens per optimizer step (across all GPUs)
gradient_accumulation_steps = None  # derived from tokens_per_step if not set

# Model
vocab_size = 100278  # MUST match tokenizer used for data prep

# Optimizer (nanoGPT defaults)
learning_rate = 6e-4
max_iters = None  # derived from total_tokens if not set
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
lr_schedule = "cosine"  # "cosine" (nanoGPT default) or "wsd" (warmup-stable-decay)
warmup_iters = 2000
lr_decay_iters = None  # defaults to max_iters
cooldown_fraction = 0.2  # fraction of max_iters for final decay (wsd only)
min_lr = 6e-5

# Logging
wandb_log = False
wandb_project = "nanodistill"
wandb_run_name = "baseline"

# System
device = "cuda"
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
    if mode == "distill" and not logprob_dir:
        # Online distill: GPU 0 reserved for teacher; student ranks start at GPU 1
        device = f"cuda:{ddp_local_rank + 1}"
    else:
        device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Derive gradient_accumulation_steps, max_iters, lr_decay_iters from high-level params
tokens_per_micro = batch_size * n_ctx
if gradient_accumulation_steps is None:
    # Round to nearest multiple of world_size so DDP divides evenly
    raw = tokens_per_step / (tokens_per_micro * ddp_world_size)
    gradient_accumulation_steps = max(1, round(raw)) * ddp_world_size
    assert gradient_accumulation_steps >= 1, (
        f"tokens_per_step={tokens_per_step} too small for batch_size={batch_size} "
        f"* n_ctx={n_ctx} * world_size={ddp_world_size} = {tokens_per_micro * ddp_world_size}"
    )
assert gradient_accumulation_steps % ddp_world_size == 0
gradient_accumulation_steps //= ddp_world_size

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * n_ctx
if max_iters is None:
    max_iters = total_tokens // tokens_per_iter
if lr_decay_iters is None:
    lr_decay_iters = max_iters
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"max_iters: {max_iters:,} ({max_iters * tokens_per_iter / 1e9:.2f}B tokens)")
    print(f"mode: {mode}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"

# --- Data loading setup ---
data_dir = os.path.join("data", dataset)

# Load token_bytes for BPB evaluation
token_bytes_path = os.path.join(data_dir, "token_bytes.pt")
token_bytes = None
if os.path.exists(token_bytes_path):
    token_bytes = torch.load(token_bytes_path, weights_only=True).to(device)
    if master_process:
        print(f"Loaded token_bytes.pt (shape: {token_bytes.shape})")


eval_batch_size = min(batch_size, 5)

def get_batch(split, bs=None):
    """nanoGPT-style memmap random sampling. Returns (x, y) with pre-shifted targets."""
    if bs is None:
        bs = batch_size
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint32, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint32, mode="r")
    ix = torch.randint(len(data) - n_ctx, (bs,))
    x = torch.stack([torch.from_numpy((data[i : i + n_ctx]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + n_ctx]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# --- Offline distillation data loading ---
# Reads parquet shards written by generate_logprobs.py (columns: input_ids,
# topk_indices, topk_logprobs). Each row carries its own input_ids, so student
# input and teacher logprobs are aligned by construction.
_top_k = 0
_offline_shards = []
_offline_shard_pos = 0
_offline_ids = _offline_idx = _offline_lp = None
_offline_perm = None
_offline_row = 0

if mode == "distill" and logprob_dir:
    import glob
    import random as pyrandom
    import pyarrow.parquet as pq

    _offline_shards = sorted(glob.glob(os.path.join(logprob_dir, "*.parquet")))
    assert _offline_shards, f"No parquet files found in {logprob_dir}"
    _shard_rng = pyrandom.Random(1337)
    _shard_rng.shuffle(_offline_shards)
    if ddp:  # disjoint shard subsets per rank
        _offline_shards = [s for i, s in enumerate(_offline_shards) if i % ddp_world_size == ddp_rank]
    _row_rng = np.random.default_rng(1337 + seed_offset)

    def _read_shard(path):
        table = pq.read_table(path)
        n = len(table)
        ids = table["input_ids"].combine_chunks().flatten().to_numpy().astype(np.int64).reshape(n, -1)
        idx = table["topk_indices"].combine_chunks().flatten().flatten().to_numpy().astype(np.int64)
        lp = table["topk_logprobs"].combine_chunks().flatten().flatten().to_numpy().astype(np.float32)
        k = idx.size // (n * ids.shape[1])
        return ids, idx.reshape(n, ids.shape[1], k), lp.reshape(n, ids.shape[1], k)

    import threading
    from concurrent.futures import ThreadPoolExecutor
    _shard_pool = ThreadPoolExecutor(max_workers=1)
    _next_shard_future = None

    def _submit_next_shard():
        global _next_shard_future, _offline_shard_pos
        path = _offline_shards[_offline_shard_pos % len(_offline_shards)]
        _offline_shard_pos += 1
        _next_shard_future = _shard_pool.submit(_read_shard, path)

    def _load_next_shard():
        """Swap in the prefetched shard (decoding parquet takes ~1 min) and prefetch the next."""
        global _offline_ids, _offline_idx, _offline_lp, _offline_perm, _offline_row, _top_k
        _offline_ids, _offline_idx, _offline_lp = _next_shard_future.result()
        _submit_next_shard()
        assert _offline_ids.shape[1] == n_ctx, f"Teacher seq_len {_offline_ids.shape[1]} != n_ctx {n_ctx}"
        _top_k = _offline_idx.shape[2]
        _offline_perm = _row_rng.permutation(len(_offline_ids))
        _offline_row = 0

    _submit_next_shard()
    _load_next_shard()
    if master_process:
        print(f"Loaded offline teacher logprobs from {logprob_dir}: "
              f"{len(_offline_shards) * ddp_world_size} shards, top_k={_top_k}")

def get_distill_batch():
    """Next batch of (input_ids, teacher_topk_ids, teacher_topk_logprobs) from parquet shards."""
    global _offline_row
    if _offline_row + batch_size > len(_offline_perm):
        _load_next_shard()
    rows = _offline_perm[_offline_row : _offline_row + batch_size]
    _offline_row += batch_size
    x = torch.from_numpy(_offline_ids[rows])
    t_ids = torch.from_numpy(_offline_idx[rows])
    t_lps = torch.from_numpy(_offline_lp[rows])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        t_ids = t_ids.pin_memory().to(device, non_blocking=True)
        t_lps = t_lps.pin_memory().to(device, non_blocking=True)
    else:
        x, t_ids, t_lps = x.to(device), t_ids.to(device), t_lps.to(device)
    return x, t_ids, t_lps


# --- Online distillation pipeline ---
def teacher_worker(teacher_ref, data, n_ctx, total_bs, t_device, temperature, queue, seed):
    """Daemon thread: run teacher inference, push (input_ids_cpu, logprobs_cpu) to queue."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    data_len = len(data) - n_ctx
    while True:
        ix = torch.randint(data_len, (total_bs,), generator=rng)
        x = torch.stack([torch.from_numpy(data[i : i + n_ctx].astype(np.int64)) for i in ix])
        x_gpu = x.to(t_device)
        with torch.no_grad():
            logits = teacher_ref(x_gpu).logits.float()
            if temperature != 1.0:
                logits.div_(temperature)
            logprobs = F.log_softmax(logits, dim=-1)
            del logits
        queue.put((x, logprobs.cpu()))


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
    # init_from is a path to a pretrained checkpoint (model weights only, fresh optimizer)
    ckpt_path = init_from if os.path.isfile(init_from) else os.path.join(init_from, "ckpt.pt")
    if master_process:
        print(f"Loading pretrained weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = create_student(checkpoint["model_args"]["vocab_size"])
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    checkpoint = None  # fresh optimizer, iter_num=0

# Muon: float32 weights + autocast(bf16) forward. OLMo3's ungated residuals
# amplify backward gradients ~1000x/layer → bf16 precision collapse at layer 0.
# Weight decay (0.1, per Moonlight) bounds weight growth but can't prevent the
# architectural gradient amplification. Float32 backward is needed.
# AdamW: pure bf16 (grad_clip + weight_decay keep things stable).
if optimizer_type == "muon":
    model.to(device=device)  # float32
else:
    model.to(device=device, dtype=torch.bfloat16)

# Print parameter counts
from model import count_parameters  # noqa: E402
if master_process:
    counts = count_parameters(model)
    print(f"Total params: {counts['total']:,}")
    print(f"Embedding params: {counts['embedding']:,}")
    print(f"Transformer core: {counts['transformer_core']:,}")


# Optimizer
if optimizer_type == "muon":
    optimizer = configure_muon_adamw(model, matrix_lr=matrix_lr, embedding_lr=embedding_lr,
                                      unembedding_lr=unembedding_lr, weight_decay=weight_decay)
else:
    optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free memory

# --- Teacher loading (online distillation) ---
teacher = None
teacher_device = None
if mode == "distill" and not logprob_dir:
    from transformers import AutoModelForCausalLM
    assert teacher_model, "teacher_model required for online distill mode"
    teacher_device = "cuda:0" if ddp else device
    if not ddp or ddp_local_rank == 0:
        quant_config = None
        if teacher_quantize == "int8":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif teacher_quantize == "int4":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        if master_process:
            print(f"Loading teacher: {teacher_model} ({teacher_quantize or 'bf16'}) on {teacher_device}")
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": teacher_device},
            quantization_config=quant_config,
        )
        teacher.eval()
        teacher.config.use_cache = False
        assert teacher.config.vocab_size == vocab_size, (
            f"Teacher vocab {teacher.config.vocab_size} != config vocab_size {vocab_size}"
        )

# Compile
if compile:
    if master_process:
        print("Compiling the model... (takes a ~minute)")
    model.model = torch.compile(model.model)  # backbone only — lm_head handled explicitly

# DDP — wrap backbone only so fused CE can handle lm_head directly
if ddp:
    ddp_device_id = ddp_local_rank + 1 if (mode == "distill" and not logprob_dir) else ddp_local_rank
    model.model = DDP(model.model, device_ids=[ddp_device_id])


# --- Evaluation ---
amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if optimizer_type == "muon" else nullcontext()
fused_ce = LigerFusedLinearCrossEntropyLoss(reduction="none")

@torch.no_grad()
def estimate_loss():
    """Estimate CE loss and BPB on train/val splits.

    Uses LigerFusedLinearCrossEntropyLoss to avoid materializing the full
    (B, T, vocab_size) logits tensor — computes CE directly from hidden states
    and lm_head weight in chunks. Also bypasses torch.compile for eval to avoid
    recompilation from batch size changes.
    """
    out = {}
    model.eval()
    # Access model components directly (bypasses torch.compile + DDP),
    # so we can run the base transformer without lm_head.
    lm_weight = raw_model.lm_head.weight
    lm_bias = getattr(raw_model.lm_head, "bias", None)
    # Bypass DDP for eval (only master_process calls this; DDP forward needs all ranks)
    eval_backbone = raw_model.model.module if ddp else raw_model.model

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        total_nats = 0.0
        total_bytes_count = 0
        for k in range(eval_iters):
            X, Y = get_batch(split, bs=eval_batch_size)
            with amp_ctx:
                # Run base transformer only — no lm_head, no (B, T, V) logits
                hidden = eval_backbone(X).last_hidden_state
                y_flat = Y.view(-1)
            # Fused linear + CE: projects hidden -> logits in chunks,
            # never materializes full (B*T, vocab_size) tensor
            per_token_loss = fused_ce(
                lm_weight, hidden.reshape(-1, hidden.size(-1)), y_flat, bias=lm_bias,
            )
            losses[k] = per_token_loss.mean().item()

            # BPB accumulation
            if token_bytes is not None:
                num_bytes = token_bytes[y_flat]
                total_nats += (per_token_loss.float() * (num_bytes > 0).float()).sum().item()
                total_bytes_count += num_bytes.sum().item()

        out[split] = losses.mean().item()
        if token_bytes is not None and total_bytes_count > 0:
            out[f"{split}_bpb"] = total_nats / (math.log(2) * total_bytes_count)

    model.train()
    return out


# --- LR schedule ---
cooldown_iters = int(max_iters * cooldown_fraction)  # for wsd schedule

def get_lr(it):
    # Warmup (shared by both schedules)
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if lr_schedule == "wsd":
        # Warmup-Stable-Decay: hold peak LR, then cosine cooldown at the end
        cooldown_start = max_iters - cooldown_iters
        if it < cooldown_start:
            return learning_rate
        decay_ratio = (it - cooldown_start) / cooldown_iters
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        # Cosine (nanoGPT default)
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

def _unwrap_backbone(m):
    """Get the raw backbone, unwrapping DDP and torch.compile."""
    backbone = m.model
    if isinstance(backbone, DDP):
        backbone = backbone.module
    return getattr(backbone, "_orig_mod", backbone)

def _save_pretrained(m, path):
    """save_pretrained that handles DDP-wrapped and/or compiled backbone."""
    orig = m.model
    m.model = _unwrap_backbone(m)
    m.save_pretrained(path)
    m.model = orig

def _clean_state_dict(m):
    """State dict with DDP/compile prefixes removed."""
    orig = m.model
    m.model = _unwrap_backbone(m)
    sd = m.state_dict()
    m.model = orig
    return sd

# --- Training loop ---
if mode == "ce":
    X, Y = get_batch("train")  # prefetch first batch
elif mode == "distill" and logprob_dir:
    X, t_ids, t_lps = get_distill_batch()  # prefetch first batch
elif mode == "distill":
    from queue import Queue
    from threading import Thread
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint32, mode="r")
    total_batch_size_teacher = ddp_world_size * batch_size
    if not ddp or ddp_local_rank == 0:
        teacher_queue = Queue(maxsize=4)
        teacher_thread = Thread(
            target=teacher_worker,
            args=(teacher, train_data, n_ctx, total_batch_size_teacher,
                  teacher_device, temperature, teacher_queue, seed),
            daemon=True,
        )
        teacher_thread.start()
        teacher = None  # thread owns the reference
        if master_process:
            print(f"Teacher pipeline started (queue maxsize=4, batch={total_batch_size_teacher})")
    else:
        teacher_queue = None

t0 = time.time()
local_iter_num = 0
raw_model = model  # CausalLM is never DDP-wrapped; model.model is the DDP backbone

while True:
    # Set LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        if optimizer_type == "muon":
            # Scale each group's initial_lr by the schedule multiplier
            lr_mult = lr / learning_rate  # normalized multiplier from schedule
            param_group["lr"] = param_group["initial_lr"] * lr_mult
            if param_group.get("kind") == "muon":
                # Momentum warmup: 0.85 -> 0.95 over first 300 steps
                frac = min(iter_num / 300, 1)
                param_group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        else:
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
                    "model": _clean_state_dict(raw_model),
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
                _save_pretrained(raw_model, hf_dir)
                print(f"saved HF model to {hf_dir}")

    if iter_num == 0 and eval_only:
        break

    # Forward/backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

        # Online distill: pull teacher logprobs from queue + broadcast to all ranks
        if mode == "distill" and not logprob_dir:
            if not ddp or ddp_local_rank == 0:
                all_X_cpu, all_logprobs_cpu = teacher_queue.get()
                all_X = all_X_cpu.to(device)
                all_logprobs = all_logprobs_cpu.to(device)
            if ddp:
                if ddp_local_rank != 0:
                    all_X = torch.empty(total_batch_size_teacher, n_ctx, dtype=torch.long, device=device)
                    all_logprobs = torch.empty(total_batch_size_teacher, n_ctx, vocab_size,
                                               dtype=torch.float32, device=device)
                torch.distributed.broadcast(all_X, src=0)
                torch.distributed.broadcast(all_logprobs, src=0)
                X = all_X[ddp_rank * batch_size : (ddp_rank + 1) * batch_size].clone()
                my_teacher_logprobs = all_logprobs[ddp_rank * batch_size : (ddp_rank + 1) * batch_size].clone()
                del all_X, all_logprobs
            else:
                X = all_X
                my_teacher_logprobs = all_logprobs

        with amp_ctx:
            if mode == "ce":
                # Backbone forward — goes through DDP wrapper if ddp, handles gradient sync
                hidden = model.model(X).last_hidden_state
                # Fused CE handles lm_head — works for both DDP and non-DDP
                if cce_impl in ("cce", "cce_exact"):
                    # CCE backward requires bf16/fp16 embeddings; autocast promotes
                    # RMSNorm to float32, so cast back explicitly.
                    h = hidden.reshape(-1, hidden.size(-1))
                    if h.dtype not in (torch.bfloat16, torch.float16):
                        h = h.bfloat16()
                    loss = linear_cross_entropy(
                        h,
                        raw_model.lm_head.weight,
                        Y.reshape(-1),
                        impl=cce_impl,
                        filter_eps=cce_filter_eps,
                    )
                else:  # "liger"
                    loss = fused_ce(
                        raw_model.lm_head.weight,
                        hidden.reshape(-1, hidden.size(-1)),
                        Y.reshape(-1),
                        bias=getattr(raw_model.lm_head, "bias", None),
                    ).mean()
            elif mode == "distill" and logprob_dir:
                # Offline sparse KL from pre-computed top-k teacher logprobs (see sparse_kl.py):
                # true student log-probs (logit - full-vocab lse from CCE) plus an explicit
                # tail bucket for the mass outside the top-k. Renormalizing both sides over
                # the top-k alone is invariant to the student's tail mass and cannot
                # calibrate the full-vocab distribution.
                assert temperature == 1.0, "temperature != 1 not supported for sparse offline KL"
                hidden = model.model(X).last_hidden_state  # (B, T, D)
                h = hidden.reshape(-1, hidden.size(-1))
                if h.dtype not in (torch.bfloat16, torch.float16):
                    h = h.bfloat16()
                targets = None
                if alpha < 1.0:
                    # next-token targets; -1 marks the last position (no target)
                    targets = torch.cat([X[:, 1:], X.new_full((X.size(0), 1), -1)], dim=1).reshape(-1)
                kl_loss, ce_loss = sparse_kl_loss(
                    h, raw_model.lm_head.weight,
                    t_ids.reshape(-1, _top_k), t_lps.reshape(-1, _top_k), targets=targets,
                )
                loss = kl_loss if alpha >= 1.0 else alpha * kl_loss + (1 - alpha) * ce_loss

            elif mode == "distill":
                # Full-vocab online KL distillation
                hidden = model.model(X).last_hidden_state
                student_logits = raw_model.lm_head(hidden)
                if temperature != 1.0:
                    student_logits = student_logits / temperature
                student_logprobs = F.log_softmax(student_logits.float(), dim=-1)
                del student_logits
                kl_loss = F.kl_div(student_logprobs, my_teacher_logprobs,
                                   log_target=True, reduction="batchmean") / n_ctx
                del student_logprobs, my_teacher_logprobs
                if temperature != 1.0:
                    kl_loss = kl_loss * (temperature ** 2)

                if alpha < 1.0:
                    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
                    shift_labels = X[:, 1:].reshape(-1)
                    if cce_impl in ("cce", "cce_exact"):
                        h = shift_hidden
                        if h.dtype not in (torch.bfloat16, torch.float16):
                            h = h.bfloat16()
                        ce_loss = linear_cross_entropy(
                            h, raw_model.lm_head.weight, shift_labels,
                            impl=cce_impl, filter_eps=cce_filter_eps,
                        )
                    else:
                        ce_loss = fused_ce(
                            raw_model.lm_head.weight, shift_hidden, shift_labels,
                            bias=getattr(raw_model.lm_head, "bias", None),
                        ).mean()
                    loss = alpha * kl_loss + (1 - alpha) * ce_loss
                else:
                    loss = kl_loss

        loss = loss / gradient_accumulation_steps

        # Prefetch next batch (online distill pulls from queue above)
        if mode == "ce":
            X, Y = get_batch("train")
        elif mode == "distill" and logprob_dir:
            X, t_ids, t_lps = get_distill_batch()

        loss.backward()
        # lm_head is outside DDP — manually sync its gradients
        if ddp and model.model.require_backward_grad_sync:
            torch.distributed.all_reduce(raw_model.lm_head.weight.grad, op=torch.distributed.ReduceOp.AVG)

    # Gradient clipping — required for both optimizers.
    # Without clipping, gradients explode ~1000x per layer through the v_proj→o_proj
    # residual path (measured: layer 11 = 6.4, layer 0 = 8.1e+36 at NaN step).
    if grad_clip != 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        grad_norm_f = grad_norm.item() if (grad_clip != 0.0 or optimizer_type == "muon") else 0.0
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}, grad_norm {grad_norm_f:.4f}")
        if wandb_log:
            log_dict = {"iter": iter_num, "train/loss": lossf, "train/time_ms": dt * 1000, "lr": lr}
            if grad_clip != 0.0 or optimizer_type == "muon":
                log_dict["train/grad_norm"] = grad_norm_f
            log_dict["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if mode == "distill":
                log_dict["train/kl_loss"] = kl_loss.item()
                if alpha < 1.0:
                    log_dict["train/ce_loss"] = ce_loss.item()
            wandb.log(log_dict)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# Final eval
if master_process:
    losses = estimate_loss()
    ce_msg = f"step {iter_num} (final): train CE {losses['train']:.4f}, val CE {losses['val']:.4f}"
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
    best_val_loss = min(best_val_loss, losses["val"])

# Final checkpoint
if master_process:
    checkpoint = {
        "model": _clean_state_dict(raw_model),
        "optimizer": optimizer.state_dict(),
        "model_args": {"vocab_size": vocab_size},
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    print(f"saving final checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    hf_dir = os.path.join(out_dir, "hf")
    _save_pretrained(raw_model, hf_dir)
    print(f"saved final HF model to {hf_dir}")

if wandb_log and master_process:
    wandb.finish()

if ddp:
    destroy_process_group()
