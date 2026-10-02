"""
Diagnostic: bf16 weights + autocast + weight_decay=0.1.

Tests whether autocast's float32 norms provide enough precision even with
bf16 model weights (no float32 master weights).

Previous results (aggressive LR, no warmup):
  - pure bf16, no WD, no clip: NaN at step 61
  - pure bf16, clip=1.0: NaN at step 146
  - pure bf16, WD=0.1, clip=1.0: NaN at step 146
  - float32 weights + autocast: survived (proven)
  - bf16 weights + autocast: THIS TEST

Usage: CUDA_VISIBLE_DEVICES=1 uv run python debug_grad_nan.py
"""

import math
import os

import numpy as np
import torch

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from model import configure_muon_adamw, create_student

# --- Config (pure bf16 + aggressive LR + weight decay) ---
vocab_size = 100278
batch_size = 12
n_ctx = 2048
dataset = "fineweb_edu"
learning_rate = 6e-4
matrix_lr = 0.05         # 2.5x normal (0.02)
embedding_lr = 0.5       # 2.5x normal (0.2)
unembedding_lr = 0.01    # 2.5x normal (0.004)
weight_decay = 0.1       # Moonlight-style: bounds weight spectral norms
device = "cuda"

# --- Data loading ---
data_dir = os.path.join("data", dataset)

def get_batch():
    """Load a random batch of (input, target) pairs from the training memmap."""
    data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint32, mode="r")
    ix = torch.randint(len(data) - n_ctx, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + n_ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + n_ctx].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

# --- Model: bf16 weights + autocast (norms run float32) ---
print("Creating model in bf16 + autocast...")
model = create_student(vocab_size)
model.to(device=device, dtype=torch.bfloat16)
amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

# --- Optimizer: Muon with weight decay ---
optimizer = configure_muon_adamw(model, matrix_lr=matrix_lr, embedding_lr=embedding_lr,
                                unembedding_lr=unembedding_lr, weight_decay=weight_decay)

# --- Fused CE ---
fused_ce = LigerFusedLinearCrossEntropyLoss(reduction="none")

# --- Backward hooks ---
grad_stats = {}

def make_hook(name):
    """Create a backward hook that records the max gradient magnitude for a named module."""
    def hook(module, grad_input, grad_output):
        """Record max abs grad_output value; flag NaN/Inf immediately."""
        maxvals = []
        for g in grad_output:
            if g is not None:
                m = g.abs().max().item()
                maxvals.append(m)
                if math.isnan(m) or math.isinf(m):
                    grad_stats[name] = float('nan')
                    return
        if maxvals:
            grad_stats[name] = max(maxvals)
    return hook

print("Registering backward hooks...")
for name, module in model.named_modules():
    if name:
        module.register_full_backward_hook(make_hook(name))

# --- Training loop ---
print(f"Running pure bf16 + Muon + WD={weight_decay} (batch={batch_size}, n_ctx={n_ctx})...")
print(f"Aggressive LRs: matrix={matrix_lr}, embed={embedding_lr}, unembed={unembedding_lr}")
print("Previous: no WD NaN@61, clip NaN@146. WD should prevent NaN entirely.\n")

torch.manual_seed(1337)
max_steps = 200

for step in range(max_steps):
    X, Y = get_batch()

    # No warmup — full speed from step 0
    with amp_ctx:
        hidden = model.model(X).last_hidden_state
        loss = fused_ce(
            model.lm_head.weight,
            hidden.reshape(-1, hidden.size(-1)),
            Y.reshape(-1),
            bias=None,
        ).mean()

    grad_stats.clear()
    loss.backward()

    # Clip + check for NaN
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    nan_modules = [n for n, v in grad_stats.items() if math.isnan(v) or math.isinf(v)]

    if nan_modules or math.isnan(loss.item()):
        print(f"\n{'='*60}")
        print(f"NaN/Inf detected at step {step}! loss={loss.item():.4f}")
        print(f"{'='*60}")
        print(f"\nModules with NaN/Inf gradients ({len(nan_modules)}):")
        for n in sorted(nan_modules):
            print(f"  {n}")
        print("\nTop gradient magnitudes:")
        for n, v in sorted(grad_stats.items(), key=lambda x: -x[1] if not math.isnan(x[1]) else float('inf'))[:20]:
            flag = " <<<< NaN" if math.isnan(v) else (" <<<< HUGE" if v > 1000 else "")
            print(f"  {v:12.4e}  {n}{flag}")
        break

    if step % 10 == 0:
        top5 = sorted(grad_stats.items(), key=lambda x: -x[1])[:5]
        # Also track weight RMS for a Muon param to see if WD is bounding it
        w_rms = model.model.layers[0].self_attn.v_proj.weight.float().pow(2).mean().sqrt().item()
        print(f"step {step:4d} | loss {loss.item():.4f} | grad_norm {grad_norm.item():.2e} | v_proj[0] w_rms {w_rms:.4f} | top grads:")
        for n, v in top5:
            print(f"    {v:.2e}  {n}")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

else:
    print(f"\nCompleted {max_steps} steps without NaN!")
    w_rms = model.model.layers[0].self_attn.v_proj.weight.float().pow(2).mean().sqrt().item()
    print(f"Final v_proj[0] weight RMS: {w_rms:.4f}")

print("\nDone.")
