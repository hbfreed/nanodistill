"""
Student model factory for off-policy knowledge distillation experiments.

Uses HuggingFace's Olmo3ForCausalLM as the student architecture:
  - GQA (grouped query attention) with QK-norm
  - SwiGLU MLP
  - RoPE positional embeddings
  - Mixed sliding/full attention pattern
  - RMSNorm

The transformer core is ~75M params; total params vary by vocab_size
since each teacher family uses a different tokenizer.
"""

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM


def create_student(
    vocab_size: int,  # REQUIRED -- varies per teacher tokenizer
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 6,
    num_key_value_heads: int = 2,
    intermediate_size: int = 2048,
    max_position_embeddings: int = 2048,
    **kwargs,
) -> Olmo3ForCausalLM:
    """Create a randomly initialized OLMo3 student model.

    Design notes:
      - head_dim = hidden_size / num_attention_heads = 768/6 = 128
        (FlashInfer supports {64, 128, 256}; Flash Attention requires head_size % 8 == 0)
      - GQA ratio: 6 Q heads, 2 KV heads (3:1)
        (vLLM requires num_heads % num_kv_heads == 0)
      - tie_word_embeddings=False (matching OLMo 3 convention)
    """
    config = Olmo3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=False,
        **kwargs,
    )

    model = Olmo3ForCausalLM(config)
    return model


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """nanoGPT-style optimizer with weight decay grouping.

    Two parameter groups:
      1. Decay group: 2D+ params (weight matrices) -- gets weight_decay
      2. No-decay group: 1D params (biases, norms) -- no weight_decay

    Returns an AdamW optimizer, using the fused CUDA kernel when available.
    """
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")

    import inspect
    fuse_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fuse_available and device_type == 'cuda'
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    return optimizer


def configure_muon_adamw(model, matrix_lr=0.02, embedding_lr=0.2, unembedding_lr=0.008,
                          weight_decay=0.0, adam_betas=(0.8, 0.95), adam_eps=1e-10):
    """Create MuonAdamW optimizer with nanochat-style parameter routing.

    Parameter routing:
      - embed_tokens.weight → AdamW at embedding_lr
      - lm_head.weight → AdamW at unembedding_lr
      - RMSNorm weights (1D) → AdamW at matrix_lr (no weight decay)
      - Q/K/V projections → Muon at matrix_lr with per-head Muon Split
      - All other 2D weight matrices → Muon at matrix_lr
    """
    from optim import MuonAdamW

    # Extract head counts from model config for Muon Split
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads

    embed_params = []
    unembed_params = []
    scalar_params = []
    muon_params_by_shape = {}  # shape -> list of params
    muon_split_params = {}  # (shape, n_heads) -> list of params

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embed_tokens" in name:
            embed_params.append(param)
        elif "lm_head" in name:
            unembed_params.append(param)
        elif param.dim() < 2:
            scalar_params.append(param)
        elif "q_proj" in name:
            key = (param.shape, num_q_heads)
            muon_split_params.setdefault(key, []).append(param)
        elif "k_proj" in name or "v_proj" in name:
            key = (param.shape, num_kv_heads)
            muon_split_params.setdefault(key, []).append(param)
        else:
            # 2D params -> Muon, grouped by shape for batch orthogonalization
            shape = param.shape
            if shape not in muon_params_by_shape:
                muon_params_by_shape[shape] = []
            muon_params_by_shape[shape].append(param)

    param_groups = []

    # AdamW groups
    if embed_params:
        param_groups.append({
            "params": embed_params, "kind": "adamw", "lr": embedding_lr,
            "betas": (0.8, 0.995), "eps": adam_eps, "weight_decay": 0.001,
            "initial_lr": embedding_lr,
        })
    if unembed_params:
        param_groups.append({
            "params": unembed_params, "kind": "adamw", "lr": unembedding_lr,
            "betas": (0.8, 0.96), "eps": adam_eps, "weight_decay": 0.01,
            "initial_lr": unembedding_lr,
        })
    if scalar_params:
        param_groups.append({
            "params": scalar_params, "kind": "adamw", "lr": matrix_lr,
            "betas": adam_betas, "eps": adam_eps, "weight_decay": 0.0,
            "initial_lr": matrix_lr,
        })

    # Muon Split groups (Q/K/V — per-head orthogonalization)
    for (shape, n_heads), params in sorted(muon_split_params.items()):
        param_groups.append({
            "params": params, "kind": "muon", "lr": matrix_lr,
            "momentum": 0.95, "ns_steps": 5, "weight_decay": weight_decay,
            "initial_lr": matrix_lr, "n_heads": n_heads,
        })

    # Muon groups (MLP, o_proj — standard orthogonalization)
    for shape, params in sorted(muon_params_by_shape.items()):
        param_groups.append({
            "params": params, "kind": "muon", "lr": matrix_lr,
            "momentum": 0.95, "ns_steps": 5, "weight_decay": weight_decay,
            "initial_lr": matrix_lr,
        })

    # Print summary
    total_muon = sum(p.numel() for pg in param_groups if pg["kind"] == "muon" for p in pg["params"])
    total_adam = sum(p.numel() for pg in param_groups if pg["kind"] == "adamw" for p in pg["params"])
    n_scalar = sum(p.numel() for p in scalar_params)
    print(f"MuonAdamW: {total_muon:,} Muon params, {total_adam:,} AdamW params, {n_scalar} scalar params")
    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        shapes = {p.shape for p in pg["params"]}
        split_info = f" split={pg['n_heads']}h" if pg.get("n_heads") else ""
        print(f"  {pg['kind']:>5s} lr={pg['lr']:.4f}: {len(pg['params'])} tensors, {n:,} params, shapes={shapes}{split_info}")

    return MuonAdamW(param_groups)


def count_parameters(model) -> dict:
    """Count parameters split by component.

    Returns dict with:
      - 'total': all parameters
      - 'embedding': embed_tokens + lm_head
      - 'transformer_core': total - embedding (the ~75M fixed part)
    """
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(
        p.numel() for n, p in model.named_parameters()
        if "embed_tokens" in n or "lm_head" in n
    )

    return {
        "total": total,
        "embedding": embedding,
        "transformer_core": total - embedding,
    }


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    # --- Test with different vocab sizes ---
    # These show how the same ~75M transformer core gets different total
    # param counts depending on the teacher's tokenizer.
    test_configs = {
        "OLMo tokenizer": 100_278,
        "Qwen tokenizer": 152_064,
        "GPT-2 tokenizer": 50_304,
    }

    for name, vs in test_configs.items():
        model = create_student(vs)
        counts = count_parameters(model)
        print(f"\n{name} (vocab_size={vs:,}):")
        print(f"  Total params:      {counts['total']:>12,}")
        print(f"  Embedding params:  {counts['embedding']:>12,}")
        print(f"  Transformer core:  {counts['transformer_core']:>12,}")

    print("\n--- Architecture Config ---")
    print(model.config)

    print("\n--- Save/Load Round-trip ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = Olmo3ForCausalLM.from_pretrained(tmpdir)
        assert loaded.config.vocab_size == model.config.vocab_size
        loaded_total = count_parameters(loaded)["total"]
        assert loaded_total == counts["total"], f"param count mismatch: {loaded_total} vs {counts['total']}"
        print(f"Save/load round-trip OK ({counts['total']:,} params)")
