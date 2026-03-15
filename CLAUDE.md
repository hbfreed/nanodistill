# nanodistill

"Does the Teacher Matter?" — off-policy knowledge distillation experiments comparing teacher families (OLMo base/instruct/think, Qwen, GPT-OSS, etc.).

## Project Structure

- `model.py` — Student model factory (`Olmo3ForCausalLM`), `configure_optimizers`, `configure_muon_adamw`, `count_parameters`
- `optim.py` — `MuonAdamW` combined optimizer (Muon for 2D matrices, AdamW for rest). Newton-Schulz5 orthogonalization from nanochat. Supports Muon Split (per-head Q/K/V orthogonalization).
- `configurator.py` — nanoGPT's "poor man's configurator" (exec's config files + CLI args)
- `train.py` — Training loop with two modes: `ce` (cross-entropy baseline) and `distill` (sparse KL from pre-computed logprobs). Eval always uses CE + BPB. Final eval runs automatically after training loop.
- `generate_logprobs.py` — Pre-compute teacher top-k logprobs. Streams from memmap, saves parquet shards. Supports torchrun DDP, `--device_map auto`, `--quantize {int8,int4}`.
- `data/fineweb_edu/prepare.py` — Tokenizes FineWeb-Edu into uint32 memmap `.bin` files. Saves `token_bytes.pt` and `meta.pkl`.
- `config/baseline_olmo.py` — CE baseline config (AdamW, ~5B tokens)
- `config/baseline_cce_adamw.py` — CE baseline with CCE + AdamW
- `config/baseline_cce_muon.py` — CE baseline with CCE-exact + Muon (sweep winner: mlr=0.02, wd=0.01)
- `config/distill_olmo.py` — Distillation config template (pure KL)
- `config/distill_olmo_mixed.py` — Mixed CE+KL config (alpha=0.9, per Peng et al. 2024)
- `config/sweep_base.py` — Short 250M-token sweep config for hyperparameter search (Muon + CCE-exact)
- `run_sweep.sh` — 3x3 matrix_lr x weight_decay sweep runner (3 GPUs)
- `run_eval_sweep.sh` — Eval-only pass for sweep runs missing final eval

## Key Design Decisions

- Vocabularies are uint32 (OLMo vocab_size=100,278 exceeds uint16)
- Top-k=128 logprobs stored as fp16 (more mantissa precision than bf16, cast to float32 at train time)
- Storage: ~772 bytes/token at k=128 (~720 GB per 1B tokens)
- Parquet shards ~2-3 GB each (2048 rows default), within HF's recommended 1-5 GB range
- BPB (bits-per-byte) as tokenizer-independent eval metric
- `attn_implementation="sdpa"` everywhere (avoids flash_attn install pain, dispatches to FA2 kernel when available)
- `dtype=` not `torch_dtype=` (deprecated in transformers)
- Distill mode uses less VRAM per sample than CE baseline: we gather student logits at the teacher's top-k indices *before* softmax, so the student never materializes the full (B, T, vocab_size) tensor. CE must compute softmax over the full 100K vocab. This means distill can run larger batch sizes than CE on the same hardware.

- `alpha` parameter: controls CE/KL blend in distill mode. `alpha=1.0` = pure KL (default), `alpha=0.9` = 90% KL + 10% CE (recommended per Peng et al. 2024). Pure KL causes gradient collapse when training from scratch — needs CE anchor.
- Peng et al. 2024 ("Scalable Knowledge Distillation for Pre-trained Language Models"): pure KL +1.6%, alpha=0.9 +3.6%, WSD schedule +8.0% over CE baseline
- CE mode uses fused CE (Liger or Apple CCE) to avoid materializing full (B,T,V) logit tensor (~4 GB saved). Backbone compiled separately so fused CE can handle lm_head internally. DDP falls back to standard F.cross_entropy.
- `optimizer_type`: `"adamw"` (default) or `"muon"`. Muon uses Newton-Schulz orthogonalization for 2D weight matrices, AdamW for embeddings and RMSNorm weights. Muon LRs are ~33x higher than AdamW (matrix_lr=0.02 vs 6e-4) because orthogonal updates have unit spectral norm.
- OLMo3 uses **post-norm** architecture (RMSNorm after sublayer, before residual add). Learnable RMSNorm weights (gamma) are essential — they let the model control sublayer output magnitude and prevent gradient amplification through the post-norm residual stream. Earlier attempts to remove learnable norms (matching nanochat's pre-norm convention) caused NaN at step ~1780 due to ~1000x/layer gradient amplification through the v_proj→o_proj path exceeding bf16 range.
- Gradient clipping (`grad_clip=1.0`) is enabled for both optimizers.
- Muon uses float32 weights + `torch.amp.autocast(bf16)`. AdamW uses pure bf16 weights (grad_clip + weight_decay keep things stable). Optimizer states are float32 for both.
- Muon embedding hyperparameters (nanochat-matched): `embed_tokens` betas `(0.8, 0.995)` wd `0.001`, `lm_head` betas `(0.8, 0.96)` wd `0.01` lr `0.008`, Muon matrix groups use `adam_betas=(0.8, 0.95)`, `adam_eps=1e-10`, `embedding_lr=0.2`.
- `cce_impl`: `"liger"` (default, Liger fused linear CE), `"cce"` (Apple CCE), or `"cce_exact"` (CCE with Kahan summation + no gradient filtering). CCE v25.1.1 had a bf16 gradient accumulation bug (+0.077 BPB vs Liger); `cce_exact` (alias for `cce_kahan_full_c_full_e`) fixes this with fp32 accumulators. Old `cce` with `filter_eps` caused NaN with Muon around step 458 (zero-gradient matrices break Newton-Schulz). CCE under autocast requires explicit `.bfloat16()` cast on hidden states (autocast promotes RMSNorm output to float32).

## Training Notes

- KL loss uses `reduction="batchmean"` which sums over (seq_len, top_k) and divides by batch_size only — ~seq_len larger than per-token CE. With from-scratch training (unlike warm-start pruning), early KL is large, grad clipping fires constantly, and Adam's per-parameter adaptivity is lost. Options to address:
  1. Normalize by n_ctx (`loss / n_ctx`) — per-token KL, directly comparable to CE, cleanest fix
  2. Use a separate (lower) LR for distill mode — pragmatic but two configs to maintain
  3. Raise grad_clip for distill mode — lets more gradient signal through so Adam can adapt
  4. Temperature scaling — divide logits by T before softmax on both sides, softens distributions and spreads mass beyond top-k
- These are not mutually exclusive (e.g. temperature is an independent knob from loss normalization)

## Muon Stability Notes

Our Muon implementation uses Newton-Schulz5 (5 steps, bf16 NS internally) with Muon Split (per-head Q/K/V orthogonalization, from GLM-5). At scale, other labs use additional techniques:
- **Kimi K2**: MuonClip optimizer + QK-Clip for attention logit stability
- **GLM-5**: Muon Split — per-head orthogonalization of Q/K/V matrices (avoids instability "without any clipping strategy")
- **Moonlight** (Liu et al. 2025): "Muon is Scalable for LLM Training" — details on scaling Muon

### Post-norm vs Pre-norm (critical for stability)
OLMo3 uses **post-norm** (RMSNorm after sublayer). This means sublayers receive raw unnormalized residual stream input, causing ~1000x gradient amplification per layer through the v_proj→o_proj path. Over 12 layers this compounds to ~1e36, near bf16 max (3.4e38). **Learnable RMSNorm gamma is required** — it lets the model scale down sublayer outputs, reducing the amplification factor. This is different from pre-norm architectures (GPT-2, nanochat) where removing learnable norms is safe because sublayers always see normalized input.

## Commands

```bash
# Tokenize data
uv run python data/fineweb_edu/prepare.py --tokenizer <model_name>

# Generate logprobs (single GPU)
uv run python generate_logprobs.py --teacher <model_name> --data data/fineweb_edu/train.bin --output <output_dir>

# Train CE baseline
uv run python train.py config/baseline_olmo.py

# Train CE baseline with Muon + CCE-exact (3 GPU DDP)
CUDA_VISIBLE_DEVICES=0,1,2 uv run torchrun --standalone --nproc_per_node=3 train.py config/baseline_cce_muon.py

# Train with distillation
uv run python train.py config/distill_olmo.py

# Run hyperparameter sweep (3 GPUs, 9 configs)
uv run bash run_sweep.sh
```

## Hardware

- 3x RTX 3090 (24 GB each), GPU 1 drives the display
- 128 GB system RAM
- 4 TB disk at /media/henry/MoreFiles for logprob storage
- Python 3.12, managed with `uv`
