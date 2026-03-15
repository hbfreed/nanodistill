# nanodistill

"Does the Teacher Matter?" — off-policy knowledge distillation experiments comparing how teacher choice affects a small student model.

Starting with OLMo 3 7B's base, instruct, and thinking variants to see if post-training matters, then testing quantized teachers, and finally larger/smarter models (Qwen, GPT-OSS, etc.). Student is a 229M-parameter transformer trained on 5B tokens of FineWeb-Edu.

## Usage

```bash
# Tokenize data
uv run python data/fineweb_edu/prepare.py --tokenizer allenai/Olmo-3-1025-7B

# Generate teacher logprobs
uv run python generate_logprobs.py \
    --teacher allenai/Olmo-3-1025-7B \
    --data data/fineweb_edu/train.bin \
    --output /path/to/logprobs

# Train CE baseline
uv run python train.py config/baseline_olmo.py

# Train with distillation
uv run python train.py config/distill_olmo.py
```

## Structure

- `train.py` — Two modes: `ce` (cross-entropy baseline) and `distill` (sparse KL from pre-computed top-k logprobs)
- `generate_logprobs.py` — Pre-compute teacher top-k=128 logprobs as parquet shards. Supports DDP, `--device_map auto`, `--quantize {int8,int4}`
- `model.py` — Student model factory (`Olmo3ForCausalLM`)
- `config/` — Training configs for baseline and distillation runs

## References

- Peng et al. 2024, ["Scalable Knowledge Distillation for Pre-trained Language Models"](https://arxiv.org/abs/2403.14276) — mixed CE+KL with alpha=0.9 outperforms both pure CE and pure KL for pre-training distillation
