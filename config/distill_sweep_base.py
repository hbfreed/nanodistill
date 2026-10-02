# Offline distillation sweep base config (250M tokens, ~1.7h per run)
# Uses pre-computed top-k=128 teacher logprobs (parquet shards from generate_logprobs.py; no teacher loaded)
# Usage: CUDA_VISIBLE_DEVICES=0 uv run python train.py config/distill_sweep_base.py --alpha=0.9 ...
mode = "distill"
logprob_dir = "/media/henry/MoreFiles/olmo3-base-logprobs"
teacher_model = ""  # empty = offline mode (no teacher loaded)
temperature = 1.0  # must be 1.0 for sparse offline KL
alpha = 0.9

dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
batch_size = 12  # matches CE baseline; sparse KL head now ~CE memory (see sparse_kl.py)
total_tokens = 250_000_000
tokens_per_step = 500_000
warmup_iters = 100
lr_schedule = "wsd"
cooldown_fraction = 0.2
eval_interval = 250

optimizer_type = "muon"
matrix_lr = 0.02
embedding_lr = 0.2
unembedding_lr = 0.008
weight_decay = 0.01
learning_rate = 6e-4
min_lr = 6e-5
cce_impl = "cce_exact"
cce_filter_eps = None

wandb_log = True
wandb_project = "nanodistill"
compile = True  # backbone compiles fine now that attention is mask-free causal SDPA (flash kernel): ~39% faster backbone
