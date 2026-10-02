# Continued pretraining base config: 250M tokens from 5B AdamW baseline
# Shared by CE control and distill experiments
# Usage: overridden by run_continue.sh per-run
init_from = "out/baseline_cce_adamw/ckpt.pt"

dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
batch_size = 5
total_tokens = 250_000_000
tokens_per_step = 500_000
warmup_iters = 100
lr_schedule = "wsd"
cooldown_fraction = 0.2
eval_interval = 100

optimizer_type = "adamw"
weight_decay = 0.1
learning_rate = 6e-4
min_lr = 6e-5
cce_impl = "cce_exact"
cce_filter_eps = None

wandb_log = True
wandb_project = "nanodistill"
compile = False
