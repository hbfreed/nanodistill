# 5B baseline: CCE + AdamW + WSD schedule
mode = "ce"
dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
out_dir = "out/baseline_cce_adamw"
wandb_run_name = "baseline-cce-adamw"
wandb_log = True

batch_size = 20
total_tokens = 5_000_000_000
tokens_per_step = 500_000
warmup_iters = 1000
lr_schedule = "wsd"
cooldown_fraction = 0.2
eval_interval = 1000

learning_rate = 6e-4
min_lr = 6e-5

cce_impl = "cce"
optimizer_type = "adamw"
