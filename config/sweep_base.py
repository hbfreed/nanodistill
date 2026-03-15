# Muon + CCE sweep base config (250M tokens, ~1.7h per run)
mode = "ce"
dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
batch_size = 5
total_tokens = 250_000_000
tokens_per_step = 500_000
warmup_iters = 100
lr_schedule = "wsd"
cooldown_fraction = 0.2
eval_interval = 250
learning_rate = 6e-4
min_lr = 6e-5
cce_impl = "cce_exact"
optimizer_type = "muon"
wandb_log = True
