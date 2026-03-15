# Baseline: OLMo student trained on CE loss to Chinchilla-optimal
mode = "ce"
dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
out_dir = "out/baseline_olmo"
wandb_run_name = "baseline-olmo-ce"
wandb_log = True

batch_size = 5
total_tokens = 5_000_000_000
tokens_per_step = 500_000
warmup_iters = 500
learning_rate = 6e-4
min_lr = 6e-5
