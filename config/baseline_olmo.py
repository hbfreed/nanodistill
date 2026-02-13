# Baseline: OLMo student trained on CE loss to Chinchilla-optimal
mode = "ce"
dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
out_dir = "out/baseline_olmo"
wandb_run_name = "baseline-olmo-ce"
wandb_log = True

# ~5B tokens / (batch_size * n_ctx * grad_accum) = max_iters
batch_size = 12
gradient_accumulation_steps = 40
# tokens_per_iter = 12 * 2048 * 40 = 983,040 ~ 1M tokens/iter
# 5B / 1M = ~5000 iters
max_iters = 5100
lr_decay_iters = 5100
warmup_iters = 500
learning_rate = 6e-4
min_lr = 6e-5
