# Mixed CE+KL distillation: OLMo-3 base teacher
# 90% KL + 10% CE per Peng et al. 2024 ("Scalable Knowledge Distillation")
# CE anchor prevents gradient collapse seen with pure KL
mode = "distill"
alpha = 0.9  # 90% KL + 10% CE (Peng et al. 2024)
dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
logprobs_path = "/media/henry/MoreFiles/olmo3-base-logprobs"
out_dir = "out/distill_olmo3_base_mixed"
wandb_run_name = "distill-olmo3-base-mixed"
wandb_log = True

eval_interval = 1000
batch_size = 10
grad_clip = 5.0
total_tokens = 1_000_000_000
tokens_per_step = 500_000
lr_schedule = "wsd"
warmup_iters = 200
cooldown_fraction = 0.2
learning_rate = 6e-4
min_lr = 6e-5
