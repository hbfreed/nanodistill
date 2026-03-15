# Distillation: student learns from OLMo-3 base teacher logits
# Clone and modify for each teacher variant:
#   - Olmo-3-1025-7B (base)         -> out/distill_olmo3_base
#   - Olmo-3-7B-Instruct (instruct) -> out/distill_olmo3_instruct
#   - Olmo-3-7B-Think (think)       -> out/distill_olmo3_think
#   - Qwen/Qwen3-8B                 -> out/distill_qwen3 (different vocab_size!)
#   - etc.
mode = "distill"
dataset = "fineweb_edu"
vocab_size = 100278  # must match teacher's tokenizer
n_ctx = 2048
logprobs_path = "/media/henry/MoreFiles/olmo3-base-logprobs"
out_dir = "out/distill_olmo3_base"
wandb_run_name = "distill-olmo3-base"
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
