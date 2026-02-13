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
logprobs_path = "hbfreed/olmo3-base-fineweb-logprobs"
out_dir = "out/distill_olmo3_base"
wandb_run_name = "distill-olmo3-base"
wandb_log = True

batch_size = 12
gradient_accumulation_steps = 40
max_iters = 5100
lr_decay_iters = 5100
warmup_iters = 500
learning_rate = 6e-4
min_lr = 6e-5
