# Online distillation: student learns from OLMo-3 base teacher (pure KL)
# Teacher on GPU 0, student DDP on GPU 1-2
# Launch: torchrun --nproc_per_node=2 train.py config/distill_olmo.py
mode = "distill"
teacher_model = "allenai/OLMo-3-1025-7B"
teacher_quantize = ""  # bf16
temperature = 1.0

dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
out_dir = "out/distill_olmo3_base"
wandb_run_name = "distill-olmo3-base"
wandb_log = True

eval_interval = 1000
batch_size = 2
total_tokens = 1_000_000_000
tokens_per_step = 500_000
lr_schedule = "wsd"
warmup_iters = 200
cooldown_fraction = 0.2
learning_rate = 6e-4
min_lr = 6e-5
