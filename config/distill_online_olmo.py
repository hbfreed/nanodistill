# Online distillation: OLMo-3 base teacher, full-vocab KL
# Teacher on GPU 0, student DDP on GPU 1-2
# Launch: torchrun --nproc_per_node=2 train.py config/distill_online_olmo.py
mode = "distill"
teacher_model = "allenai/OLMo-3-1025-7B"
teacher_quantize = ""  # bf16
temperature = 1.0
alpha = 0.9  # 90% KL + 10% CE (Peng et al. 2024)

dataset = "fineweb_edu"
vocab_size = 100278
n_ctx = 2048
out_dir = "out/distill_online_olmo3_base"
wandb_run_name = "distill-online-olmo3-base-1b"
wandb_log = True

batch_size = 2
total_tokens = 1_000_000_000
tokens_per_step = 500_000
warmup_iters = 200
lr_schedule = "wsd"
cooldown_fraction = 0.2
eval_interval = 500

learning_rate = 6e-4
min_lr = 6e-5
cce_impl = "cce_exact"
