"""
Combined Muon+AdamW optimizer for single-GPU training.

Muon (MomentUm Orthogonalized by Newton-schulz) from Keller et al.
Newton-Schulz5 orthogonalization from nanochat/modded-nanogpt.

Muon is used for 2D weight matrices (linear layers), AdamW for everything
else (embeddings, norms, biases). Param groups should have a "kind" field
set to "muon" or "adamw".
"""

import torch
from torch import Tensor


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the slope
    at zero. This does not produce UV^T exactly but rather US'V^T where S' is diagonal
    with S_{ii}' ~ Uniform(0.5, 1.5), which empirically doesn't hurt performance.

    From modded-nanogpt by @kellerjordan, batched implementation by @scottjmaddox.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonAdamW(torch.optim.Optimizer):
    """Combined Muon+AdamW optimizer for single-GPU training.

    Each param group must have a "kind" field:
      - "muon": SGD-momentum + Newton-Schulz orthogonalization (for 2D weight matrices)
      - "adamw": standard AdamW (for embeddings, norms, biases)

    Muon groups support: lr, momentum, nesterov, ns_steps, weight_decay
    AdamW groups support: lr, betas, eps, weight_decay
    """

    def __init__(self, param_groups, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5):
        """Set shared defaults; each param group must include kind='muon' or kind='adamw'."""
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Dispatch each param group to its Muon or AdamW update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("kind") == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        """SGD-momentum step with Newton-Schulz orthogonalization and optional per-head split."""
        lr = group["lr"]
        momentum = group.get("momentum", 0.95)
        nesterov = group.get("nesterov", True)
        ns_steps = group.get("ns_steps", 5)
        wd = group.get("weight_decay", 0.0)
        n_heads = group.get("n_heads")  # Muon Split: per-head orthogonalization

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.float()

            # Weight decay (decoupled, applied before update)
            if wd != 0:
                p.mul_(1 - lr * wd)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)  # float32
            buf = state["momentum_buffer"]
            buf.lerp_(g, 1 - momentum)
            g = g.lerp_(buf, momentum) if nesterov else buf

            if n_heads is not None:
                # Muon Split (GLM-5): reshape to [n_heads, head_dim, hidden] so NS
                # orthogonalizes each head independently, preventing cross-head
                # coupling that causes attention logit explosion.
                orig_shape = g.shape
                g = g.view(n_heads, -1, g.size(-1))
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                g = g.view(orig_shape)
                p.add_(g, alpha=-lr * max(1, p.size(-2) / p.size(-1)) ** 0.5)
            else:
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                p.add_(g, alpha=-lr * max(1, p.size(-2) / p.size(-1)) ** 0.5)

    def _adamw_step(self, group):
        """Bias-corrected AdamW with decoupled weight decay in float32."""
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = group.get("eps", 1e-8)
        wd = group.get("weight_decay", 0.0)

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.float()
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            t = state["step"]

            # Decoupled weight decay
            if wd != 0:
                p.mul_(1 - lr * wd)

            # Update running averages (float32)
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            # Bias-corrected step (float32), cast update to param dtype
            bias1 = 1 - beta1 ** t
            bias2 = 1 - beta2 ** t
            denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
            step_size = lr / bias1
            update = exp_avg / denom
            p.add_(update.to(p.dtype), alpha=-step_size)
