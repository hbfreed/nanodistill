"""Memory-light sparse KL for offline distillation from top-k teacher logprobs.

The student's logits at the teacher's top-k indices are computed by a custom
autograd Function that gathers lm_head rows in chunks and does NOT keep the
gathered (N, K, D) tensor for backward; the backward re-gathers per chunk.
The full-vocab log-partition comes from CCE's fused kernel (differentiable lse).
"""

import warnings

import torch
from cut_cross_entropy import linear_cross_entropy

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")


class SparseLogits(torch.autograd.Function):
    """out[n, k] = <h[n], w[ids[n, k]]> without saving the gathered (N, K, D) rows.

    Backward uses one sparse (V x N) matrix A with A[ids[n,k], n] = g[n,k]:
    grad_w = A @ h and grad_h = A^T @ w, both as CSR SpMM (no gather, no outer products).
    """

    @staticmethod
    def forward(ctx, h, w, ids, chunk):
        N, K = ids.shape
        out = torch.empty(N, K, device=h.device, dtype=torch.float32)
        for s in range(0, N, chunk):
            gw = w[ids[s : s + chunk]]  # (c, K, D), freed after the bmm
            out[s : s + chunk] = torch.bmm(gw, h[s : s + chunk].unsqueeze(-1)).squeeze(-1).float()
        ctx.save_for_backward(h, w, ids)
        return out

    @staticmethod
    def backward(ctx, g):
        h, w, ids = ctx.saved_tensors
        N, K = ids.shape
        V = w.shape[0]
        rows = ids.reshape(-1)
        cols = torch.arange(N, device=ids.device).repeat_interleave(K)
        vals = g.reshape(-1).float()
        A = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (V, N)).to_sparse_csr()
        At = torch.sparse_coo_tensor(torch.stack([cols, rows]), vals, (N, V)).to_sparse_csr()
        grad_w = torch.sparse.mm(A, h.float())
        grad_h = torch.sparse.mm(At, w.float())
        return grad_h.to(h.dtype), grad_w.to(w.dtype), None, None


def sparse_kl_loss(h, w, t_ids, t_lps, targets=None, chunk=1024, tail_eps=1e-6):
    """Per-position KL(teacher || student) over the teacher's top-k plus a tail bucket,
    and (optionally) the ordinary next-token cross-entropy, sharing one full-vocab lse.

    h: (N, D) bf16 hidden states; w: (V, D) lm_head weight (any float dtype);
    t_ids: (N, K) int64 teacher top-k token ids; t_lps: (N, K) float32 TRUE teacher logprobs;
    targets: optional (N,) int64 next-token ids, -1 where there is no target (last position).
    Returns (mean KL, mean CE or None). CE costs one extra dot per position (not a second
    100K-vocab pass): CE = lse - logit[target].
    """
    w_b = w if w.dtype == h.dtype else w.to(h.dtype)  # one cast; gathering from fp32 is ~3x slower
    student_topk_logits = SparseLogits.apply(h, w_b, t_ids, chunk)  # (N, K) fp32
    _, lse = linear_cross_entropy(h, w_b, t_ids[:, 0], impl="cce_exact", reduction="none", return_lse=True)
    lse = lse.float()
    student_topk_lps = student_topk_logits - lse.unsqueeze(-1)
    student_tail_lp = torch.log1p(-student_topk_lps.exp().sum(-1).clamp(max=1 - tail_eps))

    teacher_p = t_lps.exp()
    teacher_tail_p = (1 - teacher_p.sum(-1)).clamp(min=tail_eps)
    kl = (teacher_p * (t_lps - student_topk_lps)).sum(-1) \
        + teacher_tail_p * (teacher_tail_p.log() - student_tail_lp)

    ce = None
    if targets is not None:
        valid = targets >= 0
        tgt = targets.clamp(min=0).unsqueeze(-1)  # (N, 1)
        target_logit = SparseLogits.apply(h, w_b, tgt, chunk).squeeze(-1)
        ce = ((lse - target_logit) * valid).sum() / valid.sum().clamp(min=1)
    return kl.mean(), ce
