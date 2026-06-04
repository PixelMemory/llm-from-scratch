"""Reference implementation of a small, *modern* decoder-only Transformer (GPT).

This is the ORACLE. It is complete and correct, and it is what makes the whole
project runnable on day one: you can `train.py` and `generate.py` immediately.

As you work through the milestones, you will re-implement these pieces yourself
in `nanollm/student/`. The tests compare your version against this one. Treat
this file as the answer key: try not to read it until after you've attempted a
milestone (or are checking your work).

Modern choices baked in (these are the things interviewers probe):
  - RoPE rotary positional embeddings (no learned position table)
  - Grouped-Query Attention (n_kv_head <= n_head)
  - Pre-norm residual blocks
  - RMSNorm instead of LayerNorm
  - SwiGLU feed-forward instead of ReLU/GeLU MLP
  - Tied input/output embeddings, no biases
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# ----------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ----------------------------------------------------------------------------
def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0,
                    device=None, dtype=torch.float32):
    """Return (cos, sin), each of shape (seq_len, head_dim // 2).

    theta_i = base ** (-2i / head_dim) gives a geometric spectrum of rotation
    frequencies; multiplying by position `m` gives the rotation angle m*theta_i.
    """
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, theta)            # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions of x by position-dependent angles.

    x:   (B, n_heads, T, head_dim)
    cos: (T, head_dim/2), sin: (T, head_dim/2)
    Interleaved-pair convention: dims (0,1), (2,3), ... are rotated together.
    """
    x1 = x[..., 0::2]                          # even dims -> (B, h, T, hd/2)
    x2 = x[..., 1::2]                          # odd dims
    cos = cos[None, None, :, :]                # (1, 1, T, hd/2)
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.stack([rx1, rx2], dim=-1)      # (B, h, T, hd/2, 2)
    return out.flatten(-2)                      # interleave back -> (B, h, T, hd)


# ----------------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ----------------------------------------------------------------------------
# Attention (causal, multi-head, GQA, RoPE)
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.hd = cfg.head_dim
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.hd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.hd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.hd, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)     # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)  # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_kv_head != self.n_head:                # GQA: broadcast kv heads
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)   # (B, nh, T, T)
        causal = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v                                            # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ----------------------------------------------------------------------------
# SwiGLU feed-forward
# ----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, bias: bool = False):
        super().__init__()
        if hidden is None:
            # ~ (2/3) * 4 * dim, rounded to a multiple of 8 (keeps param count ~ a 4x ReLU MLP)
            hidden = int(8 / 3 * dim)
            hidden = ((hidden + 7) // 8) * 8
        self.w_gate = nn.Linear(dim, hidden, bias=bias)
        self.w_up = nn.Linear(dim, hidden, bias=bias)
        self.w_down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ----------------------------------------------------------------------------
# Transformer block (pre-norm)
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight   # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos, sin = precompute_rope(T, self.cfg.head_dim, self.cfg.rope_base,
                                   device=idx.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def num_params(self) -> int:
        # subtract tied head (shares storage with tok_emb)
        n = sum(p.numel() for p in self.parameters())
        return n - self.lm_head.weight.numel()

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """Weight decay on 2D+ tensors only (matrices), not on norms/biases (1D)."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)
