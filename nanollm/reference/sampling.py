"""Reference implementation of decoding / sampling.

This is the OUTERMOST layer of the system — the thing you call to make the model
produce text. Milestone 1 asks you to re-implement everything here yourself in
`nanollm/student/sampling.py`.

The standard sampling pipeline applies transforms in this order:
    logits  ->  temperature  ->  top-k  ->  top-p  ->  softmax  ->  sample
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by T. T<1 sharpens, T>1 flattens. T->0 approaches argmax."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy=True for argmax)")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits per row; set the rest to -inf."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]   # (..., 1)
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability mass >= p; set the rest to -inf. Always keep the top-1 token."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # Remove tokens whose *preceding* cumulative mass already exceeds p.
    remove = (cum - probs) > p
    remove[..., 0] = False                       # never drop the most likely token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter the filtered logits back to their original vocab positions.
    out = torch.empty_like(logits).fill_(float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def sample_next(logits: torch.Tensor, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None,
                greedy: bool = False) -> torch.Tensor:
    """Given last-position logits (B, vocab), return next token ids (B, 1)."""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             greedy: bool = False) -> torch.Tensor:
    """Autoregressively extend `idx` (B, T) by max_new_tokens tokens.

    Crops context to the model's block_size each step (no KV cache yet — that's
    a later optimization milestone).
    """
    model.eval()
    block_size = model.cfg.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]                # (B, vocab)
        next_id = sample_next(logits, temperature, top_k, top_p, greedy)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
