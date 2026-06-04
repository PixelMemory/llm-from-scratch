"""MILESTONE 1 — Sampling & the generation loop  (the outermost layer).

You are re-implementing the decoding logic that turns a trained model's logits
into text. This is where temperature / top-k / top-p / greedy live, plus the
autoregressive loop itself.

HOW TO WORK THIS MILESTONE
  1. Read the spec in each function below. Do NOT open the reference file yet.
  2. Implement each function, replacing `raise NotImplementedError`.
  3. Run:  pytest tests/test_sampling.py -v
  4. When green, see YOUR code drive the real model:
         python generate.py --impl student --prompt "The " --tokens 200
  5. Only then, diff against nanollm/reference/sampling.py to compare.

You may use torch ops (softmax, topk, sort, cumsum, multinomial, masked_fill,
argmax, scatter). No copying from the reference.

INTERVIEW PROBES to answer out loud when done (write answers in NOTES.md):
  - Why does the standard pipeline order matter (temperature -> top-k -> top-p)?
  - Why is beam search a bad fit for open-ended chat generation?
  - top-k vs top-p: when does each fail, and why is top-p adaptive?
  - In `generate`, why crop the context to block_size? What breaks if you don't?
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature.

    Spec: return logits / temperature. Raise ValueError if temperature <= 0
    (true argmax is handled by the `greedy` path, not T=0).

    Intuition: softmax(z/T). T<1 sharpens toward the argmax; T>1 flattens
    toward uniform.
    """
    # TODO(milestone-1): implement temperature scaling
    raise NotImplementedError


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k largest logits along the last dim; set the rest to -inf.

    logits: (..., vocab). Return same shape. Clamp k to the vocab size.
    Hint: torch.topk gives you the k-th largest value to threshold against.
    """
    # TODO(milestone-1): implement top-k filtering
    raise NotImplementedError


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability mass is >= p; set the rest to -inf. Always keep the single most
    likely token (so p very small still yields a valid distribution).

    logits: (..., vocab). Return same shape, with kept logits at their ORIGINAL
    positions (i.e. if you sort, remember to scatter back).

    Steps:
      1. sort logits descending (keep the original indices)
      2. probs = softmax(sorted_logits); cum = cumsum(probs)
      3. mark tokens to REMOVE where the cumulative mass *before* them already
         exceeds p, i.e. (cum - probs) > p ; never remove index 0
      4. set removed sorted-logits to -inf, scatter back to original order
    """
    # TODO(milestone-1): implement top-p / nucleus filtering
    raise NotImplementedError


def sample_next(logits: torch.Tensor, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None,
                greedy: bool = False) -> torch.Tensor:
    """Given last-position logits (B, vocab), return next token ids (B, 1).

    Pipeline:
      - greedy: return argmax (ignore temperature/top_k/top_p)
      - else: temperature (if != 1.0) -> top_k (if set) -> top_p (if set)
              -> softmax -> torch.multinomial(probs, 1)

    NOTE for reproducible tests: do NOT reseed inside this function. The tests
    seed the RNG right before calling you, then call the reference the same way.
    """
    # TODO(milestone-1): compose the helpers above into a single next-token step
    raise NotImplementedError


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             greedy: bool = False) -> torch.Tensor:
    """Autoregressively extend idx (B, T) by max_new_tokens tokens and return
    the full (B, T + max_new_tokens) sequence.

    Each step:
      - crop context to the last `model.cfg.block_size` tokens
      - forward: logits, _ = model(idx_cond)   # logits is (B, t, vocab)
      - take the last position's logits (B, vocab)
      - next_id = sample_next(...) ; append along dim=1

    Put model in eval() mode first.
    """
    # TODO(milestone-1): implement the autoregressive decoding loop
    raise NotImplementedError
