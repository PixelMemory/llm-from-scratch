"""Central configuration for the from-scratch LLM project.

Two configs live here:
  - GPTConfig:   the *architecture* (what the model IS)
  - TrainConfig: the *training run* (how we fit it)

Keep models small. The whole point is to train in seconds-to-minutes on a
laptop (CPU or Apple MPS) so the feedback loop is fast. You are learning the
mechanics, not chasing benchmark numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- set at runtime from the tokenizer ---
    vocab_size: int = 256

    # --- architecture ---
    n_layer: int = 4          # number of transformer blocks
    n_head: int = 4           # number of *query* attention heads
    n_kv_head: int = 2        # number of *key/value* heads (GQA: n_kv_head < n_head)
    n_embd: int = 128         # residual-stream / model dimension (d_model)
    block_size: int = 128     # max context length (sequence length)
    rope_base: float = 10000.0  # RoPE theta base
    dropout: float = 0.0
    bias: bool = False        # modern LLMs drop biases in Linear/Norm layers

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head (GQA)"
        head_dim = self.n_embd // self.n_head
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    # data
    data_path: str = "data/input.txt"
    out_dir: str = "out"

    # optimization
    batch_size: int = 32
    max_iters: int = 2000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    seed: int = 1337


# A deliberately tiny config for fast smoke tests / CI.
SMOKE_GPT = GPTConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=32, block_size=32)

# A bigger config worth running on a real GPU (A100/L4/T4). ~11M params.
# Pair it with a real dataset (colab/get_data.py fetches tiny-shakespeare).
GPU_GPT = GPTConfig(n_layer=6, n_head=6, n_kv_head=6, n_embd=384, block_size=256, dropout=0.2)
