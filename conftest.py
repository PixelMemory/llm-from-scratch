"""Shared pytest fixtures and import-path setup.

Ensures the repo root is importable (so `import config` / `import nanollm`
work no matter where pytest is invoked from), and provides small, seeded,
CPU-only objects the milestone tests reuse.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GPTConfig            # noqa: E402
from nanollm.reference.model import GPT  # noqa: E402


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def tiny_cfg():
    # small, GQA-enabled, even head_dim — exercises the real code paths fast
    return GPTConfig(vocab_size=17, n_layer=2, n_head=2, n_kv_head=1,
                     n_embd=32, block_size=16)


@pytest.fixture
def ref_model(tiny_cfg, device):
    torch.manual_seed(0)
    model = GPT(tiny_cfg).to(device)
    model.eval()
    return model


@pytest.fixture
def logits():
    """A fixed batch of last-position logits, shape (4, 17)."""
    torch.manual_seed(123)
    return torch.randn(4, 17)
