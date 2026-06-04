"""Milestone 1 tests: sampling & generation.

Run:  pytest tests/test_sampling.py -v

Every test compares YOUR implementation (nanollm.student.sampling) against the
reference and/or checks an invariant. They will error/fail until you implement
the stubs. Make them all green.
"""

import pytest
import torch

from nanollm.reference import sampling as ref
from nanollm.student import sampling as stu

torch.set_grad_enabled(False)


# --- temperature -----------------------------------------------------------
def test_temperature_matches_reference(logits):
    for t in (0.5, 0.8, 1.0, 2.0):
        assert torch.allclose(stu.apply_temperature(logits, t),
                              ref.apply_temperature(logits, t), atol=1e-6)


def test_temperature_rejects_nonpositive(logits):
    with pytest.raises(ValueError):
        stu.apply_temperature(logits, 0.0)


# --- top-k -----------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3, 5, 17, 100])
def test_top_k_keeps_exactly_k(logits, k):
    out = stu.top_k_filter(logits, k)
    finite = torch.isfinite(out).sum(dim=-1)
    expected = min(k, logits.size(-1))
    assert (finite == expected).all(), f"expected {expected} kept, got {finite.tolist()}"


@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_k_matches_reference(logits, k):
    assert torch.equal(torch.isfinite(stu.top_k_filter(logits, k)),
                       torch.isfinite(ref.top_k_filter(logits, k)))


# --- top-p -----------------------------------------------------------------
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
def test_top_p_matches_reference(logits, p):
    s = stu.top_p_filter(logits, p)
    r = ref.top_p_filter(logits, p)
    # same set of kept positions, same finite values
    assert torch.equal(torch.isfinite(s), torch.isfinite(r))
    mask = torch.isfinite(r)
    assert torch.allclose(s[mask], r[mask], atol=1e-6)


def test_top_p_keeps_at_least_one(logits):
    out = stu.top_p_filter(logits, 0.0)
    assert (torch.isfinite(out).sum(dim=-1) >= 1).all()


# --- sample_next -----------------------------------------------------------
def test_greedy_is_argmax(logits):
    out = stu.sample_next(logits, greedy=True)
    assert out.shape == (logits.size(0), 1)
    assert torch.equal(out.squeeze(-1), logits.argmax(dim=-1))


@pytest.mark.parametrize("kw", [
    {"temperature": 1.0},
    {"temperature": 0.7},
    {"top_k": 3},
    {"top_p": 0.9},
    {"temperature": 0.8, "top_k": 5, "top_p": 0.95},
])
def test_sample_next_matches_reference_seeded(logits, kw):
    torch.manual_seed(42)
    s = stu.sample_next(logits, **kw)
    torch.manual_seed(42)
    r = ref.sample_next(logits, **kw)
    assert torch.equal(s, r), f"mismatch for {kw}"


# --- generate loop ---------------------------------------------------------
def test_generate_shape_and_prefix(ref_model):
    idx = torch.randint(0, ref_model.cfg.vocab_size, (2, 3))
    out = stu.generate(ref_model, idx, max_new_tokens=10, greedy=True)
    assert out.shape == (2, 13)
    assert torch.equal(out[:, :3], idx)            # prefix preserved


def test_generate_matches_reference_seeded(ref_model):
    idx = torch.randint(0, ref_model.cfg.vocab_size, (2, 3))
    torch.manual_seed(7)
    s = stu.generate(ref_model, idx.clone(), 12, temperature=0.9, top_k=5)
    torch.manual_seed(7)
    r = ref.generate(ref_model, idx.clone(), 12, temperature=0.9, top_k=5)
    assert torch.equal(s, r)


def test_generate_respects_block_size(ref_model):
    # starting longer than block_size must not crash (context gets cropped)
    bs = ref_model.cfg.block_size
    idx = torch.randint(0, ref_model.cfg.vocab_size, (1, bs + 5))
    out = stu.generate(ref_model, idx, max_new_tokens=4, greedy=True)
    assert out.shape == (1, bs + 9)
