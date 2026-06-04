"""Shared utilities: device selection and checkpoint (de)serialization."""

from __future__ import annotations

import torch

from config import GPTConfig
from nanollm.reference.model import GPT
from nanollm.reference.tokenizer import CharTokenizer


def get_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_checkpoint(path: str, model: GPT, tokenizer: CharTokenizer, cfg: GPTConfig):
    torch.save(
        {
            "model": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "gpt_config": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_state_dict(ckpt["tokenizer"])
    return model, tok, cfg
