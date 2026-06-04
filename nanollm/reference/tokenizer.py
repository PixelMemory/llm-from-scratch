"""Reference tokenizer: a dead-simple character-level tokenizer.

This is intentionally trivial so the early milestones (sampling, model,
attention) have something to run against. A real Byte-Pair Encoding (BPE)
tokenizer is its own milestone later in the project, at which point you'll
replace this with your own implementation in `nanollm/student/`.
"""

from __future__ import annotations


class CharTokenizer:
    def __init__(self, chars: list[str]):
        # `chars` must be a sorted, de-duplicated list for reproducibility.
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # --- (de)serialization for checkpoints ---
    def state_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        return cls(state["chars"])
