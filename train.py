"""Train the reference GPT on a character-level corpus.

This trains the ORACLE model so you have something real to generate from while
you work the milestones. The training loop already includes the "production"
touches that Milestone 5 will ask you to re-derive yourself:
  - AdamW with weight decay only on 2D params
  - linear warmup + cosine decay LR schedule
  - gradient clipping
  - (on CUDA) bf16 autocast for speed

Usage:
  python train.py                          # tiny model on the bundled corpus (~1-2 min, laptop)
  python train.py --smoke                  # ~10s sanity run
  python train.py --profile gpu --iters 6000   # ~11M-param model (for an A100/L4/T4)
  python train.py --data data/input.txt    # point at a different corpus
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os

import torch

from config import GPTConfig, TrainConfig, SMOKE_GPT, GPU_GPT
from nanollm.reference.model import GPT
from nanollm.reference.tokenizer import CharTokenizer
from nanollm.util import get_device, save_checkpoint


def get_lr(it: int, tc: TrainConfig) -> float:
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / tc.warmup_iters
    if it >= tc.max_iters:
        return tc.min_lr
    ratio = (it - tc.warmup_iters) / (tc.max_iters - tc.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


def make_batch_fn(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)
    return get_batch


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters: int, amp_ctx) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch()
        with amp_ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["tiny", "gpu"], default="tiny",
                    help="'tiny' for laptop, 'gpu' for a bigger model on a real GPU")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast sanity run")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--data", type=str, default=None, help="path to a UTF-8 text corpus")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    tc = TrainConfig()
    if args.data:
        tc.data_path = args.data

    if args.smoke:
        base = SMOKE_GPT
        tc.max_iters, tc.warmup_iters, tc.eval_interval, tc.eval_iters, tc.batch_size = 30, 5, 15, 5, 16
    elif args.profile == "gpu":
        base = GPU_GPT
        tc.max_iters, tc.warmup_iters, tc.eval_interval = 6000, 200, 500
        tc.batch_size, tc.learning_rate, tc.min_lr = 64, 3e-4, 3e-5
    else:
        base = GPTConfig()
    if args.iters is not None:
        tc.max_iters = args.iters

    torch.manual_seed(tc.seed)
    device = get_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    amp_ctx = (torch.autocast(device_type="cuda", dtype=dtype)
               if device == "cuda" else contextlib.nullcontext())
    print(f"device: {device} | profile: {'smoke' if args.smoke else args.profile} | amp: {dtype}")

    # --- data ---
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = tc.data_path if os.path.isabs(tc.data_path) else os.path.join(here, tc.data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(text):,} chars, vocab: {tokenizer.vocab_size}")

    # --- model ---
    cfg = GPTConfig(**{**base.__dict__, "vocab_size": tokenizer.vocab_size})
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M | block_size: {cfg.block_size} | batch: {tc.batch_size}")

    get_train = make_batch_fn(train_data, cfg.block_size, tc.batch_size, device)
    get_val = make_batch_fn(val_data, cfg.block_size, tc.batch_size, device)
    optimizer = model.configure_optimizers(tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2))

    # --- train loop ---
    model.train()
    for it in range(tc.max_iters):
        lr = get_lr(it, tc)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % tc.eval_interval == 0 or it == tc.max_iters - 1:
            tl = estimate_loss(model, get_train, tc.eval_iters, amp_ctx)
            vl = estimate_loss(model, get_val, tc.eval_iters, amp_ctx)
            print(f"iter {it:5d} | lr {lr:.2e} | train {tl:.4f} | val {vl:.4f}", flush=True)

        optimizer.zero_grad(set_to_none=True)
        for _ in range(tc.grad_accum_steps):
            x, y = get_train()
            with amp_ctx:
                _, loss = model(x, y)
            (loss / tc.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

    os.makedirs(os.path.join(here, tc.out_dir), exist_ok=True)
    ckpt_path = os.path.join(here, tc.out_dir, "ckpt.pt")
    save_checkpoint(ckpt_path, model, tokenizer, cfg)
    print(f"saved checkpoint -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
