"""Generate text from a trained checkpoint.

The `--impl` flag is the heart of the top-down workflow: it chooses WHOSE
sampling code drives the model.

  python generate.py --impl reference --prompt "The " --tokens 300
  python generate.py --impl student  --prompt "The " --tokens 300   # uses YOUR code

`--impl reference` works out of the box. `--impl student` will raise
NotImplementedError until you finish Milestone 1 — then it just works, and you
are watching code you wrote produce text from a model you'll soon also own.
"""

from __future__ import annotations

import argparse
import os

import torch

from nanollm.util import get_device, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["reference", "student"], default="reference")
    ap.add_argument("--prompt", type=str, default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.impl == "reference":
        from nanollm.reference.sampling import generate
    else:
        from nanollm.student.sampling import generate

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, "out", "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit("No checkpoint found. Run `python train.py` first.")

    model, tok, cfg = load_checkpoint(ckpt_path, device=device)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(
        model, ids, args.tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        greedy=args.greedy,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
