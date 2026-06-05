# Understanding the console (the web UI)

A guide to `app.py` / `frontend.html` — what every part does and how to use it
to actually learn.

## Start / stop

```bash
cd llm-from-scratch
python3 app.py            # serves http://127.0.0.1:5050
# Ctrl-C in that terminal to stop it
```

Open **http://127.0.0.1:5050**. It's a local-only server (nobody else can reach
it). It's just a wrapper around `train.py` / `generate.py` / `pytest` — anything
you do in the UI you could also do from the terminal.

---

## The top bar (status pills)

| Pill | Meaning |
|---|---|
| `device mps` | where the model runs — `mps` (your Apple GPU), `cuda`, or `cpu`. |
| `checkpoint ready` / `none` | whether a trained model exists at `out/ckpt.pt`. "none" → use the **Train** panel first. |
| `⚙︎ training…` | shows only while a training run you started from the UI is in progress. |

---

## ① Generate — run the model

This is where you turn the model into text. Controls:

- **implementation toggle — `reference` vs `student`** ← the most important switch.
  - `reference` = my complete, correct code. Always works. Use it to see what
    "good" output looks like.
  - `student` = **your** code in `nanollm/student/sampling.py`. Until you finish
    Milestone 1 it politely says "not implemented yet." Once your tests pass, the
    same button runs *your* decoding loop. The server **re-reads your file on
    every click**, so: edit → save → Generate, no restart.
- **prompt** — the starting text. ⚠️ This is a *character-level* model, so it
  only knows the characters in its training corpus. Unknown characters are
  dropped (the UI tells you). With the bundled model, start prompts with words
  from the sample text (e.g. "The ", "Attention ").
- **tokens** — how many characters to generate.
- **temperature** — randomness of each pick (see the cheat sheet below).
- **top-k** — keep only the *k* most likely next characters (blank = off).
- **top-p** — keep the smallest set of characters whose probability adds up to
  *p* (blank = off). Adaptive alternative to top-k.
- **greedy** — always take the single most likely character. Deterministic;
  **ignores temperature / top-k / top-p** (they grey out).

The line under the button is feedback: green = which implementation ran, amber =
a warning (e.g. dropped characters), red = an error.

### Sampling cheat sheet (this is also interview gold)

The pipeline is **temperature → top-k → top-p → sample** (greedy skips all of it).

| Setting | Effect | Try it |
|---|---|---|
| `greedy` on | one fixed, "safest" continuation; repetitive | reproduces memorized text on this tiny model |
| low temp (0.2–0.5) | confident, repetitive | `temp 0.3` |
| mid temp (0.7–0.9) | the usual sweet spot | `temp 0.8` |
| high temp (1.2+) | wild, often incoherent | `temp 1.5` |
| top-k = 10 | only the 10 likeliest chars each step | `temp 1.0, top_k 10` |
| top-p = 0.9 | keep the top 90% mass — fewer when confident, more when unsure | `temp 1.0, top_p 0.9` |

Watching the *same prompt* change as you move these knobs is the fastest way to
build intuition for what each one does.

---

## ② Tests — Milestone progress

One button runs the current milestone's pytest suite (`tests/test_sampling.py`)
and shows a badge:

- 🔴 `✗ 0 passed, 24 failed` — the starting state (your worklist).
- 🟢 `✓ 24 passed, 0 failed` — milestone done.

Expand **"full pytest output"** to see exactly which assertions fail and why —
that's your guide for what to fix next. Identical to running
`pytest tests/test_sampling.py` in the terminal.

---

## ③ Train — make / refresh the model

- **iters** — how many training steps. More = better samples (and longer wait).
- **Start training** — runs `train.py` in the background and streams the live
  log: each line is `iter N | lr … | train <loss> | val <loss>`. Watch **train
  loss fall**. When it finishes, the checkpoint pill flips to **ready** and the
  new model is used immediately.

> On this tiny 4 KB corpus the model *memorizes* (train loss → ~0, val loss
> rises). That's expected and is itself a lesson in overfitting — for real
> samples you'd train on more data (the Colab GPU path uses tiny-shakespeare).

---

## The intended learning loop (Milestone 1)

1. Open `nanollm/student/sampling.py`, read the spec for one function.
2. Implement it. Save.
3. UI → **Tests** → *Run Milestone 1 tests*. Read failures, fix, repeat → green.
4. UI → **Generate**, toggle to **student**, click Generate — watch *your* code
   produce text from the real model.
5. Compare against the `reference` toggle. Then answer the milestone's interview
   probes in `NOTES.md`, out loud.

That loop — implement → test → see it run → explain — is the whole point of the
project. The console just makes steps 3–4 one click instead of a terminal trip.
