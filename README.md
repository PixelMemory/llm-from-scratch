# Build a Modern LLM From Scratch — Top-Down

A learning project for your ML-engineer interview prep. You start with a
**complete, runnable** small language model and then **peel it apart from the
outside in**, re-implementing each layer yourself against tests. You always
have a working system; you progressively come to *own* every line of it —
tokenizer, attention, training loop, alignment.

This maps directly onto your study plan's "live ML coding from scratch" core
(weeks 1–2) plus the alignment/PEFT material (week 4). Phase 2 then turns the
model into a served ML system (week 3).

---

## The idea: top-down, not bottom-up

Your weeks 1–4 built things bottom-up: backprop → attention → block → assemble.
This project runs the other direction. The reference model already works:

```
generate.py  ──calls──▶  sampling loop  ──calls──▶  model.forward
                                                       ├─ token embedding
                                                       ├─ N × Block
                                                       │    ├─ RMSNorm
                                                       │    ├─ Attention (RoPE, GQA, causal)
                                                       │    └─ SwiGLU MLP
                                                       └─ final norm + LM head
```

You peel from the top of that call stack downward. Each **milestone** hands you
a stubbed file in `nanollm/student/` and a set of failing tests. You implement
the layer, turn the tests green, then flip a switch to watch *your* code run
inside the real model.

---

## Quickstart

```bash
cd llm-from-scratch
# deps already present in your env; if on a fresh machine:
#   pip install -r requirements.txt

# 1. Train the reference model (~1–2 min on your MPS GPU). Produces out/ckpt.pt
python train.py
#    (or `python train.py --smoke` for a ~10s sanity run)

# 2. Watch the whole system work, end to end:
python generate.py --impl reference --prompt "The " --tokens 300

# 3. Start Milestone 1: open MILESTONES.md, then implement
#    nanollm/student/sampling.py until the tests pass:
pytest tests/test_sampling.py -v

# 4. See YOUR code drive the model:
python generate.py --impl student --prompt "The " --tokens 300
```

> Train for longer (`python train.py --iters 5000`) any time you want nicer
> samples. The default tiny model is built for fast iteration, not quality.

---

## The workflow for every milestone

1. **Read the spec** in the `nanollm/student/<layer>.py` stub. *Don't open the
   `reference/` version yet* — attempting before peeking is where the learning
   is.
2. **Implement** the stub.
3. **`pytest tests/test_<layer>.py -v`** until green. Tests check your code
   against the reference oracle and against invariants (shapes, softmax sums to
   1, causal mask zeroes the future, RoPE relative-position property, …).
4. **Run it live**: a flag swaps your implementation into the real model so you
   see it generate / train.
5. **Diff against the reference** and read it critically. Note anything you did
   differently.
6. **Answer the interview probes** for that milestone in `NOTES.md`, out loud.
   This is the part that turns "I implemented it" into "I can whiteboard it."

The honor-code rule: the `reference/` directory is the answer key. Using it
before you've attempted the milestone yourself defeats the entire point.

---

## Layout

```
llm-from-scratch/
├── README.md            ← you are here
├── MILESTONES.md        ← the full top-down sequence + concepts + probes
├── NOTES.md             ← your answers to the interview probes (you fill this)
├── config.py            ← GPTConfig (architecture) + TrainConfig (the run)
├── train.py             ← trains the reference model -> out/ckpt.pt
├── generate.py          ← generate text; --impl {reference, student}
├── data/input.txt       ← tiny char-level corpus
├── nanollm/
│   ├── reference/       ← complete, correct ORACLE (answer key + makes it run)
│   │   ├── model.py     ← GPT: RoPE, GQA attention, RMSNorm, SwiGLU
│   │   ├── sampling.py  ← decoding: greedy / temperature / top-k / top-p
│   │   └── tokenizer.py ← char tokenizer (BPE comes as a later milestone)
│   ├── student/         ← YOUR implementations (start as stubs)
│   │   └── sampling.py  ← Milestone 1
│   └── util.py          ← device + checkpoint helpers
└── tests/               ← one test module per milestone
    └── test_sampling.py ← Milestone 1 tests
```

New student stubs + tests get added as you reach each milestone (so we keep the
codebase honest and you're never staring at ten red files at once). Milestone 1
(sampling) is live now.

---

## Phase 2 (later): from model to ML system

Once you own the model, we turn it into a served system — the week-3 material:
a tokenizer/embedding service, a small RAG pipeline over a document store, a
tool-using agent loop, and an eval harness (faithfulness, retrieval recall,
LLM-as-judge). Same scaffold-and-test style. See the end of `MILESTONES.md`.
