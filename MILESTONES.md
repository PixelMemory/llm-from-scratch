# Milestones — the top-down peel

You re-implement the model from the **outside in**. Each milestone replaces one
layer of `nanollm/student/` and turns a test module green. Cross-references to
your study plan are in (parentheses).

Legend:  ✅ live now   ·   🔒 unlocked when you reach it (stub + tests added then)

---

## ✅ Milestone 1 — Sampling & the generation loop  (week 2, Day 10)

**The outermost layer**: how logits become text.

- **Implement:** `nanollm/student/sampling.py`
  — `apply_temperature`, `top_k_filter`, `top_p_filter`, `sample_next`, `generate`
- **Test:** `pytest tests/test_sampling.py -v`
- **Run live:** `python generate.py --impl student --prompt "The " --tokens 200`
- **Concept recap:** greedy is deterministic and repetitive; temperature scales
  logits (`softmax(z/T)`); top-k keeps a fixed number of candidates; top-p
  (nucleus) keeps a *probability-adaptive* set. Pipeline order is
  temperature → top-k → top-p → softmax → sample.
- **Probes (write answers in NOTES.md):**
  - Why is beam search bad for open-ended chat but fine for translation?
  - top-k vs top-p — concretely, when does each misbehave?
  - Why crop context to `block_size` in the loop? What breaks otherwise?
  - How would a KV cache change this loop's complexity? (foreshadows M3.5)

---

## 🔒 Milestone 2 — The model skeleton & the block  (week 1, Day 7)

Peel open `model.forward`: embedding → stacked pre-norm blocks → final norm →
tied LM head, and the residual wiring `x = x + attn(norm(x))`.

- **Implement:** `nanollm/student/model.py` — `GPT`, `Block` (you'll be *given*
  attention/norm/mlp as reference imports at first, then replace them in M3–M4).
- **Test:** parameter count, output shape `(B,T,vocab)`, weight tying
  (head and embedding share storage), loss is finite, a few steps of training
  on a fixed batch drive the loss down.
- **Run live:** `python train.py --impl student --smoke`
- **Probes:** Why pre-norm over post-norm for deep stacks? Why tie embeddings?
  Why no bias terms? Where do most of the parameters live (and why)?

---

## 🔒 Milestone 3 — Attention internals  (week 1, Days 5–6)

The heart. Build it in graded steps so each is independently testable:
1. scaled dot-product attention (`QKᵀ/√d_k`, softmax, weighted sum)
2. causal mask (no attending to the future)
3. multi-head split / merge
4. grouped-query attention (`n_kv_head < n_head`, broadcast KV)

- **Implement:** `nanollm/student/attention.py`
- **Test:** output shape; attention weights sum to 1; causal mask zeroes the
  upper triangle; GQA broadcasting matches reference; full-model generation
  still matches after swap-in.
- **Probes:** Derive the `√d_k` scaling from the variance argument. MHA vs MQA
  vs GQA memory/quality tradeoff. Compute KV-cache size for a given config.

### 🔒 Milestone 3.5 — RoPE & a KV cache (week 2, Day 8)
Add rotary embeddings (`apply_rope`) and an incremental-decode KV cache, then
make `generate` use it. **Test:** RoPE relative-position invariance
(`⟨q_m,k_n⟩` depends only on `m−n`); cached vs uncached logits match.
- **Probes:** Why does RoPE make attention relative? Why does it play nicely
  with a KV cache? Sketch PI vs NTK-aware vs YaRN for context extension.

---

## 🔒 Milestone 4 — Normalization & activation  (week 1, Days 3–4)

Replace `RMSNorm` and `SwiGLU` with your own.

- **Implement:** `nanollm/student/layers.py`
- **Test:** RMSNorm matches reference and is mean-shift-invariant; SwiGLU shapes
  and the `silu(W₁x) ⊙ (W₂x)` gating; hidden-dim sizing (~⅔·4d).
- **Probes:** Why did RMSNorm replace LayerNorm in LLMs (what does dropping the
  mean cost)? Why SwiGLU over GeLU/ReLU? Why the ⅔ hidden-dim shrink?

---

## 🔒 Milestone 5 — Training internals  (week 1 Day 1 + week 2 Day 10)

Re-derive the loop: cross-entropy from logits, AdamW (with bias correction),
warmup→cosine LR, grad clipping, grad accumulation, weight-decay param groups.

- **Implement:** `nanollm/student/optim.py` + `student/train_step.py`
- **Test:** your cross-entropy matches `F.cross_entropy`; your AdamW step
  matches `torch.optim.AdamW` for a few steps; LR schedule values; decay applied
  only to 2D params.
- **Probes:** Why bias-correct Adam's moments? Why warmup for transformers? Why
  no weight decay on norms/biases? AdamW vs Adam (decoupled decay).

---

## 🔒 Milestone 6 — Tokenizer: BPE from scratch  (week 2, Day 9)

The input boundary. Train a byte-pair-encoding tokenizer; swap it for the char
tokenizer and retrain.

- **Implement:** `nanollm/student/bpe.py` — `train`, `encode`, `decode`
- **Test:** merges learned in frequency order; encode/decode round-trips; vocab
  size respected; a known tiny corpus produces the expected merges.
- **Probes:** BPE vs WordPiece vs Unigram. Why byte-level? Name 3 tokenization
  failure modes (digits, rare tokens, multilingual). Training/encoding cost.

---

## 🔒 Milestone 7 — LoRA fine-tuning  (week 2, Day 12)

Adapt the trained base cheaply. Wrap `nn.Linear` with a low-rank update.

- **Implement:** `nanollm/student/lora.py` — `LoRALinear`
- **Test:** only A, B receive gradients (base frozen); `B=0` init means initial
  output equals the base; `α/r` scaling; shapes.
- **Probes:** Why does low-rank adaptation work (intrinsic rank)? QLoRA's NF4 +
  double quantization + paged optimizers. Param savings math.

---

## 🔒 Milestone 8 — DPO alignment  (week 2, Day 12)

Turn the base into a preference-aligned model without a reward model or RL loop.

- **Implement:** `nanollm/student/dpo.py` — `dpo_loss` (+ a tiny preference set)
- **Test:** loss matches the closed form on fixed inputs; gradient pushes chosen
  logprob up / rejected down; reference model stays frozen.
- **Probes:** Derive DPO from the RLHF objective (where does `Z(x)` cancel?).
  Compare PPO vs DPO vs GRPO. What does β control?

---

## 🔒 Stretch milestones (your week-4 "monthly artifact" menu)

Pick by target company / interest. Each is a self-contained add-on:
- **GRPO toy loop** (week 4, Day 22) — group-relative advantage, no value model.
- **Tiny MoE FFN** (week 2, Day 13) — top-2 routing + load-balancing loss.
- **Speculative decoding** — draft + verify accept/reject loop.
- **FlashAttention-style tiled attention** — block-wise QK + online softmax.
- **INT8 weight quantization** — per-channel, with a quality check.

---

## The active-inference bridge (your differentiator)

Wherever it's honest, the milestones connect back to your FEP/active-inference
work — these are exactly the "taste" bridges week 4 wants you fluent in:
- **M1 / generation:** iterative sample→evaluate→refine ↔ active inference's
  belief-refinement loop and test-time compute (state the parallel, note where
  scaling regimes differ — don't oversell).
- **M5 / cross-entropy:** next-token NLL ↔ prediction-error minimization.
- **M8 / DPO's KL-to-reference:** the `β·KL(π‖π_ref)` term ↔ the KL-to-prior in
  free-energy minimization / the ELBO regularizer.
- **(future VAE add-on):** the FEP ↔ ELBO equivalence (free energy = −ELBO),
  the single most reusable bridge in your project pitch.

Jot these in `NOTES.md` as you hit each milestone; by the end you'll have a
project narrative that ties your research to every part of a modern LLM.

---

## Phase 2 — From model to ML system  (week 3)

After you own the model, we build outward into a served system, same
scaffold + tests style:
1. **Serving** — load the checkpoint behind a tiny generation API; batch
   requests; measure latency (TTFT, tokens/s).
2. **RAG** — chunk a small corpus, embed, retrieve (dense + BM25 hybrid),
   rerank, stuff context, generate with citations.
3. **Agent** — a ReAct tool-use loop with a whitelisted tool surface and
   prompt-injection guards.
4. **Eval harness** — faithfulness, context/answer relevance, retrieval
   recall@k, LLM-as-judge with position-bias controls.

We'll spec these out when you get there.
