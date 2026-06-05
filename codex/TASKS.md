# Task queue — train the LLM on a Colab GPU

> Read `codex/README.md` first (protocol + safety rules). Do `[TODO]` tasks in
> order; update Status; append to the RUN LOG at the bottom.

**Goal:** run `colab/train_a100.ipynb` on a Colab GPU (A100 if available), train
the ~11M-param model on tiny-shakespeare, and save the checkpoint to the user's
Google Drive at `MyDrive/llm-from-scratch/ckpt.pt`.

**Repo:** `https://github.com/PixelMemory/llm-from-scratch` (public)
**Open-in-Colab:** `https://colab.research.google.com/github/PixelMemory/llm-from-scratch/blob/main/colab/train_a100.ipynb`

---

## Task 0 — Prerequisites  ·  Status: `[DONE]`
- Confirm the browser is signed into the user's Google account (Colab loads
  without a login prompt). If not → `[BLOCKED]`, stop.
- Repo is now **public** — **no GitHub token or Colab Secret needed.** Skip any
  token setup; the clone and the Open-in-Colab link work without authorization.
- **Acceptance:** logged-in Colab session confirmed; repo is public (no token).

## Task 1 — Open the training notebook  ·  Status: `[DONE]`
- Navigate to the **Open-in-Colab** link above. If Colab asks to authorize
  GitHub access for a private repo, approve it with the user's account.
- **Fallback** if the link fails: open `https://colab.research.google.com`,
  create a **New notebook**, and paste the single bootstrap cell from
  *Appendix A* below (it does clone → data → train → save in one cell).
- **Acceptance:** the notebook (or bootstrap cell) is open in Colab.

## Task 2 — Select a GPU runtime  ·  Status: `[DONE]`
- Runtime → Change runtime type → Hardware accelerator = **GPU** (choose
  **A100** if the dropdown offers it; otherwise any GPU is fine). Save.
- **Acceptance:** runtime shows a GPU. (Cell 1 will print the GPU name and will
  `assert` CUDA is available.)

## Task 3 — Run all cells  ·  Status: `[DONE]`
- Runtime → **Run all**.
- When the Drive cell runs, complete the **Google Drive authorization** popup
  using the user's account (no passwords typed — just approve).
- **Acceptance:** every cell runs without error; cell 1 printed a GPU name;
  the train cell is streaming `iter … | train … | val …` lines.

## Task 4 — Wait for training & capture results  ·  Status: `[DONE]`
- Let training finish (look for `saved checkpoint -> .../out/ckpt.pt`).
- Note the **final train and val loss**, and copy the text **sample** printed by
  the last cell.
- **Acceptance:** training completed; final losses and a sample captured.

## Task 5 — Verify checkpoint on Drive  ·  Status: `[DONE]`
- Confirm the save cell printed a Drive listing containing `ckpt.pt` (and
  `train_gpu.log`). Optionally check `MyDrive/llm-from-scratch/` in Drive.
- **Acceptance:** `MyDrive/llm-from-scratch/ckpt.pt` exists.

## Task 6 — Report back  ·  Status: `[DONE]`
- Append a RUN LOG entry: date, GPU used, final train/val loss, the sample, the
  Drive checkpoint path, and anything that went wrong.
- **Acceptance:** RUN LOG updated. Done.

## Task 7 — Bring the trained checkpoint to the LOCAL machine  ·  Status: `[TODO]`
Goal: copy the GPU checkpoint from Google Drive into the user's **local** project
so the local web UI (`app.py`) and `generate.py` use the A100-trained model.

- **Local project folder** (note the spaces — quote it in shell):
  `/Users/felixhao/Desktop/Job Hunt Docs/Study Plans/llm-from-scratch`
- **Target file:** `<project>/out/ckpt.pt`  (`out/` is git-ignored, so this is a
  local-only file — do NOT commit it.)

Steps:
1. If `out/ckpt.pt` already exists, back it up first: `cp out/ckpt.pt out/ckpt.laptop.pt`
2. Put `MyDrive/llm-from-scratch/ckpt.pt` at `<project>/out/ckpt.pt`, whichever is available:
   - **Google Drive for Desktop synced:** copy from the local mount, e.g.
     `~/Library/CloudStorage/GoogleDrive-*/My Drive/llm-from-scratch/ckpt.pt`.
   - **Otherwise (browser):** open `https://drive.google.com` → `My Drive/llm-from-scratch`
     → download `ckpt.pt` (lands in `~/Downloads`) → move it into `<project>/out/`.
3. Verify it's the GPU model, not the old laptop one. From `<project>`:
   `python generate.py --impl reference --prompt "ROMEO:" --tokens 300 --temperature 0.8`
   → expect Shakespeare-style text (CAPS speaker names, archaic English), NOT the
   "The student sat down to build a language model…" sample. Optionally confirm config:
   `python -c "import torch;print(torch.load('out/ckpt.pt',map_location='cpu',weights_only=False)['gpt_config'])"`
   → expect `n_layer=6, n_embd=384, block_size=256`.
- **Acceptance:** `<project>/out/ckpt.pt` is the GPU model, loads, and generates
  Shakespeare-style text; the old model is backed up at `out/ckpt.laptop.pt`.
- **Report back:** acquisition method, file size, config, and a sample, in the RUN LOG.

---

## Appendix A — one-cell bootstrap (fallback for Task 1)

Paste into a blank Colab notebook (GPU runtime) and run. Same effect as the
prepared notebook.

```python
import os, subprocess
REPO = "PixelMemory/llm-from-scratch"
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")   # private repo only
except Exception:
    token = None
url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"
if not os.path.isdir("/content/llm-from-scratch"):
    subprocess.run(["git","clone","--depth","1",url], check=True, cwd="/content")
%cd /content/llm-from-scratch
!python colab/get_data.py
from google.colab import drive; drive.mount("/content/drive")
os.makedirs("/content/drive/MyDrive/llm-from-scratch", exist_ok=True)
os.makedirs("out", exist_ok=True)   # so `tee out/...` below has a directory to write to
!python train.py --profile gpu --data data/input.txt --iters 6000 2>&1 | tee out/train_gpu.log
import shutil
shutil.copy("out/ckpt.pt", "/content/drive/MyDrive/llm-from-scratch/ckpt.pt")
shutil.copy("out/train_gpu.log", "/content/drive/MyDrive/llm-from-scratch/train_gpu.log")
!python generate.py --impl reference --prompt "ROMEO:" --tokens 400 --temperature 0.8
```

---

## RUN LOG
<!-- Codex: append entries below. Example:
### 2026-06-05 — run by Codex
- GPU: A100 40GB
- final: train 1.21 | val 1.49
- sample: "ROMEO: ..."
- checkpoint: MyDrive/llm-from-scratch/ckpt.pt (12.3 MB)
- notes: Drive auth needed one click; otherwise clean.
-->

### 2026-06-05 — run by Codex
- Status: blocked at Task 0.
- Colab account: confirmed signed in to the user's Google account; Colab loaded without a login prompt.
- GitHub notebook link: failed with Colab "Notebook not found" / GitHub API 404 because the repo is private and Colab is not authorized for it.
- Colab Secrets: opened a blank notebook and checked the Secrets pane; it reported "No secrets saved", so `GITHUB_TOKEN` is absent.
- Blocker: the private repo requires a Colab Secret named `GITHUB_TOKEN` with notebook access enabled. Per protocol, stopped here rather than guessing past the missing credential.

### 2026-06-05 — blocker resolved (repo made public)
- `PixelMemory/llm-from-scratch` is now **public**, so the Task 0 blocker is cleared: no `GITHUB_TOKEN` and no Colab↔GitHub authorization are required.
- Codex: please resume from **Task 1** — the Open-in-Colab link now opens directly, and the clone cell works token-free.

### 2026-06-05 — Colab GPU training completed by Codex
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB.
- Final: train 0.6525 | val 1.6641 at iter 5999.
- Checkpoint: `/content/llm-from-scratch/out/ckpt.pt`; copied to `MyDrive/llm-from-scratch/ckpt.pt`.
- Drive verification: copy cell printed `saved to Drive: ['ckpt.pt', 'train_gpu.log']`.
- Sample: `ROMEO: All this did swear, and fetch a creature of that hand, My heart he written, undo my brother live, I will appear my apparel than this toft. NORTHUMBERLAND: With all my heart; and many overspeak the day More words than never will use my hand: Upon his place, my life I cannot blow To be my widow-tongue and so long exile. LORD FITZWATER: Shall I be here, for untimely be drunk. LORD FITZWATER: I kn`
- Notes: first visible Colab "Connect GPU" path assigned a T4; user clarified A100 was available, so Codex changed Runtime -> Change runtime type -> A100 GPU and reran. The notebook's original training cell tried to `tee out/train_gpu.log` before `out/` existed; Codex interrupted early, added `!mkdir -p out` before the training command in the Colab cell, and reran training successfully. Google Drive mount required OAuth approval and completed without password/2FA.
