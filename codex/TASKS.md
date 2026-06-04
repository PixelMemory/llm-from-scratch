# Task queue — train the LLM on a Colab GPU

> Read `codex/README.md` first (protocol + safety rules). Do `[TODO]` tasks in
> order; update Status; append to the RUN LOG at the bottom.

**Goal:** run `colab/train_a100.ipynb` on a Colab GPU (A100 if available), train
the ~11M-param model on tiny-shakespeare, and save the checkpoint to the user's
Google Drive at `MyDrive/llm-from-scratch/ckpt.pt`.

**Repo:** `https://github.com/PixelMemory/llm-from-scratch` (private)
**Open-in-Colab:** `https://colab.research.google.com/github/PixelMemory/llm-from-scratch/blob/main/colab/train_a100.ipynb`

---

## Task 0 — Prerequisites  ·  Status: `[TODO]`
- Confirm the browser is signed into the user's Google account (Colab loads
  without a login prompt). If not → `[BLOCKED]`, stop.
- Repo is **private**, so confirm a Colab Secret named `GITHUB_TOKEN` exists
  (Colab left sidebar → 🔑 Secrets), with "Notebook access" enabled. If missing,
  the user must create a GitHub token (scope `repo`) and add it — `[BLOCKED]`
  with that note if absent.
  *(If the repo is later made public, this task is N/A — skip it.)*
- **Acceptance:** logged-in Colab session confirmed; `GITHUB_TOKEN` present (or repo public).

## Task 1 — Open the training notebook  ·  Status: `[TODO]`
- Navigate to the **Open-in-Colab** link above. If Colab asks to authorize
  GitHub access for a private repo, approve it with the user's account.
- **Fallback** if the link fails: open `https://colab.research.google.com`,
  create a **New notebook**, and paste the single bootstrap cell from
  *Appendix A* below (it does clone → data → train → save in one cell).
- **Acceptance:** the notebook (or bootstrap cell) is open in Colab.

## Task 2 — Select a GPU runtime  ·  Status: `[TODO]`
- Runtime → Change runtime type → Hardware accelerator = **GPU** (choose
  **A100** if the dropdown offers it; otherwise any GPU is fine). Save.
- **Acceptance:** runtime shows a GPU. (Cell 1 will print the GPU name and will
  `assert` CUDA is available.)

## Task 3 — Run all cells  ·  Status: `[TODO]`
- Runtime → **Run all**.
- When the Drive cell runs, complete the **Google Drive authorization** popup
  using the user's account (no passwords typed — just approve).
- **Acceptance:** every cell runs without error; cell 1 printed a GPU name;
  the train cell is streaming `iter … | train … | val …` lines.

## Task 4 — Wait for training & capture results  ·  Status: `[TODO]`
- Let training finish (look for `saved checkpoint -> .../out/ckpt.pt`).
- Note the **final train and val loss**, and copy the text **sample** printed by
  the last cell.
- **Acceptance:** training completed; final losses and a sample captured.

## Task 5 — Verify checkpoint on Drive  ·  Status: `[TODO]`
- Confirm the save cell printed a Drive listing containing `ckpt.pt` (and
  `train_gpu.log`). Optionally check `MyDrive/llm-from-scratch/` in Drive.
- **Acceptance:** `MyDrive/llm-from-scratch/ckpt.pt` exists.

## Task 6 — Report back  ·  Status: `[TODO]`
- Append a RUN LOG entry: date, GPU used, final train/val loss, the sample, the
  Drive checkpoint path, and anything that went wrong.
- **Acceptance:** RUN LOG updated. Done.

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
