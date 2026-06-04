"""Generates colab/train_a100.ipynb. Run:  python colab/_build_notebook.py

Kept in-repo so the notebook is reproducible and reviewable as plain Python.
"""

import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

REPO = "PixelMemory/llm-from-scratch"
HERE = os.path.dirname(os.path.abspath(__file__))

cells = [
    new_markdown_cell(
        f"# Train the from-scratch LLM on a Colab GPU\n"
        f"Repo: `{REPO}` · saves the checkpoint to your Google Drive.\n\n"
        "**To run:** set the runtime to a GPU (Runtime → Change runtime type → A100, "
        "if your plan offers it), then **Runtime → Run all**. You'll be asked to "
        "authorize Google Drive once.\n\n"
        "If the repo is *private*, add a GitHub token to Colab Secrets named "
        "`GITHUB_TOKEN` (with notebook access) first — the clone cell uses it. "
        "For a public repo, no token is needed."
    ),
    new_code_cell(
        "# 1) Check the GPU we were assigned\n"
        "import torch\n"
        "!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true\n"
        "assert torch.cuda.is_available(), 'No CUDA GPU — set Runtime → Change runtime type → GPU'\n"
        "print('torch', torch.__version__, '| device:', torch.cuda.get_device_name(0))"
    ),
    new_code_cell(
        "# 2) Get the code (works for public OR private repo)\n"
        "import os, subprocess\n"
        f'REPO = "{REPO}"\n'
        "token = None\n"
        "try:\n"
        "    from google.colab import userdata\n"
        '    token = userdata.get("GITHUB_TOKEN")   # only needed if the repo is private\n'
        "except Exception:\n"
        "    token = None\n"
        'url = f"https://{token}@github.com/{REPO}.git" if token else f"https://github.com/{REPO}.git"\n'
        'if not os.path.isdir("/content/llm-from-scratch"):\n'
        '    subprocess.run(["git", "clone", "--depth", "1", url], check=True, cwd="/content")\n'
        'print("cloned:", os.path.isdir("/content/llm-from-scratch"))'
    ),
    new_code_cell(
        "# 3) Fetch a real dataset (tiny-shakespeare ~1MB) — replaces the 4KB sample\n"
        "%cd /content/llm-from-scratch\n"
        "!python colab/get_data.py"
    ),
    new_code_cell(
        "# 4) Mount Google Drive (one OAuth click) for checkpoint persistence\n"
        "from google.colab import drive\n"
        'drive.mount("/content/drive")\n'
        "import os\n"
        'os.makedirs("/content/drive/MyDrive/llm-from-scratch", exist_ok=True)\n'
        'print("drive ready")'
    ),
    new_code_cell(
        "# 5) Train on the GPU (bf16 autocast on CUDA). ~a few minutes on an A100.\n"
        "!python train.py --profile gpu --data data/input.txt --iters 6000 2>&1 | tee out/train_gpu.log"
    ),
    new_code_cell(
        "# 6) Copy the checkpoint + log to Drive so they survive the session\n"
        "import shutil, glob, os\n"
        'DRIVE = "/content/drive/MyDrive/llm-from-scratch"\n'
        'shutil.copy("out/ckpt.pt", os.path.join(DRIVE, "ckpt.pt"))\n'
        'for f in glob.glob("out/*.log"):\n'
        "    shutil.copy(f, DRIVE)\n"
        'print("saved to Drive:", os.listdir(DRIVE))'
    ),
    new_code_cell(
        "# 7) Quick sample from the trained model\n"
        '!python generate.py --impl reference --prompt "ROMEO:" --tokens 400 --temperature 0.8'
    ),
]

nb = new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"name": "train_a100.ipynb", "provenance": []},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}

out = os.path.join(HERE, "train_a100.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out)
