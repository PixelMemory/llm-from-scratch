"""Tiny local web UI for the from-scratch LLM project.

Run:
    python app.py
    # then open http://127.0.0.1:5050

Three panels:
  - Generate : prompt + sampling controls + reference/student toggle.
               Your nanollm/student/sampling.py is hot-reloaded on every call,
               so edit -> click Generate -> see your code run (no restart).
  - Tests    : run the current milestone's pytest suite, see pass/fail.
  - Train    : start training with a live streaming log.

This is a convenience wrapper around train.py / generate.py / pytest — it does
not change any of the learning code.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402

from nanollm.util import get_device, load_checkpoint  # noqa: E402
import nanollm.reference.sampling as ref_sampling  # noqa: E402

app = Flask(__name__)
CKPT = os.path.join(HERE, "out", "ckpt.pt")
PORT = int(os.environ.get("PORT", "5050"))

# Serialize all torch access (the Flask dev server is multi-threaded).
_GEN_LOCK = threading.Lock()
_model_cache = {"mtime": None, "model": None, "tok": None, "cfg": None, "device": None}

# Background training subprocess state.
_train = {"proc": None, "log": [], "lock": threading.Lock()}


def _load_model():
    """Load (and cache) the checkpoint, reloading if the file changed on disk."""
    if not os.path.exists(CKPT):
        return None
    mtime = os.path.getmtime(CKPT)
    if _model_cache["model"] is None or _model_cache["mtime"] != mtime:
        device = get_device()
        model, tok, cfg = load_checkpoint(CKPT, device=device)
        _model_cache.update(mtime=mtime, model=model, tok=tok, cfg=cfg, device=device)
    return _model_cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    with open(os.path.join(HERE, "frontend.html"), encoding="utf-8") as f:
        return f.read()


@app.route("/api/status")
def status():
    running = _train["proc"] is not None and _train["proc"].poll() is None
    return jsonify(
        checkpoint=os.path.exists(CKPT),
        device=get_device(),
        training=running,
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    d = request.get_json(force=True) or {}
    impl = d.get("impl", "reference")
    prompt = d.get("prompt", "\n")
    tokens = max(1, min(int(d.get("tokens", 200)), 1000))
    greedy = bool(d.get("greedy", False))
    temperature = float(d.get("temperature", 0.8))
    temperature = max(0.05, temperature)
    top_k = d.get("top_k")
    top_p = d.get("top_p")
    top_k = int(top_k) if top_k not in (None, "", 0, "0") else None
    top_p = float(top_p) if top_p not in (None, "") else None
    if top_p is not None and not (0.0 < top_p < 1.0):
        top_p = None

    with _GEN_LOCK:
        cache = _load_model()
        if cache is None:
            return jsonify(ok=False, error="No checkpoint yet — train a model first (Train panel).")

        # pick implementation; hot-reload student so edits take effect live
        if impl == "student":
            try:
                import nanollm.student.sampling as stu
                importlib.reload(stu)
                generate = stu.generate
            except Exception as e:  # syntax error etc. while editing
                return jsonify(ok=False, error=f"Couldn't load your student/sampling.py — {type(e).__name__}: {e}")
        else:
            generate = ref_sampling.generate

        tok = cache["tok"]
        # the char tokenizer only knows characters from data/input.txt
        known = [ch for ch in prompt if ch in tok.stoi]
        dropped = sorted(set(prompt) - set(tok.stoi))
        if not known:
            known = ["\n"] if "\n" in tok.stoi else [tok.chars[0]]
        warning = None
        if dropped:
            shown = "".join(dropped)[:20]
            warning = f"Dropped {len(dropped)} char(s) not in this tiny vocab: {shown!r}"

        try:
            ids = torch.tensor([tok.encode("".join(known))], dtype=torch.long, device=cache["device"])
            out = generate(cache["model"], ids, tokens, temperature=temperature,
                           top_k=top_k, top_p=top_p, greedy=greedy)
            text = tok.decode(out[0].tolist())
            return jsonify(ok=True, text=text, warning=warning, impl=impl)
        except NotImplementedError:
            return jsonify(ok=False, error="Milestone 1 isn't implemented yet. Fill in nanollm/student/sampling.py, then try again.")
        except Exception as e:
            return jsonify(ok=False, error=f"{type(e).__name__}: {e}")


@app.route("/api/test", methods=["POST"])
def api_test():
    d = request.get_json(force=True) or {}
    target = d.get("target", "tests/test_sampling.py")
    if not re.fullmatch(r"tests/[\w/]+\.py", target):  # don't run arbitrary paths
        target = "tests/test_sampling.py"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", target, "-q", "--no-header"],
            cwd=HERE, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return jsonify(ok=False, error="pytest timed out")
    out = proc.stdout + proc.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = sum(int(x) for x in re.findall(r"(\d+) (?:failed|error)", out))
    return jsonify(ok=True, passed=passed, failed=failed, returncode=proc.returncode, output=out)


def _reader(proc):
    for line in proc.stdout:
        with _train["lock"]:
            _train["log"].append(line.rstrip())


@app.route("/api/train/start", methods=["POST"])
def train_start():
    if _train["proc"] is not None and _train["proc"].poll() is None:
        return jsonify(ok=False, error="Training is already running.")
    d = request.get_json(force=True) or {}
    iters = max(50, min(int(d.get("iters", 2500)), 50000))
    with _train["lock"]:
        _train["log"] = []
    proc = subprocess.Popen(
        [sys.executable, "train.py", "--iters", str(iters)],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    _train["proc"] = proc
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    return jsonify(ok=True, iters=iters)


@app.route("/api/train/status")
def train_status():
    proc = _train["proc"]
    running = proc is not None and proc.poll() is None
    with _train["lock"]:
        log = "\n".join(_train["log"][-300:])
    rc = None if (proc is None or running) else proc.returncode
    return jsonify(running=running, log=log, returncode=rc)


if __name__ == "__main__":
    print(f"\n  LLM-from-scratch UI  ->  http://127.0.0.1:{PORT}\n")
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False, threaded=True)
