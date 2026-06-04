# Codex cooperation protocol

This folder is a **handoff between two agents**:

- **Claude (planner)** prepares the project and writes a task queue to `TASKS.md`.
- **Codex (executor)** — an agent with a **browser / computer-use** capability —
  picks up `TASKS.md` and runs the tasks (e.g. driving Google Colab to train on
  a GPU), then writes results back.

## How to pick up work (Codex)

1. Read `TASKS.md` top to bottom.
2. Do the tasks whose **Status is `[TODO]`**, in order. Set a task to `[DOING]`
   when you start and `[DONE]` when its acceptance criteria are met.
3. Append a dated entry to the **RUN LOG** at the bottom of `TASKS.md` with what
   happened (GPU used, final losses, a sample, the checkpoint path, any errors).
4. If you cannot proceed, set the task to `[BLOCKED]`, write why in the RUN LOG,
   and stop — do not guess past a blocker.

## Status legend

`[TODO]` not started · `[DOING]` in progress · `[DONE]` done & verified · `[BLOCKED]` needs a human

## Safety rules (important)

- **Never type the user's password or 2FA codes.** Use a browser that is
  *already signed in* to their Google account. If a sign-in or 2FA screen
  appears, set the task `[BLOCKED]` and stop.
- **Never commit secrets.** The GitHub token (only needed if the repo is
  private) lives in **Colab Secrets** as `GITHUB_TOKEN`. Do not paste it into a
  cell literally, a file, or the RUN LOG.
- **Don't take destructive actions** (deleting Drive files, changing account
  settings, installing browser extensions). The only writes expected are: a
  Colab notebook run, and the checkpoint saved to `MyDrive/llm-from-scratch/`.
- Prefer the lowest-effort reliable path. If "Run all" on the prepared notebook
  works, you're done — no need to improvise.

## Environment assumptions

- A browser logged into the user's Google account (Colab access).
- Colab **Pro / Pro+** if an A100 is required (the notebook adapts to whatever
  GPU is assigned, so any GPU runtime works).
- Repo: `PixelMemory/llm-from-scratch`. If it is **private**, a `GITHUB_TOKEN`
  Colab Secret must exist (see Task 0). If public, no token is needed.

## Handback to Claude / the user

When training is done, the checkpoint is at
`Google Drive → MyDrive/llm-from-scratch/ckpt.pt` plus `train_gpu.log`.
Record the final train/val loss and a text sample in the RUN LOG. The user (or
Claude) pulls the checkpoint into `llm-from-scratch/out/` to use locally.
