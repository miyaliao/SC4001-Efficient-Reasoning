# Qwen3-4B GSM8K pilot runner

Status: **CPU dry-run verified in isolated synthetic tests; real GPU inference has NOT been executed.**

## Goal and data

Use only `data/processed/pilot.jsonl` (200 GSM8K TRAIN questions) to compare a frozen Qwen3-4B checkpoint in `fast=non-thinking` and `slow=thinking` modes. The script refuses filenames other than `pilot.jsonl` and datasets not containing exactly 200 rows. Official GSM8K test and SVAMP are not used during development.

## CPU only (safe while all GPUs are busy)

```bash
cd /data1/lmy/SC4001-efficient-reasoning
git pull --ff-only origin main
uv pip install --python .venv/bin/python -r requirements-eval.txt
.venv/bin/python scripts/audit_splits.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --dry-run
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --dry-run --limit 1
```

The dry-run does not import torch, load model weights, use a GPU, or write any output. Expected: **200 questions, 400 tasks (two modes), 400 pending** on a fresh output directory.

## GPU smoke when an exclusive free GPU is available

Before starting, inspect `nvidia-smi`; NEVER evict running workloads. The runner refuses a GPU with any existing compute process according to `nvidia-smi`. Keep project dependencies separate from MedDiagnose's environment. `requirements-eval.txt` covers only CPU operation; real inference needs a compatible CUDA-enabled `torch` and `transformers` installation in the project's `.venv`. The model has not yet been loaded with this runner.

```bash
cd /data1/lmy/SC4001-efficient-reasoning
# Replace N with a confirmed free physical GPU index. Do not type N literally.
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --limit 1
# Inspect outputs/pilot-v1/cases/*.json for mode markers and truncation.
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src .venv/bin/python scripts/run_pilot.py
# Only after all 400 tasks finish:
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --export-only
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairs.py --dataset data/processed/pilot.jsonl --predictions outputs/pilot-v1/predictions.jsonl --output outputs/pilot-v1/summary.json
```

## Resume and provenance

- Successful `(sample ID, mode)` cases are saved atomically under `outputs/pilot-v1/cases/` with raw response, count of generated tokens, measured generation time, finish reason, truncation, and run fingerprint.
- Re-run the same command after interruption: already valid checkpoints are skipped. `--limit 1` writes only 2 real outputs and does not export an incomplete pilot as a completed result.
- The fingerprint includes dataset SHA256, local model `config.json` / `tokenizer_config.json` SHA256, seed, prompt, modes, and decoding settings. **It does not hash the full multi-GB model shards**; record the precise upstream ModelScope revision separately before any formal results.
- Changing settings refuses to reuse existing checkpoints; start a new output directory.
- Ground truth is never passed to the generation backend, only the question.
- Output length uses the same `max_new_tokens` cap (currently 2048) and greedy decoding in both modes. Count and analyze truncated generations; this is a pilot setting, not a proof of optimal compute budget.
- Real `finish_reason` behavior and correctness of response parsing for `<think>` blocks need verification on real model generations.
- Generation latency excludes model load and input-token preprocessing. A test on a busy GPU is invalid for accurate efficiency evaluation.
- All outputs stay git-ignored; back up experimental artifacts as needed.
