# Qwen3-4B paired GSM8K pilot: sampled v2

## Why v2 (October 2026)

The first 5-question smoke used greedy decoding and saved its outputs under `outputs/pilot-v1/`. That smoke established that GPU inference and Fast/Slow switching work, but it is NOT a reference baseline: Qwen3 advises **against greedy thinking decoding**. The 200-question study must use v2.

For the official Qwen3-4B hybrid model, follow [Qwen3-4B sampling recommendations](https://huggingface.co/Qwen/Qwen3-4B):
- Fast / non-thinking: do_sample=true, temperature=0.7, top_p=0.8, top_k=20, min_p=0.
- Slow / thinking: do_sample=true, temperature=0.6, top_p=0.95, top_k=20, min_p=0.
- Equal `max_new_tokens=4096` safety caps. This is a **pilot design choice**, NOT the manufacturer's full-length recommendation. Track truncation. Actual output length (and time) is the compute cost.
- SHA256-derived independent seed per (base seed, sample ID, mode) makes sampling resumable without depending on generation order. We use one sampled response per mode as an exploratory feasibility study; larger experiments should check variation across seeds.
- Prompt and local model are fixed. The mode comparison combines **thinking toggle plus Qwen's mode-specific sampling defaults**. A later same-sampler ablation can isolate the toggle if needed.
- Fingerprints include config, data hashes, prompt, tokenizer config and generation config. Record checkpoint revision and package versions separately; fingerprint does not hash multi-GB weights.

## Step 1: CPU checks (no GPU)

```bash
cd /data1/lmy/SC4001-efficient-reasoning
git status --short
git pull --ff-only origin main
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/audit_splits.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --dry-run --limit 5
```

If processed splits have never been created, run `.venv/bin/python scripts/prepare_splits.py --seed 42` **once**, then audit. Do not re-split if validated files already exist. The pilot reads exactly the same 200 GSM8K training examples used previously and never accesses official test labels.

`requirements-eval.txt` covers CPU-only utilities. Real GPU inference also requires a compatible CUDA-enabled torch + transformers installation inside the project's `.venv`. Verify packages before running.

## Step 2: sampled v2 smoke before full pilot

Check `nvidia-smi` and select a GPU that is **fully free** and available for your use. The runner intentionally refuses a target GPU with another compute process; never evict another job.

```bash
cd /data1/lmy/SC4001-efficient-reasoning
nvidia-smi
# Replace N with a real free physical GPU index.
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --limit 1
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --limit 5
PYTHONPATH=src .venv/bin/python scripts/evaluate_smoke.py --limit 5
```

Review `outputs/pilot-v2/smoke_summary_n5.json` and raw `outputs/pilot-v2/cases/` for: parsing correctness after `</think>`, invalid answers, truncation, generation token counts, finish reasons and latency. Use these five **only as a pipeline check** (not accuracy evidence).

## Step 3: run the paired 200-question pilot

After sampled v2 smoke passes, keeping the same v2 config and output directory:

```bash
CUDA_VISIBLE_DEVICES=N PYTHONPATH=src .venv/bin/python scripts/run_pilot.py
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --export-only
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairs.py \
  --dataset data/processed/pilot.jsonl \
  --predictions outputs/pilot-v2/predictions.jsonl \
  --output outputs/pilot-v2/summary.json
```

Each `(sample ID, mode)` is saved separately and atomically; rerunning the same command with the same fingerprint skips completed cases. Do not mix `pilot-v1` greedy artifacts with `pilot-v2` sampled results.

Metrics include Fast/Slow accuracy, both / fast-only / slow-only / neither counts, mean generated tokens and latency, invalid-final-answer counts, and truncation counts. Inspect per-case failures. A 200-question, single-sample study is exploratory, not an unbiased estimate of final test performance. The official GSM8K test and SVAMP remain held out.

## Notes for later research

- 4096 is a common maximum for both modes, **not** a matched token-cost comparison. Compare a learned router against a random/uniform routing baseline at matched actual mean cost.
- A fixed sample seed makes results reproducible but does not eliminate sampling variance.
- To study the causal effect of the thinking toggle alone, repeat a subset using the SAME temperature/top-p/top-k in both modes (separate fingerprint and outputs).
- Before LoRA/SFT, keep training questions disjoint from this 200-question development sample.
- The measured `latency_sec` includes generation/prefill but excludes model loading and input preprocessing. Avoid GPU contention when comparing timings.

## Two-GPU data parallel execution (physical GPUs 0 and 3)

Run the **same** model, dataset, seed, and output directory in both workers. `--num-shards 2` partitions by question index modulo 2; each question keeps its Fast and Slow pair on one worker. Per-case seeds are independent of shard order, and completed case checkpoints are reused. The fingerprint must match. A worker does not export incomplete results.

First, confirm both GPUs are free with `nvidia-smi` and that no other compute processes are using them. To inspect planned work without loading the model:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --dry-run --num-shards 2 --shard-id 0
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --dry-run --num-shards 2 --shard-id 1
```

For full pilot, use two shell sessions in the same repository:

```bash
# Terminal A: GPU 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --num-shards 2 --shard-id 0
# Terminal B: GPU 3
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --num-shards 2 --shard-id 1
```

Only **after both workers finish successfully**:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --export-only
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairs.py --dataset data/processed/pilot.jsonl --predictions outputs/pilot-v2/predictions.jsonl --output outputs/pilot-v2/summary.json
```

Do not start an unsharded inference worker at the same time as shard workers, since it may duplicate work. Shared output is safe only when shards partition questions disjointly and configuration is identical. Generation latency per question is not equivalent to end-to-end two-GPU throughput.
