# SC4001 Efficient Reasoning

NTU SC4001 Neural Networks and Deep Learning — group project, 2026.

**Current research question:** Can post-training or adaptive compute improve mathematical task success per generated token? SFT vs RL vs router remains undecided.

## Scope

- Frozen, official post-trained Qwen3-4B in thinking and non-thinking modes for the paired pilot; optional SFT + adaptive routing investigated only after the feasibility study.
- Feasibility pilot uses 200 examples from the GSM8K training split only.
- A 5-question greedy functional smoke has been run on the server (results are local, not committed); the 200-question sampled-v2 pilot has since completed (user-reported); SFT/Router experiments remain pending.
- Official GSM8K test (1,319) and SVAMP (1,000) are evaluation-only; no threshold or architecture tuning on either.

## Local artifact paths (not committed)

- Qwen3-4B: `/data1/lmy/models/Qwen3-4B`
- Qwen3-1.7B candidate: `/data1/lmy/models/Qwen3-1.7B` (ModelScope downloaded; config/tokenizer load verified, GPU generation pending)
- GSM8K: `data/raw/gsm8k/{train,test}.parquet`
- SVAMP: `data/raw/svamp/SVAMP.json`

## CPU-only setup

```bash
cd /data1/lmy/SC4001-efficient-reasoning
uv venv --python 3.13 .venv  # if absent
uv pip install --python .venv/bin/python -r requirements-eval.txt
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/prepare_splits.py --seed 42
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Qwen3 pilot runner (GPU not required for dry-run)

The paired v2 pilot is planned on **200 GSM8K training questions** (400 sampled Fast/Slow tasks). The original 5-question greedy smoke is kept separately under `outputs/pilot-v1/`; do not reuse it in v2 results.

```bash
# Only in the project's own environment:
uv pip install --python .venv/bin/python -r requirements-eval.txt
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --dry-run
PYTHONPATH=src .venv/bin/python scripts/run_pilot.py --dry-run --limit 1
```

See [Pilot Runner Runbook](docs/pilot_runner.md) for GPU guardrails, resume behavior, smoke/full inference commands, and evaluation.

## Evaluate paired generations (when available)

Prediction JSONL: one record per example and mode, with `id`, `mode` (`fast`/`slow`), `response`, and optionally `output_tokens`, `latency_sec`, `truncated`.

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairs.py \
  --dataset data/processed/pilot.jsonl \
  --predictions outputs/pilot-v2/predictions.jsonl \
  --output outputs/pilot-v2/summary.json
```

The evaluation currently requires **complete paired** predictions; it deliberately refuses incomplete or duplicate IDs. Test-time labels must never be available to the inference router.


## Mixed GSM-Plus + MATH-500 boxed format evaluation (2026-10)

A frozen 160-question evaluation-only mix (80 GSM-Plus and 80 MATH-500) is prepared locally. A separate runner uses the same Qwen3-4B backend, but **a new prompt** requiring a final boxed mathematical answer. Old GSM8K results under outputs/pilot-v2 are not a matched-prompt baseline for these new results.

**Always run from repo root, in module form:**

    export PYTHONPATH=src:.
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m scripts.evaluate_mixed --gold-audit-only
    .venv/bin/python -m scripts.run_mixed --dry-run

For the 10+10 question smoke, use two eligible, free GPUs in separate terminals:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_mixed --num-shards 2 --shard-id 0
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u -m scripts.run_mixed --num-shards 2 --shard-id 1

Then run:

    .venv/bin/python -m scripts.evaluate_mixed

Read docs/mixed_boxed_pilot.md for the full runbook. Qwen3-4B has completed the 160-question mixed baseline and a selective 16K recovery experiment; results are consolidated in [Experimental Results and Model Selection](docs/experiment_results_and_model_selection.md).

## Canonical boxed scoring (pre-training frozen policy)

The first 20-question mixed smoke kept the boxed-answer prompt but revealed standard Markdown/LaTeX formatting variants. The revised, **separately versioned** scorer accepts final-line Markdown headings/bold and paired math delimiters around an explicit boxed answer, then checks mathematical equivalence with Math-Verify. It still marks missing final box, incomplete Thinking, and wrong math as failure. Exact string-format compliance is separately reported. It preserves the original strict summary and all model generations.

From repository root:

    export PYTHONPATH=src:.
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m scripts.evaluate_mixed_canonical
    .venv/bin/python -m json.tool outputs/mixed-boxed-v1/smoke_canonical_summary.json

Once the smoke rescoring and semantic review pass, use the existing dual GPU --full workflow documented in docs/mixed_boxed_pilot.md; then run:

    .venv/bin/python -m scripts.evaluate_mixed_canonical --full

Do not tune scoring on SFT/RL outputs after training; save protocol, package versions, and data fingerprint.

## Current results and Qwen3-1.7B candidate (2026-10-21)

See **[Experimental Results & Model Selection](docs/experiment_results_and_model_selection.md)** for the single source of truth: GSM8K pilot, 4B mixed baseline, 4096 vs 8192 smoke, 16384 selective recovery, and 1.7B next steps. The external mixed baseline (80 GSM-Plus + 80 MATH-500; 4096 max new tokens; canonical-boxed-v2) recorded Fast **131/160 = 81.875%**, Slow **120/160 = 75%** with **25 Slow truncations**. Selective 16K recovery of those 25 truncations produced **14 correct answers**, at ~7486 generated tokens per selected case; it is *not* an all-160 16K baseline.

**Qwen3-1.7B status:** ModelScope weight files are locally present, AutoConfig and AutoTokenizer load (qwen3, vocabulary 151643); a 20-question dry-run shows 40 pending generations. Full model GPU inference and mathematical results have *not* yet been tested. Use independent configuration and output folder, without modifying the 4B baseline. No SFT/RL/Router training has started.

## AI use

Document model/tool-assisted ideation, coding, debugging, and editing in the final report. Review claims, test all code and report only experiments actually performed.
