# SC4001 Efficient Reasoning

NTU SC4001 Neural Networks and Deep Learning — group project, 2026.

**Provisional research question:** Can a learned neural router predict when a small language model benefits from extra thinking, improving accuracy at matched compute budgets?

## Scope

- Frozen, official post-trained Qwen3-4B in thinking and non-thinking modes.
- Feasibility pilot uses 200 examples from the GSM8K training split only.
- Neural routing is conditional on pilot results; **no model inference or training has been run yet**.
- Official GSM8K test (1,319) and SVAMP (1,000) are evaluation-only; no threshold or architecture tuning on either.

## Local artifact paths (not committed)

- Qwen3-4B: `/data1/lmy/models/Qwen3-4B`
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

The paired pilot is planned on **200 GSM8K training questions** (400 fast/slow tasks). No model inference results have been collected yet.

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
  --predictions outputs/pilot_predictions.jsonl \
  --output outputs/pilot_summary.json
```

The evaluation currently requires **complete paired** predictions; it deliberately refuses incomplete or duplicate IDs. Test-time labels must never be available to the inference router.

## AI use

Document model/tool-assisted ideation, coding, debugging, and editing in the final report. Review claims, test all code and report only experiments actually performed.
