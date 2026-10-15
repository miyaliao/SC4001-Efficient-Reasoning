# Pilot readiness checklist

Status: **GPU inference has NOT been run.** This is a protocol draft.

## Verified CPU prerequisites (reported 2026-10-15)
- Qwen3-4B local basic shard check passed; actual model load pending.
- GSM8K 7,473 train and 1,319 test rows present; SVAMP 1,000 rows present.
- Deterministic disjoint splits: pilot 200, router_train 5,000, router_val 1,000, internal_holdout 1,273.
- 7 evaluation unit tests and the Qwen3 thinking/non-thinking tokenizer template smoke passed.
- All 8 server GPUs were occupied; do not evict or interrupt existing tasks.

## Run these checks with no GPU
```bash
cd /data1/lmy/SC4001-efficient-reasoning
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/prepare_splits.py --seed 42
.venv/bin/python scripts/audit_splits.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/smoke_evaluation_cpu.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/smoke_tokenizer.py
```

The synthetic evaluation smoke output is a software test, not research evidence.

## Freeze before real pilot inference
1. Ensure permission and actual free GPU memory; do not kill or suspend existing jobs.
2. Use the official post-trained Qwen3-4B with one frozen model revision.
3. Record python, torch, transformers, tokenizer, model revision, CUDA and device.
4. Validate one real question in each thinking mode first.
5. Fix a shared answer instruction and comparable decoding protocol before the 200 paired questions.
6. Persist per-question, per-mode raw output, finish reason, generated tokens, elapsed time, errors and settings; enable resume.
7. Audit invalid numeric answers and truncation; do not confuse cap-induced truncation with reasoning failure.
8. Never tune on GSM8K official test or SVAMP.

## Primary analysis
- Accuracy, invalid parsing rate and truncation by mode.
- Four paired groups: both correct, fast-only, slow-only and neither correct.
- Oracle paired accuracy is an upper bound computed with gold labels, not a real deployable router.
- Generated tokens and wall-time; router cost must later be included.
- Router must beat random and lexical heuristics at **matched mean compute budgets**, not simply at arbitrary thresholds.
- Do not assume a slow-only rate automatically implies a predictable routing signal.

## Go / No-Go
Run only the GSM8K-train pilot first. If most differences come from truncation or there is little incremental benefit from slow reasoning, reconsider model, task difficulty or project scope before training a router.

## Research integrity
Label simulated samples and examples explicitly; never present them as real model results. Preserve an AI-use statement in the final report.