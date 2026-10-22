# Qwen3-1.7B mixed smoke — frozen error audit (2026-10-22)

Status: **frozen observational diagnostic**, before extending to the full 160 questions. This does NOT alter any model output, scoring contract, prompt, or reference gold labels.

## Frozen experiment
- Model: Qwen3-1.7B (local path `/data1/lmy/models/Qwen3-1.7B`)
- Dataset: `data/processed/mixed_eval_v1.jsonl`, 10 GSM-Plus + 10 MATH-500 questions for smoke
- Dataset SHA256 (user-verified earlier): `102940cf50a34bbdcd2632b1ca836bc5dc8593879724cfcf900ade2f01a53bfd`
- Config: `configs/mixed_1p7b.yaml`; seed 42; `max_new_tokens: 4096`; both fast and slow.
- Primary scorer: `canonical-boxed-v2`, Math-Verify. Official task-success denominator includes truncated and format-invalid outputs.

## Observed results (20 questions per mode)

| Metric | Fast | Slow |
| --- | ---: | ---: |
| Canonical task success | 9/20 | 12/20 |
| Canonical format | 14/20 | 15/20 |
| Truncated | 2/20 | 3/20 |
| Mean output tokens | 702.75 | 1885.35 |
| Mean latency (sec) | 20.344 | 54.702 |

Paired outcomes: both=7, fast-only=2, slow-only=5, neither=6. Observed oracle upper bound =14/20 (70%); not achievable without an oracle.

## Mutually exclusive failure/success symptoms (scorer output)

| Category | Fast | Slow |
| --- | ---: | ---: |
| success | 9 | 12 |
| truncated | 2 | 3 |
| non-truncated format failure | 4 | 2 |
| valid format, mathematical failure or equivalence dispute | 5 | 3 |
| math parse failure | 0 | 0 |

A failure category is an **observed symptom, not established cognitive cause**. Truncation overrides format in the mutually exclusive accounting; it may coincide with an invalid final response.

## Human-inspection notes (diagnostic only)
- `gsmplus-006713`: question states 30 cows × 50 liters per cow vs target 250 liters; gold=20, while Slow's 0 is consistent with the printed question. Flag **question/gold dispute**; retain official scorer result unchanged.
- `math500-0082`: Slow generation explicitly explores the correct slope 3/2 but hits token budget before a valid final answer. Flag **correct intermediate / no valid final**, but official truncated failure remains.
- Other examples of formatting, repetition, and mathematics concerns were identified in exploratory review. These labels are *not* independently verified ground-truth error annotations and must not be treated as a certified count of training-correctable failures.

## Frozen policy for the full evaluation
1. Do **not** edit `configs/mixed_1p7b.yaml`, `src/sc4001/mixed_protocol.py`, or the canonical scorer during this run.
2. Preserve the existing 20-question case checkpoints; the full run is resumable under the same output directory and manifest fingerprint.
3. Use two disjoint question shards, `--num-shards 2 --shard-id 0/1 --full`; verify two different idle physical GPUs and run no other writer in parallel.
4. Run `--full --dry-run` to establish task counts; do not manually overwrite/merge checkpoints.
5. After BOTH shards finish, run `--full --export-only`, then the canonical-v2 `--full` scorer.
6. Treat this 160-question set as **exploratory**; preserve an untouched independent final test for post-training claims. Report benchmark-specific performance, paired outcomes, formatting, truncation, tokens, latency, and human-audit flags.

Frozen source: user-exported `failure_audit/summary.json`, `cases.jsonl`, and `review.txt` from completed 20-question smoke (local artifacts, not committed).
