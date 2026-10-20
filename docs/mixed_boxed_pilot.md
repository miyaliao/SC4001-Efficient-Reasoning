# Boxed-Answer Mixed Benchmark v1

## Fixed protocol

The independent evaluation dataset has 80 GSM-Plus and 80 MATH-500 questions, prepared with seed 42 and SHA256 102940cf50a34bbdcd2632b1ca836bc5dc8593879724cfcf900ade2f01a53bfd (user-reported 2026-10-20). GSM-Plus is stratified across seven perturbation categories; MATH-500 by levels 1-5. The reference prompt and model settings are frozen by a run fingerprint. Model responses and local datasets are gitignored.

Old GSM8K outputs/pilot-v2 has a DIFFERENT prompt and scorer; do not mix outputs or claim matched-baseline comparability. This protocol is for evaluation only; do not use these benchmark examples for SFT/router training. The 0 flagged visual dependencies result is a heuristic check, not proof that every question is diagram-independent.

Default is 10 GSM-Plus + 10 MATH-500 questions (20 x 2 modes). Only --full can run all 160 (320 generations).

Scoring contract: one final line exactly Final answer: \boxed{EXPRESSION}. Ignore reasoning inside think. Count format compliance, parsing, correctness, and token truncation separately. Do not interpret an arbitrary intermediate result as the answer. Source: https://github.com/huggingface/Math-Verify

## Step 1: CPU-only test and 160-gold audit

Run from repo root using module invocation, not direct scripts.

    cd /data1/lmy/SC4001-efficient-reasoning
    git pull --ff-only origin main
    export PYTHONPATH=src:.
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m scripts.evaluate_mixed --gold-audit-only
    .venv/bin/python -m scripts.run_mixed --dry-run
    .venv/bin/python -m scripts.run_mixed --dry-run --num-shards 2 --shard-id 0
    .venv/bin/python -m scripts.run_mixed --dry-run --num-shards 2 --shard-id 1

Stop if a gold answer cannot be parsed by Math-Verify; inspect its official answer and the parser configuration BEFORE spending GPU. Do not drop it after seeing accuracy.

## Step 2: 20-question smoke, GPU 0 and 3

Check GPUs are free and approved. Use two SSH/tmux windows:

    nvidia-smi

    # Terminal A, GPU 0
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_mixed --num-shards 2 --shard-id 0

    # Terminal B, GPU 3
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u -m scripts.run_mixed --num-shards 2 --shard-id 1

When BOTH complete, evaluate offline:

    .venv/bin/python -m scripts.evaluate_mixed
    .venv/bin/python -m json.tool outputs/mixed-boxed-v1/smoke_summary.json

This report includes 20 per-case records with Fast/Slow, format compliance, parse success, correctness, truncation, mean output tokens, and latency, grouped by benchmark and difficulty/perturbation type. Examine all noncompliant or suspicious records manually. A 20-question smoke only validates the pipeline, not model superiority.

## Step 3: Full evaluation, after validation only

    # Terminal A
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_mixed --full --num-shards 2 --shard-id 0

    # Terminal B
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u -m scripts.run_mixed --full --num-shards 2 --shard-id 1

    # After both workers finish:
    .venv/bin/python -m scripts.run_mixed --full --export-only
    .venv/bin/python -m scripts.evaluate_mixed --full

The results are stored separately under outputs/mixed-boxed-v1; resumable atomic JSON case outputs share an immutable fingerprint. Each question is assigned by its index modulo shard count. Never run an unsharded writer while sharded workers are running. Export refuses incomplete cases.

## Analysis guardrails

- Report GSM-Plus and MATH-500 independently, plus perturbation categories and MATH levels.
- Inspect fraction, coordinate, set, and symbolic LaTeX output for extraction and equivalence mistakes. Failing format is distinct from failing mathematics.
- This mix is an exploratory evaluation set, not a training or hyperparameter tuning pool. Do not claim it is an untouched test set if it informs later design.
- Record exact local package versions, seed, dataset SHA256, model revision, prompt and token cap. Parallel execution changes wall throughput, not single-query cost.
