# Boxed-Answer Mixed Benchmark v1

## CURRENT OFFICIAL RUN: 4096 tokens, canonical-boxed-v2

This section supersedes the earlier smoke-only instructions. The model
prompt and generation protocol are unchanged. Only the offline parser now
accepts a colonless final-answer heading such as a Markdown header on a line
of its own, followed by a display-math boxed answer. Standard Markdown/LaTeX
wrappers are permitted. A single balanced final box is still mandatory.

Gold 18 vs generated boxed answer 12 for GSM-Plus 008566 remains mathematically
WRONG, even though v2 now recognizes its answer structure.

Generation: existing configs/mixed.yaml specifies max_new_tokens=4096.
Use the original outputs/mixed-boxed-v1 directory and its validated manifest.
Saved 4096 checkpoints remain usable because prompt, dataset, seed, sampling,
model and budget are unchanged. Never merge independent 8192 outputs.

Scorer: canonical-boxed-v2. It writes smoke_canonical_v2_summary.json and
full_canonical_v2_summary.json instead of overwriting prior v1 reports.
Exact-format compliance is preserved as a secondary, strict final-line metric.

### CPU verification before starting GPUs

Run from the repository root:

    cd /data1/lmy/SC4001-efficient-reasoning
    git pull --ff-only origin main
    export PYTHONPATH=src:.
    grep max_new_tokens configs/mixed.yaml
    sha256sum data/processed/mixed_eval_v1.jsonl
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m scripts.evaluate_mixed_canonical --gold-audit-only
    .venv/bin/python -m scripts.evaluate_mixed_canonical
    .venv/bin/python -m scripts.run_mixed --full --dry-run

Expect 160 selected questions and 320 tasks total; with the previous complete
4096 smoke, 40 tasks are complete and 280 pending. If manifest fingerprint
mismatches or the gold audit fails, STOP; do not bypass a safety check.

### Official GPU run

GPU 3 was reported occupied. Only run on a free GPU 0. The unsharded
full runner resumes automatically, skips existing 4096 checkpoints, and
writes per-task files atomically:

    nvidia-smi -i 0
    PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_mixed --full

Do not run another writer concurrently or use GPU 3 while occupied.

### CPU evaluation once GPU generation has finished

    .venv/bin/python -m scripts.run_mixed --full --dry-run
    .venv/bin/python -m scripts.run_mixed --full --export-only
    .venv/bin/python -m scripts.evaluate_mixed_canonical --full

Final v2 report: outputs/mixed-boxed-v1/full_canonical_v2_summary.json.
Inspect each benchmark separately and all per-case review_flags. Report task
success, format, truncation, output tokens and latency. Truncated cases without
valid final answers stay in the denominator and count as task failures.
The 20 smoke questions informed the budget choice: report the remaining
140 exploratory questions independently as well.

No server GPU job was initiated by this documentation change.


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


## First 20-question smoke: findings and offline diagnostic

The user-reported first run scored Fast 8/20 (40%) and Slow 10/20 (50%); format compliance was only Fast 11/20 and Slow 12/20, with 4 Slow truncations. Breakdown: GSM-Plus 10 (Fast 5/10, Slow 5/10) and MATH-500 10 (Fast 3/10, Slow 5/10). These are **strict-format scorer** results; they are not verified mathematical error rates, and n=10 per benchmark is not statistically conclusive.

**Do not launch --full yet.** Before changing the prompt or the format acceptance rule, inspect the actual raw generations. Math-Verify also scored a case with gold 36 and parsed output 36*(1/100) as correct, and interpreted gold '(2,4)' as Interval.open(2,4). Both warrant question-aware manual review (percentage semantics and ordered-pair vs interval interpretation).

Commands from the repo root:

    git pull --ff-only origin main
    export PYTHONPATH=src:.
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m scripts.audit_mixed_smoke
    .venv/bin/python -m json.tool outputs/mixed-boxed-v1/smoke_format_audit.json

The audit saves post-thinking response tails for noncompliant predictions and flags ambiguous Math-Verify interpretations without modifying the original results. Check whether models emitted bare boxes, normal prose, math-delimited boxes, incomplete thinking, or repeated final labels.

If the prompt is revised after reviewing the audit, bump the protocol version and use a **new output directory** so it cannot reuse v1 responses with a different prompt. Repeat the 20-question smoke and verify format success before starting the 160-question evaluation.


## Frozen scoring decision: canonical boxed v1 (2026-10-20)

The generation prompt and checkpoint are unchanged from mixed-boxed-v1.
Do NOT rerun the 20 smoke generations just because the scorer is revised.

PRIMARY: canonical task success. Require a completed post-thinking final answer, exactly one final-answer label, exactly one balanced boxed expression and no substantive trailing text. Accept ordinary Markdown emphasis/heading and conventional inline/display LaTeX wrappers around the box. Then Math-Verify decides mathematical equivalence to gold. Unfinished thinking, missing/ambiguous final answer, missing box and wrong math all count as task failures.

SECONDARY: exact-format compliance, exact task success, canonical format compliance, Math-Verify parse success, truncation, output tokens and latency. Standard LaTeX formatting is not counted as substantive instruction failure, but exact-format matching is still separately measured for research.

Percentage questions: A bare numeric percentage output such as 36 percent can be compared with numeric gold 36 percentage points ONLY if the question explicitly asks for a percentage, recording an interpretation flag. Do not drop percentages in count questions. Manually review flagged cases. Structured MATH-500 answers (coordinates interpreted as intervals) and large integers with LaTeX digit grouping also require independent review.

The new CPU-only scorer writes new files, preserving the old strict summary and all saved model generations:

    cd /data1/lmy/SC4001-efficient-reasoning
    git pull --ff-only origin main
    export PYTHONPATH=src:.
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m scripts.evaluate_mixed_canonical
    .venv/bin/python -m json.tool outputs/mixed-boxed-v1/smoke_canonical_summary.json

Check per_case and review_flags before running the 160-question baseline. Use the same prompt, data, sampled decoding and seed as smoke. GPU commands remain unchanged in Step 3 above. Once both GPU workers finish:

    .venv/bin/python -m scripts.run_mixed --full --export-only
    .venv/bin/python -m scripts.evaluate_mixed_canonical --full

The output is full_canonical_summary.json, grouped by benchmark and stratum. Report all metrics without pooling GSM-Plus and MATH-500 into one headline number. Do not train on these benchmark samples. If evaluation results guide method selection, document the set as exploratory rather than an untouched final test.
