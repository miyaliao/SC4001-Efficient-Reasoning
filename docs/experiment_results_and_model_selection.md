# SC4001 Efficient Reasoning — Results and Model Selection

Updated 2026-10-21. All numerical results are from user-reported runs on ubuntu49, not rerun by this documentation change. Models, datasets and generated predictions stay local.

## Research question

Can SFT, RL/GRPO or adaptive inference improve mathematical task success per generated token in a small post-trained LLM? The training method has **not** been selected. Qwen3-4B was the initial benchmark model; Qwen3-1.7B is under evaluation because 4B Fast accuracy may leave limited headroom for training gains on the selected questions.

Environment: project /data1/lmy/SC4001-efficient-reasoning; Python .venv/bin/python; run from root with PYTHONPATH=src:. and module invocation (python -m scripts.NAME). GPU availability must be checked per run. Original model: /data1/lmy/models/Qwen3-4B. New candidate: /data1/lmy/models/Qwen3-1.7B.

## Dataset and evaluation protocol

- GSM8K pilot: 200 cases from GSM8K **training** data, using earlier final-number format, NOT directly comparable to the boxed mixed-benchmark protocol.
- External mixed set: 80 GSM-Plus questions stratified across perturbations, and 80 MATH-500 questions with 16 at each difficulty level 1–5. Seed 42. Dataset path data/processed/mixed_eval_v1.jsonl; user-reported SHA256 102940cf50a34bbdcd2632b1ca836bc5dc8593879724cfcf900ade2f01a53bfd.
- Mixed Prompt requires final boxed answer; Canonical Boxed Scorer v2 accepts standard LaTeX math wrappers and Markdown final-answer headings (including colonless heading), but only ONE final boxed answer. Math-Verify 0.9.0 grades the isolated expression.
- Primary metric: canonical task success = compliant final answer AND mathematically equivalent to gold; denominator includes truncation and format failures. Secondary: format compliance, truncations, mean tokens/latency, Fast-only/Slow-only paired counts. Review semantic ambiguities.
- Mixed generation seed 42, sampled Qwen3 settings: Fast temperature 0.7, top_p 0.8, top_k 20, min_p 0; Slow temperature 0.6, top_p 0.95, top_k 20, min_p 0; max_new_tokens 4096, unless named sensitivity/recovery experiment.

## 1. GSM8K training pilot — Qwen3-4B, 200 paired cases

Legacy regex scores: Fast 183/200 = 91.5%, Slow 178/200 = 89.0%.
Format-corrected audit: Fast 186/200 = 93.0%, Slow 187/200 = 93.5%, including seven Slow truncations. A Math-Verify audit exposed nine parser disagreements and motivated the revised boxed-answer format. These GSM8K results belong to the *older prompt/scorer*. Local outputs: outputs/pilot-v2/.

## 2. Official 4096-budget mixed baseline — Qwen3-4B

Generation fingerprint: ca21a1fc9b3f9aa5bca84345e8f92288f16c7e5cf5635c227ba2b1d0690ab500. Local summary: outputs/mixed-boxed-v1/full_canonical_v2_summary.json.

| Benchmark | Mode | Task success | Canonical format | Truncations | Mean output tokens | Mean latency |
| --- | --- | --- | --- | --- | --- | --- |
| GSM-Plus (80) | Fast | 63/80 = 78.75% | 97.5% | 0 | 197.4 | 7.3 s |
| GSM-Plus (80) | Slow | 62/80 = 77.5% | 85.0% | 10 | 1539.6 | 56.9 s |
| MATH-500 (80) | Fast | 68/80 = 85.0% | 93.8% | 0 | 444.9 | 16.4 s |
| MATH-500 (80) | Slow | 58/80 = 72.5% | 72.5% | 15 | 2071.2 | 76.5 s |
| Overall (160) | Fast | 131/160 = 81.875% | 95.6% | 0 | 321.2 | 11.9 s |
| Overall (160) | Slow | 120/160 = 75.0% | 78.8% | 25 | 1805.4 | 66.7 s |

Paired: Both 113, Fast-only 18, Slow-only 7, Neither 22. Oracle best-of-two = 138/160 = 86.25%, just +4.375 percentage points over always Fast. This is **only** an oracle bound for the two fixed 4K responses, not a limit on alternative models, budgets or retraining.

Interpretation: Fast achieves higher *end-to-end* success at 4096 with far fewer output tokens. Slow is limited in part by failing to finish within budget (25/160 truncated); do not conclude that unrestricted thinking must be worse.

Math semantic review flags: percentage points (GSM-Plus 003667, MATH-500 0067), interval vs coordinate (MATH-500 0314, 0312), digit-grouping LaTeX (MATH-500 0217). Preserve audit decisions rather than changing scorer based on model performance.

## 3. Budget sensitivity on 20 smoke questions (10 + 10)

| Run | Task success | Format | Truncated | Mean tokens | Mean latency |
| --- | --- | --- | --- | --- | --- |
| Fast 4096 | 16/20=80% | 100% | 0 | 294.9 | 11.1 s |
| Slow 4096 | 14/20=70% | 80% | 4 | 2131.5 | 79.6 s |
| Fast 8192 | 16/20=80% | 100% | 0 | 294.9 | 12.1 s |
| Slow 8192 | 14/20=70% | 80% | 3 | 2796.9 | 123.6 s |

The 20 cases were used to develop protocol and choose budgets; do not call them independent test samples. One Slow response stopped at 5117 tokens but gave a wrong final answer (gold 18, prediction 12); increasing budget does not necessarily fix reasoning.

## 4. Selective 16K Slow truncation recovery — Qwen3-4B

Only the 25 cases that truncated in the full Slow 4096 baseline were rerun with max_new_tokens=16384; not an exhaustive 160-question 16K evaluation. Independent fingerprint 76bf120cd4c04854b1184e5b179d82abd43842505d09243a72f5f6c499a94fd1. Report: outputs/recovery-slow-16384/recovery_summary.json.

| Original truncation cohort | n | Correct at 16K | Still truncated | Canonical format | Avg tokens | Avg latency |
| --- | --- | --- | --- | --- | --- | --- |
| GSM-Plus | 10 | 3 | 0 | 9 | 7204.5 | 266.7 s |
| MATH-500 | 15 | 11 | 2 | 11 | 7672.9 | 288.7 s |
| Total | 25 | **14/25=56%** | 2 | 20 | **7485.5** | **279.9 s** |

11 unrecovered cases: six mathematically wrong despite parseable final answer; three ended without compliant final answer; two still truncated. Hypothetically replacing the original 25 truncated failures with these recoveries produces Slow 134/160=83.75%; this is a **selective replacement counterfactual**, NOT full 16K accuracy nor a free compute improvement. Real retry cost must include the 4096-token unsuccessful attempt.

## 5. Model choice under consideration — Qwen3-1.7B

Qwen3-4B Fast is already successful on 78.75% GSM-Plus and 85% MATH-500; demonstrating significant mathematical accuracy improvement from SFT/RL on 80-case subsets may be difficult. Qwen3-1.7B might offer more learning headroom with lower compute cost, but this must be measured.

User downloaded Qwen/Qwen3-1.7B through ModelScope to /data1/lmy/models/Qwen3-1.7B. Reported files include two safetensors shards (~3.3GB and 594MB), a safetensors index, config, generation config and tokenizer assets. Local checks passed AutoConfig/AutoTokenizer: model_type=qwen3, vocab_size=151643. **Weight integrity and CUDA inference have not been verified.**

Local candidate config configs/mixed_1p7b.yaml was created from configs/mixed.yaml replacing model_path only; this config's presence in the user's working copy is inferred from successful dry-run, but is not yet committed. New independent result directory: outputs/mixed-boxed-1p7b. Reported 20-question dry run: selected_questions_total=20, pending_tasks=40, fingerprint=a9cdf2a754f8cb3c877310c4f634d8d59eb57514461d28a36e86a7b53d73f6b7; no GPU/model used during dry-run. **No actual Qwen3-1.7B results yet.**

Suggested next commands, all from repository root:

    export PYTHONPATH=src:.
    nvidia-smi
    grep -E 'model_path|max_new_tokens' configs/mixed_1p7b.yaml
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m scripts.run_mixed --config configs/mixed_1p7b.yaml --output-dir outputs/mixed-boxed-1p7b --dry-run

On a free, exclusive GPU (substitute its index):

    PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u -m scripts.run_mixed --config configs/mixed_1p7b.yaml --output-dir outputs/mixed-boxed-1p7b

After the 40 predictions are complete:

    PYTHONPATH=src:. .venv/bin/python -m scripts.evaluate_mixed_canonical --config configs/mixed_1p7b.yaml --output-dir outputs/mixed-boxed-1p7b

Examine task success, format, truncation and model behavior on exactly the same 20 sampled mixed questions as 4B. The smoke is an environment/protocol check, NOT sufficient alone to rank model capability. For a matched 160-case comparison, run both modes on the full mixed dataset, same sampling settings, output cap, prompt and scorer, with independent directories and manifests.

## 6. Future method validation (not yet chosen)

SFT: Baseline vs short-CoT SFT, optionally full-CoT comparator; evaluate task success, output length, truncation under matched 4K budget.
RL/GRPO: baseline vs correctness+format reward vs correctness+format+length; monitor reward variance and math-performance degradation.
Router: compare always Fast/Slow, random, heuristic, learned routing, and oracle theoretical ceiling; never use test labels as features.

The mixed 160-case sample has informed project/model selection: **exploratory**, not pristine final test. Keep a separate de-duplicated held-out final evaluation set. Do not train on GSM-Plus or MATH-500 evaluation questions.
