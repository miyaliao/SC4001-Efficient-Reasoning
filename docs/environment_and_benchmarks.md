# Environment & external evaluation protocol

Updated: 2026-10-20. This is a working inventory; **not** a claim that current GPU packages, downloads, or new evaluator runs have been verified.

## Known environment

- SSH user/host used in prior commands: lmy@ubuntu49.
- Project: /data1/lmy/SC4001-efficient-reasoning.
- Python: repo-local .venv/bin/python; Python 3.13 was used in previous setup, but exact installed version needs rechecking.
- User reports physical GPUs 0 and 3 available. Previous context indicated RTX 4090 24GB, but check hardware and occupancy with nvidia-smi.
- Model: /data1/lmy/models/Qwen3-4B (not committed).
- Data: data/raw/gsm8k/train.parquet, test.parquet; data/raw/svamp/SVAMP.json (gitignored).
- Fixed splits: pilot=200; router_train=5000; router_val=1000; internal_holdout=1273; seed=42.
- Paired v2 output: outputs/pilot-v2/cases with 400 Fast/Slow samples from 200 GSM8K train questions.
- Previous user-provided corrected result: Fast 93.0%, Slow 93.5%, 7 Slow truncated. Do not treat these as newly executed metrics.
- Actual package versions, installed driver/CUDA, and availability of external datasets are unverified here. Do not assume direct connectivity to Hugging Face.

Record versions locally (avoid committing private logs):

    cd /data1/lmy/SC4001-efficient-reasoning
    mkdir -p logs
    git rev-parse HEAD
    .venv/bin/python --version
    .venv/bin/python -m pip freeze > logs/environment-pip-freeze.txt
    nvidia-smi
    .venv/bin/python -c "import torch,transformers;print(torch.__version__,torch.version.cuda,transformers.__version__)"

## Standardized offline Math-Verify audit

Official tool: https://github.com/huggingface/Math-Verify

Install only in the project virtualenv:

    uv pip install --python .venv/bin/python 'math-verify[antlr4_13_2]'
    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
    PYTHONPATH=src .venv/bin/python scripts/evaluate_math_verify.py

The script reads existing outputs and writes outputs/pilot-v2/math_verify_audit.json. It does not modify the original summary or require a GPU. Compare discrepancies manually. Special handling removes incomplete thinking, Qwen end tokens and the observed LaTeX final-answer wrapper. Scorer agreement is not guaranteed.

## Dataset mirror / local download

The third-party mirror https://hf-mirror.com documents use of HF_ENDPOINT. It is **not** the original Hugging Face domain. Verify network reachability and dataset revisions from the server; do not put credentials into unofficial services.

    cd /data1/lmy/SC4001-efficient-reasoning
    export HF_ENDPOINT=https://hf-mirror.com
    uv pip install --python .venv/bin/python huggingface_hub
    mkdir -p data/raw/gsm_plus data/raw/math_500
    .venv/bin/hf download qintongli/GSM-Plus --repo-type dataset --local-dir data/raw/gsm_plus
    .venv/bin/hf download HuggingFaceH4/MATH-500 --repo-type dataset --local-dir data/raw/math_500
    find data/raw/gsm_plus data/raw/math_500 -type f

If the executable is instead huggingface-cli, use that installed CLI. HF_ENDPOINT may not work with every client/version and a third-party mirror may be unreachable; do not claim download success before checking actual files. GSM-Plus authors also host a GitHub repository: https://github.com/qtli/GSM-Plus; verify its version against v1.

The preparation script reads local JSON, JSONL, or Parquet files. Locate the real downloaded filenames first:

    PYTHONPATH=src .venv/bin/python scripts/prepare_mixed_eval.py --gsm-plus PATH_TO_GSMPLUS_FILE --math-500 PATH_TO_MATH500_FILE --per-dataset 80 --seed 42

It produces data/processed/mixed_eval_v1.jsonl and a SHA256/strata manifest. It does not run generation and does not train.

## Evaluation set design

Initial proposal: GSM-Plus 80 + MATH-500 80, kept as **separate benchmark groups** within a single manifest. Select a fixed seed before using them for model comparison.

- GSM-Plus is derived from GSM8K **test** questions and is labeled test-only on its dataset card; do not use for SFT or router training. Sample at most one variation per seed_question to avoid repeated near-duplicate source questions. Exclude critical-thinking (unanswerable) variants because numeric equivalence does not grade abstention. Report excluded type explicitly; keep perturbation labels.
- MATH-500 is a 500-question test subset spanning different levels and topics; stratify by level. Expressions, fractions, sets, symbolic answers and tuples require mathematical rather than numeric string verification.
- Do not tune SFT, router architecture, temperature or thresholds on this evaluation set and then claim it is an untouched final test.
- Independently report accuracy, output tokens, latency, invalid responses, and truncation by benchmark and stratum. A simple pooled number obscures different distributions.
- Use separate training/dev resources for SFT and routing. Maintain seed-question lineage when comparing GSM-Plus with GSM8K Test.
- Before full generation: inspect 5-10 real rows of each dataset, validate answer parsing on gold strings, and confirm the official task instructions. The current run_pilot.py only supports pilot.jsonl and cannot run mixed_eval_v1.jsonl without a new benchmark-specific runner. This is deliberate to protect existing checkpoints.

Primary sources:
- https://huggingface.co/datasets/qintongli/GSM-Plus
- https://github.com/qtli/GSM-Plus
- https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- https://github.com/huggingface/Math-Verify


## Visual dependency screening (2026-10-20)

The 80+80 mixed sampler now automatically excludes MATH-500 problems whose **problem text itself** contains embedded image/Asymptote/TikZ/HTML image markers. Such rows are skipped before level-balanced sampling and are replaced from the same available level pools where possible. Mere presence of [asy] in the **reference solution** is not grounds for exclusion.

Possible natural-language references ("diagram below", "use the figure") are recorded with `visual_review_required=true` and listed separately in `data/processed/mixed_eval_v1.visual_review.json`. **Do not use flagged questions for GPU inference until reviewed.** For each flagged question, keep only if all necessary geometry/graph facts are contained in its problem text; otherwise exclude and replace from the same level, preserving sample size, and record IDs and reason in the manifest. This heuristic is not a guarantee that every visual-dependent question is detected. The selected set must be manually spot-checked before results are published.

To rebuild deterministically using downloaded files:

```bash
git pull --ff-only origin main
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/prepare_mixed_eval.py \
  --gsm-plus data/raw/gsm_plus/data/test-00000-of-00001.jsonl \
  --math-500 data/raw/math_500/test.jsonl --per-dataset 80 --seed 42
.venv/bin/python -m json.tool data/processed/mixed_eval_v1.visual_review.json
```

The manifest SHA256 will change due to new visual-audit fields. Do not merge responses from older sample hashes into this new set. New benchmark runner is not yet implemented.
