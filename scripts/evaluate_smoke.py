"""Evaluate a SMALL, real, paired GSM8K pilot smoke run on CPU.

Run after scripts/run_pilot.py --limit 5 (or --limit 10). This script
never loads the model. Smoke results are diagnostic, not final claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc4001.answers import gold_answer, predicted_answer
from sc4001.metrics import evaluate_pairs
from sc4001.pilot import (
    atomic_json, case_path, check_manifest, load_completed, load_examples,
    run_manifest,
)


def build_smoke_report(examples: list[dict], output_dir: Path, fingerprint: str) -> dict:
    predictions = []
    per_case = []
    for sample in examples:
        pair = {}
        for mode in ("fast", "slow"):
            ckpt = load_completed(
                case_path(output_dir, sample["id"], mode),
                sample["id"], mode, fingerprint
            )
            if ckpt is None:
                raise ValueError(
                    f"Missing {mode} output for {sample['id']}. "
                    "Run the same --limit with run_pilot.py first."
                )
            pair[mode] = ckpt
            predictions.append(ckpt)
        gold = gold_answer(sample["answer"])
        row = {"id": sample["id"], "gold": str(gold)}
        for mode in ("fast", "slow"):
            ckpt = pair[mode]
            parsed = predicted_answer(ckpt["response"])
            row[mode] = {
                "answer": str(parsed) if parsed is not None else None,
                "correct": parsed is not None and parsed == gold,
                "tokens": ckpt.get("output_tokens"),
                "latency_sec": ckpt.get("latency_sec"),
                "truncated": ckpt.get("truncated", False),
            }
        per_case.append(row)

    report = evaluate_pairs(examples, predictions)
    return {
        "kind": "REAL_SMOKE_DIAGNOSTIC_NOT_FINAL_EVALUATION",
        "fingerprint": fingerprint,
        "metrics": report,
        "per_case": per_case,
        "warning": (
            "Small smoke samples cannot establish a robust accuracy difference. "
            "Inspect full raw generations, parse failures, and truncation before drawing conclusions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--output-dir", default="outputs/pilot-v2")
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("--limit must be between 1 and 200")

    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = Path(config["dataset"])
    all_examples = load_examples(dataset)
    if len(all_examples) != 200:
        raise ValueError("Pilot dataset must contain 200 train-split samples")
    manifest = run_manifest(config, dataset)
    output = Path(args.output_dir)
    check_manifest(output, manifest, create=False)
    report = build_smoke_report(
        all_examples[:args.limit], output, manifest["fingerprint"]
    )
    filename = output / f"smoke_summary_n{args.limit}.json"
    atomic_json(filename, report)

    print(f"REAL SMOKE: {args.limit} paired questions; report: {filename}")
    metrics = report["metrics"]
    print(f"{'METRIC':<24} {'FAST':>12} {'SLOW':>12}")
    print(f"{'Accuracy':<24} {metrics['fast_accuracy']:>11.1%} {metrics['slow_accuracy']:>11.1%}")
    print(f"{'Mean output tokens':<24} {str(metrics['mean_output_tokens']['fast']):>12} {str(metrics['mean_output_tokens']['slow']):>12}")
    print(f"{'Mean latency (s)':<24} {str(metrics['mean_latency_sec']['fast']):>12} {str(metrics['mean_latency_sec']['slow']):>12}")
    print(f"{'Invalid answers':<24} {metrics['invalid_final_answer']['fast']:>12} {metrics['invalid_final_answer']['slow']:>12}")
    print(f"{'Truncations':<24} {metrics['truncated']['fast']:>12} {metrics['truncated']['slow']:>12}")
    print("Paired categories:", json.dumps(metrics["paired_counts"], sort_keys=True))
    print("NOTE: Small-sample exploratory smoke only; inspect raw responses before interpreting.")


if __name__ == "__main__":
    main()
