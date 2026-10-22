"""Audit canonical mixed benchmark errors from saved GPU generations (CPU only).

This diagnostic NEVER changes or rescues the official boxed-v2 scores.
It works on a 20-question smoke or the full 160-question report.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sc4001.mixed_protocol import case_path, completed, load_mixed
from scripts.analyze_mixed_errors import analyze, categorize


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("scorer_protocol") != "canonical-boxed-v2":
        raise ValueError(f"Expected canonical-boxed-v2 report: {path}")
    cases = report.get("per_case", [])
    if not cases or len(cases) != report.get("n"):
        raise ValueError(f"Missing/incomplete per-case report: {path}")
    if len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate case IDs in report")
    return report


def audit(report: dict, raw_dir: Path, dataset: Path, comparison: dict | None):
    dataset_rows = {r["id"]: r for r in load_mixed(dataset)}
    ref_cases = ({r["id"]: r for r in comparison["per_case"]}
                 if comparison is not None else {})
    groups = {"fast": Counter(), "slow": Counter()}
    flags = {"fast": Counter(), "slow": Counter()}
    details = []

    for case in report["per_case"]:
        sid = case["id"]
        if sid not in dataset_rows:
            raise ValueError(f"Case not present in specified dataset: {sid}")
        sample = dataset_rows[sid]
        detail = {
            "id": sid, "benchmark": case["benchmark"],
            "stratum": case["stratum"], "paired_category": case["paired_category"],
            "question": sample["question"], "gold": str(sample["answer"]),
        }
        if sid in ref_cases:
            detail["reference"] = {
                mode: {
                    "correct": ref_cases[sid][mode]["correct"],
                    "failure_type": categorize(ref_cases[sid][mode]),
                } for mode in ("fast", "slow")
            }
        for mode in ("fast", "slow"):
            scored = case[mode]
            raw = completed(case_path(raw_dir, sid, mode), sid, mode,
                            report["generation_fingerprint"])
            if raw is None:
                raise ValueError(f"Missing raw generation: {sid} {mode}")
            cause = categorize(scored)
            groups[mode][cause] += 1
            flags[mode][scored["format_status"]] += 1
            detail[mode] = {
                "failure_type": cause,
                "correct": scored["correct"],
                "canonical_format_ok": scored["canonical_format_ok"],
                "format_status": scored["format_status"],
                "math_parse_ok": scored["parse_ok"],
                "prediction": scored["prediction"],
                "review_flags": scored["review_flags"],
                "truncated": scored["truncated"],
                "tokens": scored["output_tokens"],
                "latency_sec": scored["latency_sec"],
                "response": raw["response"],
            }
        details.append(detail)

    return {
        "n": len(details),
        "failure_counts": {mode: dict(count) for mode, count in groups.items()},
        "format_status_counts": {mode: dict(count) for mode, count in flags.items()},
        "group_summary": analyze(report)["group_summary"],
        "cases": details,
    }


def format_review(results: dict, *, excerpt_chars: int = 2600) -> str:
    lines = [
        "# Canonical mixed failure review",
        "",
        "Categories are mutually exclusive: success > truncated > "
        "format_failure > math_parse_failure > math_wrong_or_verification_dispute.",
        "These are observed failure symptoms, NOT verified root causes.",
        "Inspect full raw generations in cases.jsonl before labeling arithmetic "
        "or reasoning errors. Do not change scores based on this audit.",
        "",
    ]
    for case in results["cases"]:
        if case["fast"]["correct"] and case["slow"]["correct"]:
            continue
        lines += [
            "=" * 90,
            f'{case["id"]} [{case["benchmark"]} / {case["stratum"]}]',
            f'paired_category: {case["paired_category"]}',
            "QUESTION: " + case["question"],
            "GOLD: " + case["gold"],
        ]
        if "reference" in case:
            lines.append("REFERENCE: " + json.dumps(case["reference"], ensure_ascii=False))
        for mode in ("fast", "slow"):
            record = case[mode]
            response = record["response"]
            excerpt = (response if len(response) <= excerpt_chars * 2
                       else response[:excerpt_chars] +
                       "\n\n...[MIDDLE OMITTED; full response in cases.jsonl]...\n\n" +
                       response[-excerpt_chars:])
            lines += [
                "",
                f'[{mode.upper()}] classification={record["failure_type"]} '
                f'format={record["format_status"]} '
                f'parse_ok={record["math_parse_ok"]} '
                f'truncated={record["truncated"]} tokens={record["tokens"]}',
                f'prediction: {record["prediction"]}',
                f'review_flags: {record["review_flags"]}',
                "RAW RESPONSE (beginning + end):",
                excerpt,
                "",
            ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default="outputs/mixed-boxed-1p7b/smoke_canonical_v2_summary.json")
    p.add_argument("--cases-dir", default="outputs/mixed-boxed-1p7b/cases")
    p.add_argument("--dataset", default="data/processed/mixed_eval_v1.jsonl")
    p.add_argument("--reference-summary",
                   default="outputs/mixed-boxed-v1/full_canonical_v2_summary.json")
    p.add_argument("--output-dir", default="outputs/mixed-boxed-1p7b/failure_audit")
    args = p.parse_args()
    report = load_report(Path(args.summary))
    comparison = (load_report(Path(args.reference_summary))
                  if args.reference_summary else None)
    if comparison is not None:
        missing = {r["id"] for r in report["per_case"]} - {
            r["id"] for r in comparison["per_case"]
        }
        if missing:
            raise ValueError(f"Reference report missing {len(missing)} cases")
    result = audit(report, Path(args.cases_dir), Path(args.dataset), comparison)

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "cases"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (destination / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in result["cases"]:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    (destination / "review.txt").write_text(
        format_review(result), encoding="utf-8")

    for mode in ("fast", "slow"):
        print(f"\n=== {mode.upper()} ===")
        print(json.dumps(result["failure_counts"][mode], indent=2))
        print("Format status:", result["format_status_counts"][mode])
    print("\nFailed-pair overview (including matched reference when available):")
    for case in result["cases"]:
        if case["fast"]["correct"] and case["slow"]["correct"]:
            continue
        ref = (" ref4B=" + "".join(
            ("F" if case["reference"]["fast"]["correct"] else "-",
             "S" if case["reference"]["slow"]["correct"] else "-"))
            if "reference" in case else "")
        print(f'{case["id"]}: fast={case["fast"]["failure_type"]} '
              f'slow={case["slow"]["failure_type"]}{ref}')
    print("\nSaved:", destination / "summary.json")
    print("Full raw generations:", destination / "cases.jsonl")
    print("Readable failed-case review:", destination / "review.txt")


if __name__ == "__main__":
    main()
