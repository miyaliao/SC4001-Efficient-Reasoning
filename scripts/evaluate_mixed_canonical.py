"""Rescore saved mixed benchmark under canonical boxed format (CPU only).

V1 strict report remains untouched; no GPU generation or prompting changes.
Run -m scripts.evaluate_mixed_canonical from repo root with PYTHONPATH=src:.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean

from sc4001.boxed_scoring import canonical_box, final_body, numeric_percent_points
from sc4001.mixed_protocol import (
    MODES, case_path, completed, load_mixed, manifest, selected_questions,
)
from sc4001.pilot import atomic_json, check_manifest
from scripts.evaluate_mixed import audit_gold


def math_grade(gold: str, boxed: str, question: str):
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig

    adjusted, percent_points = numeric_percent_points(str(gold), boxed, question)
    settings = [LatexExtractionConfig(boxed_match_priority=0)]
    ref = parse(r"\boxed{" + str(gold) + "}", extraction_config=settings,
                fallback_mode="no_fallback")
    pred = parse(adjusted, extraction_config=settings, fallback_mode="no_fallback")
    if not ref or not pred:
        return False, bool(pred), [str(x) for x in pred], percent_points
    result = bool(verify(ref, pred))
    # Basic numeric sanity check: a literal numeric result must equal a literal
    # numeric gold, never accept e.g. 36% == 36 via fuzzy verification.
    import re
    plain_gold = re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", str(gold).strip())
    plain_pred = re.fullmatch(r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", adjusted)
    if plain_gold and plain_pred:
        result = (Decimal(str(gold).replace(",", "")) ==
                  Decimal(plain_pred.group(1).replace(",", "")))
    elif plain_gold and (r"\%" in adjusted or "%" in adjusted) and not percent_points:
        # An isolated percentage is not the same unit as a raw count.
        result = False
    return result, True, [str(x) for x in pred], percent_points


def score(rows, root: Path, fingerprint: str):
    scored = []
    groups = defaultdict(list)
    for row in rows:
        case = {"id": row["id"], "benchmark": row["benchmark"],
                "stratum": row["stratum"], "gold": str(row["answer"])}
        pair = {}
        for mode in MODES:
            saved = completed(case_path(root, row["id"], mode), row["id"], mode, fingerprint)
            if saved is None:
                raise ValueError(f"Missing generation for {row['id']} {mode}")
            box, status, wrapper = canonical_box(saved["response"])
            if box is None:
                correct, parsed, extracted, normalized_percent = False, False, [], False
            else:
                correct, parsed, extracted, normalized_percent = math_grade(
                    row["answer"], box, row["question"])
            body = final_body(saved["response"])
            last_line = body.splitlines()[-1].strip() if body else ""
            strict_compliant = bool(
                box is not None and wrapper == "none"
                and last_line == "Final answer: " + box
            )
            # Exact-format task success also requires a mathematically correct answer.
            review_flags = []
            if normalized_percent:
                review_flags.append("percentage_points_interpretation")
            if any("Interval." in p for p in extracted):
                review_flags.append("interval_or_coordinate_ambiguity")
            if str(row["answer"]).find(r"\!") >= 0:
                review_flags.append("latex_digit_grouping")
            item = {
                "review_flags": review_flags,
                "correct": correct,
                "canonical_format_ok": status == "ok",
                "exact_format_ok": strict_compliant,
                "exact_task_success": strict_compliant and correct,
                "format_status": status, "wrapper": wrapper,
                "parse_ok": parsed, "prediction": extracted,
                "percent_points_normalized": normalized_percent,
                "truncated": bool(saved.get("truncated")),
                "output_tokens": saved.get("output_tokens"),
                "latency_sec": saved.get("latency_sec"),
            }
            case[mode] = item
            pair[mode] = bool(correct)
        case["paired_category"] = ("both" if all(pair.values()) else
                                   "fast_only" if pair["fast"] else
                                   "slow_only" if pair["slow"] else "neither")
        scored.append(case)
        groups["overall"].append(case)
        groups[row["benchmark"]].append(case)
        groups[row["benchmark"] + ":" + str(row["stratum"])].append(case)

    def aggregate(examples):
        result = {"n": len(examples),
                  "paired_counts": dict(Counter(x["paired_category"] for x in examples))}
        for mode in MODES:
            values = [x[mode] for x in examples]
            result[mode] = {
                "task_success": sum(x["correct"] for x in values) / len(values),
                "math_accuracy_given_canonical_format": (
                    sum(x["correct"] for x in values) /
                    sum(x["canonical_format_ok"] for x in values)
                    if any(x["canonical_format_ok"] for x in values) else None
                ),
                "canonical_format_compliance": sum(x["canonical_format_ok"] for x in values) / len(values),
                "exact_format_compliance": sum(x["exact_format_ok"] for x in values) / len(values),
                "exact_task_success": sum(x["exact_task_success"] for x in values) / len(values),
                "math_parse_count": sum(x["parse_ok"] for x in values),
                "truncated_count": sum(x["truncated"] for x in values),
                "percent_points_normalized_count": sum(x["percent_points_normalized"] for x in values),
                "format_failures": dict(Counter(x["format_status"] for x in values if not x["canonical_format_ok"])),
                "mean_output_tokens": mean(x["output_tokens"] for x in values),
                "mean_latency_sec": mean(x["latency_sec"] for x in values),
            }
        return result

    from importlib.metadata import version
    return {
        "scorer_protocol": "canonical-boxed-v2",
        "math_verify_version": version("math-verify"),
        "generation_fingerprint": fingerprint,
        "n": len(rows),
        "groups": {k: aggregate(v) for k, v in sorted(groups.items())},
        "per_case": scored,
        "note": ("Presentation wrappers are allowed, but a unique final boxed "
                 "answer is mandatory. Math correctness is evaluated only when "
                 "the final box exists. Review percent and structured answers.")
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mixed.yaml")
    p.add_argument("--output-dir", default="outputs/mixed-boxed-v1")
    p.add_argument("--smoke-per-benchmark", type=int, default=10)
    p.add_argument("--full", action="store_true")
    p.add_argument("--gold-audit-only", action="store_true")
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_file = Path(cfg["dataset"])
    rows = load_mixed(data_file)
    selected = selected_questions(
        rows, None if args.full or args.gold_audit_only else args.smoke_per_benchmark)
    errors = audit_gold(selected)
    if errors:
        print(json.dumps({"gold_parse_failures": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(f"Gold parse failed: {len(errors)} entries")
    print(f"Gold audit PASS: {len(selected)} records", flush=True)
    if args.gold_audit_only:
        return

    spec = manifest(cfg, data_file)
    output_dir = Path(args.output_dir)
    check_manifest(output_dir, spec, create=False)
    report = score(selected, output_dir, spec["fingerprint"])
    destination = output_dir / ("full_canonical_v2_summary.json" if args.full
                                else "smoke_canonical_v2_summary.json")
    atomic_json(destination, report)
    print(json.dumps(report["groups"]["overall"], indent=2))
    print("Per-benchmark and per-case details:", destination)


if __name__ == "__main__":
    main()
