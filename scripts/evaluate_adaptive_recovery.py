"""Create 160-question official *adaptive 4K->16K* evaluation.

Replaces the scored outcome of only baseline Slow-truncated cases with their
16K retry. Full inference cost counts BOTH original 4K attempt and retry.
Does not modify either source experiment or claim full Slow-16K evaluation.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from sc4001.mixed_protocol import load_mixed, manifest, case_path, completed
from sc4001.boxed_scoring import canonical_box, final_body
from sc4001.pilot import atomic_json, check_manifest
from scripts.evaluate_mixed_canonical import math_grade


def summarize(cases):
    buckets = defaultdict(list)
    for item in cases:
        buckets["overall"].append(item)
        buckets[item["benchmark"]].append(item)
        buckets[item["benchmark"] + ":" + str(item["stratum"])].append(item)
    groups = {}
    for name, rows in sorted(buckets.items()):
        g = {"n": len(rows), "paired_counts": dict(Counter(x["paired_category"] for x in rows))}
        for mode in ("fast", "slow"):
            r = [x[mode] for x in rows]
            good_format = sum(x["canonical_format_ok"] for x in r)
            correct = sum(x["correct"] for x in r)
            g[mode] = {
                "task_success": correct / len(r),
                "math_accuracy_given_canonical_format": correct / good_format if good_format else None,
                "canonical_format_compliance": good_format / len(r),
                "exact_format_compliance": sum(x["exact_format_ok"] for x in r)/len(r),
                "exact_task_success": sum(x["exact_task_success"] for x in r)/len(r),
                "math_parse_count": sum(x["parse_ok"] for x in r),
                "truncated_count": sum(x["truncated"] for x in r),
                "percent_points_normalized_count": sum(x["percent_points_normalized"] for x in r),
                "format_failures": dict(Counter(x["format_status"] for x in r if not x["canonical_format_ok"])),
                "mean_output_tokens": mean(x["output_tokens"] for x in r),
                "mean_latency_sec": mean(x["latency_sec"] for x in r),
            }
        groups[name] = g
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-config", default="configs/mixed_1p7b.yaml")
    p.add_argument("--baseline-dir", default="outputs/mixed-boxed-1p7b")
    p.add_argument("--recovery-dir", default="outputs/recovery-1p7b-slow-16384")
    p.add_argument("--output-dir", default="outputs/mixed-boxed-1p7b/adaptive-4k-to-16k")
    p.add_argument("--expected-cohort", type=int, default=28)
    args = p.parse_args()
    import yaml
    baseline_config = yaml.safe_load(Path(args.baseline_config).read_text())
    if baseline_config["decode"]["max_new_tokens"] != 4096:
        raise ValueError("Expected original 4096-token config")
    data_file = Path(baseline_config["dataset"])
    rows = load_mixed(data_file)
    if len(rows) != 160:
        raise ValueError("Expected 160 rows")
    baseline_spec = manifest(baseline_config, data_file)
    base_root = Path(args.baseline_dir)
    check_manifest(base_root, baseline_spec, create=False)
    base_report = json.loads((base_root/"full_canonical_v2_summary.json").read_text())
    if base_report["generation_fingerprint"] != baseline_spec["fingerprint"]:
        raise ValueError("Baseline report fingerprint mismatch")
    if base_report["scorer_protocol"] != "canonical-boxed-v2":
        raise ValueError("Unexpected scorer")
    updated_config = json.loads(json.dumps(baseline_config))
    updated_config["decode"]["max_new_tokens"] = 16384
    recovery_spec = manifest(updated_config, data_file)
    rec_root = Path(args.recovery_dir)
    check_manifest(rec_root, recovery_spec, create=False)

    case_map = {x["id"]: json.loads(json.dumps(x)) for x in base_report["per_case"]}
    if len(case_map) != 160:
        raise ValueError("Baseline per-case records missing")
    selected = []
    recovered_correct = 0
    retry_tokens = 0
    retry_latency = 0.0
    for row in rows:
        sid = row["id"]
        b = completed(case_path(base_root,sid,"slow"),sid,"slow",baseline_spec["fingerprint"])
        if b is None:
            raise ValueError(f"Missing baseline slow case: {sid}")
        if not b["truncated"]:
            continue
        selected.append(sid)
        r = completed(case_path(rec_root,sid,"slow"),sid,"slow",recovery_spec["fingerprint"])
        if r is None:
            raise ValueError(f"Missing recovered slow case: {sid}")
        box,status,wrapper = canonical_box(r["response"])
        correct,parsed,extracted,percent = (math_grade(row["answer"],box,row["question"])
             if box is not None else (False,False,[],False))
        body = final_body(r["response"])
        last = body.splitlines()[-1].strip() if body else ""
        exact = bool(box is not None and wrapper == "none" and last == "Final answer: " + box)
        flags = []
        if percent: flags.append("percentage_points_interpretation")
        if any("Interval." in x for x in extracted):
            flags.append("interval_or_coordinate_ambiguity")
        if r"\!" in str(row["answer"]):
            flags.append("latex_digit_grouping")
        case_map[sid]["slow"] = {
            "review_flags": flags, "correct": bool(correct),
            "canonical_format_ok": status == "ok",
            "exact_format_ok": exact, "exact_task_success": bool(exact and correct),
            "format_status": status, "wrapper": wrapper,
            "parse_ok": bool(parsed), "prediction": extracted,
            "percent_points_normalized": percent,
            "truncated": bool(r["truncated"]),
            "output_tokens": r["output_tokens"], "latency_sec": r["latency_sec"],
        }
        recovered_correct += int(correct)
        retry_tokens += r["output_tokens"]
        retry_latency += r["latency_sec"]
    if len(selected) != args.expected_cohort:
        raise ValueError(f"Expected {args.expected_cohort} truncations, got {len(selected)}")
    cases=[]
    for row in rows:
        case=case_map[row["id"]]
        f,s=bool(case["fast"]["correct"]),bool(case["slow"]["correct"])
        case["paired_category"]="both" if f and s else "fast_only" if f else "slow_only" if s else "neither"
        cases.append(case)
    groups=summarize(cases)
    initial_slow_tokens=sum(x["slow"]["output_tokens"] for x in base_report["per_case"])
    initial_slow_latency=sum(x["slow"]["latency_sec"] for x in base_report["per_case"])
    report={
        "policy":"adaptive-4k-to-16k-on-slow-truncation-v1",
        "scorer_protocol":"canonical-boxed-v2",
        "baseline_fingerprint":baseline_spec["fingerprint"],
        "recovery_fingerprint":recovery_spec["fingerprint"],
        "n":len(rows), "recovery_cohort_count":len(selected),
        "recovery_cohort_ids":selected,
        "recovered_success_count":recovered_correct,
        "original_slow_success":base_report["groups"]["overall"]["slow"]["task_success"],
        "adaptive_slow_success":groups["overall"]["slow"]["task_success"],
        "groups":groups, "per_case":cases,
        "efficiency":{
            "slow_mean_total_tokens_per_question":(initial_slow_tokens+retry_tokens)/len(rows),
            "slow_mean_total_compute_latency_per_question_sec":(initial_slow_latency+retry_latency)/len(rows),
            "retry_total_output_tokens":retry_tokens,
            "retry_total_latency_sec":retry_latency,
            "note":"Total costs include unsuccessful original 4K generation AND 16K retry. Mean output tokens in groups describe retained final outputs only."
        },
        "disclaimer":"Adaptive selective recovery, NOT full 16K Slow baseline. Exploratory 160-question set; baseline unchanged."
    }
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"official_adaptive_summary.json",report)
    print(json.dumps({
        "policy":report["policy"],
        "recovery_cohort":len(selected),
        "recovered_correct":recovered_correct,
        "slow_success_original":report["original_slow_success"],
        "slow_success_adaptive":report["adaptive_slow_success"],
        "fast_success":groups["overall"]["fast"]["task_success"],
        "adaptive_slow_truncated":groups["overall"]["slow"]["truncated_count"],
        "paired_counts":groups["overall"]["paired_counts"],
        "slow_mean_total_tokens":report["efficiency"]["slow_mean_total_tokens_per_question"],
        "slow_mean_total_compute_latency_sec":report["efficiency"]["slow_mean_total_compute_latency_per_question_sec"]
    },indent=2))
    print("Saved:",out/"official_adaptive_summary.json")

if __name__=="__main__":main()
