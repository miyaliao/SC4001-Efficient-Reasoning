"""Offline, paired error analysis of frozen canonical-boxed-v2 outputs.

Uses the persisted full score and generation records; no GPU. Separates
truncation, format failure, and math failures, with overlap made explicit.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def categorize(row):
    if row["correct"]:
        return "success"
    if row["truncated"]:
        return "truncated"
    if not row["canonical_format_ok"]:
        return "format_failure"
    if not row["parse_ok"]:
        return "math_parse_failure"
    return "math_wrong_or_verification_dispute"


def analyze(report):
    cases = report["per_case"]
    groups = defaultdict(list)
    outcomes = defaultdict(list)
    details = []
    for case in cases:
        fast, slow = case["fast"], case["slow"]
        tags = {
            "fast": categorize(fast), "slow": categorize(slow),
        }
        paired = case["paired_category"]
        row = {
            "id":case["id"],"benchmark":case["benchmark"],"stratum":case["stratum"],
            "paired_category":paired,
            "gold":case["gold"],
            "fast_failure":tags["fast"],"slow_failure":tags["slow"],
            "fast_correct":fast["correct"],"slow_correct":slow["correct"],
            "fast_format":fast["canonical_format_ok"],"slow_format":slow["canonical_format_ok"],
            "fast_truncated":fast["truncated"],"slow_truncated":slow["truncated"],
            "fast_tokens":fast["output_tokens"],"slow_tokens":slow["output_tokens"],
            "fast_prediction":"; ".join(fast["prediction"]),
            "slow_prediction":"; ".join(slow["prediction"]),
            "fast_flags":"; ".join(fast["review_flags"]),
            "slow_flags":"; ".join(slow["review_flags"]),
        }
        details.append(row)
        for key in ("overall",case["benchmark"],f'{case["benchmark"]}:{case["stratum"]}'):
            groups[key].append(row)
        outcomes[paired].append(case["id"])

    def stat(rows):
        total=len(rows)
        d={
            "n":total,
            "paired_counts":dict(Counter(x["paired_category"] for x in rows)),
            "fast_failure_types":dict(Counter(x["fast_failure"] for x in rows)),
            "slow_failure_types":dict(Counter(x["slow_failure"] for x in rows)),
            "slow_truncation_by_pair":dict(Counter(x["paired_category"] for x in rows if x["slow_truncated"])),
            "slow_truncated_n":sum(x["slow_truncated"] for x in rows),
            "fast_success_n":sum(x["fast_correct"] for x in rows),
            "slow_success_n":sum(x["slow_correct"] for x in rows),
        }
        d["oracle_success_n"] = d["paired_counts"].get("both",0)+d["paired_counts"].get("fast_only",0)+d["paired_counts"].get("slow_only",0)
        d["oracle_success"] = d["oracle_success_n"]/total
        d["fast_success"] = d["fast_success_n"]/total
        d["slow_success"] = d["slow_success_n"]/total
        return d
    return {
        "scorer_protocol":report["scorer_protocol"],
        "generation_fingerprint":report["generation_fingerprint"],
        "group_summary":{k:stat(v) for k,v in sorted(groups.items())},
        "error_case_ids":{k:v for k,v in sorted(outcomes.items())},
        "per_case":details,
        "note":"Truncation overrides format-failure category; mathematical failure is a residual after valid final/parse. Case IDs are NOT labels of underlying cognitive error. Review original response and question before causal claims."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="outputs/mixed-boxed-v1/full_canonical_v2_summary.json")
    p.add_argument("--output-dir",default="outputs/mixed-boxed-v1/error_analysis")
    args=p.parse_args()
    report=json.loads(Path(args.input).read_text(encoding="utf-8"))
    if report.get("scorer_protocol")!="canonical-boxed-v2" or report.get("n")!=160:
        raise ValueError("Expected complete canonical-boxed-v2 N=160 report")
    result=analyze(report)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps({k:v for k,v in result.items() if k!="per_case"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (out/"cases.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(result["per_case"][0]))
        writer.writeheader()
        writer.writerows(result["per_case"])
    print(json.dumps(result["group_summary"]["overall"],indent=2))
    print("Group-level summary:",out/"summary.json")
    print("Per-case CSV:",out/"cases.csv")
    print("Inspect Fast-only, Slow-only, truncations, and parser flags manually.")


if __name__=="__main__":
    main()
