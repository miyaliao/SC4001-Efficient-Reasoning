"""Inspect boxed-format failures and suspicious math scores without changing metrics."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from scripts.evaluate_mixed import after_thinking, boxed_final

_BOX = re.compile(r"\\boxed\s*\{")
_FINAL = re.compile(r"final\s+answer\b", re.IGNORECASE)
_MATH_WRAP = re.compile(r"(?:\$+|\\\(|\\\[)\s*\\boxed\s*\{")


def classify(response: str, truncated: bool) -> str:
    """Diagnostic only: never promotes a noncompliant response to correct."""
    _, status = boxed_final(response)
    if status == "ok":
        return "strict_format_ok"
    if truncated:
        return "truncated_before_valid_final"
    body = after_thinking(response)
    n_boxes = len(_BOX.findall(body))
    n_labels = len(_FINAL.findall(body))
    if n_boxes == 0:
        return "no_boxed_expression"
    if n_labels == 0:
        return "box_without_final_answer_label"
    if n_boxes > 1 or n_labels > 1:
        return "multiple_boxes_or_final_labels"
    if _MATH_WRAP.search(body):
        return "boxed_expression_wrapped_in_math_delimiters"
    return "other_boxed_format_mismatch"


def analyze(summary: dict, cases_dir: Path, dataset: Path, tail_len: int = 1200):
    source = {r["id"]: r for r in map(json.loads, dataset.read_text(encoding="utf-8").splitlines()) if r}
    issues, categories = [], Counter()
    for item in summary["per_case"]:
        sid = item["id"]
        for mode in ("fast", "slow"):
            scored = item[mode]
            path = cases_dir / f"{sid}__{mode}.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            reason = classify(raw["response"], bool(raw.get("truncated")))
            parsed = scored.get("predicted", [])
            ambiguity = any("*(1/100)" in x or "Interval." in x for x in parsed)
            if reason == "strict_format_ok" and not ambiguity:
                continue
            categories[f"{mode}:{reason}"] += 1
            if ambiguity:
                categories[f"{mode}:ambiguous_math_parse"] += 1
            issues.append({
                "id": sid, "mode": mode, "benchmark": item["benchmark"],
                "stratum": item["stratum"], "gold": item["gold"],
                "question_excerpt": source[sid]["question"][:600],
                "format_status": scored["format_status"],
                "diagnosis": reason,
                "truncated": bool(raw.get("truncated")),
                "scored_correct": scored["correct"],
                "math_verify_parsed": parsed,
                "math_parse_needs_review": ambiguity,
                "post_thinking_response_tail": after_thinking(raw["response"])[-tail_len:],
            })
    return {"counts": dict(sorted(categories.items())), "issues": issues,
            "note": "For diagnosis only; neither scores nor original outputs were changed."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="outputs/mixed-boxed-v1/smoke_summary.json")
    ap.add_argument("--cases-dir", default="outputs/mixed-boxed-v1/cases")
    ap.add_argument("--dataset", default="data/processed/mixed_eval_v1.jsonl")
    ap.add_argument("--output", default="outputs/mixed-boxed-v1/smoke_format_audit.json")
    args = ap.parse_args()
    result = analyze(json.loads(Path(args.summary).read_text()), Path(args.cases_dir), Path(args.dataset))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "total_flagged": len(result["issues"])}, indent=2))
    print("Review:", output)


if __name__ == "__main__":
    main()
