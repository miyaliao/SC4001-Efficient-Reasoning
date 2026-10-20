"""CPU-only Math-Verify audit of existing GSM8K paired results."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sc4001.answers import gold_answer, predicted_answer


def final_text(response):
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    elif "<think>" in response:
        return ""
    response = re.sub(r"<\|[^>]+\|>", "", response)
    return re.sub(r"\\text\s*\{\s*Final\s+answer\s*:\s*\}",
                  "Final answer:", response, flags=re.I)


def math_score(gold, response):
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    text = final_text(response)
    reference = parse("Final answer: " + str(gold),
                      extraction_config=[ExprExtractionConfig()],
                      fallback_mode="no_fallback")
    prediction = (parse(text, extraction_config=[
                      LatexExtractionConfig(), ExprExtractionConfig()],
                      fallback_mode="no_fallback") if text else [])
    return bool(reference and prediction and verify(reference, prediction)), prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/pilot.jsonl")
    parser.add_argument("--cases", default="outputs/pilot-v2/cases")
    parser.add_argument("--output", default="outputs/pilot-v2/math_verify_audit.json")
    args = parser.parse_args()
    samples = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    counts, categories, flags = Counter(), Counter(), []
    for example in samples:
        gold = gold_answer(example["answer"])
        if gold is None:
            raise ValueError(f"Unparseable gold: {example['id']}")
        pair = {}
        for mode in ("fast", "slow"):
            case = Path(args.cases) / f"{example['id']}__{mode}.json"
            row = json.loads(case.read_text())
            accurate, parsed = math_score(gold, row["response"])
            pair[mode] = accurate
            counts[f"{mode}_correct"] += int(accurate)
            counts[f"{mode}_unparsed"] += int(not parsed)
            counts[f"{mode}_truncated"] += int(bool(row.get("truncated")))
            old = predicted_answer(row["response"])
            old_correct = bool(old is not None and old == gold)
            if accurate != old_correct or not parsed:
                flags.append({"id": example["id"], "mode": mode,
                              "gold": str(gold), "old_correct": old_correct,
                              "math_verify_correct": accurate,
                              "math_verify_parsed": [str(x) for x in parsed],
                              "truncated": bool(row.get("truncated")),
                              "final_tail": final_text(row["response"])[-350:]})
        cat = ("both" if pair["fast"] and pair["slow"] else
               "fast_only" if pair["fast"] else
               "slow_only" if pair["slow"] else "neither")
        categories[cat] += 1
    result = {
        "n": len(samples), "scorer": "math-verify",
        "accuracy": {mode: counts[f"{mode}_correct"] / len(samples) for mode in ("fast", "slow")},
        "unparsed": {mode: counts[f"{mode}_unparsed"] for mode in ("fast", "slow")},
        "truncated": {mode: counts[f"{mode}_truncated"] for mode in ("fast", "slow")},
        "paired_counts": dict(categories), "flagged_cases": flags,
        "warning": "Independent audit; inspect scorer disagreements before publishing results.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k!="flagged_cases"}, indent=2))
    print("Flagged cases:", len(flags), "See:", out)


if __name__ == "__main__":
    main()
