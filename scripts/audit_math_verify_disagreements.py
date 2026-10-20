"""CPU-only 3-way comparison: legacy regex, corrected-format audit, Math-Verify.

No result is silently declared authoritative. Reads saved predictions only.
"""
import argparse
import json
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from sc4001.answers import gold_answer, predicted_answer
from scripts.evaluate_math_verify import math_score, final_text

def read_gold(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    return {r["id"]: gold_answer(r["answer"]) for r in rows}

def read_corrections(path):
    if not Path(path).exists():
        return {}
    report = json.loads(Path(path).read_text())
    return {(r["id"], r["mode"]): r for r in report.get("corrections", [])}

def analyze(dataset, cases, corrections):
    golds = read_gold(dataset)
    old_corrections = read_corrections(corrections)
    totals = Counter()
    disagreements = []
    for sid, gold in golds.items():
        if gold is None:
            raise ValueError(f"Invalid gold answer: {sid}")
        for mode in ("fast", "slow"):
            p = Path(cases) / f"{sid}__{mode}.json"
            if not p.exists():
                raise FileNotFoundError(p)
            record = json.loads(p.read_text())
            old = predicted_answer(record["response"])
            regex_ok = old == gold if old is not None else False
            mv_ok, parsed = math_score(gold, record["response"])
            corrected_entry = old_corrections.get((sid, mode))
            audited_ok = bool(corrected_entry["correct"]) if corrected_entry else regex_ok
            totals[f"{mode}_legacy"] += regex_ok
            totals[f"{mode}_corrected"] += audited_ok
            totals[f"{mode}_math_verify"] += mv_ok
            if audited_ok != mv_ok or (not parsed and not record.get("truncated", False)):
                disagreements.append({
                    "id": sid, "mode": mode, "gold": str(gold),
                    "corrected_audit_correct": audited_ok,
                    "math_verify_correct": mv_ok,
                    "math_verify_parsed": [str(x) for x in parsed],
                    "prior_correction": corrected_entry,
                    "truncated": bool(record.get("truncated", False)),
                    "final_text_tail": final_text(record["response"])[-900:],
                })
    return {"n": len(golds), "totals": dict(totals),
            "disagreements": disagreements,
            "disagreement_count": len(disagreements),
            "note": "Compare all disagreements manually before choosing a canonical scorer."}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/processed/pilot.jsonl")
    p.add_argument("--cases", default="outputs/pilot-v2/cases")
    p.add_argument("--corrections", default="outputs/pilot-v2/audit_corrected.json")
    p.add_argument("--output", default="outputs/pilot-v2/scorer_disagreements.json")
    a=p.parse_args()
    result=analyze(a.dataset, a.cases, a.corrections)
    out=Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"n":result["n"],"totals":result["totals"],
                      "disagreement_count":result["disagreement_count"]},indent=2))
    print("Review each flagged case:",out)

if __name__=="__main__":
    main()
