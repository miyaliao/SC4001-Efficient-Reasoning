"""CPU-only synthetic integration smoke. NOT a real model experiment."""
import json
from decimal import Decimal
from pathlib import Path

from sc4001.answers import gold_answer
from sc4001.metrics import evaluate_pairs

root = Path(__file__).resolve().parents[1]
pilot_path = root / "data/processed/pilot.jsonl"
if not pilot_path.exists():
    raise SystemExit("Run scripts/prepare_splits.py first")

with pilot_path.open("r", encoding="utf-8", newline="") as stream:
    samples = [json.loads(line) for _, line in zip(range(4), stream)]
assert len(samples) == 4
predictions = []

for i, item in enumerate(samples):
    gold = gold_answer(item["answer"])
    assert isinstance(gold, Decimal)
    for mode in ("fast", "slow"):
        correct = (i == 0) or (i == 1 and mode == "fast") or (i == 2 and mode == "slow")
        number = gold if correct else gold + 1
        predictions.append({
            "id": item["id"],
            "mode": mode,
            "response": f"Final answer: {number}",
            "output_tokens": 10 if mode == "fast" else 40,
            "latency_sec": 0.1 if mode == "fast" else 0.4,
            "truncated": False,
        })

result = evaluate_pairs(samples, predictions)
assert result["paired_counts"] == {
    "both": 1, "fast_only": 1, "slow_only": 1, "neither": 1,
}
assert result["fast_accuracy"] == 0.5
assert result["slow_accuracy"] == 0.5
assert result["oracle_upper_bound"] == 0.75
print("SYNTHETIC / NOT EMPIRICAL RESULTS")
print(json.dumps(result, indent=2))
print("PASS: paired metrics pipeline works without loading Qwen3 or using GPU.")
