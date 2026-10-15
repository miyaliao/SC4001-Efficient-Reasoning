"""Strict paired Fast/Slow evaluation."""
from statistics import mean

from sc4001.answers import gold_answer, predicted_answer


def evaluate_pairs(examples, predictions):
    if not examples:
        raise ValueError("Empty dataset")
    by_id = {row["id"]: row for row in examples}
    if len(by_id) != len(examples):
        raise ValueError("Duplicate dataset IDs")

    preds = {}
    for row in predictions:
        key = (row["id"], row["mode"])
        if row["id"] not in by_id or row["mode"] not in ("fast", "slow"):
            raise ValueError(f"Unknown sample/mode: {key}")
        if key in preds:
            raise ValueError(f"Duplicate prediction: {key}")
        preds[key] = row

    expected = {(sample_id, mode) for sample_id in by_id for mode in ("fast", "slow")}
    if set(preds) != expected:
        missing = expected - set(preds)
        extra = set(preds) - expected
        raise ValueError(f"Incomplete predictions (missing={len(missing)}, extra={len(extra)})")

    counts = {"both": 0, "fast_only": 0, "slow_only": 0, "neither": 0}
    invalid = {"fast": 0, "slow": 0}
    truncated = {"fast": 0, "slow": 0}
    tokens = {"fast": [], "slow": []}
    latency = {"fast": [], "slow": []}

    for sample_id, example in by_id.items():
        answer = gold_answer(example["answer"])
        if answer is None:
            raise ValueError(f"Unparseable gold answer: {sample_id}")
        correct = {}
        for mode in ("fast", "slow"):
            row = preds[(sample_id, mode)]
            prediction = predicted_answer(row.get("response"))
            invalid[mode] += prediction is None
            correct[mode] = prediction is not None and prediction == answer
            truncated[mode] += bool(row.get("truncated", False))
            if row.get("output_tokens") is not None:
                value = int(row["output_tokens"])
                if value < 0:
                    raise ValueError("Negative output_tokens")
                tokens[mode].append(value)
            if row.get("latency_sec") is not None:
                value = float(row["latency_sec"])
                if value < 0:
                    raise ValueError("Negative latency_sec")
                latency[mode].append(value)

        fast, slow = correct["fast"], correct["slow"]
        category = "both" if fast and slow else "fast_only" if fast else "slow_only" if slow else "neither"
        counts[category] += 1

    n = len(examples)
    return {
        "n": n,
        "paired_counts": counts,
        "fast_accuracy": (counts["both"] + counts["fast_only"]) / n,
        "slow_accuracy": (counts["both"] + counts["slow_only"]) / n,
        "oracle_upper_bound": (n - counts["neither"]) / n,
        "invalid_final_answer": invalid,
        "truncated": truncated,
        "mean_output_tokens": {k: mean(v) if len(v) == n else None for k, v in tokens.items()},
        "mean_latency_sec": {k: mean(v) if len(v) == n else None for k, v in latency.items()},
    }
