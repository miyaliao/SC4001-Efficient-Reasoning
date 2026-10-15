"""Audit that development splits are disjoint, complete, and match GSM8K train."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"pilot": 200, "router_train": 5000, "router_val": 1000, "internal_holdout": 1273}


def main():
    source = ROOT / "data/raw/gsm8k/train.parquet"
    rows = pq.read_table(source, columns=["question", "answer"]).to_pylist()
    assert len(rows) == 7473
    out_dir = ROOT / "data/processed"
    manifest = json.loads((out_dir / "split_manifest.json").read_text())
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    seen = set()
    for split, count in EXPECTED.items():
        path = out_dir / f"{split}.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert len(records) == count, (split, len(records))
        ids = []
        for item in records:
            sample_id = item["id"]
            assert sample_id.startswith("gsm8k-train-")
            index = int(sample_id.split("-")[-1])
            assert 0 <= index < len(rows)
            assert sample_id not in seen, f"Duplicate sample: {sample_id}"
            seen.add(sample_id)
            assert item["question"] == rows[index]["question"]
            assert item["answer"] == rows[index]["answer"]
            ids.append(sample_id)
        actual_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        assert actual_hash == manifest["splits"][split]["id_sha256"]
        print(f"{split}: {count} samples; source and hash PASS")
    expected = {f"gsm8k-train-{i:06d}" for i in range(7473)}
    assert seen == expected, f"Coverage mismatch; missing={len(expected - seen)}"
    print("PASS: all 7473 train records covered exactly once")
    print("Official test was not read.")


if __name__ == "__main__":
    main()
