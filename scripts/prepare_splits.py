"""Create deterministic, disjoint development splits from GSM8K train only."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SPLITS = [
    ("pilot", 200),
    ("router_train", 5000),
    ("router_val", 1000),
    ("internal_holdout", 1273),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = ROOT / "data/raw/gsm8k/train.parquet"
    rows = pq.read_table(source, columns=["question", "answer"]).to_pylist()
    if len(rows) != 7473:
        raise ValueError(f"Expected 7473 GSM8K train examples, got {len(rows)}")

    ids = list(range(len(rows)))
    random.Random(args.seed).shuffle(ids)
    destination = ROOT / "data/processed"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": args.seed,
        "splits": {},
    }
    offset = 0
    for name, count in SPLITS:
        indices = ids[offset:offset + count]
        offset += count
        entries = [{"id": f"gsm8k-train-{i:06d}", **rows[i]} for i in indices]
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for entry in entries:
                out.write(json.dumps(entry, ensure_ascii=True) + "\n")
        ids_text = "\n".join(entry["id"] for entry in entries)
        manifest["splits"][name] = {
            "count": count,
            "id_sha256": hashlib.sha256(ids_text.encode("utf-8")).hexdigest(),
        }
        print(f"{name}: {count}")
    assert offset == len(rows)
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote development splits to data/processed (excluded from Git)")


if __name__ == "__main__":
    main()
