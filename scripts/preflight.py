"""CPU-only validation of locally downloaded datasets."""
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

for name, expected in (("train", 7473), ("test", 1319)):
    source = ROOT / "data/raw/gsm8k" / (name + ".parquet")
    if not source.is_file():
        raise SystemExit(f"Missing dataset file: {source}")
    table = pq.read_table(source, columns=["question", "answer"])
    assert table.num_rows == expected, (name, table.num_rows)
    rows = table.to_pylist()
    assert all(isinstance(x["question"], str) and isinstance(x["answer"], str) for x in rows)
    print(f"GSM8K {name}: {len(rows)} records PASS")

svamp = ROOT / "data/raw/svamp/SVAMP.json"
if not svamp.is_file():
    raise SystemExit(f"Missing SVAMP file: {svamp}")
data = json.loads(svamp.read_text(encoding="utf-8"))
assert len(data) == 1000 and all("Answer" in x and "Question" in x for x in data)
print("SVAMP: 1000 records PASS")
print("CPU preflight PASS (no model inference performed)")
