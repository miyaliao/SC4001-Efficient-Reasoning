"""Pilot plumbing. Safe to import without torch or model access."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

MODES = ("fast", "slow")
FINAL_INSTRUCTION = (
    "Solve the following math word problem. "
    "Write your final numerical result on its own line in the format "
    "'Final answer: <number>'."
)
VALID_ID = re.compile(r"^gsm8k-train-[0-9]{6}$")


def load_examples(path: Path) -> list[dict]:
    """Use physical JSONL lines; str.splitlines would split Unicode separators."""
    examples, seen = [], set()
    with path.open("r", encoding="utf-8", newline="") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            sid = item.get("id")
            if not isinstance(sid, str) or not VALID_ID.fullmatch(sid):
                raise ValueError(f"Invalid sample id at {line_number}: {sid!r}")
            if sid in seen:
                raise ValueError(f"Duplicate sample ID: {sid}")
            if not isinstance(item.get("question"), str) or not isinstance(item.get("answer"), str):
                raise ValueError(f"Missing question/answer in {sid}")
            seen.add(sid)
            examples.append(item)
    if not examples:
        raise ValueError(f"Empty dataset: {path}")
    return examples


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", prefix=".partial-", delete=False) as f:
            tmp = Path(f.name)
            json.dump(payload, f, ensure_ascii=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", prefix=".partial-", delete=False) as f:
            tmp = Path(f.name)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def run_manifest(config: dict, dataset_file: Path) -> dict:
    if dataset_file.name != "pilot.jsonl":
        raise ValueError("Pilot runner accepts only pilot.jsonl, not test or router data")
    if config.get("modes") != ["fast", "slow"]:
        raise ValueError("Pilot modes must be ['fast', 'slow']")
    if config.get("decode", {}).get("do_sample") is not False:
        raise ValueError("The deterministic pilot requires do_sample=false")
    cap = config.get("decode", {}).get("max_new_tokens")
    if type(cap) is not int or not 256 <= cap <= 16384:
        raise ValueError("max_new_tokens must be an integer in [256, 16384]")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    model = Path(str(config.get("model_path", "")))
    if not model.is_absolute():
        raise ValueError("model_path must be absolute")
    details = {
        "protocol": "sc4001-pilot-v1",
        "dataset_file": str(dataset_file.resolve()),
        "dataset_sha256": sha256_file(dataset_file),
        "model_path": str(model),
        "model_config_sha256": sha256_file(model / "config.json") if (model / "config.json").exists() else None,
        "tokenizer_config_sha256": sha256_file(model / "tokenizer_config.json") if (model / "tokenizer_config.json").exists() else None,
        "modes": config["modes"],
        "seed": config["seed"],
        "decode": config["decode"],
        "final_answer_instruction": FINAL_INSTRUCTION,
    }
    return {"fingerprint": json_hash(details), "settings": details}


def check_manifest(output_dir: Path, manifest: dict, *, create: bool) -> None:
    path = output_dir / "manifest.json"
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError("Run fingerprint mismatch; choose a NEW output directory")
    elif create:
        if (output_dir / "cases").exists() and any((output_dir / "cases").iterdir()):
            raise ValueError("Case checkpoints exist without run manifest")
        atomic_json(path, manifest)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Nonempty run directory without manifest")


def case_path(root: Path, sample_id: str, mode: str) -> Path:
    if not VALID_ID.fullmatch(sample_id) or mode not in MODES:
        raise ValueError("Unsafe sample ID or mode")
    return root / "cases" / f"{sample_id}__{mode}.json"


def load_completed(path: Path, sample_id: str, mode: str, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("id"), record.get("mode"), record.get("fingerprint"), record.get("status")) != (
        sample_id, mode, fingerprint, "ok"
    ):
        raise ValueError(f"Unrecognized or stale case checkpoint: {path}")
    if not isinstance(record.get("response"), str) or type(record.get("output_tokens")) is not int:
        raise ValueError(f"Malformed case checkpoint: {path}")
    return record


def plan(examples: list[dict], output_dir: Path, fingerprint: str, *, limit: int | None = None):
    selected = examples if limit is None else examples[:limit]
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    pending, skipped = [], 0
    for row in selected:
        for mode in MODES:
            if load_completed(case_path(output_dir, row["id"], mode), row["id"], mode, fingerprint):
                skipped += 1
            else:
                pending.append((row["id"], mode, row["question"]))
    return selected, pending, skipped


def export_results(examples: list[dict], output_dir: Path, fingerprint: str, *, complete_only: bool = False):
    """Never export an incomplete set as a final paired dataset."""
    records = []
    for row in examples:
        for mode in MODES:
            value = load_completed(case_path(output_dir, row["id"], mode), row["id"], mode, fingerprint)
            if value is None:
                if complete_only:
                    raise ValueError("Results incomplete; cannot export paired dataset")
                continue
            records.append({k: v for k, v in value.items() if k != "fingerprint"})
    if complete_only:
        atomic_jsonl(output_dir / "predictions.jsonl", records)
    return len(records)
