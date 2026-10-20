"""Independent mixed-benchmark experiment protocol; no GPU needed to import."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from sc4001.pilot import atomic_jsonl, check_manifest, json_hash, sha256_file

MIXED_INSTRUCTION = (
    r"Solve the following math problem carefully. "
    r"You may show reasoning, but your final answer MUST be on exactly one final line "
    r"in the format: Final answer: \boxed{YOUR_ANSWER}. "
    r"Replace YOUR_ANSWER with the actual mathematical expression in valid LaTeX. "
    r"Do not put words or units inside the box. "
    r"Do not output anything after that final-answer line."
)
MODES = ("fast", "slow")
IDS = re.compile(r"^(gsmplus-\d{6}|math500-\d{4})$")


def load_mixed(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("id")
            if not isinstance(sid, str) or not IDS.fullmatch(sid):
                raise ValueError(f"Invalid mixed sample ID at line {lineno}: {sid}")
            if sid in seen:
                raise ValueError(f"Duplicate sample ID: {sid}")
            if row.get("benchmark") not in ("gsm_plus", "math_500"):
                raise ValueError(f"Unsupported benchmark: {sid}")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"Missing question: {sid}")
            if row.get("answer") is None or not str(row["answer"]).strip():
                raise ValueError(f"Missing gold: {sid}")
            if row.get("visual_review_required"):
                raise ValueError(f"Unreviewed visual-dependent problem: {sid}")
            seen.add(sid)
            rows.append(row)
    if not rows:
        raise ValueError("Empty mixed evaluation set")
    return rows


def selected_questions(rows: list[dict], per_benchmark: int | None) -> list[dict]:
    if per_benchmark is None:
        return list(rows)
    if per_benchmark < 1:
        raise ValueError("Smoke sample size must be positive")
    seen = {"gsm_plus": 0, "math_500": 0}
    result = []
    for row in rows:
        name = row["benchmark"]
        if seen[name] < per_benchmark:
            result.append(row)
            seen[name] += 1
    if any(n != per_benchmark for n in seen.values()):
        raise ValueError(f"Smoke requires {per_benchmark} questions from each benchmark: {seen}")
    return result


def case_seed(seed: int, sid: str, mode: str) -> int:
    if type(seed) is not int or not IDS.fullmatch(sid) or mode not in MODES:
        raise ValueError("Invalid seed, sample ID, or mode")
    digest = hashlib.sha256(f"mixed-v1:{seed}:{sid}:{mode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def case_path(root: Path, sid: str, mode: str) -> Path:
    if not IDS.fullmatch(sid) or mode not in MODES:
        raise ValueError("Invalid mixed case path")
    return root / "cases" / f"{sid}__{mode}.json"


def manifest(config: dict, data_path: Path) -> dict:
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        raise ValueError("model_path must be absolute")
    decode = config["decode"]
    if decode.get("do_sample") is not True:
        raise ValueError("Sampled pilot requires do_sample=true")
    for mode in MODES:
        params = decode["sampling"][mode]
        if not {"temperature", "top_p", "top_k", "min_p"} <= set(params):
            raise ValueError(f"Missing sampling parameters for {mode}")
    if type(decode.get("max_new_tokens")) is not int or decode["max_new_tokens"] < 256:
        raise ValueError("Invalid max_new_tokens")
    settings = {
        "protocol": "mixed-boxed-v1",
        "data_sha256": sha256_file(data_path),
        "data_file": str(data_path.resolve()),
        "model_path": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json") if (model_path / "config.json").is_file() else None,
        "tokenizer_config_sha256": sha256_file(model_path / "tokenizer_config.json") if (model_path / "tokenizer_config.json").is_file() else None,
        "generation_config_sha256": sha256_file(model_path / "generation_config.json") if (model_path / "generation_config.json").is_file() else None,
        "seed": config["seed"],
        "seed_scheme": "mixed-v1 sha256 per sample and mode",
        "decode": decode,
        "prompt": MIXED_INSTRUCTION,
    }
    return {"fingerprint": json_hash(settings), "settings": settings}


def completed(path: Path, sid: str, mode: str, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("id"), row.get("mode"), row.get("fingerprint"), row.get("status")) != (
        sid, mode, fingerprint, "ok"
    ):
        raise ValueError(f"Stale or invalid checkpoint: {path}")
    if not isinstance(row.get("response"), str) or type(row.get("output_tokens")) is not int:
        raise ValueError(f"Malformed checkpoint: {path}")
    return row


def export_paired(rows: list[dict], root: Path, fingerprint: str) -> int:
    output = []
    for row in rows:
        for mode in MODES:
            saved = completed(case_path(root, row["id"], mode), row["id"], mode, fingerprint)
            if saved is None:
                raise ValueError(f"Missing completed task: {row['id']} {mode}")
            output.append(saved)
    atomic_jsonl(root / "predictions.jsonl", output)
    return len(output)
