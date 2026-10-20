"""Offline preparation of a stratified 80+80 evaluation-only set."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path



# Only mark-up IN THE QUESTION is an automatic exclusion.
VISUAL_MARKUP = re.compile(
    r"\[(?:asy|asydef|asybegin)\b|\\begin\s*\{(?:tikzpicture|asy|overpic)\}"
    r"|!\[[^\]]*\]\([^)]+\)|<img\b|<svg\b|\\includegraphics\b",
    re.IGNORECASE,
)
VISUAL_REFERENCE = re.compile(
    r"\b(?:shown|pictured|illustrated|depicted|drawn)\s+"
    r"(?:in\s+)?(?:the\s+)?(?:following|below|above|accompanying)?\s*"
    r"(?:figure|diagram|graph|picture|image)\b"
    r"|\b(?:figure|diagram|graph|picture|image)\s+(?:below|above|shown|provided)\b"
    r"|\b(?:see|refer to|use)\s+(?:the\s+)?(?:figure|diagram|graph|picture|image)\b",
    re.IGNORECASE,
)


def visual_status(question):
    """Return (auto_exclude, requires_manual_review, reason)."""
    if VISUAL_MARKUP.search(question):
        return True, False, "embedded visual markup in question"
    if VISUAL_REFERENCE.search(question):
        return False, True, "question references a visual; check text sufficiency"
    return False, False, ""


def read_rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for name in ("test", "testmini", "data"):
            if isinstance(data.get(name), list):
                return data[name]
    raise ValueError(f"Unsupported file layout: {path}")


def balanced(groups, n, rng):
    groups = {k: list(v) for k,v in groups.items()}
    for pool in groups.values():
        rng.shuffle(pool)
    result = []
    while len(result) < n:
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(result) < n:
                result.append(groups[key].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"Only {len(result)} usable rows, requested {n}")
    return result


def build(gsm_rows, math_rows, per_dataset=80, seed=42):
    rng = random.Random(seed)
    by_kind = defaultdict(list)
    seen_seeds = set()
    shuffled = list(enumerate(gsm_rows))
    rng.shuffle(shuffled)
    for i, row in shuffled:
        kind = str(row.get("perturbation_type", "")).strip()
        base = str(row.get("seed_question", "")).strip()
        if not kind or not base or not row.get("question") or row.get("answer") is None:
            continue
        if "critical" in kind.lower():
            continue
        if base in seen_seeds:
            continue
        seen_seeds.add(base)
        by_kind[kind].append({
            "id": f"gsmplus-{i:06d}", "benchmark": "gsm_plus", "stratum": kind,
            "question": row["question"], "answer": str(row["answer"]),
            "seed_question": base, "source_index": i
        })
    gsm = balanced(by_kind, per_dataset, rng)
    by_level = defaultdict(list)
    for i, row in enumerate(math_rows):
        if not row.get("problem") or row.get("answer") is None:
            continue
        excluded, review, reason = visual_status(row["problem"])
        if excluded:
            continue
        by_level[str(row.get("level", "unknown"))].append({
            "id": f"math500-{i:04d}", "benchmark": "math_500",
            "stratum": str(row.get("level", "unknown")),
            "topic": row.get("subject", row.get("type")),
            "unique_id": row.get("unique_id"),
            "visual_review_required": review, "visual_review_reason": reason, "question": row["problem"],
            "answer": str(row["answer"]), "source_index": i,
        })
    return gsm + balanced(by_level, per_dataset, rng)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gsm-plus", type=Path, required=True)
    p.add_argument("--math-500", type=Path, required=True)
    p.add_argument("--per-dataset", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path,
                   default=Path("data/processed/mixed_eval_v1.jsonl"))
    a = p.parse_args()
    if a.per_dataset < 1:
        p.error("--per-dataset must be >=1")
    rows = build(read_rows(a.gsm_plus), read_rows(a.math_500), a.per_dataset, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    review_rows = [r for r in rows if r.get("visual_review_required")]
    review_path = a.output.with_suffix(".visual_review.json")
    review_path.write_text(json.dumps(review_rows, indent=2, ensure_ascii=False)+"\n")
    kinds = Counter((r["benchmark"], r["stratum"]) for r in rows)
    manifest = {"seed": a.seed, "n": len(rows),
                "gsmplus_sha256": sha(a.gsm_plus), "math500_sha256": sha(a.math_500),
                "output_sha256": sha(a.output),
                "visual_review_count": len(review_rows),
                "visual_review_path": str(review_path),
                "strata": {f"{k[0]}:{k[1]}":v for k,v in sorted(kinds.items())},
                "policy": "evaluation only; one GSM-Plus variant per seed; critical thinking and embedded-visual MATH questions excluded; visual references require manual review"}
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
