"""Strict boxed-answer evaluation for GSM-Plus + MATH-500 (CPU-only).

Does NOT accept unanchored intermediate calculations as a final answer.
Run --gold-audit-only before GPU inference, then smoke and full evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from sc4001.mixed_protocol import (
    MODES, case_path, completed, load_mixed, manifest, selected_questions,
)
from sc4001.pilot import atomic_json, check_manifest

FINAL_BOX = re.compile(r"Final\s+answer\s*:\s*(\\boxed\s*\{)", flags=re.I)
QWEN_TOKEN = re.compile(r"<\|[^>]+\|>")


def after_thinking(response: str) -> str:
    if "</think>" in response:
        return response.rsplit("</think>", 1)[1]
    if "<think>" in response:
        return ""
    return response


def boxed_final(response: str) -> tuple[str | None, str]:
    """Return full balanced boxed expression and status.

    The sole Final answer label must contain a complete boxed expression
    at the end of the post-thinking response. Handles nested LaTeX braces.
    """
    text = QWEN_TOKEN.sub("", after_thinking(response))
    if not text.strip():
        return None, "missing_final"
    hits = list(FINAL_BOX.finditer(text))
    if len(hits) != 1:
        return None, "missing_or_multiple_final_labels"
    start = hits[0].start(1)
    opened = text.find("{", start)
    depth = 0
    end = None
    for j in range(opened, len(text)):
        if text[j] == "{" and (j == 0 or text[j - 1] != "\\"):
            depth += 1
        elif text[j] == "}" and (j == 0 or text[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end is None:
        return None, "unclosed_box"
    if text[end:].strip():
        return None, "trailing_content"
    return text[start:end].replace(" ", "", 1) if False else text[start:end], "ok"


def math_verify_objects(gold: str, extracted: str):
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig

    cfg = [LatexExtractionConfig(boxed_match_priority=0)]
    # For the numerical GSM-Plus subset and symbolic MATH-500 golds alike,
    # a boxed gold expression can be parsed under the same LaTeX contract.
    reference = parse(r"\boxed{" + str(gold) + "}", extraction_config=cfg, fallback_mode="no_fallback")
    candidate = parse(extracted, extraction_config=cfg, fallback_mode="no_fallback")
    return reference, candidate, bool(reference and candidate and verify(reference, candidate))


def audit_gold(rows: list[dict]) -> list[dict]:
    from math_verify import parse
    from math_verify.parser import LatexExtractionConfig
    config = [LatexExtractionConfig(boxed_match_priority=0)]
    errors = []
    for row in rows:
        gold = str(row["answer"])
        parsed = parse(r"\boxed{" + gold + "}", extraction_config=config,
                       fallback_mode="no_fallback")
        if not parsed:
            errors.append({"id": row["id"], "benchmark": row["benchmark"],
                           "stratum": row["stratum"], "gold": gold})
    return errors


def summarize(rows, output_root, fingerprint):
    examples = []
    by_group = defaultdict(list)
    for item in rows:
        result = {"id": item["id"], "benchmark": item["benchmark"],
                  "stratum": item["stratum"], "gold": str(item["answer"])}
        correctness = {}
        for mode in MODES:
            ckpt = completed(case_path(output_root, item["id"], mode),
                             item["id"], mode, fingerprint)
            if ckpt is None:
                raise ValueError(f"Missing {item['id']} {mode}; run generation first")
            boxed, status = boxed_final(ckpt["response"])
            reference, candidate, correct = (math_verify_objects(item["answer"], boxed)
                                              if boxed is not None else ([], [], False))
            record = {"correct": bool(correct), "format_ok": status == "ok",
                      "format_status": status, "parse_ok": bool(candidate),
                      "predicted": [str(x) for x in candidate],
                      "truncated": bool(ckpt.get("truncated")),
                      "tokens": ckpt.get("output_tokens"),
                      "latency_sec": ckpt.get("latency_sec")}
            result[mode] = record
            correctness[mode] = bool(correct)
        result["paired_category"] = ("both" if all(correctness.values()) else
                                      "fast_only" if correctness["fast"] else
                                      "slow_only" if correctness["slow"] else "neither")
        examples.append(result)
        by_group["overall"].append(result)
        by_group[item["benchmark"]].append(result)
        by_group[f"{item['benchmark']}:{item['stratum']}"].append(result)

    def aggregation(items):
        out = {"n": len(items), "paired_counts": dict(Counter(x["paired_category"] for x in items))}
        for mode in MODES:
            r = [x[mode] for x in items]
            out[mode] = {
                "accuracy": sum(v["correct"] for v in r) / len(r),
                "format_compliance": sum(v["format_ok"] for v in r) / len(r),
                "parsed_count": sum(v["parse_ok"] for v in r),
                "truncated_count": sum(v["truncated"] for v in r),
                "format_failure_reasons": dict(Counter(v["format_status"] for v in r if not v["format_ok"])),
                "mean_output_tokens": mean(v["tokens"] for v in r) if all(v["tokens"] is not None for v in r) else None,
                "mean_latency_sec": mean(v["latency_sec"] for v in r) if all(v["latency_sec"] is not None for v in r) else None,
            }
        return out

    return {"n": len(rows), "fingerprint": fingerprint, "protocol": "mixed-boxed-v1",
            "groups": {key: aggregation(value) for key, value in sorted(by_group.items())},
            "per_case": examples,
            "warning": "Smoke is diagnostic, not final test. Inspect parse failures and mathematical equivalence disputes."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mixed.yaml")
    ap.add_argument("--output-dir", default="outputs/mixed-boxed-v1")
    ap.add_argument("--smoke-per-benchmark", type=int, default=10)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-audit-only", action="store_true")
    args = ap.parse_args()

    import yaml
    conf = yaml.safe_load(Path(args.config).read_text())
    data_path = Path(conf["dataset"])
    rows = load_mixed(data_path)
    focus = selected_questions(rows, None if args.full or args.gold_audit_only else args.smoke_per_benchmark)
    errors = audit_gold(focus)
    if errors:
        print(json.dumps({"gold_parse_failures": errors}, indent=2, ensure_ascii=False))
        raise SystemExit(f"Gold audit failed for {len(errors)} entries: review before GPU inference")
    print(f"Gold parse audit PASS: {len(focus)} answers", flush=True)
    if args.gold_audit_only:
        return

    spec = manifest(conf, data_path)
    root = Path(args.output_dir)
    check_manifest(root, spec, create=False)
    result = summarize(focus, root, spec["fingerprint"])
    filename = root / ("full_summary.json" if args.full else "smoke_summary.json")
    atomic_json(filename, result)
    print(json.dumps(result["groups"]["overall"], indent=2))
    print("Saved complete grouped and per-case results:", filename)


if __name__ == "__main__":
    main()
