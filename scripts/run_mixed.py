"""Qwen3 mixed GSM-Plus / MATH-500 paired runner.

Default mode is the 10+10 question smoke; --full is explicit.
Use -m from the repo root with PYTHONPATH=src:. .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc4001.mixed_protocol import (
    MIXED_INSTRUCTION, MODES, case_path, case_seed, completed,
    export_paired, load_mixed, manifest, selected_questions,
)
from sc4001.pilot import atomic_json, check_manifest
from scripts.run_pilot import QwenBackend, check_exclusive_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mixed.yaml")
    parser.add_argument("--output-dir", default="outputs/mixed-boxed-v1")
    parser.add_argument("--smoke-per-benchmark", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="Use all 160 questions; deliberately explicit")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("Require num-shards >= 1 and 0 <= shard-id < num-shards")
    if args.export_only and (not args.full or args.num_shards != 1):
        parser.error("Export only after full run, unsharded: --full --export-only")

    import yaml
    conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_path = Path(conf["dataset"])
    samples = load_mixed(data_path)
    if len(samples) != 160 or sum(r["benchmark"] == "gsm_plus" for r in samples) != 80:
        raise ValueError("Expected frozen mixed evaluation set of 80 GSM-Plus and 80 MATH-500 questions")
    spec = manifest(conf, data_path)
    root = Path(args.output_dir)
    check_manifest(root, spec, create=False)

    focus = selected_questions(samples, None if args.full else args.smoke_per_benchmark)
    shard = [row for i, row in enumerate(focus) if i % args.num_shards == args.shard_id]
    pending = []
    done = 0
    for row in shard:
        for mode in MODES:
            file = case_path(root, row["id"], mode)
            if completed(file, row["id"], mode, spec["fingerprint"]):
                done += 1
            else:
                pending.append((row, mode))
    print(json.dumps({
        "fingerprint": spec["fingerprint"],
        "selection": "full" if args.full else "smoke",
        "selected_questions_total": len(focus),
        "this_shard_questions": len(shard),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "complete_tasks": done,
        "pending_tasks": len(pending),
        "preview": [[row["id"], mode] for row, mode in pending[:6]],
    }, indent=2), flush=True)

    if args.dry_run:
        print("DRY RUN: no torch/model import and no output writes", flush=True)
        return

    if args.export_only:
        count = export_paired(samples, root, spec["fingerprint"])
        print(f"Exported {count} complete paired predictions", flush=True)
        return

    if pending:
        check_exclusive_gpu()
    check_manifest(root, spec, create=True)
    if not pending:
        print("Selected tasks already completed", flush=True)
        return

    backend = QwenBackend(conf["model_path"], conf["seed"])
    for i, (row, mode) in enumerate(pending, 1):
        seed = case_seed(conf["seed"], row["id"], mode)
        output = backend.generate(
            row["question"], mode, conf["decode"], seed,
            instruction=MIXED_INSTRUCTION,
        )
        save = {
            "id": row["id"], "benchmark": row["benchmark"],
            "stratum": row["stratum"], "mode": mode, "status": "ok",
            "fingerprint": spec["fingerprint"], **output,
        }
        atomic_json(case_path(root, row["id"], mode), save)
        print(f"[{i}/{len(pending)}] {row['id']} {mode} "
              f"tokens={output['output_tokens']} truncated={output['truncated']}",
              flush=True)
    print("Shard complete; no automatic export of partial results", flush=True)


if __name__ == "__main__":
    main()
