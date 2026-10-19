"""Reproducibly sampled Qwen3-4B pilot; --dry-run does not import torch."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from sc4001.pilot import (
    FINAL_INSTRUCTION, atomic_json, case_path, check_manifest,
    case_seed, export_results, load_examples, plan, run_manifest,
)


def check_exclusive_gpu() -> None:
    """Refuse to start inference if selected GPU has any existing compute job."""
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical.isdigit():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly ONE free physical GPU index")
    try:
        gpu_data = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, timeout=10,
        )
        app_data = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot verify GPU exclusivity; refusing inference") from exc
    identities = [line.split(",", 1) for line in gpu_data.splitlines() if "," in line]
    matching = [uuid.strip() for index, uuid in identities if index.strip() == physical]
    if len(matching) != 1:
        raise RuntimeError(f"Could not uniquely identify physical GPU {physical}")
    used = [
        line for line in app_data.splitlines()
        if line.split(",", 1)[0].strip() == matching[0]
    ]
    if used:
        raise RuntimeError(f"GPU {physical} already has {len(used)} compute process(es); refusing to start")


class QwenBackend:
    def __init__(self, model_path: str, seed: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.manual_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        ).to("cuda:0").eval()

    def generate(self, question: str, mode: str, decode: dict, sample_seed: int) -> dict:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": FINAL_INSTRUCTION + "\n\n" + question}],
            tokenize=True, add_generation_prompt=True,
            enable_thinking=(mode == "slow"), return_tensors="pt",
        ).to("cuda:0")
        max_new_tokens = decode["max_new_tokens"]
        params = decode["sampling"][mode]
        # Per-case seeding guarantees resumable sampling: order does not change a case.
        self.torch.manual_seed(sample_seed)
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                prompt, do_sample=True, max_new_tokens=max_new_tokens,
                temperature=params["temperature"], top_p=params["top_p"],
                top_k=params["top_k"], min_p=params["min_p"],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = output[0, prompt.shape[-1]:].tolist()
        end_ids = self.model.generation_config.eos_token_id
        if isinstance(end_ids, int):
            end_ids = [end_ids]
        if end_ids is None:
            end_ids = [self.tokenizer.eos_token_id]
        ended = bool(generated and generated[-1] in end_ids)
        truncated = len(generated) >= max_new_tokens and not ended
        return {
            "response": self.tokenizer.decode(generated, skip_special_tokens=False),
            "prompt_tokens": int(prompt.shape[-1]),
            "sample_seed": sample_seed,
            "sampling_params": dict(params),
            "output_tokens": len(generated),
            "latency_sec": elapsed,
            "finish_reason": "eos" if ended else "length" if truncated else "other",
            "truncated": truncated,
        }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", default="configs/pilot.yaml")
    cli.add_argument("--output-dir", default="outputs/pilot-v2")
    cli.add_argument("--dry-run", action="store_true", help="Plan only; no torch/model import or artifact writes")
    cli.add_argument("--limit", type=int, default=None, help="First N questions for smoke test")
    cli.add_argument("--export-only", action="store_true", help="Export complete 200-question paired results")
    cli.add_argument("--num-shards", type=int, default=1, help="Number of disjoint question shards")
    cli.add_argument("--shard-id", type=int, default=0, help="Zero-based shard index")
    args = cli.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        cli.error("Require num-shards >= 1 and 0 <= shard-id < num-shards")
    if args.export_only and (args.num_shards != 1 or args.shard_id != 0):
        cli.error("Use unsharded --export-only after all workers complete")

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Install requirements-eval.txt (PyYAML) first") from exc

    conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = Path(conf["dataset"])
    samples = load_examples(dataset)
    if len(samples) != 200:
        raise ValueError("Pilot dataset must contain exactly 200 GSM8K training questions")
    manifest = run_manifest(conf, dataset)
    output_dir = Path(args.output_dir)
    check_manifest(output_dir, manifest, create=False)
    selected = samples if args.limit is None else samples[:args.limit]
    shard_samples = [sample for i, sample in enumerate(selected) if i % args.num_shards == args.shard_id]
    selection, pending, skipped = plan(shard_samples, output_dir, manifest["fingerprint"])
    print(json.dumps({
        "mode": "dry-run" if args.dry_run else "export" if args.export_only else "inference",
        "fingerprint": manifest["fingerprint"],
        "selected_questions": len(selection),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_tasks": len(selection) * 2,
        "complete_tasks": skipped,
        "pending_tasks": len(pending),
        "preview": [list(task[:2]) for task in pending[:8]],
    }, indent=2))

    if args.dry_run:
        print("DRY RUN: no GPU used, no model loaded, no artifacts written")
        return

    if args.export_only:
        if args.limit is not None:
            raise ValueError("Export requires the whole 200-question pilot")
        n = export_results(samples, output_dir, manifest["fingerprint"], complete_only=True)
        print(f"Exported {n} verified paired predictions")
        return

    if pending:
        check_exclusive_gpu()
    check_manifest(output_dir, manifest, create=True)
    if not pending:
        print("Requested tasks complete; --export-only exports the full 200-question run")
        return

    backend = QwenBackend(conf["model_path"], conf["seed"])
    for i, (sid, mode, question) in enumerate(pending, 1):
        output = backend.generate(question, mode, conf["decode"], case_seed(conf["seed"], sid, mode))
        record = {
            "id": sid, "mode": mode, "status": "ok",
            "fingerprint": manifest["fingerprint"], **output,
        }
        atomic_json(case_path(output_dir, sid, mode), record)
        print(f"[{i}/{len(pending)}] {sid} {mode}: "
              f"tokens={output['output_tokens']} truncated={output['truncated']}", flush=True)

    if args.limit is None and args.num_shards == 1:
        export_results(samples, output_dir, manifest["fingerprint"], complete_only=True)
        print("Full pilot exported; ready for evaluation")
    else:
        print("Shard/partial run complete; use unsharded --export-only after all workers finish")


if __name__ == "__main__":
    main()
