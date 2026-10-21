"""Selective Slow truncation-recovery experiment (GPU only when --run).

Source cohort is frozen from completed 4096-token paired baseline.
Independent checkpoint fingerprints per budget; no baseline files are changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from sc4001.mixed_protocol import MIXED_INSTRUCTION, case_seed, load_mixed, manifest, case_path, completed
from sc4001.pilot import atomic_json, check_manifest
from scripts.run_pilot import QwenBackend, check_exclusive_gpu
from sc4001.boxed_scoring import canonical_box
from scripts.evaluate_mixed_canonical import math_grade

def cohort(rows, baseline_root, baseline_manifest):
    source = []
    for row in rows:
        p=completed(case_path(baseline_root,row["id"],"slow"),row["id"],"slow",baseline_manifest["fingerprint"])
        if p is None:
            raise ValueError(f"Baseline is incomplete: {row['id']}")
        if p["truncated"]:
            source.append(row)
    return source

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-config",default="configs/mixed.yaml")
    p.add_argument("--baseline-dir",default="outputs/mixed-boxed-v1")
    p.add_argument("--budget",type=int,choices=(8192,16384,32768),default=8192)
    p.add_argument("--output-dir",default=None)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--evaluate-only",action="store_true")
    p.add_argument("--num-shards",type=int,default=1)
    p.add_argument("--shard-id",type=int,default=0)
    args=p.parse_args()
    if args.num_shards<1 or not 0<=args.shard_id<args.num_shards:
        p.error("Invalid shard settings")
    import yaml
    base_config=yaml.safe_load(Path(args.baseline_config).read_text())
    if base_config["decode"]["max_new_tokens"]!=4096:
        raise ValueError("Baseline config must be the original 4096-token one")
    data_file=Path(base_config["dataset"])
    rows=load_mixed(data_file)
    if len(rows)!=160:
        raise ValueError("Expected 160-question baseline")
    original=manifest(base_config,data_file)
    baseline_root=Path(args.baseline_dir)
    check_manifest(baseline_root,original,create=False)
    selected=cohort(rows,baseline_root,original)
    if len(selected)!=25:
        raise ValueError(f"Expected frozen 25-case baseline truncation cohort; found {len(selected)}")
    adjusted=json.loads(json.dumps(base_config))
    adjusted["decode"]["max_new_tokens"]=args.budget
    spec=manifest(adjusted,data_file)
    root=Path(args.output_dir or f"outputs/recovery-slow-{args.budget}")
    check_manifest(root,spec,create=False)
    shard=[r for i,r in enumerate(selected) if i%args.num_shards==args.shard_id]
    pending=[r for r in shard if completed(case_path(root,r["id"],"slow"),r["id"],"slow",spec["fingerprint"]) is None]
    print(json.dumps({"baseline_cohort":len(selected),"budget":args.budget,
                      "fingerprint":spec["fingerprint"],"selected_shard":len(shard),
                      "complete_tasks":len(shard)-len(pending),"pending_tasks":len(pending),
                      "preview":[r["id"] for r in pending[:8]]},indent=2),flush=True)
    if args.dry_run:return
    if not args.evaluate_only:
        if pending:check_exclusive_gpu()
        check_manifest(root,spec,create=True)
        if pending:
            backend=QwenBackend(adjusted["model_path"],adjusted["seed"])
            for index,row in enumerate(pending,1):
                out=backend.generate(row["question"],"slow",adjusted["decode"],
                                     case_seed(adjusted["seed"],row["id"],"slow"),
                                     instruction=MIXED_INSTRUCTION)
                record={"id":row["id"],"benchmark":row["benchmark"],
                        "stratum":row["stratum"],"mode":"slow","status":"ok",
                        "fingerprint":spec["fingerprint"],**out}
                atomic_json(case_path(root,row["id"],"slow"),record)
                print(f"[{index}/{len(pending)}] {row['id']} slow "
                      f"tokens={out['output_tokens']} truncated={out['truncated']}",flush=True)
        return
    if pending:
        raise ValueError(f"Recovery outputs incomplete: {len(pending)} pending in shard")
    if args.num_shards!=1:
        raise ValueError("Evaluate without shard options after all shards complete")
    per_case=[]
    for row in selected:
        saved=completed(case_path(root,row["id"],"slow"),row["id"],"slow",spec["fingerprint"])
        if saved is None:raise ValueError(f"Missing recovery output: {row['id']}")
        box,status,_=canonical_box(saved["response"])
        correct,parsed,values,percent=(math_grade(str(row["answer"]),box,row["question"])
                                      if box is not None else (False,False,[],False))
        per_case.append({"id":row["id"],"benchmark":row["benchmark"],"stratum":row["stratum"],
                         "gold":str(row["answer"]),"task_success":correct,"canonical_format_ok":status=="ok",
                         "format_status":status,"math_parse_ok":parsed,"prediction":values,
                         "percent_normalized":percent,"truncated":bool(saved["truncated"]),
                         "output_tokens":saved["output_tokens"],"latency_sec":saved["latency_sec"]})
    groups={}
    for name in ("overall","gsm_plus","math_500"):
        records=per_case if name=="overall" else [x for x in per_case if x["benchmark"]==name]
        groups[name]={"n":len(records),"success_count":sum(x["task_success"] for x in records),
                      "truncated_count":sum(x["truncated"] for x in records),
                      "format_count":sum(x["canonical_format_ok"] for x in records),
                      "mean_tokens":sum(x["output_tokens"] for x in records)/len(records),
                      "mean_latency_sec":sum(x["latency_sec"] for x in records)/len(records)}
    report={"protocol":"selective-slow-recovery-v1","budget":args.budget,
            "baseline_fingerprint":original["fingerprint"],"recovery_fingerprint":spec["fingerprint"],
            "note":"Selected on baseline truncation: conditional diagnostic only, NOT all-160 accuracy.",
            "groups":groups,"per_case":per_case}
    atomic_json(root/"recovery_summary.json",report)
    print(json.dumps(groups,indent=2))
    print(root/"recovery_summary.json")

if __name__=="__main__":main()
