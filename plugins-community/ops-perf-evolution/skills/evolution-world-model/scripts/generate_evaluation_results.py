#!/usr/bin/env python3
"""
Generate evaluation_results.json from ops-profiling performance.json
for a lingxi-partial variant directory.
"""
import argparse
import json
import os
import sys


def build(perf_path: str, variant_dir: str, implementation_note: str = "",
          local_refinement_rounds: int = 0, local_refinement_gain: float = 1.0) -> dict:
    with open(perf_path, "r", encoding="utf-8") as f:
        perf = json.load(f)

    per_case = perf.get("per_case", [])
    valid = [c for c in per_case if c.get("speedup") is not None]
    n_valid = len(valid)
    n_total = len(per_case)

    if n_valid > 0:
        speedup = sum(c["speedup"] for c in valid) / n_valid
        base_time_ms = sum(c.get("ref_us", 0) for c in valid) / n_valid / 1000.0
        gen_time_ms = sum(c.get("asc_us", 0) for c in valid) / n_valid / 1000.0
    else:
        speedup = 0.0
        base_time_ms = 0.0
        gen_time_ms = 0.0

    # Determine precision: ops-profiling sets speedup None for failed cases.
    # We treat precision_passed true only if all cases produced a speedup.
    precision_passed = n_valid == n_total and n_total > 0
    compilation_success = n_valid > 0

    # Build per_case list normalized to evaluation schema
    eval_per_case = []
    for c in per_case:
        eval_per_case.append({
            "case": c.get("case", -1),
            "shape": str(c.get("shape", [])),
            "dtype": c.get("dtype", "unknown"),
            "ref_us": c.get("ref_us", 0.0),
            "asc_us": c.get("asc_us", 0.0),
            "speedup": c.get("speedup", 0.0) if c.get("speedup") is not None else None
        })

    return {
        "comparison": {
            "compilation_success": compilation_success,
            "precision_passed": precision_passed,
            "speedup": speedup,
            "measurement_quality": "medium",
            "ref_base": "pytorch",
            "note": implementation_note,
            "n_cases_total": n_total,
            "n_cases_valid": n_valid
        },
        "evolved": {
            "pipeline": None,
            "pipeline_note": "Quick-mode profiling on NPU 8 (no pipeline metrics)."
        },
        "base_time_ms": base_time_ms,
        "gen_time_ms": gen_time_ms,
        "local_refinement_rounds": local_refinement_rounds,
        "local_refinement_gain": local_refinement_gain,
        "implementation_note": implementation_note,
        "per_case": eval_per_case
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-json", required=True)
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument("--implementation-note", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = build(args.perf_json, args.variant_dir, args.implementation_note)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] {args.output}  n_valid={result['comparison']['n_cases_valid']}/{result['comparison']['n_cases_total']}  speedup={result['comparison']['speedup']:.3f}x")


if __name__ == "__main__":
    main()
