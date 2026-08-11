#!/usr/bin/env python3
"""evolution-knowledge（query CLI）— 轻量结构化 OKF 查询。

只读操作。从 skills/evolution-strategies/references/INDEX.json 中读取 OKF 卡片元数据，
按 kind / op_family / bottleneck / phenomenon / tags / keyword 过滤，输出 JSON。
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
INDEX_PATH = ROOT / "skills" / "evolution-strategies" / "references" / "INDEX.json"


def _load_index() -> dict:
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _matches(card: dict, kind: str | None, op_family: str | None,
             bottleneck: str | None, phenomenon: str | None,
             failed_combination: list[str] | None, status: str | None,
             keyword: str | None) -> bool:
    if kind and card.get("kind") != kind:
        return False
    if status and status != "all" and card.get("status") != status:
        return False
    if op_family and op_family not in (card.get("op_families") or []):
        return False
    if bottleneck and bottleneck not in (card.get("bottlenecks") or []):
        return False
    if phenomenon and card.get("phenomenon") != phenomenon:
        return False
    if failed_combination:
        fc = card.get("failed_combination") or []
        if not set(failed_combination).issubset(set(fc)):
            return False
    if keyword:
        hay = " ".join([
            str(card.get(k, "")) for k in
            ["id", "title", "description", "path", "tags"]
        ]).lower()
        if keyword.lower() not in hay:
            return False
    return True


def _sufficiency(results: list[dict]) -> bool:
    active = [r for r in results if r.get("status") == "active"]
    verified = [r for r in active if r.get("confidence") == "cross_op_verified"]
    single = [r for r in active if r.get("confidence") == "single_op"]
    return bool(verified) or len(single) >= 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Query ops-perf-evolution OKF KB")
    parser.add_argument("--kind", choices=["discovered_strategy", "proven_solution", "failure_pattern"])
    parser.add_argument("--op-family")
    parser.add_argument("--bottleneck")
    parser.add_argument("--phenomenon")
    parser.add_argument("--failed-combination", help="comma-separated strategy IDs")
    parser.add_argument("--status", default="active")
    parser.add_argument("--keyword")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    args = parser.parse_args()

    index = json.loads(args.index_path.read_text(encoding="utf-8"))
    cards = index.get("cards", [])
    okf_cards = [c for c in cards if c.get("source_key", "").startswith("evolution-knowledge#okf/")]

    fc = [s.strip() for s in args.failed_combination.split(",") if s.strip()] if args.failed_combination else None

    results = [
        {
            "id": c.get("id"),
            "title": c.get("title"),
            "path": c.get("doc_path"),
            "kind": c.get("kind"),
            "confidence": c.get("confidence"),
            "status": c.get("status"),
            "op_families": c.get("op_families", []),
            "bottlenecks": c.get("bottlenecks", []),
            "phenomenon": c.get("phenomenon"),
            "failed_combination": c.get("failed_combination", []),
            "tags": c.get("tags", []),
        }
        for c in okf_cards
        if _matches(c, args.kind, args.op_family, args.bottleneck,
                    args.phenomenon, fc, args.status, args.keyword)
    ]

    out = {
        "query": {
            "kind": args.kind,
            "op_family": args.op_family,
            "bottleneck": args.bottleneck,
            "phenomenon": args.phenomenon,
            "failed_combination": fc,
            "status": args.status,
            "keyword": args.keyword,
        },
        "results": results,
        "sufficiency": _sufficiency(results),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
