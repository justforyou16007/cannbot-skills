#!/usr/bin/env python3
"""批量迁移 proven_solutions/INDEX.md 条目到 OKF 单文件。

用法：
    python3 skills/evolution-knowledge/scripts/migrate_proven_to_okf.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "skills" / "evolution-knowledge" / "references" / "a3" / "proven_solutions" / "INDEX.md"
DST_DIR = ROOT / "skills" / "evolution-knowledge" / "references" / "okf" / "runbooks" / "proven-solutions"
VOCAB_PATH = ROOT / "skills" / "evolution-strategies" / "references" / "INDEX.json"


def load_vocab() -> tuple[set[str], set[str]]:
    try:
        data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        vocab = data.get("vocabulary", {})
        return set(vocab.get("op_families", [])), set(vocab.get("bottleneck_labels", []))
    except Exception:
        return set(), set()


def kebab(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s.lower()[:60]


def extract_field(entry: str, name: str) -> str:
    m = re.search(rf"\*\*{name}\*\*: ?(.*?)\n", entry)
    return m.group(1).strip() if m else ""


def infer_op_families(entry: str, vocab: set[str]) -> list[str]:
    ot = extract_field(entry, "Operator type").lower()
    mapping = {
        "attention": ["attention", "flash_attention"],
        "elementwise": ["elementwise"],
        "reduction": ["reduction"],
        "normalization": ["normalization", "reduction"],
        "sort": ["sort"],
        "topk": ["topk"],
        "nms": ["sort", "topk"],
        "matmul": ["matmul"],
    }
    found = set()
    for k, families in mapping.items():
        if k in ot:
            for f in families:
                if f in vocab:
                    found.add(f)
    return sorted(found)


def main() -> int:
    vocab_ops, _ = load_vocab()
    DST_DIR.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8")
    # split by ### headings after ## Recorded Solutions
    pos = text.find("## Recorded Solutions")
    body = text[pos:] if pos >= 0 else text
    entries = re.split(r"\n(?=### )", body)

    for entry in entries:
        if not entry.startswith("### "):
            continue
        title_line = entry.split("\n")[0].replace("### ", "").strip()
        if title_line.startswith("{") or not title_line:
            continue

        filename = kebab(title_line) + ".md"
        source = extract_field(entry, "Source")
        speedup = extract_field(entry, "Speedup")
        op_type = extract_field(entry, "Operator type")
        hardware = extract_field(entry, "Hardware")
        strategy_ids = extract_field(entry, "Strategy IDs")
        op_families = infer_op_families(entry, vocab_ops)

        # Heuristic confidence: cross-operator if multiple op names in Source
        confidence = "cross_op_verified" if len(re.findall(r"[A-Z][a-zA-Z0-9_]+", source)) > 1 else "single_op"

        fm = [
            "---",
            "schema_version: okf.v1",
            "kind: proven_solution",
            f"title: \"{title_line}\"",
            f"source: \"{source}\"",
            f"speedup: \"{speedup}\"",
            f"operator_type: \"{op_type}\"",
            f"hardware: \"{hardware}\"",
            f"confidence: {confidence}",
            "status: active",
            f"op_families: {op_families}",
            f"strategy_ids: {strategy_ids}",
            "---",
        ]
        dst = DST_DIR / filename
        dst.write_text("\n".join(fm) + "\n\n" + entry, encoding="utf-8")
        print(f"migrated {title_line[:50]} -> {filename} op_families={op_families} confidence={confidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
