#!/usr/bin/env python3
"""批量迁移 disc_X*.md 到 OKF 双轨，增强 frontmatter。

用法：
    python3 skills/evolution-knowledge/scripts/migrate_discovered_to_okf.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT / "skills" / "evolution-strategies" / "references" / "discovered"
DST_DIR = ROOT / "skills" / "evolution-knowledge" / "references" / "okf" / "strategies" / "discovered"
VOCAB_PATH = ROOT / "skills" / "evolution-strategies" / "references" / "INDEX.json"


def load_vocab() -> tuple[set[str], set[str]]:
    try:
        data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        vocab = data.get("vocabulary", {})
        op_families = set(vocab.get("op_families", []))
        bottleneck_labels = set(vocab.get("bottleneck_labels", []))
        return op_families, bottleneck_labels
    except Exception:
        return set(), set()


OP_FAMILY_KEYWORDS = {
    "attention": ["attention", "flash", "causal", "gqa", "mqa", "mla", "decode", "prefill", "softmax"],
    "flash_attention": ["flash attention", "flashattention", "flash_decode", "flashdecode", "fa-"],
    "matmul": ["matmul", "gemm", "gemv", "cube", "quant_matmul", "weight_quant", "grouped_matmul"],
    "reduction": ["reduction", "reduce", "groupnorm", "layernorm", "rmsnorm", "topk", "sort"],
    "normalization": ["groupnorm", "layernorm", "rmsnorm", "normalization"],
    "elementwise": ["elementwise", "foreach", "dtype", "passthrough"],
    "sort": ["sort", "topk", "nms"],
    "topk": ["topk", "nms"],
    "moe": ["moe", "expert", "group_list"],
    "cv_fusion": ["cv", "img2col", "conv2d"],
}

BOTTLENECK_KEYWORDS = {
    "mte2_stall": ["mte2", "hbm", "bandwidth", "memory", "gm read", "gm write", "搬运"],
    "mte3_stall": ["mte3", "gm", "global memory", "memcpy"],
    "compute_bound": ["compute", "cube", "gemm", "mac", "aic", "利用率"],
    "scalar_compute": ["scalar", "vector", "aiv", "vec_"],
    "tiling_imbalance": ["tiling", "负载", "不均衡", "imbalance"],
    "no_overlap": ["overlap", "pipeline", "流水"],
    "bus_contention": ["contention", "冲突"],
    "l2_cache_thrash": ["l2", "cache"],
}


def infer_op_families(text: str, vocab: set[str]) -> list[str]:
    text_lo = text.lower()
    found = set()
    for family, kws in OP_FAMILY_KEYWORDS.items():
        if family not in vocab:
            continue
        for kw in kws:
            if kw in text_lo:
                found.add(family)
                break
    return sorted(found)


def infer_bottlenecks(text: str, vocab: set[str]) -> list[str]:
    text_lo = text.lower()
    found = set()
    for label, kws in BOTTLENECK_KEYWORDS.items():
        if label not in vocab:
            continue
        for kw in kws:
            if kw in text_lo:
                found.add(label)
                break
    return sorted(found)


def infer_tags(op_families: list[str], bottlenecks: list[str], text: str) -> list[str]:
    extra = set()
    text_lo = text.lower()
    if "decode" in text_lo:
        extra.add("decode")
    if "prefill" in text_lo:
        extra.add("prefill")
    if "gqa" in text_lo or "grouped query" in text_lo:
        extra.add("gqa")
    if "mla" in text_lo:
        extra.add("mla")
    if "quant" in text_lo:
        extra.add("quantization")
    if "moe" in text_lo:
        extra.add("moe")
    if "conv2d" in text_lo or "img2col" in text_lo:
        extra.add("conv2d")
    tags = sorted(set(op_families) | set(bottlenecks) | extra)
    # 去重，避免 tag 和 op_families/bottlenecks 完全重复时无意义；保留即可
    return tags


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        m = re.match(r"---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if m:
            body = raw[m.end():]
            fm_text = m.group(1)
            fm = {}
            for line in fm_text.splitlines():
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip()
            return fm, body
    return {}, raw


def build_okf_frontmatter(fm: dict, text: str, body: str, vocab_ops: set[str], vocab_bottlenecks: set[str]) -> str:
    op_families = infer_op_families(text, vocab_ops)
    bottlenecks = infer_bottlenecks(text, vocab_bottlenecks)
    tags = infer_tags(op_families, bottlenecks, text)

    title = ""
    m = re.search(r"# Strategy X\d+: (.+)", body)
    if m:
        title = m.group(1).strip()

    description = ""
    m = re.search(r"## 核心思路\s*\n(.+?)(?:\n##|\n\n##|$)", body, re.DOTALL)
    if m:
        description = " ".join(m.group(1).split())[:200]

    def q(s: str) -> str:
        s = s.strip()
        if any(c in s for c in [':', '"', "'", "[", "]", "{", "}"]):
            return '"' + s.replace('"', '\\"') + '"'
        if s == "":
            return '""'
        return s

    lines = [
        "---",
        "schema_version: okf.v1",
        "kind: discovered_strategy",
        f"id: {fm.get('id', '')}",
        f"origin: {fm.get('origin', 'discovered')}",
        f"title: {q(title)}",
        f"description: {q(description)}",
        "confidence: single_op",
        "status: active",
        f"op_families: {op_families}",
        f"bottlenecks: {bottlenecks}",
        f"tags: {tags}",
    ]
    for k in ["discovered_round", "discovered_from", "base_speedup"]:
        if k in fm:
            lines.append(f"{k}: {fm[k]}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def main() -> int:
    vocab_ops, vocab_bottlenecks = load_vocab()
    DST_DIR.mkdir(parents=True, exist_ok=True)

    for src in sorted(SRC_DIR.glob("disc_X*.md")):
        raw = src.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(raw)
        text = raw.lower()
        new_front = build_okf_frontmatter(fm, text, body, vocab_ops, vocab_bottlenecks)
        dst = DST_DIR / src.name
        dst.write_text(new_front + body, encoding="utf-8")
        print(f"migrated {src.name} -> {dst.relative_to(ROOT)}  op_families={infer_op_families(text, vocab_ops)} bottlenecks={infer_bottlenecks(text, vocab_bottlenecks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
