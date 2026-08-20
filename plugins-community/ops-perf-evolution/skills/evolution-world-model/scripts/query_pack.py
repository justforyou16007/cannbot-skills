#!/usr/bin/env python3
"""
Query Pack — Compressed evolution context for LLM idea generation.

Generates a ≤8000 char summary of the evolution knowledge graph that includes:
1. Operator summary + baseline evidence
2. Top 5 gaps (search seeds for next ideas)
3. Banlist (failed ideas — highest anti-repetition value)
4. Key experiments (best performing variants)
5. Verified claims (accumulated knowledge)
6. Recent edges (relationship context)

Modeled after research-wiki's query_pack.md mechanism.

Usage:
    python3 query_pack.py build <kg_root> [--max-chars 8000] [--kg-json <path>]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Import sibling modules
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from kg_wiki import (  # noqa: E402
    load_ideas, load_experiments, load_claims, parse_gap_map,
    load_edges, get_banlist, append_log,
)

logger = logging.getLogger(__name__)

# Section char budgets (total ≤ 8000)
_BUDGET = {
    "summary": 800,       # operator summary + baseline
    "gaps": 1200,         # top 5 gaps
    "banlist": 1400,      # failed ideas (never pruned first)
    "experiments": 1800,  # key experiments
    "claims": 900,        # verified claims
    "edges": 500,         # recent edges
    "reserve": 1400,      # flex buffer
}


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def _build_summary_section(
    kg_root: str, kg_json_path: str | None = None,
) -> str:
    """Build the operator summary + baseline section."""
    # Try loading knowledge_graph.json
    kg_data: dict = {}
    if kg_json_path:
        try:
            kg_data = json.loads(Path(kg_json_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    else:
        for candidate in [
            Path(kg_root) / "knowledge_graph.json",
            Path(kg_root).parent / "knowledge_graph.json",
        ]:
            if candidate.exists():
                try:
                    kg_data = json.loads(candidate.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass

    parts = []
    kernel_summary = kg_data.get("kernel_summary", "")
    if kernel_summary:
        parts.append(f"**算子**: {kernel_summary}")

    baseline = kg_data.get("baseline_performance", {})
    if baseline:
        parts.append(f"**基线**: speedup={baseline.get('speedup', '?')}, "
                     f"time_ms={baseline.get('time_ms', '?')}")

    best_score = kg_data.get("best_score")
    if best_score and best_score > 1.0:
        parts.append(f"**当前最优**: {best_score}x")

    hw = kg_data.get("hw_params")
    if hw:
        parts.append(f"**硬件**: {hw.get('chip_model', '?')}, "
                     f"UB={hw.get('ub_size_bytes', '?')}B, "
                     f"cores={hw.get('core_num', '?')}")

    baseline_ev = kg_data.get("baseline_evidence")
    if baseline_ev and isinstance(baseline_ev, dict):
        parts.append(f"**基线瓶颈**: {baseline_ev.get('bottleneck_type', '?')} "
                     f"(D类={baseline_ev.get('d_class_pct', '?')}%, "
                     f"C类={baseline_ev.get('c_class_pct', '?')}%)")

    questions = kg_data.get("open_questions", [])
    if isinstance(questions, list) and questions:
        rendered_questions = []
        for question in questions[-3:]:
            if isinstance(question, dict):
                rendered_questions.append(
                    str(question.get("bottleneck") or question.get("question") or "")
                )
            else:
                rendered_questions.append(str(question))
        rendered_questions = [q for q in rendered_questions if q]
        if rendered_questions:
            parts.append("**开放问题**: " + "；".join(rendered_questions))

    return "\n".join(parts) if parts else "_算子信息暂缺._"


def _build_gaps_section(kg_root: str) -> str:
    """Build the top-gap section, retaining the canonical untried gap."""
    gaps = parse_gap_map(kg_root)
    open_gaps = [g for g in gaps if g.get("status", "open") == "open"]

    if not open_gaps:
        return "_暂无已知知识缺口._"

    # Sort by priority descending, take top 5.  Keep the canonical
    # untried-strategies gap in the search context even when its numeric
    # priority is below five more local gaps: during a stall, losing the
    # signal that a large part of the strategy space is untouched defeats the
    # purpose of the gap map.  Closed gaps remain excluded above.
    open_gaps.sort(key=lambda g: g.get("priority", 1.0), reverse=True)
    top = open_gaps[:5]
    untried_gap = next(
        (gap for gap in open_gaps if gap.get("key") == "untried_strategies"),
        None,
    )
    if untried_gap is not None and untried_gap not in top:
        top = [untried_gap] + top

    lines = []
    for gap in top:
        gid = gap.get("id", "?")
        desc = gap.get("description", "")
        evidence = gap.get("evidence", "")
        lenses = gap.get("suggested_lenses", [])
        line = f"- **{gid}**: {desc}"
        if evidence:
            line += f" | 证据: {evidence[:80]}"
        if lenses:
            line += f" | 镜头: {', '.join(lenses)}"
        lines.append(line)

    return "\n".join(lines)


def _build_banlist_section(kg_root: str) -> str:
    """Build the banlist section using get_banlist() (strategy_infeasible only).

    Only includes ideas from the canonical banlist — those whose failure
    was strategy_infeasible. impl_error failures are shown separately as
    a "repair queue" note so the LLM knows the direction may still work.
    """
    # Strategy banlist (canonical — only strategy_infeasible)
    banned = get_banlist(kg_root)
    experiments = load_experiments(kg_root)
    ideas = load_ideas(kg_root)

    # Also collect impl_error failures for the repair queue
    impl_error_ideas = []
    for idea in ideas.values():
        if idea.get("outcome") != "negative":
            continue
        idea_nid = idea.get("node_id", "")
        for exp in experiments.values():
            if exp.get("idea_id") == idea_nid and exp.get("failure_type") == "impl_error":
                impl_error_ideas.append(idea)
                break

    if not banned and not impl_error_ideas:
        return "_暂无失败 idea._"

    lines = []

    # Banlist proper (strategy_infeasible) — these directions should NOT be retried
    if banned:
        lines.append("**策略禁忌（不可行，勿重试）：**")
        for idea in banned[:6]:
            title = idea.get("title", "?")
            stags = idea.get("strategy_tags", [])
            stags_str = ", ".join(stags) if isinstance(stags, list) else str(stags)
            reason = ""
            idea_nid = idea.get("node_id", "")
            for exp in experiments.values():
                if exp.get("idea_id") == idea_nid and exp.get("failure_reason"):
                    reason = exp["failure_reason"]
                    break
            line = f"- ~~{title}~~ [{stags_str}]"
            if reason:
                line += f" — {reason[:60]}"
            lines.append(line)

    # Repair queue (impl_error) — these might work with a corrected implementation
    if impl_error_ideas:
        lines.append("")
        lines.append("**修复队列（实现错误，策略本身可能有效）：**")
        for idea in impl_error_ideas[:4]:
            title = idea.get("title", "?")
            stags = idea.get("strategy_tags", [])
            stags_str = ", ".join(stags) if isinstance(stags, list) else str(stags)
            reason = ""
            idea_nid = idea.get("node_id", "")
            for exp in experiments.values():
                if exp.get("idea_id") == idea_nid and exp.get("failure_reason"):
                    reason = exp["failure_reason"]
                    break
            line = f"- {title} [{stags_str}] ⚠️ impl_error"
            if reason:
                line += f" — {reason[:60]}"
            lines.append(line)

    return "\n".join(lines)


def _build_experiments_section(kg_root: str, kg_json_path: str | None = None) -> str:
    """Build the key experiments section with mixed sampling.

    Instead of just top-N by speedup, uses a multi-slot strategy:
    - Slot 1: Best performers (highest speedup) — what works
    - Slot 2: Most recent round — freshest evidence
    - Slot 3: Key failures (strategy_infeasible) — what to avoid
    - Slot 4: Bottleneck shifts — evolution of the performance landscape

    This prevents losing critical failure/shift information that a
    pure speedup ranking would bury.
    """
    experiments = load_experiments(kg_root)
    if not experiments:
        return "_暂无实验结果._"

    all_exps = list(experiments.values())

    # --- Slot 1: Best performers (top 3 by speedup) ---
    passed = [e for e in all_exps if e.get("verdict") == "yes"]
    passed.sort(key=lambda e: e.get("speedup") or 0, reverse=True)
    best = passed[:3]

    # --- Slot 2: Most recent round experiments (up to 2) ---
    max_round = max((e.get("round") or 0 for e in all_exps), default=0)
    recent = [e for e in all_exps
              if (e.get("round") or 0) == max_round
              and e not in best][:2]

    # --- Slot 3: Key failures — strategy_infeasible only (up to 2) ---
    infeasible = [e for e in all_exps
                  if e.get("failure_type") == "strategy_infeasible"
                  and e not in best and e not in recent][:2]

    # --- Slot 4: Bottleneck shifts (up to 1) ---
    # Experiments where profiling shows a DIFFERENT bottleneck than baseline
    # or the experiment's parent idea's experiment.
    baseline_bn = None
    if kg_json_path:
        try:
            _kg = json.loads(Path(kg_json_path).read_text(encoding="utf-8"))
            be = _kg.get("baseline_evidence")
            if isinstance(be, dict):
                baseline_bn = be.get("bottleneck_type")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    if not baseline_bn:
        # Try from the first experiment as a proxy
        for e in passed:
            pe = e.get("profiling_evidence")
            if isinstance(pe, dict) and pe.get("bottleneck_type"):
                baseline_bn = pe["bottleneck_type"]
                break

    shifts = []
    if baseline_bn:
        for exp in passed:
            if exp in best or exp in recent:
                continue
            pe = exp.get("profiling_evidence")
            if isinstance(pe, dict) and pe.get("bottleneck_type"):
                if pe["bottleneck_type"] != baseline_bn:
                    shifts.append(exp)
                    if len(shifts) >= 1:
                        break

    # Combine with dedup
    seen_slugs: set[str] = set()
    selected: list[dict] = []
    for group in [best, recent, infeasible, shifts]:
        for exp in group:
            slug = exp.get("node_id", "")
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                selected.append(exp)

    # Format
    lines = []
    for exp in selected:
        title = exp.get("title", "?")
        speedup = exp.get("speedup")
        speedup_str = f"{speedup}x" if speedup is not None else "N/A"
        verdict = exp.get("verdict", "?")
        idea_id = exp.get("idea_id", "?")
        rnd = exp.get("round", "?")

        # Tag the experiment's role for the LLM reader
        tags = []
        if exp in best:
            tags.append("最优")
        if exp in recent:
            tags.append("最近")
        if exp in infeasible:
            tags.append("不可行")
        if exp in shifts:
            tags.append("瓶颈迁移")
        tag_str = f" [{'/'.join(tags)}]" if tags else ""

        line = f"- **{title}**: {speedup_str} [{verdict}]{tag_str} (R{rnd}, idea: {idea_id})"

        # Add profiling highlight
        pe = exp.get("profiling_evidence")
        if isinstance(pe, dict) and pe.get("bottleneck_type"):
            line += f" | 瓶颈: {pe['bottleneck_type']}"

        # Add failure reason for failed experiments
        if exp.get("failure_reason"):
            line += f" | 原因: {exp['failure_reason'][:50]}"

        lines.append(line)

    return "\n".join(lines)


def _build_claims_section(kg_root: str) -> str:
    """Build the verified claims section."""
    claims = load_claims(kg_root)
    verified = [c for c in claims.values() if c.get("status") == "verified"]

    if not verified:
        return "_暂无已验证的 claim._"

    lines = []
    for claim in verified[:5]:  # top 5
        name = claim.get("name", "?")
        strat = claim.get("strategy_implications", [])
        conf = claim.get("confidence", 0)
        supp = claim.get("supporting_experiments", [])
        line = f"- **{name}** (置信度: {conf:.2f})"
        if strat:
            line += f" → 策略: {', '.join(strat)}"
        if supp:
            line += f" | 支撑: {len(supp)} 个实验"
        lines.append(line)

    return "\n".join(lines)


def _build_edges_section(kg_root: str) -> str:
    """Build the recent edges section."""
    edges = load_edges(kg_root)
    if not edges:
        return "_暂无关系链路._"

    # Take the most recent 10 edges
    recent = edges[-10:]
    lines = []
    for edge in recent:
        line = f"- `{edge['from']}` --{edge['type']}--> `{edge['to']}`"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_query_pack(
    kg_root: str,
    max_chars: int = 8000,
    kg_json_path: str | None = None,
) -> str:
    """Build the complete query_pack.md content.

    Returns the markdown text. Also writes to kg_root/query_pack.md.
    """
    sections = []

    # 1. Summary
    summary = _build_summary_section(kg_root, kg_json_path)
    sections.append(("## 算子概要", _truncate(summary, _BUDGET["summary"])))

    # 2. Gaps (never pruned first)
    gaps = _build_gaps_section(kg_root)
    sections.append(("## 知识缺口 (搜索种子)", _truncate(gaps, _BUDGET["gaps"])))

    # 3. Banlist (never pruned first)
    banlist = _build_banlist_section(kg_root)
    sections.append(("## 禁忌表 (已失败方向)", _truncate(banlist, _BUDGET["banlist"])))

    # 4. Experiments
    exps = _build_experiments_section(kg_root, kg_json_path)
    sections.append(("## 关键实验", _truncate(exps, _BUDGET["experiments"])))

    # 5. Claims
    claims = _build_claims_section(kg_root)
    sections.append(("## 已验证结论", _truncate(claims, _BUDGET["claims"])))

    # 6. Edges
    edges = _build_edges_section(kg_root)
    sections.append(("## 最近链路", _truncate(edges, _BUDGET["edges"])))

    # Assemble
    header = "# Query Pack — 演化知识压缩摘要\n"
    body = "\n\n".join(f"{title}\n{content}" for title, content in sections)
    full = f"{header}\n{body}\n"

    # Graduated pruning if over budget.
    # Priority: keep gaps + banlist (highest anti-repetition value),
    # prune edges first, then claims, then experiments.
    if len(full) > max_chars:
        prune_order = ["## 最近链路", "## 已验证结论", "## 关键实验"]
        for prune_title in prune_order:
            if len(full) <= max_chars:
                break
            pruned_sections = []
            for title, content in sections:
                if title == prune_title:
                    # Halve this section
                    half = len(content) // 2
                    pruned_sections.append((title, _truncate(content, half)))
                else:
                    pruned_sections.append((title, content))
            sections = pruned_sections
            body = "\n\n".join(f"{t}\n{c}" for t, c in sections)
            full = f"{header}\n{body}\n"

        # Final hard truncation if still over
        if len(full) > max_chars:
            full = full[:max_chars - 3] + "…\n"

    # Write to disk
    qp_path = Path(kg_root) / "query_pack.md"
    qp_path.write_text(full, encoding="utf-8")
    append_log(kg_root, f"rebuild_query_pack: {len(full)} chars")
    logger.info("Query pack built: %d chars", len(full))

    return full


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    result = build_query_pack(args.kg_root, args.max_chars, args.kg_json)
    print(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query Pack builder for evolution knowledge graph."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build", help="Build query pack")
    p_build.add_argument("kg_root")
    p_build.add_argument("--max-chars", type=int, default=8000)
    p_build.add_argument("--kg-json", default=None,
                         help="Path to knowledge_graph.json")
    p_build.set_defaults(func=cmd_build)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    result = args.func(args)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
