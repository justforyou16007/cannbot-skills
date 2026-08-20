#!/usr/bin/env python3
"""
Multi-Lens Idea Generation Engine — generates optimization ideas from multiple
analytical perspectives.

Lenses:
1. profiling_driven  — BOTTLENECK_STRATEGY_MAP deterministic mapping + LLM refinement
2. method_transfer   — strategy index matching to operator characteristics
3. gap_analysis      — gap_map driven idea proposals
4. continuation      — deepening successful experiment directions
5. open_exploration  — LLM free-form reasoning (fallback when others exhaust)
6. workflow_guided   — family-specific workflow branch ideas
7. supervisor_sourced — Supervisor new_nodes[] mapped to KG ideas by kg_ops

The engine is primarily LLM-driven with deterministic fallback: when running as
a CLI tool (no LLM), it uses profiling_driven and method_transfer lenses only.
When called from an agent, the LLM supplements with richer idea descriptions.

Usage:
    python3 idea_generation.py generate <kg_root> \\
        --baseline-evidence <json> --parallel-num <int> \\
        [--strategy-index <path>] [--round <int>]
    python3 idea_generation.py from_experiment <kg_root> \\
        --experiment-slug <slug> --parallel-num <int>
    python3 idea_generation.py dedup <kg_root>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Import sibling modules
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from kg_wiki import (  # noqa: E402
    load_ideas, load_experiments,
    parse_gap_map, get_banlist, upsert_idea, append_log, _slugify,
)
from profiling_evidence import BOTTLENECK_STRATEGY_MAP  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diversity / dedup helpers
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _infer_optimization_type(strategy_tags: list[str]) -> str:
    """Infer optimization_type from strategy tags.

    Follows the existing wm_ops.py convention:
    P1/P7/P10/P11 → bandwidth; P2/P4/P5/P8 → tiling; else → algorithm
    """
    bandwidth_strats = {"P1", "P7", "P10", "P11", "P19", "P25", "P66"}
    tiling_strats = {"P2", "P4", "P5", "P8", "P47", "P51"}

    bw_count = sum(1 for s in strategy_tags if s in bandwidth_strats)
    tl_count = sum(1 for s in strategy_tags if s in tiling_strats)

    if bw_count >= tl_count and bw_count > 0:
        return "bandwidth"
    if tl_count > 0:
        return "tiling"
    return "algorithm"


def _estimate_difficulty(strategy_tags: list[str]) -> int:
    """Estimate difficulty from strategy count."""
    n = len(strategy_tags)
    if n <= 1:
        return 2
    if n == 2:
        return 3
    if n == 3:
        return 4
    return min(5, n + 1)


def dedup_ideas(
    ideas: list[dict],
    existing_ideas: dict[str, dict] | None = None,
    banlist: list[dict] | None = None,
    collision_threshold: float = 0.6,
) -> list[dict]:
    """Remove duplicate and banned ideas.

    Dedup criteria:
    1. Jaccard similarity of strategy_tags > collision_threshold
    2. Ideas matching banlist entries (failed ideas)

    Special case: open_exploration ideas with empty strategy_tags are
    never deduplicated against each other (they are differentiated by
    their thesis/title, not by strategy tags).
    """
    existing_ideas = existing_ideas or {}
    banlist = banlist or []

    # Build banlist signatures
    ban_sigs = []
    for banned in banlist:
        stags = banned.get("strategy_tags", [])
        if isinstance(stags, list):
            ban_sigs.append(set(stags))

    # Build existing idea signatures
    existing_sigs = []
    for idea in existing_ideas.values():
        stags = idea.get("strategy_tags", [])
        if isinstance(stags, list):
            existing_sigs.append(set(stags))

    accepted = []
    accepted_sigs: list[set[str]] = []

    for idea in ideas:
        sig = set(idea.get("strategy_tags", []))
        is_open_exploration = idea.get("mode") == "open_exploration" and not sig
        idea_tags = idea.get("tags", [])
        is_retry = isinstance(idea_tags, (list, tuple, set)) and "retry" in idea_tags

        # Skip Jaccard-based dedup for open_exploration with empty tags —
        # they are differentiated by thesis/title, not by strategy tags.
        # Empty set vs empty set has Jaccard=1.0 which would incorrectly
        # collapse all open exploration ideas into one.
        if not is_open_exploration:
            # Check banlist (skip comparing against empty banlist entries)
            is_banned = (
                False if is_retry else any(
                    _jaccard(sig, bs) > collision_threshold
                    for bs in ban_sigs if bs
                )
            )
            if is_banned:
                logger.info("Idea filtered (banlist): %s", idea.get("title", "?"))
                continue

            # Check existing ideas (skip comparing against empty existing sigs)
            is_dup_existing = any(
                _jaccard(sig, es) > collision_threshold
                for es in existing_sigs if es
            )
            if is_dup_existing:
                logger.info("Idea filtered (existing): %s", idea.get("title", "?"))
                continue

            # Check within current batch — only skip empty-vs-empty to avoid
            # collapsing non-open-exploration empty-tag ideas; but non-empty
            # sigs must still be checked against empty ones
            is_dup_batch = any(
                _jaccard(sig, as_) > collision_threshold
                for as_ in accepted_sigs
                if sig or as_  # skip only the empty-vs-empty case
            )
            if is_dup_batch:
                logger.info("Idea filtered (batch dedup): %s", idea.get("title", "?"))
                continue

        accepted.append(idea)
        accepted_sigs.append(sig)

    return accepted


# ---------------------------------------------------------------------------
# Lens 1: Profiling-driven (deterministic + LLM supplement)
# ---------------------------------------------------------------------------

def profiling_driven_lens(
    baseline_evidence: dict | None = None,
    experiment_evidence: dict | None = None,
    parent_facts: dict | None = None,  # noqa: ARG001 — reserved for future LLM use
    parent_diagnosis: dict | None = None,
) -> list[dict]:
    """Generate ideas from profiling evidence using BOTTLENECK_STRATEGY_MAP.

    This is the deterministic core — works without LLM.
    """
    ideas = []
    evidence = baseline_evidence or experiment_evidence or {}
    bottleneck_type = evidence.get("bottleneck_type")

    if not bottleneck_type or bottleneck_type not in BOTTLENECK_STRATEGY_MAP:
        return ideas

    bn_info = BOTTLENECK_STRATEGY_MAP[bottleneck_type]
    primary = bn_info.get("primary", [])
    secondary = bn_info.get("secondary", [])
    anti = set(bn_info.get("anti", []))
    description = bn_info.get("description", "")

    # Idea 1: Primary strategies only
    if primary:
        ideas.append({
            "title": f"针对 {bottleneck_type} 瓶颈的主要优化: {', '.join(primary)}",
            "strategy_tags": [s for s in primary if s not in anti],
            "optimization_type": _infer_optimization_type(primary),
            "difficulty": _estimate_difficulty(primary),
            "mode": "strategy_guided",
            "generation_lens": "profiling_driven",
            "target_gaps": [],
            "thesis": f"基于 profiling 证据 ({bottleneck_type}): {description}",
            "tags": [bottleneck_type],
        })

    # Idea 2: Primary + top secondary combination
    if primary and secondary:
        combo = [s for s in primary + secondary[:2] if s not in anti]
        ideas.append({
            "title": (f"针对 {bottleneck_type} 的组合优化: "
                      f"{', '.join(combo)}"),
            "strategy_tags": combo,
            "optimization_type": _infer_optimization_type(combo),
            "difficulty": _estimate_difficulty(combo),
            "mode": "strategy_guided",
            "generation_lens": "profiling_driven",
            "target_gaps": [],
            "thesis": (f"在主策略基础上叠加辅助策略, "
                       f"综合解决 {bottleneck_type} 瓶颈"),
            "tags": [bottleneck_type],
        })

    # Idea 3: If diagnosis has specific bottleneck_labels, refine
    if parent_diagnosis and isinstance(parent_diagnosis, dict):
        labels = parent_diagnosis.get("bottleneck_labels", [])
        if isinstance(labels, list) and labels:
            for label in labels[:2]:
                if label in BOTTLENECK_STRATEGY_MAP and label != bottleneck_type:
                    sub_info = BOTTLENECK_STRATEGY_MAP[label]
                    sub_primary = sub_info.get("primary", [])
                    sub_anti = set(sub_info.get("anti", []))
                    combined = [s for s in primary + sub_primary
                                if s not in anti and s not in sub_anti]
                    combined = list(dict.fromkeys(combined))  # dedup preserving order
                    if combined:
                        ideas.append({
                            "title": (f"多瓶颈优化 ({bottleneck_type} + {label}): "
                                      f"{', '.join(combined)}"),
                            "strategy_tags": combined,
                            "optimization_type": _infer_optimization_type(combined),
                            "difficulty": _estimate_difficulty(combined),
                            "mode": "profiling_driven",
                            "generation_lens": "profiling_driven",
                            "target_gaps": [],
                            "thesis": f"同时解决 {bottleneck_type} 和 {label} 瓶颈",
                            "tags": [bottleneck_type, label],
                        })

    return ideas


# ---------------------------------------------------------------------------
# Lens 2: Method transfer (strategy index matching)
# ---------------------------------------------------------------------------

def method_transfer_lens(
    kernel_traits: dict | None = None,
    existing_strategies: set[str] | None = None,
) -> list[dict]:
    """Generate ideas by transferring strategies based on operator characteristics.

    Looks at operator traits (memory_bound, compute_bound, reduction, broadcast,
    etc.) and suggests strategies that match but haven't been tried.
    """
    ideas = []
    traits = kernel_traits or {}
    existing = existing_strategies or set()

    # Trait-to-strategy mapping (simplified; full version reads strategy-index.md)
    trait_strategies = {
        "memory_bound": [["P1", "P10"], ["P25", "P66"], ["P7", "P19"]],
        "compute_bound": [["D1", "P46"], ["P84", "P95"]],
        "reduction": [["P94", "P5"], ["P84", "P67"]],
        "broadcast": [["P99", "P10"], ["P97"]],
        "elementwise": [["P10", "P95"], ["P1", "P7"]],
        "has_tail": [["P11", "P100"], ["P47"]],
        "multi_core": [["P4", "P2"], ["P51", "P73"]],
    }

    for trait, combos in trait_strategies.items():
        if not traits.get(trait, False):
            continue
        for combo in combos:
            untested = [s for s in combo if s not in existing]
            if len(untested) == len(combo):  # Entirely new
                ideas.append({
                    "title": f"方法迁移 ({trait}): {', '.join(combo)}",
                    "strategy_tags": combo,
                    "optimization_type": _infer_optimization_type(combo),
                    "difficulty": _estimate_difficulty(combo),
                    "mode": "strategy_guided",
                    "generation_lens": "method_transfer",
                    "target_gaps": [],
                    "thesis": f"基于算子 {trait} 特征, 迁移 {', '.join(combo)} 策略",
                    "tags": [trait, "method_transfer"],
                })

    return ideas


# ---------------------------------------------------------------------------
# Lens 3: Gap analysis driven
# ---------------------------------------------------------------------------

def gap_analysis_lens(
    kg_root: str,
    max_ideas: int = 5,
) -> list[dict]:
    """Generate ideas targeting open gaps in the gap map."""
    gaps = parse_gap_map(kg_root)
    open_gaps = [g for g in gaps if g.get("status") == "open"]

    if not open_gaps:
        return []

    # Sort by priority, take top N
    open_gaps.sort(key=lambda g: g.get("priority", 1.0), reverse=True)
    top_gaps = open_gaps[:max_ideas]

    ideas = []
    for gap in top_gaps:
        gap_id = gap.get("id", "G?")
        desc = gap.get("description", "")

        # Try to extract strategy suggestions from gap description
        strategy_tags = []
        for m in __import__("re").finditer(r"\b([PDAXR]\d+)\b", desc):
            strategy_tags.append(m.group(1))

        if not strategy_tags:
            # Generic gap — produce an open_exploration idea
            ideas.append({
                "title": f"Gap {gap_id}: {desc[:60]}",
                "strategy_tags": [],
                "optimization_type": "algorithm",
                "difficulty": 3,
                "mode": "open_exploration",
                "generation_lens": "gap_analysis",
                "target_gaps": [gap_id],
                "thesis": f"针对知识缺口 {gap_id}: {desc}",
                "tags": ["gap_driven"],
            })
        else:
            ideas.append({
                "title": f"Gap {gap_id}: {', '.join(strategy_tags[:3])} 优化",
                "strategy_tags": strategy_tags[:4],
                "optimization_type": _infer_optimization_type(strategy_tags),
                "difficulty": _estimate_difficulty(strategy_tags),
                "mode": "strategy_guided",
                "generation_lens": "gap_analysis",
                "target_gaps": [gap_id],
                "thesis": f"针对知识缺口 {gap_id}: {desc}",
                "tags": ["gap_driven"],
            })

    return ideas


# ---------------------------------------------------------------------------
# Lens 4: Continuation (deepen successful experiments)
# ---------------------------------------------------------------------------

def continuation_lens(
    kg_root: str,
    experiment_slug: str,
    max_ideas: int = 3,
) -> list[dict]:
    """Generate ideas that deepen a successful experiment.

    Looks at what the experiment achieved and what bottlenecks remain,
    then proposes follow-up optimizations.
    """
    experiments = load_experiments(kg_root)
    ideas_db = load_ideas(kg_root)

    exp = experiments.get(experiment_slug, {})
    if not exp or exp.get("verdict") != "yes":
        return []

    # Get parent idea's strategies
    idea_id = exp.get("idea_id", "")
    idea_slug = idea_id.replace("idea:", "") if idea_id.startswith("idea:") else ""
    parent_idea = ideas_db.get(idea_slug, {})
    parent_stags = parent_idea.get("strategy_tags", [])
    if not isinstance(parent_stags, list):
        parent_stags = []

    ideas = []
    exp_node_id = exp.get("node_id", f"exp:{experiment_slug}")

    # Look at profiling evidence for remaining bottlenecks
    pe = exp.get("profiling_evidence")
    if isinstance(pe, dict):
        bn_type = pe.get("bottleneck_type")
        if bn_type and bn_type in BOTTLENECK_STRATEGY_MAP:
            bn_info = BOTTLENECK_STRATEGY_MAP[bn_type]
            suggested = bn_info.get("primary", []) + bn_info.get("secondary", [])[:2]
            anti = set(bn_info.get("anti", []))
            new_stags = [s for s in suggested
                         if s not in parent_stags and s not in anti]

            if new_stags:
                combo = parent_stags + new_stags[:2]
                ideas.append({
                    "title": (f"深化 {experiment_slug}: 叠加 "
                              f"{', '.join(new_stags[:2])} 解决残余 {bn_type}"),
                    "strategy_tags": combo,
                    "optimization_type": _infer_optimization_type(combo),
                    "difficulty": _estimate_difficulty(combo),
                    "mode": "strategy_guided",
                    "generation_lens": "continuation",
                    "target_gaps": [],
                    "parent_experiment": exp_node_id,
                    "parent_idea": idea_id if idea_id else None,
                    "thesis": (f"在 {experiment_slug} (speedup={exp.get('speedup', '?')}x) "
                               f"基础上解决残余 {bn_type} 瓶颈"),
                    "tags": ["continuation", bn_type],
                })

    # Also try adding orthogonal strategies
    orthogonal_combos = [
        (["P1"], "双缓冲"),
        (["P2", "P4"], "分块+均衡"),
        (["P10"], "向量化搬运"),
        (["D1"], "混合精度"),
    ]
    for orth_stags, orth_name in orthogonal_combos:
        new_tags = [s for s in orth_stags if s not in parent_stags]
        if new_tags:
            combo = parent_stags + new_tags
            ideas.append({
                "title": (f"深化 {experiment_slug}: 叠加{orth_name} "
                          f"({', '.join(new_tags)})"),
                "strategy_tags": combo,
                "optimization_type": _infer_optimization_type(combo),
                "difficulty": _estimate_difficulty(combo),
                "mode": "strategy_guided",
                "generation_lens": "continuation",
                "target_gaps": [],
                "parent_experiment": exp_node_id,
                "parent_idea": idea_id if idea_id else None,
                "thesis": f"在已成功方案上叠加正交优化 ({orth_name})",
                "tags": ["continuation"],
            })
            if len(ideas) >= max_ideas:
                break

    return ideas[:max_ideas]


# ---------------------------------------------------------------------------
# Lens 5: Open exploration (placeholder for LLM)
# ---------------------------------------------------------------------------

def open_exploration_lens(
    kernel_summary: str = "",
    num_ideas: int = 2,
) -> list[dict]:
    """Generate placeholder ideas for LLM-driven open exploration.

    When called from the agent, the LLM fills in the thesis and strategy_tags.
    The deterministic version just creates scaffold ideas.
    """
    ideas = []
    for i in range(num_ideas):
        ideas.append({
            "title": f"开放探索方案 {i + 1}",
            "strategy_tags": [],
            "optimization_type": "algorithm",
            "difficulty": 3,
            "mode": "open_exploration",
            "generation_lens": "open_exploration",
            "target_gaps": [],
            "thesis": (f"基于内核代码和 profiling 数据的第一原理推理。"
                       f"算子: {kernel_summary[:100]}" if kernel_summary else
                       "LLM 自主推理优化方向"),
            "tags": ["open_exploration"],
        })
    return ideas


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_ideas_for_init(
    kg_root: str,
    parallel_num: int,
    baseline_evidence: dict | None = None,
    kernel_traits: dict | None = None,
    kernel_summary: str = "",
    strategy_index_path: str | None = None,
    generation_round: int = 0,
) -> list[dict]:
    """Generate initial ideas (Init phase).

    Produces parallel_num * 2 ideas from multiple lenses.
    """
    target_count = parallel_num * 2
    all_ideas: list[dict] = []

    # Lens 1: Profiling-driven (from baseline evidence)
    all_ideas.extend(profiling_driven_lens(baseline_evidence=baseline_evidence))

    # Lens 2: Method transfer
    existing_strats: set[str] = set()
    all_ideas.extend(method_transfer_lens(kernel_traits, existing_strats))

    # Lens 3: Gap analysis
    all_ideas.extend(gap_analysis_lens(kg_root, max_ideas=3))

    # Lens 5: Open exploration (fill remaining slots)
    remaining = max(2, target_count - len(all_ideas))
    all_ideas.extend(open_exploration_lens(kernel_summary, remaining))

    # Set generation round
    for idea in all_ideas:
        idea["generation_round"] = generation_round

    # Dedup against banlist and existing
    existing = load_ideas(kg_root)
    banlist_ideas = get_banlist(kg_root)
    deduped = dedup_ideas(all_ideas, existing, banlist_ideas)

    # Ensure we have enough diversity
    opt_types = defaultdict(int)
    for idea in deduped:
        opt_types[idea.get("optimization_type", "algorithm")] += 1

    logger.info("Generated %d ideas (deduped from %d) for init. "
                "Types: %s", len(deduped), len(all_ideas), dict(opt_types))

    return deduped[:target_count]


def generate_ideas_from_experiment(
    kg_root: str,
    experiment_slug: str,
    parallel_num: int = 3,
    generation_round: int = 0,
) -> list[dict]:
    """Generate new ideas from a completed experiment (Refine phase).

    For passed experiments: continuation + profiling-driven + gap analysis
    For failed experiments: retry idea (if impl_error) or gap update
    """
    experiments = load_experiments(kg_root)
    exp = experiments.get(experiment_slug, {})

    if not exp:
        logger.warning("Experiment not found: %s", experiment_slug)
        return []

    all_ideas: list[dict] = []

    if exp.get("verdict") == "yes":
        # Continuation lens — deepen successful direction
        all_ideas.extend(continuation_lens(kg_root, experiment_slug, max_ideas=3))

        # Profiling-driven lens — based on new evidence
        pe = exp.get("profiling_evidence")
        if isinstance(pe, dict):
            diagnosis = exp.get("diagnosis")
            all_ideas.extend(profiling_driven_lens(
                experiment_evidence=pe,
                parent_diagnosis=diagnosis if isinstance(diagnosis, dict) else None,
            ))

        # Gap analysis — check for new gaps from this experiment
        all_ideas.extend(gap_analysis_lens(kg_root, max_ideas=2))

    elif exp.get("verdict") == "no":
        failure_type = exp.get("failure_type", "unknown")
        retry_count = exp.get("retry_count", 0)

        if failure_type == "impl_error" and retry_count < 2:
            # Create retry idea with increased difficulty
            idea_id = exp.get("idea_id", "")
            idea_slug = idea_id.replace("idea:", "") if idea_id.startswith("idea:") else ""
            parent_idea = load_ideas(kg_root).get(idea_slug, {})
            stags = parent_idea.get("strategy_tags", [])
            if isinstance(stags, list):
                all_ideas.append({
                    "title": f"重试: {parent_idea.get('title', experiment_slug)} (修复实现错误)",
                    "strategy_tags": stags,
                    "optimization_type": parent_idea.get("optimization_type", "algorithm"),
                    "difficulty": min(5, parent_idea.get("difficulty", 3) + 1),
                    "mode": parent_idea.get("mode", "strategy_guided"),
                    "generation_lens": "continuation",
                    "target_gaps": parent_idea.get("target_gaps", []),
                    "parent_idea": idea_id if idea_id else None,
                    "parent_experiment": exp.get("node_id"),
                    "thesis": (f"修复实现错误后重试: "
                               f"{exp.get('failure_reason', '原因不明')}"),
                    "tags": ["retry", f"retry_{retry_count + 1}"],
                })

    # Set generation round
    for idea in all_ideas:
        idea["generation_round"] = generation_round

    # Dedup
    existing = load_ideas(kg_root)
    banlist_ideas = get_banlist(kg_root)
    deduped = dedup_ideas(all_ideas, existing, banlist_ideas)

    return deduped[:parallel_num]


def persist_ideas(
    kg_root: str,
    ideas: list[dict],
    generation_round: int = 0,
) -> list[str]:
    """Persist a list of idea dicts as markdown files via upsert_idea.

    Returns list of created slugs.
    """
    slugs = []
    for idx, idea in enumerate(ideas):
        slug = _slugify(idea.get("title", f"idea-r{generation_round}-{idx}"))
        # Ensure uniqueness
        if (Path(kg_root) / "ideas" / f"{slug}.md").exists():
            slug = f"{slug}-{idx}"

        upsert_idea(
            kg_root, slug, idea["title"],
            strategy_tags=idea.get("strategy_tags", []),
            optimization_type=idea.get("optimization_type", "algorithm"),
            difficulty=idea.get("difficulty", 3),
            mode=idea.get("mode", "strategy_guided"),
            workflow_ref=idea.get("workflow_ref"),
            generation_lens=idea.get("generation_lens", "profiling_driven"),
            generation_round=idea.get("generation_round", generation_round),
            target_gaps=idea.get("target_gaps", []),
            tags=idea.get("tags", []),
            description=idea.get("description", ""),
            thesis=idea.get("thesis", ""),
            strategy_rationale=idea.get("strategy_rationale", ""),
            parent_idea=idea.get("parent_idea"),
            parent_experiment=idea.get("parent_experiment"),
        )
        slugs.append(slug)

    append_log(kg_root, f"persist_ideas: {len(slugs)} ideas written for round {generation_round}")
    return slugs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    baseline_evidence = None
    if args.baseline_evidence:
        try:
            baseline_evidence = json.loads(args.baseline_evidence)
        except json.JSONDecodeError:
            # Try as file path
            try:
                baseline_evidence = json.loads(
                    Path(args.baseline_evidence).read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning("Could not parse baseline evidence")

    ideas = generate_ideas_for_init(
        args.kg_root,
        parallel_num=args.parallel_num,
        baseline_evidence=baseline_evidence,
        generation_round=args.round,
    )

    if not args.dry_run:
        slugs = persist_ideas(args.kg_root, ideas, args.round)
        print(json.dumps({"generated": len(slugs), "slugs": slugs},
                          indent=2, ensure_ascii=False))
    else:
        print(json.dumps(ideas, indent=2, ensure_ascii=False))


def cmd_from_experiment(args: argparse.Namespace) -> None:
    ideas = generate_ideas_from_experiment(
        args.kg_root,
        args.experiment_slug,
        parallel_num=args.parallel_num,
        generation_round=args.round,
    )

    if not args.dry_run:
        slugs = persist_ideas(args.kg_root, ideas, args.round)
        print(json.dumps({"generated": len(slugs), "slugs": slugs},
                          indent=2, ensure_ascii=False))
    else:
        print(json.dumps(ideas, indent=2, ensure_ascii=False))


def cmd_dedup(args: argparse.Namespace) -> None:
    ideas = load_ideas(args.kg_root)
    banlist_ideas = get_banlist(args.kg_root)

    # Build list from existing ideas
    idea_list = []
    for slug, idea in ideas.items():
        idea_list.append({
            "slug": slug,
            "title": idea.get("title", slug),
            "strategy_tags": idea.get("strategy_tags", []),
        })

    deduped = dedup_ideas(idea_list, {}, banlist_ideas)
    removed = len(idea_list) - len(deduped)
    print(json.dumps({"total": len(idea_list), "deduped": len(deduped),
                       "removed": removed}, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-lens idea generation engine."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate", help="Generate initial ideas")
    p_gen.add_argument("kg_root")
    p_gen.add_argument("--parallel-num", type=int, default=3)
    p_gen.add_argument("--baseline-evidence", default=None)
    p_gen.add_argument("--round", type=int, default=0)
    p_gen.add_argument("--dry-run", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_exp = subparsers.add_parser("from_experiment",
                                   help="Generate ideas from experiment result")
    p_exp.add_argument("kg_root")
    p_exp.add_argument("--experiment-slug", required=True)
    p_exp.add_argument("--parallel-num", type=int, default=3)
    p_exp.add_argument("--round", type=int, default=0)
    p_exp.add_argument("--dry-run", action="store_true")
    p_exp.set_defaults(func=cmd_from_experiment)

    p_dedup = subparsers.add_parser("dedup", help="Check for duplicate ideas")
    p_dedup.add_argument("kg_root")
    p_dedup.set_defaults(func=cmd_dedup)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    result = args.func(args)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
