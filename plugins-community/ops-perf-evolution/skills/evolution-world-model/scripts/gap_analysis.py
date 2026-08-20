#!/usr/bin/env python3
"""
Gap Analysis — Identifies knowledge gaps in the evolution knowledge graph.

Runs after each Refine step to detect:
1. Unexplored optimization directions (strategy coverage gaps)
2. Bottleneck shifts that create new optimization opportunities
3. Untested combinations that claim-supported strategies suggest
4. Stagnation-driven gaps (when the system needs new directions)

Usage:
    python3 gap_analysis.py analyze <kg_root> [--strategy-index <path>]
    python3 gap_analysis.py detect_unexplored <kg_root> --strategy-index <path>
    python3 gap_analysis.py detect_bottleneck_shift <kg_root>
    python3 gap_analysis.py prioritize <kg_root> [--stagnation-count <int>]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any  # noqa: F401 — used by callers

# Import sibling modules
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from kg_wiki import (  # noqa: E402
    parse_gap_map, write_gap_map, load_ideas, load_experiments,
    load_claims, load_edges, append_log,
)
from profiling_evidence import BOTTLENECK_STRATEGY_MAP  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_experiment_idea(
    ideas: dict[str, dict], experiment: dict,
) -> dict | None:
    """Resolve the idea an experiment tested, by slug then by node_id."""
    idea_id = experiment.get("idea_id", "")
    if not isinstance(idea_id, str) or not idea_id:
        return None
    idea = ideas.get(idea_id.removeprefix("idea:"))
    if idea is not None:
        return idea
    return next(
        (candidate for candidate in ideas.values()
         if candidate.get("node_id") == idea_id),
        None,
    )


def _collect_tested_strategy_combos(
    ideas: dict[str, dict], experiments: dict[str, dict],
    successful_only: bool = False,
) -> list[set[str]]:
    """Collect strategy combinations proven tested by experiments or outcomes.

    An experiment is evidence that its linked idea's strategies were tried,
    even when the orchestration layer has not yet updated the idea outcome.
    Non-pending outcomes remain included for compatibility with older graphs.
    """
    combos: list[set[str]] = []
    seen_combos: set[frozenset[str]] = set()

    def add_idea_combo(idea: dict | None) -> None:
        if not idea:
            return
        stags = idea.get("strategy_tags", [])
        if not isinstance(stags, list) or not stags:
            return
        combo = frozenset(stags)
        if combo not in seen_combos:
            seen_combos.add(combo)
            combos.append(set(combo))

    for experiment in experiments.values():
        if successful_only and experiment.get("verdict") not in ("yes", "partial"):
            continue
        add_idea_combo(_resolve_experiment_idea(ideas, experiment))

    for idea in ideas.values():
        successful_outcomes = ("positive", "mixed") if successful_only else (
            "positive", "negative", "mixed"
        )
        if idea.get("outcome") in successful_outcomes:
            add_idea_combo(idea)

    return combos


def _collect_successful_opt_types(
    ideas: dict[str, dict], experiments: dict[str, dict],
) -> set[str]:
    """Collect optimization types that already have a working experiment.

    A passing (or partially passing) experiment is direct proof that its
    idea's optimization type has been covered.  Relying on ``idea.outcome``
    alone would report a type as unexplored whenever the orchestration layer
    fails to call ``update_idea_outcome`` — the same failure mode fixed for
    strategy coverage in :func:`_collect_tested_strategy_combos`.
    """
    opt_types: set[str] = set()

    for experiment in experiments.values():
        if experiment.get("verdict") not in ("yes", "partial"):
            continue
        idea = _resolve_experiment_idea(ideas, experiment)
        if idea and idea.get("optimization_type"):
            opt_types.add(idea["optimization_type"])

    for idea in ideas.values():
        if idea.get("outcome") in ("positive", "mixed") and idea.get("optimization_type"):
            opt_types.add(idea["optimization_type"])

    return opt_types


def _gap_key(kind: str, subject: str = "") -> str:
    """Return a stable gap identity from semantic kind and invariant subject."""
    kind = re.sub(r"\s+", "_", str(kind).strip().lower())
    subject = re.sub(r"\s+", " ", str(subject).strip().lower())
    return f"{kind}:{subject}" if subject else kind


def _legacy_gap_key(description: str) -> str | None:
    """Infer identities for old gap maps that predate the ``Key`` field."""
    desc = str(description or "").strip()
    if re.search(r"\d+\s*个\s*p\s*系列策略尚未尝试", desc, re.IGNORECASE):
        return _gap_key("untried_strategies")

    match = re.match(r"^(.+?)\s*瓶颈的主要策略未探索", desc)
    if match:
        return _gap_key("bottleneck_primary_unexplored", match.group(1))

    match = re.match(r"^瓶颈迁移:\s*(\S+)\s*→\s*(\S+)", desc)
    if match:
        return _gap_key("bottleneck_shift", f"{match.group(1)}->{match.group(2)}")

    match = re.match(r"^(.+?)\s*类优化方向尚未有成功的实验", desc)
    if match:
        return _gap_key("opt_type_uncovered", match.group(1))

    match = re.search(r"Claim ['\"]?(.+?)['\"]? 验证了.*?尚未与\s*(\S+)\s*策略", desc)
    if match:
        return _gap_key("claim_combination", f"{match.group(1)}|{match.group(2)}")
    return None


def _gap_identity(gap: dict) -> str | None:
    """Return the explicit identity, with a legacy description fallback."""
    explicit = str(gap.get("key") or "").strip()
    if explicit:
        return explicit.lower()
    kind = str(gap.get("kind") or "").strip()
    subject = str(gap.get("subject") or "").strip()
    if kind:
        return _gap_key(kind, subject)
    return _legacy_gap_key(str(gap.get("description", "")))


def _gap_kind_subject(key: str) -> tuple[str, str]:
    if ":" not in key:
        return key, ""
    return key.split(":", 1)


def _coalesce_existing_gaps(existing_gaps: list[dict]) -> None:
    """Keep one active record per identity while preserving referenced IDs.

    Legacy maps can contain several count-bearing descriptions for the same
    semantic gap.  The first record keeps its G<n> identity.  Later records
    remain as closed historical aliases so existing idea/edge references do
    not become dangling, but they no longer participate in open-gap context.
    """
    canonical: dict[str, dict] = {}
    for gap in existing_gaps:
        key = _gap_identity(gap)
        if not key:
            continue
        kind, subject = _gap_kind_subject(key)
        gap["key"] = key
        if not gap.get("kind"):
            gap["kind"] = kind
        if not gap.get("subject") and subject:
            gap["subject"] = subject
        previous = canonical.get(key)
        if previous is None:
            canonical[key] = gap
            continue
        if gap is previous:
            continue
        previous_id = previous.get("id", "")
        gap["status"] = "closed"
        gap["description"] = f"重复记录，已合并至 {previous_id}"
        gap["evidence"] = f"legacy gap identity {key}; canonical={previous_id}"
        gap["closure_reason"] = f"merged into {previous_id}"
        # Keep the historical G<n> record for edge/idea references, but remove
        # its active identity so the semantic gap is represented exactly once.
        gap["key"] = ""
        gap["kind"] = ""
        gap["subject"] = ""


def _successful_experiments_for_gap(
    kg_root: str, gap_id: str,
    ideas: dict[str, dict], experiments: dict[str, dict], claims: dict[str, dict],
) -> list[str]:
    """Return successful experiment refs connected through addresses_gap."""
    successful_refs: list[str] = []
    for edge in load_edges(kg_root):
        if edge.get("type") != "addresses_gap" or edge.get("to") != gap_id:
            continue
        source = str(edge.get("from") or "")
        source_experiments: list[dict] = []
        if source.startswith("idea:"):
            source_idea = ideas.get(source.removeprefix("idea:"))
            if source_idea:
                source_experiments = [
                    exp for exp in experiments.values()
                    if str(exp.get("idea_id", "")) == source
                ]
        elif source.startswith("claim:"):
            claim = claims.get(source.removeprefix("claim:"))
            if claim:
                refs = claim.get("supporting_experiments", [])
                if isinstance(refs, list):
                    for ref in refs:
                        ref_slug = str(ref).removeprefix("exp:")
                        if ref_slug in experiments:
                            source_experiments.append(experiments[ref_slug])
        for exp in source_experiments:
            if exp.get("verdict") in ("yes", "partial"):
                successful_refs.append(str(exp.get("node_id") or ""))
    return list(dict.fromkeys(ref for ref in successful_refs if ref))


def _close_addressed_gaps(
    kg_root: str, gaps: list[dict], strategy_index_path: str | None,
) -> None:
    """Close gaps only when graph evidence demonstrates they were addressed."""
    ideas = load_ideas(kg_root)
    experiments = load_experiments(kg_root)
    claims = load_claims(kg_root)
    successful_combos = _collect_tested_strategy_combos(
        ideas, experiments, successful_only=True,
    )
    successful_opt_types = _collect_successful_opt_types(ideas, experiments)
    successful_flat = set().union(*successful_combos) if successful_combos else set()
    all_strategies = _parse_strategy_index(strategy_index_path) if strategy_index_path else {}

    for gap in gaps:
        if gap.get("status", "open") != "open":
            continue
        key = _gap_identity(gap)
        if not key:
            continue
        kind, subject = _gap_kind_subject(key)
        reason_refs = _successful_experiments_for_gap(
            kg_root, f"gap:{gap.get('id', '')}", ideas, experiments, claims,
        )
        # Some gap kinds have an explicit semantic closure condition.  Those
        # conditions are authoritative: an addresses_gap edge is useful
        # provenance, but cannot close the gap on its own.  The final branch is
        # deliberately the fallback for kinds without a dedicated condition.
        if kind == "opt_type_uncovered":
            addressed = subject in successful_opt_types
            if addressed:
                reason_refs.append(f"successful optimization_type={subject}")
        elif kind == "bottleneck_primary_unexplored":
            addressed = False
            info = BOTTLENECK_STRATEGY_MAP.get(subject, {})
            primary = set(info.get("primary", []))
            if primary and any(primary.issubset(combo) for combo in successful_combos):
                addressed = True
                reason_refs.append(f"successful strategies={','.join(sorted(primary))}")
        elif kind == "untried_strategies":
            addressed = False
            if all_strategies:
                untouched = {
                    sid for sid in all_strategies
                    if sid.startswith("P") and sid not in successful_flat
                }
                if len(untouched) <= 5:
                    addressed = True
                    reason_refs.append(
                        "successful strategy coverage reduced remaining gap"
                    )
        else:
            # bottleneck_shift, claim_combination, and future unclassified
            # kinds have no stronger closure predicate yet.  A successful
            # experiment reached through addresses_gap is therefore the
            # generic, evidence-backed fallback.
            addressed = bool(reason_refs)

        if addressed:
            gap["status"] = "closed"
            gap["closure_reason"] = "closed by " + ", ".join(reason_refs)


def _parse_strategy_index(index_path: str) -> dict[str, dict]:
    """Parse strategy-index.md to extract strategy IDs and their categories.

    Returns dict[strategy_id] -> {name, category, when_to_apply, ...}
    """
    path = Path(index_path)
    if not path.exists():
        logger.warning("Strategy index not found: %s", index_path)
        return {}

    text = path.read_text(encoding="utf-8")
    strategies: dict[str, dict] = {}

    # Parse table rows: | ID | Name | Category | When to Apply | ...
    for line in text.split("\n"):
        # Match table rows like: | P1 | 双缓冲 | bandwidth | ... |
        m = re.match(r"\|\s*([PDAXR]\d+)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|", line)
        if m:
            sid = m.group(1).strip()
            strategies[sid] = {
                "id": sid,
                "name": m.group(2).strip(),
                "category": m.group(3).strip(),
                "when_to_apply": m.group(4).strip(),
            }

    return strategies


def _experiment_bottleneck(experiment: dict) -> str | None:
    """Extract the normalized bottleneck label from experiment evidence."""
    for field in ("profiling_evidence", "profiling_insight"):
        evidence = experiment.get(field)
        if not isinstance(evidence, dict):
            continue
        bottleneck = evidence.get("bottleneck_type") or evidence.get("bottleneck")
        if bottleneck:
            return str(bottleneck)
    return None


def _collect_seen_bottlenecks(kg_root: str) -> set[str]:
    """Collect bottlenecks observed in experiments and baseline evidence."""
    seen_bottlenecks: set[str] = set()
    for experiment in load_experiments(kg_root).values():
        # Reuse the canonical single-experiment extractor, and also retain
        # both evidence fields when a legacy record contains both labels.
        bottleneck = _experiment_bottleneck(experiment)
        if bottleneck:
            seen_bottlenecks.add(bottleneck)
        for field, key in (
            ("profiling_evidence", "bottleneck_type"),
            ("profiling_insight", "bottleneck"),
        ):
            evidence = experiment.get(field)
            if isinstance(evidence, dict) and evidence.get(key):
                seen_bottlenecks.add(str(evidence[key]))

    kg_json_path = Path(kg_root) / "knowledge_graph.json"
    if kg_json_path.exists():
        try:
            kg_data = json.loads(kg_json_path.read_text(encoding="utf-8"))
            baseline_evidence = kg_data.get("baseline_evidence")
            if (isinstance(baseline_evidence, dict)
                    and baseline_evidence.get("bottleneck_type")):
                seen_bottlenecks.add(str(baseline_evidence["bottleneck_type"]))
        except (json.JSONDecodeError, OSError):
            pass

    return seen_bottlenecks


# ---------------------------------------------------------------------------
# Gap detection functions
# ---------------------------------------------------------------------------

def detect_unexplored_directions(
    kg_root: str,
    strategy_index_path: str | None = None,
) -> list[dict]:
    """Detect strategy directions that haven't been tried yet.

    Compares tested strategies against the full strategy index to find
    optimization directions that haven't been explored.
    """
    ideas = load_ideas(kg_root)
    experiments = load_experiments(kg_root)
    tested_combos = _collect_tested_strategy_combos(ideas, experiments)
    tested_flat: set[str] = set()
    for combo in tested_combos:
        tested_flat.update(combo)

    gaps = []
    existing_gaps = parse_gap_map(kg_root)
    existing_gap_descs = {g["description"].lower() for g in existing_gaps}

    # 1. Check bottleneck types that haven't been explored
    seen_bottlenecks = _collect_seen_bottlenecks(kg_root)

    # Only generate gaps for bottleneck types that have actually been SEEN
    # in profiling data. Without this guard, an empty KG generates gaps for
    # every entry in BOTTLENECK_STRATEGY_MAP (16+ noise gaps).
    for bn_type, info in BOTTLENECK_STRATEGY_MAP.items():
        if bn_type == "near_optimal":
            continue
        if bn_type not in seen_bottlenecks:
            continue  # Skip bottlenecks we haven't actually observed
        primary = set(info.get("primary", []))
        if not primary:
            continue
        # If we've seen this bottleneck but haven't tried its primary strategies
        untested_primary = primary - tested_flat
        if untested_primary:
            desc = f"{bn_type} 瓶颈的主要策略未探索: {', '.join(sorted(untested_primary))}"
            gaps.append({
                "description": desc,
                "status": "open",
                "kind": "bottleneck_primary_unexplored",
                "subject": bn_type,
                "key": _gap_key("bottleneck_primary_unexplored", bn_type),
                "evidence": info.get("description", ""),
                "suggested_lenses": ["profiling_driven", "method_transfer"],
                "priority": 1.5 if bn_type in seen_bottlenecks else 0.8,
            })

    # 2. Check optimization types without coverage
    tested_opt_types = _collect_successful_opt_types(ideas, experiments)

    # Optimization-type coverage is meaningful only after at least one
    # experiment exists.  An empty KG has no evidence for declaring any type
    # unexplored.
    if experiments:
        core_opt_types = {"bandwidth", "tiling", "algorithm"}
        missing_types = core_opt_types - tested_opt_types
        for mt in missing_types:
            desc = f"{mt} 类优化方向尚未有成功的实验"
            gaps.append({
                "description": desc,
                "status": "open",
                "kind": "opt_type_uncovered",
                "subject": mt,
                "key": _gap_key("opt_type_uncovered", mt),
                "evidence": f"所有已通过实验均非 {mt} 类型",
                "suggested_lenses": ["method_transfer", "gap_analysis"],
                "priority": 1.2,
            })

    # 3. If strategy index is available, find completely untouched strategies
    if strategy_index_path:
        all_strategies = _parse_strategy_index(strategy_index_path)
        untouched_important = []
        for sid, info in all_strategies.items():
            if sid not in tested_flat and sid.startswith("P"):
                untouched_important.append(sid)
        if len(untouched_important) > 5:
            # Only report as a gap if many strategies are untouched
            desc = (f"{len(untouched_important)} 个 P 系列策略尚未尝试, "
                    f"例如: {', '.join(sorted(untouched_important)[:5])}")
            gaps.append({
                "description": desc,
                "status": "open",
                "kind": "untried_strategies",
                "subject": "",
                "key": _gap_key("untried_strategies"),
                "evidence": f"策略覆盖率: {len(tested_flat)}/{len(all_strategies)}",
                "suggested_lenses": ["method_transfer"],
                "priority": 0.8,
            })

    return gaps


def detect_bottleneck_shift_gaps(
    kg_root: str,
    baseline_bottleneck: str | None = None,
) -> list[dict]:
    """Detect experiments whose bottleneck differs from the baseline.

    The baseline comes from the explicit argument, root-level
    ``baseline_evidence.bottleneck_type``, or (as a compatibility fallback)
    the first experiment's bottleneck.  Comparing every experiment to this
    fixed reference avoids treating a parent/child lineage as the baseline.
    """
    experiments = load_experiments(kg_root)
    existing_gaps = parse_gap_map(kg_root)
    existing_gap_descs = {g["description"].lower() for g in existing_gaps}

    if baseline_bottleneck is None:
        kg_json_path = Path(kg_root) / "knowledge_graph.json"
        if kg_json_path.exists():
            try:
                kg_data = json.loads(kg_json_path.read_text(encoding="utf-8"))
                baseline_evidence = kg_data.get("baseline_evidence")
                if isinstance(baseline_evidence, dict):
                    baseline_bottleneck = baseline_evidence.get("bottleneck_type")
            except (json.JSONDecodeError, OSError):
                pass

    # Legacy/partial KGs may not have root-level baseline evidence.  Use the
    # first experiment in stable slug order as a proxy in that case.
    if baseline_bottleneck is None:
        for slug in sorted(experiments):
            baseline_bottleneck = _experiment_bottleneck(experiments[slug])
            if baseline_bottleneck:
                break

    if not baseline_bottleneck:
        return []

    shift_sources: dict[tuple[str, str], dict[str, list[str]]] = {}
    for slug, experiment in sorted(experiments.items()):
        experiment_bottleneck = _experiment_bottleneck(experiment)
        if not experiment_bottleneck or experiment_bottleneck == baseline_bottleneck:
            continue

        pair = (str(baseline_bottleneck), experiment_bottleneck)
        source_data = shift_sources.setdefault(
            pair, {"ideas": [], "experiments": []},
        )
        idea_id = experiment.get("idea_id")
        idea_ref = str(idea_id) if idea_id else ""
        experiment_ref = str(
            experiment.get("node_id") or f"exp:{slug}"
        )
        if idea_ref and idea_ref not in source_data["ideas"]:
            source_data["ideas"].append(idea_ref)
        if experiment_ref not in source_data["experiments"]:
            source_data["experiments"].append(experiment_ref)

    gaps = []
    for (baseline_bn, experiment_bn), source_data in shift_sources.items():
        desc = f"瓶颈迁移: {baseline_bn} → {experiment_bn}"
        # Accept a legacy description with a per-experiment suffix as the
        # same stable shift, so old gap maps do not receive a duplicate.
        desc_lower = desc.lower()
        # Preserve the legacy prefix match for old, unkeyed shift descriptions.
        # Keyed records are intentionally allowed through so their evidence can
        # be refreshed in place by analyze_gaps().
        if any(
            not existing.get("key")
            and str(existing.get("description", "")).lower().startswith(desc_lower)
            for existing in existing_gaps
        ):
            continue

        new_bn_info = BOTTLENECK_STRATEGY_MAP.get(experiment_bn, {})
        suggested = new_bn_info.get("primary", [])
        idea_refs = source_data["ideas"]
        source_refs = idea_refs or source_data["experiments"]
        evidence = f"来源: {', '.join(source_refs)}"
        if idea_refs:
            evidence += f"; 实验: {', '.join(source_data['experiments'])}"
        evidence += (
            f"; 基线瓶颈 {baseline_bn} 后观测到 {experiment_bn}; "
            f"建议策略: {', '.join(suggested)}"
        )
        gaps.append({
            "description": desc,
            "status": "open",
            "kind": "bottleneck_shift",
            "subject": f"{baseline_bn}->{experiment_bn}",
            "key": _gap_key("bottleneck_shift", f"{baseline_bn}->{experiment_bn}"),
            "evidence": evidence,
            "suggested_lenses": ["profiling_driven", "continuation"],
            "priority": 1.8,
        })
        existing_gap_descs.add(desc_lower)

    return gaps


def detect_claim_driven_gaps(kg_root: str) -> list[dict]:
    """Detect gaps from claim analysis.

    For example, a verified claim about P1 being essential might suggest
    testing P1 combinations that haven't been tried yet.
    """
    claims = load_claims(kg_root)
    ideas = load_ideas(kg_root)
    experiments = load_experiments(kg_root)
    existing_gaps = parse_gap_map(kg_root)
    existing_gap_descs = {g["description"].lower() for g in existing_gaps}

    gaps = []

    tested_combos = _collect_tested_strategy_combos(ideas, experiments)
    seen_bottlenecks = _collect_seen_bottlenecks(kg_root)

    for claim in claims.values():
        if claim.get("status") != "verified":
            continue

        strat_impl = claim.get("strategy_implications", [])
        if not isinstance(strat_impl, list) or not strat_impl:
            continue

        # Check if there are untested combinations including the claim's strategies
        claim_strats = set(strat_impl)
        # Look for bottleneck strategies that pair well with claim strategies
        for bn_type, info in BOTTLENECK_STRATEGY_MAP.items():
            if bn_type not in seen_bottlenecks:
                continue
            primary = set(info.get("primary", []))
            potential_combo = claim_strats | primary
            if len(potential_combo) <= len(claim_strats):
                continue  # No new strategies added

            # Check if this combo has been tested
            already_tested = any(
                combo >= potential_combo for combo in tested_combos
            )
            if not already_tested:
                desc = (f"Claim '{claim.get('name', '?')}' 验证了 "
                        f"{', '.join(sorted(claim_strats))}，"
                        f"但尚未与 {bn_type} 策略 "
                        f"{', '.join(sorted(primary - claim_strats))} 组合测试")
                gaps.append({
                    "description": desc,
                    "status": "open",
                    "kind": "claim_combination",
                    "subject": (
                        f"{claim.get('name') or claim.get('node_id', '?')}|{bn_type}"
                    ),
                    "key": _gap_key(
                        "claim_combination",
                        f"{claim.get('name') or claim.get('node_id', '?')}|{bn_type}",
                    ),
                    "evidence": f"基于 claim:{claim.get('node_id', '?')}",
                    "suggested_lenses": ["continuation", "gap_analysis"],
                    "priority": 1.3,
                })

    return gaps


# ---------------------------------------------------------------------------
# Gap prioritization
# ---------------------------------------------------------------------------

def prioritize_gaps(
    gaps: list[dict],
    stagnation_count: int = 0,
    stagnation_count_vs_base: int = 0,
) -> list[dict]:
    """Prioritize gaps, boosting priority when stagnation is detected.

    Args:
        gaps: list of gap dicts with 'priority' field
        stagnation_count: global stagnation counter
        stagnation_count_vs_base: vs-base stagnation counter

    Returns:
        gaps sorted by effective priority (descending), with priority adjusted
    """
    stagnation_boost = 0.0
    if stagnation_count >= 1:
        stagnation_boost += 0.5 * stagnation_count
    if stagnation_count_vs_base >= 1:
        stagnation_boost += 0.3 * stagnation_count_vs_base

    for gap in gaps:
        base_priority = gap.get("priority", 1.0)
        # Gaps suggesting open_exploration get extra boost during stagnation
        lenses = gap.get("suggested_lenses", [])
        exploration_boost = 0.0
        if stagnation_boost > 0 and "open_exploration" in lenses:
            exploration_boost = 0.5

        gap["effective_priority"] = base_priority + stagnation_boost + exploration_boost

    gaps.sort(key=lambda g: g.get("effective_priority", 0), reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_gaps(
    kg_root: str,
    strategy_index_path: str | None = None,
    stagnation_count: int = 0,
    stagnation_count_vs_base: int = 0,
    baseline_bottleneck: str | None = None,
) -> list[dict]:
    """Run the full gap analysis pipeline.

    1. Load existing gaps
    2. Detect new gaps from multiple sources
    3. Merge (avoid duplicates)
    4. Prioritize
    5. Write updated gap_map.md

    Returns the merged, prioritized list of gaps.
    """
    # 1. Load existing gaps.  Legacy descriptions are assigned a semantic key
    # here; duplicate legacy records become closed aliases so their referenced
    # IDs remain valid without polluting the open-gap context.
    existing_gaps = parse_gap_map(kg_root)
    _coalesce_existing_gaps(existing_gaps)

    # 2. Detect new gaps from multiple sources
    new_gaps = []
    new_gaps.extend(detect_unexplored_directions(kg_root, strategy_index_path))
    new_gaps.extend(detect_bottleneck_shift_gaps(kg_root, baseline_bottleneck))
    new_gaps.extend(detect_claim_driven_gaps(kg_root))

    # 3. Merge by stable identity.  A legacy description-only match remains a
    # fallback for maps that cannot be classified from their text.
    existing_by_key: dict[str, dict] = {}
    existing_descs: dict[str, dict] = {}
    for existing in existing_gaps:
        key = _gap_identity(existing)
        if key and key not in existing_by_key:
            existing_by_key[key] = existing
        existing_descs.setdefault(str(existing.get("description", "")).lower(), existing)

    # Compute next available ID from existing gaps
    max_gap_num = 0
    for g in existing_gaps:
        m = re.match(r"G(\d+)", g.get("id", ""))
        if m:
            max_gap_num = max(max_gap_num, int(m.group(1)))
    added_count = 0
    for gap in new_gaps:
        key = _gap_identity(gap)
        matched = existing_by_key.get(key) if key else None
        if matched is None:
            matched = existing_descs.get(str(gap.get("description", "")).lower())

        if matched is not None:
            # Keep the original G<n> and status.  In particular, a closed gap
            # must not be reopened merely because its detector still observes
            # the historical condition.
            old_status = matched.get("status", "open")
            preserved_closure_reason = matched.get("closure_reason", "")
            matched.update({
                "description": gap.get("description", matched.get("description", "")),
                "kind": gap.get("kind") or matched.get("kind", ""),
                "subject": gap.get("subject") or matched.get("subject", ""),
                "key": key or matched.get("key", ""),
                "evidence": gap.get("evidence", matched.get("evidence", "")),
                "suggested_lenses": gap.get(
                    "suggested_lenses", matched.get("suggested_lenses", []),
                ),
                "priority": gap.get("priority", matched.get("priority", 1.0)),
                "status": old_status,
            })
            # Detector evidence is intentionally refreshed, but irreversible
            # closure provenance belongs to the existing record and must
            # survive that in-place merge.
            if old_status != "open" and preserved_closure_reason:
                matched["closure_reason"] = preserved_closure_reason
            if key:
                existing_by_key[key] = matched
            existing_descs[str(matched.get("description", "")).lower()] = matched
            continue

        max_gap_num += 1
        gap["id"] = f"G{max_gap_num}"
        existing_gaps.append(gap)
        if key:
            existing_by_key[key] = gap
        existing_descs[str(gap.get("description", "")).lower()] = gap
        added_count += 1

    # 4. Close only with successful experiment evidence, explicit gap edges,
    # or a coverage condition that is now demonstrably satisfied.
    _close_addressed_gaps(kg_root, existing_gaps, strategy_index_path)

    # 5. Prioritize
    all_gaps = prioritize_gaps(
        existing_gaps,
        stagnation_count=stagnation_count,
        stagnation_count_vs_base=stagnation_count_vs_base,
    )

    # 6. Write updated gap_map.md
    write_gap_map(kg_root, all_gaps)
    append_log(kg_root, f"analyze_gaps: {added_count} new gaps found, "
               f"{len(all_gaps)} total")

    return all_gaps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    gaps = analyze_gaps(
        args.kg_root,
        strategy_index_path=args.strategy_index,
        stagnation_count=args.stagnation_count,
        stagnation_count_vs_base=args.stagnation_count_vs_base,
        baseline_bottleneck=args.baseline_bottleneck,
    )
    print(json.dumps([{
        "id": g.get("id", "?"),
        "description": g["description"],
        "status": g.get("status", "open"),
        "priority": g.get("effective_priority", g.get("priority", 1.0)),
    } for g in gaps], indent=2, ensure_ascii=False))


def cmd_detect_unexplored(args: argparse.Namespace) -> None:
    gaps = detect_unexplored_directions(args.kg_root, args.strategy_index)
    print(json.dumps(gaps, indent=2, ensure_ascii=False))


def cmd_detect_shift(args: argparse.Namespace) -> None:
    gaps = detect_bottleneck_shift_gaps(args.kg_root, args.baseline_bottleneck)
    print(json.dumps(gaps, indent=2, ensure_ascii=False))


def cmd_prioritize(args: argparse.Namespace) -> None:
    gaps = parse_gap_map(args.kg_root)
    prioritized = prioritize_gaps(gaps, args.stagnation_count,
                                   args.stagnation_count_vs_base)
    print(json.dumps([{
        "id": g.get("id", "?"),
        "description": g["description"],
        "priority": g.get("effective_priority", g.get("priority", 1.0)),
    } for g in prioritized], indent=2, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gap Analysis for evolution knowledge graph."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_analyze = subparsers.add_parser("analyze", help="Run full gap analysis")
    p_analyze.add_argument("kg_root")
    p_analyze.add_argument("--strategy-index", default=None)
    p_analyze.add_argument("--stagnation-count", type=int, default=0)
    p_analyze.add_argument("--stagnation-count-vs-base", type=int, default=0)
    p_analyze.add_argument("--baseline-bottleneck", default=None)
    p_analyze.set_defaults(func=cmd_analyze)

    p_unexp = subparsers.add_parser("detect_unexplored",
                                     help="Detect unexplored directions")
    p_unexp.add_argument("kg_root")
    p_unexp.add_argument("--strategy-index", default=None)
    p_unexp.set_defaults(func=cmd_detect_unexplored)

    p_shift = subparsers.add_parser("detect_bottleneck_shift",
                                     help="Detect bottleneck shift gaps")
    p_shift.add_argument("kg_root")
    p_shift.add_argument("--baseline-bottleneck", default=None)
    p_shift.set_defaults(func=cmd_detect_shift)

    p_prio = subparsers.add_parser("prioritize", help="Prioritize existing gaps")
    p_prio.add_argument("kg_root")
    p_prio.add_argument("--stagnation-count", type=int, default=0)
    p_prio.add_argument("--stagnation-count-vs-base", type=int, default=0)
    p_prio.set_defaults(func=cmd_prioritize)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    result = args.func(args)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
