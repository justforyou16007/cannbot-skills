#!/usr/bin/env python3
"""
Claim Engine — Generates and manages claims from accumulated experiment evidence.

Claims are evidence-backed conclusions about optimization strategies. They are
generated when sufficient experimental evidence accumulates:
- 3+ experiments confirm a strategy → verified claim
- 2+ experiments show a strategy fails → refutation claim

Claims influence the utility function (supporting evidence boosts related ideas)
and idea generation (claims narrow the search space).

Usage:
    python3 claim_engine.py check <kg_root>
    python3 claim_engine.py generate <kg_root> [--dry-run]
    python3 claim_engine.py strategy_implications <kg_root>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Import sibling modules
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from kg_wiki import (  # noqa: E402
    load_experiments, load_claims, load_ideas,
    add_claim, append_log, _slugify,
)

logger = logging.getLogger(__name__)

# Thresholds for claim generation
_MIN_SUPPORTING_EXPERIMENTS = 3   # need 3+ positive experiments to verify
_MIN_REFUTING_EXPERIMENTS = 2     # need 2+ negative experiments to refute
_MIN_CONFIDENCE_FOR_VERIFY = 0.75


# ---------------------------------------------------------------------------
# Claim condition checking
# ---------------------------------------------------------------------------

def _group_experiments_by_strategy(
    experiments: dict[str, dict],
    ideas: dict[str, dict],
) -> dict[str, list[dict]]:
    """Group experiments by individual strategy tags.

    Returns dict[strategy_id] -> list of experiment dicts (with idea info merged).
    """
    strategy_exps: dict[str, list[dict]] = defaultdict(list)

    for exp in experiments.values():
        # Find the linked idea to get strategy tags
        idea_id = exp.get("idea_id", "")
        idea_slug = idea_id.replace("idea:", "") if idea_id.startswith("idea:") else ""
        idea = ideas.get(idea_slug, {})

        stags = idea.get("strategy_tags", [])
        if not isinstance(stags, list):
            continue

        exp_with_idea = {**exp, "_idea": idea, "_idea_slug": idea_slug}
        for stag in stags:
            strategy_exps[stag].append(exp_with_idea)

    return dict(strategy_exps)


def _group_experiments_by_bottleneck(
    experiments: dict[str, dict],
) -> dict[str, list[dict]]:
    """Group experiments by bottleneck type from profiling evidence."""
    bn_exps: dict[str, list[dict]] = defaultdict(list)
    for exp in experiments.values():
        pe = exp.get("profiling_evidence")
        if isinstance(pe, dict) and pe.get("bottleneck_type"):
            bn_exps[pe["bottleneck_type"]].append(exp)
    return dict(bn_exps)


def check_claim_conditions(kg_root: str) -> list[dict]:
    """Check which claim conditions are met based on current evidence.

    Returns a list of claim candidates, each with:
    - claim_type: "strategy_verified" | "strategy_refuted" | "bottleneck_pattern"
    - key: strategy ID or bottleneck type
    - supporting_experiments: list of exp node_ids
    - refuting_experiments: list of exp node_ids
    - confidence: float
    - reason: description
    """
    experiments = load_experiments(kg_root)
    ideas = load_ideas(kg_root)
    existing_claims = load_claims(kg_root)

    candidates = []
    strategy_exps = _group_experiments_by_strategy(experiments, ideas)

    for stag, exps in strategy_exps.items():
        # Count verdicts
        positive = [e for e in exps if e.get("verdict") == "yes"]
        negative = [e for e in exps if e.get("verdict") == "no"]
        partial = [e for e in exps if e.get("verdict") == "partial"]

        # Check for strategy verification
        if len(positive) >= _MIN_SUPPORTING_EXPERIMENTS:
            # Check if claim already exists
            claim_slug = _slugify(f"{stag}-verified")
            if claim_slug in existing_claims:
                # Check if we have new supporting evidence
                existing = existing_claims[claim_slug]
                existing_supp = existing.get("supporting_experiments", [])
                new_supp = [f"exp:{e.get('node_id', '').replace('exp:', '')}"
                            for e in positive]
                if set(new_supp) <= set(existing_supp):
                    continue  # No new evidence

            avg_speedup = sum(e.get("speedup", 1.0) or 1.0 for e in positive) / len(positive)
            confidence = min(0.99, 0.5 + 0.1 * len(positive) + 0.05 * len(partial))

            candidates.append({
                "claim_type": "strategy_verified",
                "polarity": "positive",
                "key": stag,
                "supporting_experiments": [
                    e.get("node_id", "exp:?") for e in positive
                ],
                "refuting_experiments": [
                    e.get("node_id", "exp:?") for e in negative
                ],
                "confidence": confidence,
                "avg_speedup": avg_speedup,
                "reason": (f"策略 {stag} 在 {len(positive)} 个实验中成功"
                           f"（平均 speedup {avg_speedup:.2f}x），"
                           f"{len(negative)} 个失败"),
            })

        # Check for strategy refutation
        # ONLY use strategy_infeasible failures — impl_error means the strategy
        # might still work, just the implementation was wrong.
        infeasible = [e for e in negative
                      if e.get("failure_type") == "strategy_infeasible"]
        if len(infeasible) >= _MIN_REFUTING_EXPERIMENTS and len(positive) == 0:
            claim_slug = _slugify(f"{stag}-refuted")
            if claim_slug in existing_claims:
                continue

            confidence = min(0.95, 0.4 + 0.15 * len(infeasible))

            candidates.append({
                "claim_type": "strategy_refuted",
                "polarity": "negative",
                "key": stag,
                "supporting_experiments": [],
                "refuting_experiments": [
                    e.get("node_id", "exp:?") for e in infeasible
                ],
                "confidence": confidence,
                "primary_failure": "strategy_infeasible",
                "reason": (f"策略 {stag} 在 {len(infeasible)} 个实验中因策略不可行而失败"
                           f"（排除了实现错误）"),
            })

    # Check for bottleneck pattern claims
    bn_exps = _group_experiments_by_bottleneck(experiments)
    for bn_type, exps in bn_exps.items():
        if len(exps) < _MIN_SUPPORTING_EXPERIMENTS:
            continue

        claim_slug = _slugify(f"bn-{bn_type}-pattern")
        if claim_slug in existing_claims:
            continue

        # What strategies consistently work for this bottleneck?
        positive_exps = [e for e in exps if e.get("verdict") == "yes"]
        if len(positive_exps) < 2:
            continue

        # Find common strategies across positive experiments
        stag_counts: Counter = Counter()
        for exp in positive_exps:
            idea_id = exp.get("idea_id", "")
            idea_slug = idea_id.replace("idea:", "")
            idea = ideas.get(idea_slug, {})
            stags = idea.get("strategy_tags", [])
            if isinstance(stags, list):
                stag_counts.update(stags)

        common_stags = [s for s, c in stag_counts.most_common()
                        if c >= len(positive_exps) * 0.6]  # present in 60%+ of successes
        if common_stags:
            candidates.append({
                "claim_type": "bottleneck_pattern",
                "polarity": "positive",
                "key": bn_type,
                "common_strategies": common_stags,
                "supporting_experiments": [
                    e.get("node_id", "exp:?") for e in positive_exps
                ],
                "refuting_experiments": [],
                "confidence": min(0.90, 0.4 + 0.1 * len(positive_exps)),
                "reason": (f"瓶颈 {bn_type} 的 {len(positive_exps)} 个成功实验中，"
                           f"策略 {', '.join(common_stags)} 出现率 ≥60%"),
            })

    return candidates


# ---------------------------------------------------------------------------
# Claim generation
# ---------------------------------------------------------------------------

def generate_claims(kg_root: str, dry_run: bool = False) -> list[dict]:
    """Generate claims from candidates that meet conditions.

    Returns list of generated/updated claim metadata.
    """
    candidates = check_claim_conditions(kg_root)
    generated = []

    for cand in candidates:
        claim_type = cand["claim_type"]
        key = cand["key"]

        if claim_type == "strategy_verified":
            slug = _slugify(f"{key}-verified")
            name = f"策略 {key} 对此算子有效"
            status = "verified" if cand["confidence"] >= _MIN_CONFIDENCE_FOR_VERIFY else "unproven"
            statement = cand["reason"]
            scope = f"基于当前算子的 {len(cand['supporting_experiments'])} 个实验结果"
            strategy_implications = [key]
            bottleneck_labels = []  # Not bottleneck-specific

        elif claim_type == "strategy_refuted":
            slug = _slugify(f"{key}-refuted")
            name = f"策略 {key} 对此算子不可行"
            status = "verified" if cand["confidence"] >= _MIN_CONFIDENCE_FOR_VERIFY else "unproven"
            statement = cand["reason"]
            scope = (f"主要失败类型: {cand.get('primary_failure', '?')}; "
                     f"基于 {len(cand['refuting_experiments'])} 个失败实验")
            strategy_implications = [key]
            bottleneck_labels = []

        elif claim_type == "bottleneck_pattern":
            slug = _slugify(f"bn-{key}-pattern")
            name = f"瓶颈 {key} 下 {', '.join(cand.get('common_strategies', []))} 最有效"
            status = "unproven"  # Patterns need more evidence
            statement = cand["reason"]
            scope = f"基于 {len(cand['supporting_experiments'])} 个成功实验的模式识别"
            strategy_implications = cand.get("common_strategies", [])
            bottleneck_labels = [key]

        else:
            continue

        polarity = cand.get("polarity", "positive")

        if not dry_run:
            add_claim(
                kg_root, slug, name,
                status=status,
                polarity=polarity,
                statement=statement,
                scope=scope,
                supporting_experiments=cand.get("supporting_experiments", []),
                refuting_experiments=cand.get("refuting_experiments", []),
                strategy_implications=strategy_implications,
                bottleneck_labels=bottleneck_labels,
                confidence=cand["confidence"],
                update_on_exist=True,
            )

        generated.append({
            "slug": slug,
            "name": name,
            "status": status,
            "confidence": cand["confidence"],
            "claim_type": claim_type,
        })

    if generated:
        append_log(kg_root, f"generate_claims: {len(generated)} claims generated")
        logger.info("Generated %d claims", len(generated))

    return generated


# ---------------------------------------------------------------------------
# Strategy implications extraction
# ---------------------------------------------------------------------------

def get_strategy_implications(kg_root: str) -> dict[str, list[str]]:
    """提取 claim→strategy 影响，供 query_pack 和 idea 生成使用。"""
    claims = load_claims(kg_root)
    implications: dict[str, list[str]] = defaultdict(list)

    for slug, claim in claims.items():
        if claim.get("status") not in ("verified", "unproven"):
            continue
        strat_impl = claim.get("strategy_implications", [])
        if not isinstance(strat_impl, list):
            continue
        for stag in strat_impl:
            implications[stag].append(f"claim:{slug}")

    return dict(implications)


def get_refuted_strategies(kg_root: str) -> set[str]:
    """Get strategies that have been refuted by claims.

    Returns set of strategy IDs that should be penalized in utility scoring.
    """
    claims = load_claims(kg_root)
    refuted: set[str] = set()

    for claim in claims.values():
        if claim.get("status") != "verified":
            continue
        name = claim.get("name", "")
        polarity = claim.get("polarity")
        is_refuted = polarity == "negative"
        if not polarity:
            # Fallback for claims written before the polarity field existed.
            is_refuted = "不可行" in name or "refuted" in name.lower()
        if is_refuted:
            strat_impl = claim.get("strategy_implications", [])
            if isinstance(strat_impl, list):
                refuted.update(strat_impl)

    return refuted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> None:
    candidates = check_claim_conditions(args.kg_root)
    print(json.dumps(candidates, indent=2, ensure_ascii=False))


def cmd_generate(args: argparse.Namespace) -> None:
    generated = generate_claims(args.kg_root, dry_run=args.dry_run)
    print(json.dumps(generated, indent=2, ensure_ascii=False))


def cmd_implications(args: argparse.Namespace) -> None:
    impl = get_strategy_implications(args.kg_root)
    print(json.dumps(impl, indent=2, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Claim Engine for evolution knowledge graph."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_check = subparsers.add_parser("check", help="Check claim conditions")
    p_check.add_argument("kg_root")
    p_check.set_defaults(func=cmd_check)

    p_gen = subparsers.add_parser("generate", help="Generate claims from evidence")
    p_gen.add_argument("kg_root")
    p_gen.add_argument("--dry-run", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_impl = subparsers.add_parser("strategy_implications",
                                    help="Extract strategy implications from claims")
    p_impl.add_argument("kg_root")
    p_impl.set_defaults(func=cmd_implications)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    result = args.func(args)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
