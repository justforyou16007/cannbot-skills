#!/usr/bin/env python3
"""Knowledge-graph orchestration for AscendC evolution sessions.

This module is the KG-mode counterpart of :mod:`wm_ops`.  It owns the
round-level transaction (experiment creation, idea outcome updates, claims,
gaps, and query-pack refresh) while leaving entity persistence to
``kg_wiki.py``.

The canonical mode predicate is deliberately small and deterministic.  A
``world_model.json``-only directory is a legacy MCTS session; every new
directory defaults to KG, and ``knowledge_graph.json`` is the durable KG
marker.  Having both stores is rejected as state corruption.  Consumers
should use :func:`detect_mode` instead of reimplementing this test.

Examples::

    python3 kg_ops.py init --evo-dir <evo_dir> --op-name FastGELU \
        --requested-rounds 3 --parallel-num 3 \
        --baseline-performance '{"time_us": 100.0}'
    python3 kg_ops.py refine --evo-dir <evo_dir> --round 1 \
        --results /tmp/round_results.json
    python3 kg_ops.py summary --evo-dir <evo_dir>
    python3 kg_ops.py validate --evo-dir <evo_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import claim_engine  # noqa: E402
import gap_analysis  # noqa: E402
import idea_generation  # noqa: E402
from profiling_evidence import (  # noqa: E402
    BOTTLENECK_STRATEGY_MAP,
    extract_profiling_evidence,
    synthesize_analysis_from_pipeline,
)
import query_pack  # noqa: E402
from kg_wiki import (  # noqa: E402
    _parse_frontmatter,
    _read_text,
    add_experiment,
    get_stats,
    init_kg,
    load_experiments,
    load_ideas,
    parse_gap_map,
    sync_knowledge_graph,
    update_idea_outcome,
)

LOGGER = logging.getLogger(__name__)

# Keep this duplicated with wm_ops.DRIFT_THRESHOLD and
# wm_ops._decide_drift_status deliberately.  wm_ops.py is legacy-compat code;
# KG mode must preserve the same circuit-breaker semantics without importing
# or refactoring that implementation.
DRIFT_THRESHOLD = 2

_SUPERVISOR_TRIGGERS = {
    "stall",
    "profiling_driven_all_failed",
    "profiling_blind_spot",
}

_QUALITY_STAGNATION_THRESHOLD = {
    "good": 1.02,
    "acceptable": 1.03,
    "noisy": 1.05,
}
_VALID_EDGE_TYPES = {
    "tested_by",
    "supports",
    "invalidates",
    "addresses_gap",
    "inspired_by",
    "derived_from",
    "depends_on",
    "refutes",
}
_EDGE_DIRECTIONS = {
    "tested_by": ({"idea"}, {"experiment"}),
    "supports": ({"experiment"}, {"claim", "idea"}),
    "invalidates": ({"experiment"}, {"claim", "idea"}),
    "addresses_gap": ({"idea", "claim"}, {"gap"}),
    "inspired_by": ({"idea"}, {"idea"}),
    "derived_from": ({"idea"}, {"experiment"}),
    "depends_on": ({"claim"}, {"claim"}),
    "refutes": ({"claim"}, {"claim"}),
}


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort integer coercion for integrity/reporting paths."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _now_iso() -> str:
    """Return the repository's timezone-aware session timestamp."""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")


def detect_mode(evo_dir: str) -> str:
    """Return the immutable session mode; new output directories default KG."""
    root = Path(evo_dir)
    has_kg = (root / "knowledge_graph.json").is_file()
    has_mcts = (root / "world_model.json").is_file()
    if has_kg and has_mcts:
        raise ValueError(
            "conflicting evolution stores: both knowledge_graph.json and "
            "world_model.json exist; recover one immutable session mode before continuing"
        )
    return "mcts" if has_mcts else "kg"


def _decide_drift_status(kg: dict[str, Any]) -> tuple[str, str]:
    """Mirror ``wm_ops._decide_drift_status`` for a KG dictionary."""
    stagnation_count = int(kg.get("stagnation_count", 0) or 0)
    stagnation_count_vs_base = int(kg.get("stagnation_count_vs_base", 0) or 0)

    if stagnation_count >= DRIFT_THRESHOLD:
        return "replan_required", (
            f"stagnation_count={stagnation_count} ≥ {DRIFT_THRESHOLD}: "
            f"no global-best progress for {stagnation_count} consecutive rounds"
        )
    if stagnation_count_vs_base >= DRIFT_THRESHOLD:
        return "replan_required", (
            f"stagnation_count_vs_base={stagnation_count_vs_base} ≥ {DRIFT_THRESHOLD}: "
            f"no variant beat its parent for {stagnation_count_vs_base} consecutive rounds"
        )
    return "normal", (
        f"stagnation_count={stagnation_count}, "
        f"vs_base={stagnation_count_vs_base} below threshold"
    )


def _maybe_update_drift_status(evo_dir: str, kg: dict[str, Any]) -> None:
    """Best-effort KG drift writer; never lets refine fail.

    This intentionally follows wm_ops._maybe_update_drift_status: it is
    idempotent, returns early when state is absent or unchanged, uses the same
    log markers, and suppresses every exception.
    """
    try:
        state_path = Path(evo_dir) / "state.json"
        if not state_path.is_file():
            return
        from state_ops import _read_state, _write_state  # type: ignore

        state = _read_state(evo_dir)
        previous = state.get("drift_status", "normal")
        new_status, reason = _decide_drift_status(kg)
        if previous == new_status:
            return
        state["drift_status"] = new_status
        _write_state(evo_dir, state)
        marker = "[DRIFT]" if new_status == "replan_required" else "[drift cleared]"
        LOGGER.info(
            "  %s drift_status %s → %s (%s)",
            marker, previous, new_status, reason,
        )
    except Exception as exc:  # pragma: no cover - defensive circuit breaker
        LOGGER.debug("_maybe_update_drift_status suppressed: %s", exc)


def _read_drift_status(evo_dir: str) -> str:
    """Read the runtime drift flag without making it a KG hard dependency."""
    try:
        from state_ops import _read_state  # type: ignore
        state_path = Path(evo_dir) / "state.json"
        if not state_path.is_file():
            return "normal"
        return str(_read_state(evo_dir).get("drift_status", "normal"))
    except Exception:  # pragma: no cover - summary must remain best effort
        return "normal"


def _maybe_mark_profiling_skipped(
    evo_dir: str, results_dir: str | None, results: list[dict[str, Any]],
) -> None:
    """Mirror the legacy R9 gate for KG rounds using nested eval semantics."""
    if not results_dir or not (Path(evo_dir) / "state.json").is_file():
        return
    missing: list[int] = []
    root = Path(results_dir)
    for result in results:
        evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
        if evaluation.get("precision_passed") is not True:
            continue
        parallel_dir = root / f"parallel_{result['parallel_index']}"
        complete = any(
            path.name.startswith("op_summary_") and path.suffix == ".csv"
            for path in (parallel_dir / "profiling").rglob("*.csv")
        ) if (parallel_dir / "profiling").is_dir() else False
        if not complete:
            missing.append(int(result["parallel_index"]))
    if not missing:
        return
    try:
        from state_ops import _read_state, _write_state  # type: ignore
        state = _read_state(evo_dir)
        pending = state.setdefault("must_run_before_next_round", [])
        if "msprof" not in pending:
            pending.append("msprof")
            _write_state(evo_dir, state)
        LOGGER.warning(
            "[R9] KG profiling artifacts missing for passed partial(s) %s; "
            "must_run_before_next_round += msprof",
            missing,
        )
    except Exception as exc:  # pragma: no cover - state gate is best effort
        LOGGER.debug("_maybe_mark_profiling_skipped suppressed: %s", exc)


def _evo_dir_from_args(args: argparse.Namespace) -> str:
    value = getattr(args, "evo_dir", None) or getattr(args, "evo_dir_pos", None)
    if not value:
        raise ValueError("--evo-dir (or positional evo_dir) is required")
    evo_dir = os.path.abspath(value)
    if not os.path.isdir(evo_dir):
        raise ValueError(f"evo_dir does not exist or is not a directory: {evo_dir}")
    return evo_dir


def _read_json_value(value: str | None, label: str, *, default: Any = None) -> Any:
    """Read a JSON object from an inline value or a file path."""
    if value is None or value == "":
        return default
    candidate = value[1:] if value.startswith("@") else value
    path = Path(candidate)
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON or a readable file: {value[:120]}") from exc


def _read_kg(evo_dir: str) -> dict[str, Any]:
    path = Path(evo_dir) / "knowledge_graph.json"
    if not path.is_file():
        raise ValueError(f"KG session not initialized: missing {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid knowledge_graph.json at {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"knowledge_graph.json must contain an object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _default_strategy_index() -> str | None:
    candidate = _SCRIPT_DIR.parent.parent / "evolution-strategies" / "references" / "strategy-index.md"
    return str(candidate) if candidate.is_file() else None


def _extract_baseline_performance(
    explicit: dict | None,
    baseline_eval: dict | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if isinstance(explicit, dict):
        baseline = dict(explicit)
    elif isinstance(baseline_eval, dict):
        baseline = dict(baseline_eval.get("baseline") or {})
        if not baseline:
            baseline = {
                key: baseline_eval[key]
                for key in (
                    "speedup", "time_us", "time_ms", "precision_passed",
                    "compilation_success",
                )
                if key in baseline_eval
            }
        if not baseline and isinstance(baseline_eval.get("evolved"), dict):
            baseline = dict(baseline_eval["evolved"])
    else:
        baseline = {}

    if getattr(args, "baseline_time_us", None) is not None:
        baseline["time_us"] = args.baseline_time_us
    if getattr(args, "baseline_time_ms", None) is not None:
        baseline["time_ms"] = args.baseline_time_ms
    if getattr(args, "baseline_speedup", None) is not None:
        baseline["speedup"] = args.baseline_speedup

    # Lingxi's baseline_evaluation.json is intentionally flat.  Its score
    # scale is vs PyTorch, with geomean_speedup being the canonical aggregate;
    # without this normalization a new KG started at 1.0 and could mark a
    # kernel slower than the existing baseline as a global improvement.
    if isinstance(baseline_eval, dict):
        if not isinstance(baseline.get("time_us"), (int, float)) or baseline.get("time_us") <= 0:
            flat_time = baseline_eval.get("baseline_time_us")
            if isinstance(flat_time, (int, float)) and flat_time > 0:
                baseline["time_us"] = float(flat_time)
        if not isinstance(baseline.get("speedup"), (int, float)) or baseline.get("speedup") <= 0:
            profiling = baseline_eval.get("ops_profiling_result")
            candidates = []
            if isinstance(profiling, dict):
                candidates.append(profiling.get("geomean_speedup"))
            comparison = baseline_eval.get("comparison")
            if isinstance(comparison, dict):
                candidates.append(comparison.get("speedup"))
            candidates.extend((baseline_eval.get("geomean_speedup"), baseline_eval.get("avg_speedup")))
            for candidate in candidates:
                if isinstance(candidate, (int, float)) and candidate > 0:
                    baseline["speedup"] = float(candidate)
                    break
    if not isinstance(baseline.get("speedup"), (int, float)) or baseline.get("speedup") <= 0:
        baseline["speedup"] = 1.0
    if "time_ms" not in baseline:
        time_us = baseline.get("time_us")
        baseline["time_ms"] = (
            float(time_us) / 1000.0
            if isinstance(time_us, (int, float)) and time_us > 0 else None
        )
    return baseline


def _extract_baseline_evidence(
    explicit: dict | None, baseline_eval: dict | None,
) -> dict[str, Any] | None:
    if isinstance(explicit, dict):
        return explicit
    if not isinstance(baseline_eval, dict):
        return None
    for key in ("baseline_evidence", "profiling_evidence", "profiling"):
        value = baseline_eval.get(key)
        if isinstance(value, dict) and value:
            return value
    baseline = baseline_eval.get("baseline")
    if isinstance(baseline, dict):
        for key in ("baseline_evidence", "profiling_evidence", "profiling"):
            value = baseline.get(key)
            if isinstance(value, dict) and value:
                return value

    # evaluate_ops_direct baseline self-evaluation stores the measured
    # pipeline in ``evolved`` (and per-shape entries), while ``baseline`` is a
    # non-empty compatibility dict with an empty pipeline.  Recover only the
    # observed fields; do not invent a bottleneck when the evaluator did not
    # produce one.
    candidates: list[dict[str, Any]] = []
    if isinstance(baseline_eval.get("profiling_analysis"), dict):
        candidates.append({
            "profiling_analysis": baseline_eval["profiling_analysis"],
            "source": "baseline_evaluation",
        })
    shape_results = baseline_eval.get("shape_results")
    if isinstance(shape_results, dict):
        for group in ("target", "generalization"):
            entries = shape_results.get(group)
            if isinstance(entries, list):
                candidates.extend(entry for entry in entries if isinstance(entry, dict))
    evolved = baseline_eval.get("evolved")
    if isinstance(evolved, dict):
        candidates.append(evolved)
    for candidate in candidates:
        profiling_analysis = candidate.get("profiling_analysis")
        if isinstance(profiling_analysis, dict) and profiling_analysis:
            enriched = extract_profiling_evidence(
                {"profiling_analysis": profiling_analysis}
            )
            if enriched:
                enriched["source"] = str(candidate.get("source") or "baseline_evaluation")
                return enriched
        pipeline = candidate.get("pipeline")
        bottleneck = candidate.get("bottleneck")
        if isinstance(pipeline, dict) and pipeline:
            evidence: dict[str, Any] = {
                "pipeline": dict(pipeline),
                "source": "baseline_evaluation",
            }
            if isinstance(bottleneck, str) and bottleneck and bottleneck != "unknown":
                evidence["bottleneck_type"] = bottleneck
            # Ops direct evaluation normally emits only coarse pipeline ratios.
            # Promote them through the same classifier used by MCTS so KG init
            # does not silently fall back to open-exploration placeholders.
            try:
                analysis = synthesize_analysis_from_pipeline(
                    pipeline,
                    bottleneck_hint=(
                        bottleneck if isinstance(bottleneck, str) else None
                    ),
                )
                enriched = extract_profiling_evidence(
                    {"profiling_analysis": analysis}
                ) if analysis else None
            except (TypeError, ValueError, KeyError):
                enriched = None
            if enriched:
                if isinstance(bottleneck, str) and bottleneck in BOTTLENECK_STRATEGY_MAP:
                    # A deep evaluator bottleneck is authoritative when it is
                    # present; the synthesized classifier is only a fallback.
                    enriched["bottleneck_type"] = bottleneck
                    strategy_info = BOTTLENECK_STRATEGY_MAP[bottleneck]
                    enriched["suggested_strategies"] = (
                        list(strategy_info.get("primary", []))
                        + list(strategy_info.get("secondary", []))
                    )
                    enriched["anti_strategies"] = list(strategy_info.get("anti", []))
                    enriched["description"] = strategy_info.get("description", "")
                enriched["pipeline"] = dict(pipeline)
                enriched["source"] = "baseline_evaluation"
                return enriched
            return evidence
    return None


def _session_from_kg(
    kg: dict[str, Any], evo_dir: str, *, session_id: str | None = None,
    op_name: str | None = None, requested_rounds: int | None = None,
    start_time: str | None = None,
) -> dict[str, Any]:
    session = dict(kg.get("session") or {})
    session["session_id"] = session_id or session.get("session_id") or ""
    session["start_time"] = start_time or session.get("start_time") or _now_iso()
    session["requested_rounds"] = (
        requested_rounds if requested_rounds is not None
        else int(session.get("requested_rounds", 0) or 0)
    )
    session["actual_rounds_completed"] = int(session.get("actual_rounds_completed", 0) or 0)
    session["evo_dir"] = os.path.abspath(evo_dir)
    session["op_name"] = op_name or session.get("op_name") or Path(evo_dir).name
    return session


def _sync_and_finalize(
    evo_dir: str, *, session: dict[str, Any], baseline: dict[str, Any] | None = None,
    kernel_summary: str | None = None, hw_params: dict | None = None,
    baseline_evidence: dict | None = None, best_score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sync derived registries, then restore KG-only metadata untouched by sync."""
    kg_data = sync_knowledge_graph(
        evo_dir,
        session=session,
        baseline=baseline,
        best_score=best_score,
    )
    if kernel_summary is not None:
        kg_data["kernel_summary"] = kernel_summary
    if hw_params is not None:
        kg_data["hw_params"] = hw_params
    if baseline_evidence is not None:
        kg_data["baseline_evidence"] = baseline_evidence
    # sync_knowledge_graph rebuilds registries from entity files and reloads
    # the on-disk JSON.  Preserve round-level KG metadata computed in memory
    # immediately before this call, especially the stagnation/drift counters.
    if metadata is not None:
        for field in (
            "stagnation_count", "stagnation_count_vs_base",
            "supervisor_used_count", "last_supervisor_round", "open_questions",
            "round_commits",
        ):
            if field in metadata:
                kg_data[field] = metadata[field]

    # kg_wiki's registry is intentionally lightweight.  Add the round-tested
    # field here because the orchestrator knows the experiment history.
    experiments = load_experiments(evo_dir)
    for slug, entry in (kg_data.get("idea_registry") or {}).items():
        rounds = [
            _safe_int(exp.get("round", 0) or 0)
            for exp in experiments.values()
            if str(exp.get("idea_id", "")).removeprefix("idea:") == slug
        ]
        entry["round_tested"] = min(rounds) if rounds else None

    _write_json(Path(evo_dir) / "knowledge_graph.json", kg_data)
    return kg_data


def _run_gap_analysis(evo_dir: str, kg: dict[str, Any]) -> list[dict]:
    evidence = kg.get("baseline_evidence")
    baseline_bottleneck = evidence.get("bottleneck_type") if isinstance(evidence, dict) else None
    return gap_analysis.analyze_gaps(
        evo_dir,
        strategy_index_path=_default_strategy_index(),
        stagnation_count=int(kg.get("stagnation_count", 0) or 0),
        stagnation_count_vs_base=int(kg.get("stagnation_count_vs_base", 0) or 0),
        baseline_bottleneck=baseline_bottleneck,
    )


def init_session(args: argparse.Namespace) -> dict[str, Any]:
    evo_dir = _evo_dir_from_args(args)
    if int(args.requested_rounds) < 1:
        raise ValueError("--requested-rounds must be >= 1")
    if int(args.parallel_num) < 1:
        raise ValueError("--parallel-num must be >= 1")
    # Never silently convert an existing MCTS session.  A new output
    # directory may contain shared kernel files and is still eligible for KG;
    # the presence of world_model.json is the explicit legacy-session signal.
    if detect_mode(evo_dir) == "mcts" and (Path(evo_dir) / "world_model.json").is_file():
        raise ValueError(
            "existing MCTS session detected (world_model.json); use wm_ops.py "
            "or start KG init in a new evo_dir"
        )
    init_kg(evo_dir)
    kg = _read_kg(evo_dir)
    existing_session = kg.get("session") if isinstance(kg.get("session"), dict) else {}
    resumed = bool(existing_session.get("session_id"))

    baseline_eval = _read_json_value(args.baseline_eval, "--baseline-eval")
    explicit_baseline = _read_json_value(args.baseline_performance, "--baseline-performance")
    explicit_evidence = _read_json_value(args.baseline_evidence, "--baseline-evidence")
    hw_params = _read_json_value(args.hw_params, "--hw-params")
    kernel_summary = args.kernel_summary or ""
    if args.kernel_summary_file:
        try:
            kernel_summary = Path(args.kernel_summary_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"cannot read --kernel-summary-file: {args.kernel_summary_file}") from exc

    supplied_baseline = _extract_baseline_performance(explicit_baseline, baseline_eval, args)
    supplied_evidence = _extract_baseline_evidence(explicit_evidence, baseline_eval)
    if resumed:
        baseline = kg.get("baseline_performance") or supplied_baseline
        baseline_evidence = kg.get("baseline_evidence") or supplied_evidence
        session = _session_from_kg(kg, evo_dir)
        kernel_summary = kg.get("kernel_summary") or kernel_summary
        hw_params = kg.get("hw_params") if kg.get("hw_params") is not None else hw_params
    else:
        baseline = supplied_baseline
        baseline_evidence = supplied_evidence
        session_id = args.session_id or f"{args.op_name or Path(evo_dir).name}_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = _session_from_kg(
            kg,
            evo_dir,
            session_id=session_id,
            op_name=args.op_name,
            requested_rounds=args.requested_rounds,
            start_time=args.start_time,
        )
    kg.update({
        "kernel_summary": kernel_summary,
        "baseline_performance": baseline,
        "baseline_evidence": baseline_evidence,
        "hw_params": hw_params,
        "session": session,
    })
    # Seed the KG counterpart of the legacy Supervisor counters.  setdefault
    # keeps a resumed KG session's history intact across a repeated init call.
    kg.setdefault("supervisor_used_count", 0)
    kg.setdefault("last_supervisor_round", -2)
    kg.setdefault("open_questions", [])
    _write_json(Path(evo_dir) / "knowledge_graph.json", kg)

    # The baseline evidence is written before gap analysis so the initial map
    # can contain only observed bottlenecks, not the full strategy table.
    gaps = _run_gap_analysis(evo_dir, kg)
    parallel_num = max(1, int(args.parallel_num))
    if resumed and load_ideas(evo_dir):
        # A re-entry must never reseed round-0 scaffolds.  In particular, jury
        # edits may have replaced a placeholder title while retaining its slug;
        # regenerating would otherwise create a second placeholder with a
        # numeric suffix.
        idea_slugs = []
    else:
        ideas = idea_generation.generate_ideas_for_init(
            evo_dir,
            parallel_num=parallel_num,
            baseline_evidence=baseline_evidence,
            kernel_traits=_read_json_value(args.kernel_traits, "--kernel-traits"),
            kernel_summary=kernel_summary,
            strategy_index_path=_default_strategy_index(),
            generation_round=0,
        )
        idea_slugs = idea_generation.persist_ideas(evo_dir, ideas, generation_round=0)

    baseline_score = baseline.get("speedup", 1.0) if isinstance(baseline, dict) else 1.0
    if (
        not isinstance(baseline_score, (int, float))
        or isinstance(baseline_score, bool)
        or baseline_score <= 0
    ):
        baseline_score = 1.0
    raw_best = kg.get("best_score", baseline_score)
    initial_best = float(baseline_score)
    if resumed and isinstance(raw_best, (int, float)) and not isinstance(raw_best, bool) and raw_best > 0:
        initial_best = float(raw_best)
    kg_data = _sync_and_finalize(
        evo_dir,
        session=session,
        baseline=baseline,
        kernel_summary=kernel_summary,
        hw_params=hw_params,
        baseline_evidence=baseline_evidence,
        best_score=initial_best,
    )
    gaps = _run_gap_analysis(evo_dir, kg_data)
    packed = query_pack.build_query_pack(evo_dir, max_chars=args.max_chars)
    # Query-pack generation does not change entities, but the final sync keeps
    # the on-disk registry/counters authoritative after all init writes.
    kg_data = _sync_and_finalize(
        evo_dir,
        session=session,
        baseline=baseline,
        kernel_summary=kernel_summary,
        hw_params=hw_params,
        baseline_evidence=baseline_evidence,
        best_score=float(kg_data.get("best_score", 1.0) or 1.0),
    )
    return {
        "mode": "kg",
        "resumed": resumed,
        "evo_dir": evo_dir,
        "session": kg_data.get("session", session),
        "generated_ideas": idea_slugs,
        "gap_count": len(gaps),
        "query_pack_chars": len(packed),
        "counters": kg_data.get("counters", {}),
        "supervisor_used_count": kg_data.get("supervisor_used_count", 0),
        "last_supervisor_round": kg_data.get("last_supervisor_round", -2),
    }


def _normalise_parallel_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("--parallel-map must be a JSON object mapping parallel index to idea")
    normalized: dict[str, Any] = {}
    for raw_key, spec in value.items():
        key = str(raw_key).removeprefix("parallel_")
        # Keep invalid keys intact so the caller can report the useful
        # non-negative-index error, but canonicalize 01/parallel_01 to 1 so
        # result records and durable maps cannot silently miss each other.
        if key.isdigit():
            key = str(int(key))
        if key in normalized:
            raise ValueError(f"duplicate parallel_map index after normalization: {raw_key!r} -> {key}")
        normalized[key] = spec
    return normalized


def _load_parallel_map(args: argparse.Namespace, evo_dir: str) -> dict[str, Any]:
    """Load the durable round selection map, with explicit CLI override."""
    value = _read_json_value(args.parallel_map, "--parallel-map")
    if value is None:
        map_path = Path(evo_dir) / f"round_{args.round}" / "parallel_map.json"
        if map_path.is_file():
            value = _read_json_value(str(map_path), "parallel_map.json")
    return _normalise_parallel_map(value)


def _load_result_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_value = getattr(args, "results", None) or getattr(args, "parallel_results", None)
    records: list[dict[str, Any]] = []
    if raw_value:
        payload = _read_json_value(raw_value, "--results")
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            payload = payload["results"]
        if isinstance(payload, list):
            for idx, record in enumerate(payload):
                if not isinstance(record, dict):
                    raise ValueError(f"result at index {idx} is not an object")
                records.append({"parallel_index": idx, **record})
        elif isinstance(payload, dict):
            if "verdict" in payload or "evaluation_results" in payload:
                records.append({"parallel_index": 0, **payload})
            else:
                for key, record in payload.items():
                    if not isinstance(record, dict):
                        raise ValueError(f"result for parallel {key} is not an object")
                    records.append({"parallel_index": str(key).removeprefix("parallel_"), **record})
        else:
            raise ValueError("--results must contain a JSON object or array")

    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_dir():
            raise ValueError(f"--results-dir is not a directory: {results_dir}")
        for parallel_dir in sorted(results_dir.glob("parallel_*")):
            if not parallel_dir.is_dir():
                continue
            match = re.fullmatch(r"parallel_(\d+)", parallel_dir.name)
            if not match:
                continue
            path = parallel_dir / "evaluation_results.json"
            if not path.is_file():
                marker_path = parallel_dir / ".subagent_exit_status"
                if not marker_path.is_file():
                    continue
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid subagent exit marker: {marker_path}") from exc
                status = str(marker.get("exit_status", "")) if isinstance(marker, dict) else ""
                if status not in {"timeout", "error", "killed"}:
                    continue
                records.append({
                    "parallel_index": int(match.group(1)),
                    "verdict": "no",
                    "failure_type": "impl_error",
                    "failure_reason": f"subagent_{status}: evaluation_results.json not produced",
                    "solution_ref": str(parallel_dir.relative_to(results_dir.parent)),
                    "compilation_success": False,
                    "precision_passed": False,
                })
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                marker_path = parallel_dir / ".subagent_exit_status"
                marker = None
                if marker_path.is_file():
                    try:
                        marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        marker = None
                status = str(marker.get("exit_status", "")) if isinstance(marker, dict) else ""
                if status in {"timeout", "error", "killed"}:
                    records.append({
                        "parallel_index": int(match.group(1)),
                        "_parallel_dir": str(parallel_dir),
                        "verdict": "no",
                        "failure_type": "impl_error",
                        "failure_reason": (
                            f"subagent_{status}: invalid evaluation_results.json"
                        ),
                        "solution_ref": str(parallel_dir.relative_to(results_dir.parent)),
                        "compilation_success": False,
                        "precision_passed": False,
                    })
                    continue
                raise ValueError(f"invalid evaluation result: {path}") from exc
            if not isinstance(result, dict):
                marker_path = parallel_dir / ".subagent_exit_status"
                marker = None
                if marker_path.is_file():
                    try:
                        marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        marker = None
                status = str(marker.get("exit_status", "")) if isinstance(marker, dict) else ""
                if status in {"timeout", "error", "killed"}:
                    records.append({
                        "parallel_index": int(match.group(1)),
                        "_parallel_dir": str(parallel_dir),
                        "verdict": "no",
                        "failure_type": "impl_error",
                        "failure_reason": (
                            f"subagent_{status}: evaluation_results.json is not an object"
                        ),
                        "solution_ref": str(parallel_dir.relative_to(results_dir.parent)),
                        "compilation_success": False,
                        "precision_passed": False,
                    })
                    continue
                raise ValueError(f"evaluation result must be an object: {path}")
            # The durable directory name is the dispatch identity.  A stale
            # parallel_index embedded by a worker must not silently move the
            # result to another idea (or make a missing slot look complete).
            expected_index = int(match.group(1))
            embedded_index = result.get("parallel_index", result.get("parallel"))
            if embedded_index is not None:
                embedded = str(embedded_index).removeprefix("parallel_")
                if embedded.isdigit():
                    embedded = str(int(embedded))
                if embedded != str(expected_index):
                    raise ValueError(
                        f"{path}: embedded parallel_index={embedded_index!r} "
                        f"conflicts with directory parallel_{expected_index}"
                    )
            records.append({
                "parallel_index": expected_index,
                "_parallel_dir": str(parallel_dir),
                **result,
            })

    records.sort(key=lambda item: int(str(item.get("parallel_index", 0)).removeprefix("parallel_")))
    return records


def _idea_id_from_spec(spec: Any) -> str:
    if isinstance(spec, str):
        return spec if spec.startswith("idea:") else f"idea:{spec}"
    if isinstance(spec, dict):
        for key in ("idea_id", "idea_ref", "idea", "idea_slug", "slug"):
            value = spec.get(key)
            if value:
                value = str(value)
                return value if value.startswith("idea:") else f"idea:{value}"
    return ""


def _nested_profiling(record: dict[str, Any], key: str) -> dict | None:
    value = record.get(key)
    if isinstance(value, dict):
        return value
    profiling = record.get("profiling")
    if isinstance(profiling, dict):
        nested = profiling.get(key)
        if isinstance(nested, dict):
            return nested
    return None


def _canonical_failure_type(raw: Any) -> tuple[str | None, str | None]:
    """Map evaluator-specific failure labels to the wiki schema.

    Partial agents and evaluators historically used labels such as
    ``timeout``, ``compile_error`` and ``profiling_missing``.  Persisting
    those verbatim makes ``kg_wiki.add_experiment`` reject an otherwise
    recoverable round.  Keep the original label in the reason while storing
    one of the three durable taxonomy values.
    """
    if raw is None or str(raw).strip() in {"", "null", "unknown"}:
        return None, None
    value = str(raw).strip().lower()
    if value in {"impl_error", "strategy_infeasible", "profiling_lost"}:
        return value, None
    if value in {
        "profiling_missing", "profiling_incomplete", "profiling_error",
        "msprof_missing",
    }:
        return "profiling_lost", f"original_failure_type={value}"
    if value in {"strategy_error", "infeasible", "unsupported_strategy"}:
        return "strategy_infeasible", f"original_failure_type={value}"
    return "impl_error", f"original_failure_type={value}"


def _normalise_result(
    record: dict[str, Any], round_num: int, parallel_map: dict[str, Any],
) -> dict[str, Any]:
    nested = record.get("evaluation_results")
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update({k: v for k, v in record.items() if k != "evaluation_results"})
        record = merged

    pidx = str(record.get("parallel_index", record.get("parallel", record.get("index", 0))))
    pidx = pidx.removeprefix("parallel_").strip()
    if pidx.isdigit():
        pidx = str(int(pidx))
    else:
        raise ValueError(
            f"round {round_num} has invalid parallel index {pidx!r}; "
            "expected a non-negative integer"
        )
    spec = parallel_map.get(pidx)
    mapped_idea_id = _idea_id_from_spec(spec)
    explicit_idea_id = _idea_id_from_spec(
        record.get("idea_id") or record.get("idea_ref") or record.get("idea")
        or record.get("idea_slug")
    )
    if explicit_idea_id and mapped_idea_id and explicit_idea_id != mapped_idea_id:
        raise ValueError(
            f"round {round_num} parallel {pidx} idea_id {explicit_idea_id} "
            f"conflicts with parallel_map {mapped_idea_id}"
        )
    idea_id = explicit_idea_id or mapped_idea_id
    if not idea_id:
        raise ValueError(f"round {round_num} parallel {pidx} has no idea_id; pass --parallel-map")

    comparison = record.get("comparison") if isinstance(record.get("comparison"), dict) else {}
    evolved = record.get("evolved") if isinstance(record.get("evolved"), dict) else {}
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "passed", "pass", "ok"}:
                return True
            if lowered in {"false", "no", "failed", "fail", "error"}:
                return False
        return None

    compilation_ok = _coerce_bool(record.get("compilation_success"))
    if compilation_ok is None:
        compilation_ok = _coerce_bool(comparison.get("compilation_success"))
    precision_ok = _coerce_bool(record.get("precision_passed"))
    if precision_ok is None:
        precision_ok = _coerce_bool(comparison.get("precision_passed"))
    shape_results = record.get("shape_results") if isinstance(record.get("shape_results"), dict) else {}
    shape_entries = [
        item
        for group in ("target", "generalization")
        for item in (shape_results.get(group) or [])
        if isinstance(item, dict)
    ]
    if compilation_ok is None and shape_entries:
        flags = [_coerce_bool(item.get("compilation_success")) for item in shape_entries]
        if all(flag is not None for flag in flags):
            compilation_ok = all(flags)
    if precision_ok is None and shape_entries:
        flags = [_coerce_bool(item.get("precision_passed")) for item in shape_entries]
        if all(flag is not None for flag in flags):
            precision_ok = all(flags)
    if compilation_ok is None and precision_ok is True:
        candidate_time = record.get("time_us", evolved.get("time_us", comparison.get("gen_time_us")))
        compilation_ok = (
            isinstance(candidate_time, (int, float))
            and not isinstance(candidate_time, bool)
            and candidate_time > 0
        )

    raw_gating = record.get("gating")
    gating = str(raw_gating).strip().lower() if raw_gating not in (None, "") else ""
    gating_verdict = {
        "fully_passed": "yes",
        "partial_passed": "partial",
        "target_regression": "no",
        "generalization_regression": "no",
        "failed": "no",
    }.get(gating)
    if gating and gating_verdict is None:
        raise ValueError(f"round {round_num} parallel {pidx} has unknown gating value: {gating}")
    verdict = str(record.get("verdict", "")).lower()
    if gating_verdict is not None:
        if verdict in {"yes", "partial", "no"} and verdict != gating_verdict:
            raise ValueError(
                f"round {round_num} parallel {pidx} verdict={verdict} conflicts with gating={gating}"
            )
        verdict = gating_verdict
    elif verdict not in {"yes", "partial", "no"}:
        verdict = "yes" if compilation_ok is True and precision_ok is True else "no"

    # Some evaluators emit only the gating enum and omit the legacy compile /
    # precision flags.  Gating remains the authoritative aggregate result in
    # that shape; infer the missing booleans so the persisted sidecar cannot
    # claim ``fully_passed`` while reporting ``compilation_success=false``.
    if gating_verdict is not None:
        passed_gate = gating_verdict in {"yes", "partial"}
        if compilation_ok is None:
            compilation_ok = passed_gate
        if precision_ok is None:
            precision_ok = passed_gate

    speedup = record.get("speedup")
    if speedup is None:
        speedup = record.get("score", comparison.get("speedup"))
    aggregate = record.get("aggregate") if isinstance(record.get("aggregate"), dict) else {}
    target_min = aggregate.get("target_min_speedup")
    if gating and isinstance(target_min, (int, float)) and not isinstance(target_min, bool) and target_min > 0:
        # Multi-shape candidates are ranked by their worst target shape, not by
        # the legacy first-shape compatibility field.
        speedup = target_min
    if speedup is None:
        baseline_time = record.get("baseline_time_us", comparison.get("base_time_us"))
        evolved_time = record.get("time_us", evolved.get("time_us", comparison.get("gen_time_us")))
        if (
            isinstance(baseline_time, (int, float)) and not isinstance(baseline_time, bool)
            and isinstance(evolved_time, (int, float)) and not isinstance(evolved_time, bool)
            and evolved_time > 0
        ):
            speedup = float(baseline_time) / float(evolved_time)
    if speedup is not None:
        if isinstance(speedup, bool):
            raise ValueError(f"round {round_num} parallel {pidx} has invalid boolean speedup")
        try:
            speedup = float(speedup)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"round {round_num} parallel {pidx} has invalid speedup: {speedup}") from exc
        if not math.isfinite(speedup) or speedup <= 0:
            speedup = None

    failure_type, failure_type_note = _canonical_failure_type(record.get("failure_type"))
    failure_reason = record.get("failure_reason") or record.get("reason")
    if failure_type_note:
        failure_reason = (
            f"{failure_type_note}; {failure_reason}"
            if failure_reason else failure_type_note
        )

    # A positive verdict is never allowed to survive an explicit compile or
    # precision failure.  Conversely, a failed result without a taxonomy
    # would otherwise become an inert experiment that cannot enter the retry
    # queue.  Preserve evaluator-provided strategy/profiling labels, but use
    # impl_error as the safe operational default for unclassified failures.
    if verdict in {"yes", "partial"} and (
        compilation_ok is False or precision_ok is False
    ):
        verdict = "no"
        failure_type = failure_type or "impl_error"
        failure_reason = failure_reason or "positive verdict conflicts with explicit evaluation failure"
        if gating_verdict in {"yes", "partial"}:
            gating = "failed"
    if verdict == "no" and failure_type is None:
        failure_type = "impl_error"
        failure_reason = failure_reason or "evaluation failed without a failure_type"

    # Lingxi scores are explicitly defined against the PyTorch reference.  A
    # partially written comparison can nevertheless contain a parent-kernel
    # base time and a plausible-looking speedup.  Match the MCTS safety rule:
    # prefer same-run performance.json geomeans, then a declared PyTorch
    # base/gen pair.  Ops multi-shape scores are already target-relative and
    # are governed by ``gating`` above, so they are left on that path.
    score_scale_alert: str | None = None
    canonical_speedup: float | None = None
    parallel_dir = record.get("_parallel_dir")
    if not gating:
        if isinstance(parallel_dir, str):
            performance_path = Path(parallel_dir) / "performance.json"
        else:
            performance_path = None
        if performance_path is not None and performance_path.is_file():
            try:
                performance = json.loads(performance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                performance = None
            if isinstance(performance, dict):
                ref_us = performance.get("geomean_ref_us")
                asc_us = performance.get("geomean_asc_us")
                if (
                    isinstance(ref_us, (int, float)) and ref_us > 0
                    and isinstance(asc_us, (int, float)) and asc_us > 0
                ):
                    canonical_speedup = float(ref_us) / float(asc_us)
                    score_scale_alert = "same_run_performance_json"
        if canonical_speedup is None:
            ref_base = comparison.get("ref_base", record.get("ref_base"))
            base_ms = comparison.get("base_time_ms", record.get("base_time_ms"))
            gen_ms = comparison.get("gen_time_ms", record.get("gen_time_ms"))
            if (
                ref_base == "pytorch"
                and isinstance(base_ms, (int, float)) and base_ms > 0
                and isinstance(gen_ms, (int, float)) and gen_ms > 0
            ):
                canonical_speedup = float(base_ms) / float(gen_ms)
                score_scale_alert = "same_run_comparison"
            elif ref_base not in (None, "", "missing", "pytorch"):
                # Do not let a parent/baseline-relative measurement enter the
                # global KG best score.  Keep the evidence as a failed
                # implementation/evaluation record for diagnosis and retry.
                verdict = "no"
                failure_type = failure_type or "impl_error"
                failure_reason = failure_reason or (
                    f"score_scale_invalid: ref_base={ref_base!r}, expected pytorch"
                )
    if canonical_speedup is not None:
        if speedup is not None and abs(canonical_speedup - speedup) > 0.05 * max(abs(canonical_speedup), 1e-9):
            score_scale_alert = f"{score_scale_alert}: comparison.speedup replaced"
        speedup = canonical_speedup

    # Canonical same-run performance may have supplied the score after the
    # compatibility fields were inspected.  Only classify a successful
    # verdict as malformed after that fallback has had a chance to run.
    if verdict in {"yes", "partial"} and speedup is None:
        verdict = "no"
        failure_type = failure_type or "impl_error"
        failure_reason = failure_reason or "positive verdict has no finite positive speedup"
        if gating_verdict in {"yes", "partial"}:
            gating = "failed"

    profiling_insight = _nested_profiling(record, "profiling_insight")
    profiling_evidence = _nested_profiling(record, "profiling_evidence")
    if profiling_insight is None and isinstance(evolved.get("pipeline"), dict) and evolved.get("pipeline"):
        profiling_insight = dict(evolved["pipeline"])
        if evolved.get("bottleneck"):
            profiling_insight.setdefault("bottleneck", evolved["bottleneck"])
    if profiling_evidence is None:
        profiling = record.get("profiling")
        if isinstance(profiling, dict) and (
            profiling.get("bottleneck_type") or profiling.get("primary_bottleneck")
        ):
            profiling_evidence = dict(profiling)

    if verdict in {"yes", "partial"} and not profiling_insight and not profiling_evidence:
        verdict = "no"
        failure_type = failure_type or "profiling_lost"
        failure_reason = failure_reason or "compilation/precision passed but profiling evidence is missing"
        if gating_verdict in {"yes", "partial"}:
            gating = "failed"
    elif verdict == "no" and gating == "failed" and not failure_type:
        failure_type = "impl_error"
        failure_reason = failure_reason or "multi-shape evaluator gating=failed"
    solution_ref = record.get("solution_ref") or f"round_{round_num}/parallel_{pidx}"
    raw_retry_count = record.get("retry_count", 0)
    try:
        retry_count = max(0, int(raw_retry_count or 0))
    except (TypeError, ValueError):
        retry_count = 0
    raw_confidence = str(record.get("confidence") or "").strip().lower()
    confidence = raw_confidence if raw_confidence in {"high", "medium", "low"} else (
        "high" if verdict == "yes" else "medium"
    )
    raw_tags = record.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [part.strip() for part in raw_tags.split(",") if part.strip()]
    elif not isinstance(raw_tags, list):
        raw_tags = []
    return {
        "parallel_index": int(pidx),
        "idea_id": idea_id,
        "slug": record.get("experiment_slug") or record.get("slug") or f"r{round_num}-p{pidx}",
        "title": record.get("title") or f"Round {round_num} Parallel {pidx}",
        "verdict": verdict,
        "confidence": confidence,
        "speedup": speedup,
        "solution_ref": str(solution_ref),
        "round": round_num,
        "profiling_insight": profiling_insight,
        "profiling_evidence": profiling_evidence,
        "facts": record.get("facts"),
        "diagnosis": record.get("diagnosis"),
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "retry_count": retry_count,
        "tags": raw_tags,
        "measurement_quality": comparison.get("measurement_quality", record.get("measurement_quality", "good")),
        "evaluation": {
            key: record.get(key)
            for key in (
                "aggregate", "shape_results", "compilation_success",
                "precision_passed", "n_cases_total", "eval_backend", "shapes_mode",
            )
            if record.get(key) is not None
        } | {
            "gating": gating or None,
            "compilation_success": bool(compilation_ok),
            "precision_passed": bool(precision_ok),
            "measurement_quality": comparison.get(
                "measurement_quality", record.get("measurement_quality", "good")
            ),
            "n_cases_total": record.get("n_cases_total", comparison.get("n_cases_total")),
            "score_scale": score_scale_alert or "declared",
        },
    }


def _expected_case_count(results_dir: str | None) -> int | None:
    """Return the full JSONL case count for Lingxi-style shared inputs."""
    if not results_dir:
        return None
    shared = Path(results_dir).resolve().parent / "shared"
    if not shared.is_dir():
        return None
    # Only the Lingxi contract creates immutable full-case .json.bak files.
    # Reading arbitrary shared/*.json would mistake Ops configuration files
    # for JSONL cases and incorrectly reject valid multi-shape evaluations.
    candidates = sorted(shared.glob("*.json.bak"))
    total = 0
    found_jsonl = False
    for path in candidates:
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue
            for line in lines:
                if not isinstance(json.loads(line), dict):
                    raise ValueError
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        found_jsonl = True
        total += len(lines)
    return total if found_jsonl and total > 0 else None


def _apply_full_case_guard(result: dict[str, Any], expected_cases: int | None) -> None:
    if not expected_cases:
        return
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    actual = evaluation.get("n_cases_total")
    violation = None
    if not isinstance(actual, int) or isinstance(actual, bool) or actual <= 0:
        violation = (
            f"case_evidence_missing: n_cases_total is required for all {expected_cases} shared cases"
        )
    elif actual < expected_cases:
        violation = f"case_subset_evaluation: n_cases_total={actual} < full={expected_cases}"
    if violation:
        result["verdict"] = "no"
        result["failure_type"] = "impl_error"
        result["failure_reason"] = violation
        # A case-subset result is not a valid pass even when the evaluator
        # optimistically wrote fully_passed/partial_passed.  Keep the sidecar
        # and experiment verdict aligned so stop rules/reporting cannot treat
        # an incomplete measurement as a successful gate.
        if evaluation.get("gating") in {
            "fully_passed", "partial_passed", "target_regression",
            "generalization_regression",
        }:
            evaluation["gating"] = "failed"
        evaluation["case_guard_passed"] = False
        evaluation["expected_cases"] = expected_cases
        result["evaluation"] = evaluation
    else:
        evaluation["case_guard_passed"] = True
        evaluation["expected_cases"] = expected_cases


def _parent_score(
    idea_id: str, experiments: dict[str, dict], ideas: dict[str, dict], baseline_score: float,
) -> float:
    slug = idea_id.removeprefix("idea:")
    idea = ideas.get(slug, {})
    parent_exp = idea.get("parent_experiment")
    if parent_exp:
        parent_slug = str(parent_exp).removeprefix("exp:")
        parent = experiments.get(parent_slug)
        if (
            parent
            and parent.get("verdict") in {"yes", "partial"}
            and isinstance(parent.get("speedup"), (int, float))
        ):
            return float(parent["speedup"])
    scores = [
        float(exp["speedup"])
        for exp in experiments.values()
        if str(exp.get("idea_id", "")).removeprefix("idea:") == slug
        and exp.get("verdict") in {"yes", "partial"}
        and isinstance(exp.get("speedup"), (int, float))
    ]
    return max(scores, default=baseline_score)


def _recompute_round_counters(
    kg: dict[str, Any], experiments: dict[str, dict], ideas: dict[str, dict],
) -> tuple[float, float, float]:
    """Derive round counters from evidence, making refine crash/retry safe."""
    baseline = kg.get("baseline_performance") if isinstance(kg.get("baseline_performance"), dict) else {}
    baseline_score = baseline.get("speedup", 1.0)
    if not isinstance(baseline_score, (int, float)) or baseline_score <= 0:
        baseline_score = 1.0
    baseline_score = float(baseline_score)
    best = baseline_score
    stagnation = 0
    stagnation_vs_base = 0
    processed: dict[str, dict] = {}
    last_round_best = 0.0

    by_round: dict[int, list[dict]] = {}
    for experiment in experiments.values():
        round_num = int(experiment.get("round", 0) or 0)
        if round_num > 0:
            by_round.setdefault(round_num, []).append(experiment)
    for round_num in sorted(by_round):
        round_experiments = by_round[round_num]
        eligible = [
            exp for exp in round_experiments
            if exp.get("verdict") in {"yes", "partial"}
            and isinstance(exp.get("speedup"), (int, float))
            and float(exp["speedup"]) > 0
        ]
        scores = [float(exp["speedup"]) for exp in eligible]
        last_round_best = max(scores, default=0.0)
        qualities = []
        for exp in round_experiments:
            evaluation = exp.get("evaluation") if isinstance(exp.get("evaluation"), dict) else {}
            qualities.append(str(evaluation.get("measurement_quality", "good")))
        threshold = max(
            (_QUALITY_STAGNATION_THRESHOLD.get(quality, 1.02) for quality in qualities),
            default=1.02,
        )
        if last_round_best <= best * threshold:
            stagnation += 1
        else:
            stagnation = 0
        beat_parent = any(
            float(exp["speedup"]) > _parent_score(
                str(exp.get("idea_id", "")), processed, ideas, baseline_score,
            )
            for exp in eligible
        )
        stagnation_vs_base = 0 if beat_parent else stagnation_vs_base + 1
        best = max(best, last_round_best)
        for slug, exp in experiments.items():
            if int(exp.get("round", 0) or 0) == round_num:
                processed[slug] = exp

    kg["stagnation_count"] = stagnation
    kg["stagnation_count_vs_base"] = stagnation_vs_base
    kg["best_score"] = round(best, 4)
    session = dict(kg.get("session") or {})
    session["actual_rounds_completed"] = max(by_round, default=0)
    kg["session"] = session
    return baseline_score, last_round_best, best


def _refresh_idea_outcomes(evo_dir: str, idea_ids: set[str]) -> None:
    experiments = load_experiments(evo_dir)
    for idea_id in sorted(idea_ids):
        verdicts = {
            exp.get("verdict") for exp in experiments.values()
            if exp.get("idea_id") == idea_id
        }
        if verdicts == {"yes"}:
            outcome, stage = "positive", "piloted"
        elif verdicts == {"no"}:
            outcome, stage = "negative", "archived"
        else:
            outcome, stage = "mixed", "active"
        update_idea_outcome(evo_dir, idea_id.removeprefix("idea:"), outcome, stage=stage)


def _round_fingerprint(results: list[dict[str, Any]]) -> str:
    fields = (
        "parallel_index", "idea_id", "slug", "title", "confidence", "verdict",
        "speedup", "solution_ref", "failure_type", "failure_reason", "retry_count",
        "profiling_insight", "profiling_evidence", "facts", "diagnosis", "tags",
        "measurement_quality", "evaluation",
    )
    canonical = [
        {key: result.get(key) for key in fields}
        for result in sorted(results, key=lambda item: int(item["parallel_index"]))
    ]
    payload = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_dispatchable_idea(idea_id: str, idea: dict[str, Any]) -> None:
    """Reject deterministic open-exploration scaffolds before implementation."""
    if idea.get("mode") == "workflow_guided":
        workflow_ref = idea.get("workflow_ref")
        if (
            not isinstance(workflow_ref, dict)
            or not str(workflow_ref.get("workflow_file") or workflow_ref.get("file") or "").strip()
            or not isinstance(workflow_ref.get("stage_order"), list)
            or not workflow_ref.get("stage_order")
        ):
            raise ValueError(
                f"workflow_guided idea {idea_id} is missing workflow_ref.workflow_file/stage_order"
            )
    if idea.get("mode") != "open_exploration":
        return
    title = str(idea.get("title") or "")
    thesis = str(idea.get("thesis") or "")
    if title.startswith("开放探索方案") or len(thesis.strip()) < 80 or "LLM 自主" in thesis:
        raise ValueError(
            f"open_exploration idea {idea_id} is still a placeholder; "
            "jury must persist a concrete title/thesis before dispatch"
        )


def _preflight_refine(
    args: argparse.Namespace, kg: dict[str, Any], results: list[dict[str, Any]],
    ideas: dict[str, dict], experiments: dict[str, dict], parallel_map: dict[str, Any],
) -> None:
    requested_parallel = getattr(args, "parallel_num", None)
    if parallel_map:
        expected = len(parallel_map)
        if requested_parallel is not None and int(requested_parallel) != expected:
            raise ValueError(
                f"parallel_num={requested_parallel} conflicts with durable "
                f"parallel_map size={expected}"
            )
    elif requested_parallel is not None:
        expected = max(1, int(requested_parallel))
    else:
        expected = max(1, len(results))
    expected_indices = set(range(expected))
    indices = [result.get("parallel_index") for result in results]
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices):
        raise ValueError("parallel indices must be non-negative integers")
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate parallel result indices: {indices}")
    if set(indices) != expected_indices:
        raise ValueError(
            f"round {args.round} requires exactly parallel indices "
            f"{sorted(expected_indices)}, got {sorted(indices)}"
        )
    if parallel_map:
        if any(not str(key).isdigit() for key in parallel_map):
            raise ValueError("parallel_map keys must be non-negative integer indices")
        if {int(key) for key in parallel_map} != expected_indices:
            raise ValueError(
                f"parallel_map must contain exactly indices {sorted(expected_indices)}"
            )
    if len({result["slug"] for result in results}) != len(results):
        raise ValueError("experiment slugs must be unique within a round")
    if len({str(result["idea_id"]) for result in results}) != len(results):
        raise ValueError("a round cannot evaluate the same idea in multiple parallel slots")
    for result in results:
        idea_slug = str(result["idea_id"]).removeprefix("idea:")
        if idea_slug not in ideas:
            raise ValueError(f"round {args.round} references unknown idea: {result['idea_id']}")
        idea = ideas[idea_slug]
        _assert_dispatchable_idea(str(result["idea_id"]), idea)
        if not result.get("retry_count"):
            tags = idea.get("tags") if isinstance(idea.get("tags"), list) else []
            retry_levels = [
                int(match.group(1)) for tag in tags
                if (match := re.fullmatch(r"retry_(\d+)", str(tag)))
            ]
            if retry_levels:
                result["retry_count"] = max(retry_levels)

    session = kg.get("session") if isinstance(kg.get("session"), dict) else {}
    actual = int(session.get("actual_rounds_completed", 0) or 0)
    if args.round > actual + 1:
        raise ValueError(f"round skip rejected: actual_rounds_completed={actual}, requested refine={args.round}")
    existing_by_index: dict[int, list[dict]] = {}
    for experiment in experiments.values():
        if int(experiment.get("round", 0) or 0) == args.round:
            existing_by_index.setdefault(int(experiment.get("parallel_index", 0) or 0), []).append(experiment)
    extra_persisted = set(existing_by_index) - expected_indices
    if extra_persisted:
        raise ValueError(
            f"round {args.round} has persisted experiments outside the current slot set: "
            f"{sorted(extra_persisted)}"
        )
    for index, existing in existing_by_index.items():
        if len(existing) > 1:
            raise ValueError(f"round {args.round} has duplicate persisted experiments for parallel {index}")
        incoming = next(result for result in results if result["parallel_index"] == index)
        persisted = existing[0]
        for field in ("idea_id", "verdict", "solution_ref"):
            if persisted.get(field) != incoming.get(field):
                raise ValueError(
                    f"round {args.round} parallel {index} persisted {field}="
                    f"{persisted.get(field)!r} conflicts with incoming {incoming.get(field)!r}"
                )
        persisted_score = persisted.get("speedup")
        incoming_score = incoming.get("speedup")
        if persisted_score is not None or incoming_score is not None:
            try:
                same_score = abs(float(persisted_score) - float(incoming_score)) <= 1e-9
            except (TypeError, ValueError):
                same_score = False
            if not same_score:
                raise ValueError(
                    f"round {args.round} parallel {index} persisted speedup conflicts with incoming evidence"
                )


def prepare_round(args: argparse.Namespace) -> dict[str, Any]:
    """Persist jury selection before subagents start, enabling exact recovery."""
    evo_dir = _evo_dir_from_args(args)
    if detect_mode(evo_dir) != "kg":
        raise ValueError("kg_ops prepare-round requires KG mode")
    if args.round < 1:
        raise ValueError("--round must be >= 1")
    if getattr(args, "parallel_num", None) is not None and int(args.parallel_num) < 1:
        raise ValueError("--parallel-num must be >= 1")
    kg_state = _read_kg(evo_dir)
    session_state = kg_state.get("session") if isinstance(kg_state.get("session"), dict) else {}
    actual_rounds = int(session_state.get("actual_rounds_completed", 0) or 0)
    if args.round > actual_rounds + 1:
        raise ValueError(
            f"round skip rejected: actual_rounds_completed={actual_rounds}, "
            f"requested prepare-round={args.round}"
        )
    if args.round <= actual_rounds:
        raise ValueError(
            f"round {args.round} is already completed; next prepare-round is "
            f"{actual_rounds + 1}"
        )
    parallel_map = _normalise_parallel_map(_read_json_value(args.parallel_map, "--parallel-map"))
    requested_parallel = getattr(args, "parallel_num", None)
    expected = max(1, int(requested_parallel)) if requested_parallel is not None else len(parallel_map)
    if expected < 1:
        raise ValueError("parallel_map must contain at least one selected idea")
    if any(not str(key).isdigit() for key in parallel_map):
        raise ValueError("parallel_map keys must be non-negative integer indices")
    if {int(key) for key in parallel_map} != set(range(expected)):
        raise ValueError(f"parallel_map must contain exactly indices 0..{expected - 1}")

    ideas = load_ideas(evo_dir)
    normalized: dict[str, str] = {}
    for index, spec in parallel_map.items():
        idea_id = _idea_id_from_spec(spec)
        idea_slug = idea_id.removeprefix("idea:")
        if not idea_id or idea_slug not in ideas:
            raise ValueError(f"parallel {index} references unknown idea: {spec!r}")
        _assert_dispatchable_idea(idea_id, ideas[idea_slug])
        normalized[str(int(index))] = idea_id
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("a round cannot dispatch the same idea to multiple parallel slots")

    round_dir = Path(evo_dir) / f"round_{args.round}"
    map_path = round_dir / "parallel_map.json"
    if map_path.is_file():
        existing = _normalise_parallel_map(_read_json_value(str(map_path), "parallel_map.json"))
        if existing != normalized:
            raise ValueError(
                f"round {args.round} parallel_map.json already exists with a different jury selection"
            )
    else:
        round_dir.mkdir(parents=True, exist_ok=True)
        _write_json(map_path, normalized)

    for idea_id in normalized.values():
        idea = ideas[idea_id.removeprefix("idea:")]
        update_idea_outcome(
            evo_dir,
            idea_id.removeprefix("idea:"),
            str(idea.get("outcome") or "pending"),
            stage="active",
        )
    kg = _read_kg(evo_dir)
    kg = _sync_and_finalize(
        evo_dir,
        session=_session_from_kg(kg, evo_dir),
        baseline=kg.get("baseline_performance"),
        kernel_summary=kg.get("kernel_summary"),
        hw_params=kg.get("hw_params"),
        baseline_evidence=kg.get("baseline_evidence"),
        best_score=float(kg.get("best_score", 1.0) or 1.0),
        metadata=kg,
    )
    return {
        "mode": "kg",
        "round": args.round,
        "parallel_map": normalized,
        "path": str(map_path),
        "active_ideas": kg.get("counters", {}).get("active_ideas", 0),
    }


def refine_session(args: argparse.Namespace) -> dict[str, Any]:
    evo_dir = _evo_dir_from_args(args)
    if detect_mode(evo_dir) != "kg":
        raise ValueError("kg_ops refine requires KG mode (knowledge_graph.json is missing)")
    if args.round < 1:
        raise ValueError("--round must be >= 1")
    kg = _read_kg(evo_dir)
    parallel_map = _load_parallel_map(args, evo_dir)
    records = _load_result_records(args)
    if parallel_map:
        present = {
            (
                str(record.get("parallel_index", record.get("parallel", record.get("index", 0))))
                .removeprefix("parallel_")
            )
            for record in records
        }
        present = {
            str(int(index)) if index.isdigit() else index
            for index in present
        }
        for index, idea_spec in parallel_map.items():
            if index in present:
                continue
            records.append({
                "parallel_index": int(index),
                "idea_id": _idea_id_from_spec(idea_spec),
                "verdict": "no",
                "failure_type": "impl_error",
                "failure_reason": "subagent_missing: no parallel directory, result, or terminal marker",
                "solution_ref": f"round_{args.round}/parallel_{index}",
                "compilation_success": False,
                "precision_passed": False,
            })
    if not records:
        raise ValueError(
            "refine has no result evidence; pass --results/--results-dir and a durable --parallel-map"
        )
    records.sort(key=lambda item: int(str(item.get("parallel_index", 0)).removeprefix("parallel_")))
    results = [_normalise_result(record, args.round, parallel_map) for record in records]
    expected_cases = _expected_case_count(args.results_dir)
    for result in results:
        _apply_full_case_guard(result, expected_cases)
    previous_experiments = load_experiments(evo_dir)
    previous_ideas = load_ideas(evo_dir)
    _preflight_refine(
        args, kg, results, previous_ideas, previous_experiments, parallel_map,
    )
    fingerprint = _round_fingerprint(results)
    round_commits = kg.get("round_commits") if isinstance(kg.get("round_commits"), dict) else {}
    committed = round_commits.get(str(args.round))
    if isinstance(committed, dict):
        if committed.get("fingerprint") != fingerprint:
            raise ValueError(
                f"round {args.round} is already committed with different evidence; "
                "refuse to overwrite immutable experiment history"
            )
        return {
            "mode": "kg",
            "round": args.round,
            "idempotent": True,
            "experiments": [f"exp:{result['slug']}" for result in results],
            "generated_ideas": [],
            "generated_claims": [],
            "gap_count": len(parse_gap_map(evo_dir)),
            "query_pack_chars": len(
                (Path(evo_dir) / "query_pack.md").read_text(encoding="utf-8")
            ) if (Path(evo_dir) / "query_pack.md").is_file() else 0,
            "best_score": kg.get("best_score", 1.0),
            "stagnation_count": kg.get("stagnation_count", 0),
            "stagnation_count_vs_base": kg.get("stagnation_count_vs_base", 0),
            "drift_status": _read_drift_status(evo_dir),
            "counters": kg.get("counters", {}),
        }

    created_experiments: list[str] = []
    for result in results:
        add_experiment(
            evo_dir,
            str(result["slug"]),
            str(result["title"]),
            idea_id=result["idea_id"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            speedup=result["speedup"],
            solution_ref=result["solution_ref"],
            round_num=args.round,
            parallel_index=int(result["parallel_index"]),
            profiling_insight=result["profiling_insight"],
            profiling_evidence=result["profiling_evidence"],
            facts=result["facts"] if isinstance(result["facts"], dict) else None,
            diagnosis=result["diagnosis"] if isinstance(result["diagnosis"], dict) else None,
            evaluation=result["evaluation"],
            failure_type=result["failure_type"],
            failure_reason=result["failure_reason"],
            retry_count=result["retry_count"],
            tags=result["tags"] if isinstance(result["tags"], list) else [],
            update_on_exist=True,
        )
        created_experiments.append(f"exp:{result['slug']}")
    _refresh_idea_outcomes(evo_dir, {str(result["idea_id"]) for result in results})

    _recompute_round_counters(kg, load_experiments(evo_dir), load_ideas(evo_dir))
    session = _session_from_kg(kg, evo_dir)
    kg = _sync_and_finalize(
        evo_dir,
        session=session,
        baseline=kg.get("baseline_performance"),
        kernel_summary=kg.get("kernel_summary"),
        hw_params=kg.get("hw_params"),
        baseline_evidence=kg.get("baseline_evidence"),
        best_score=float(kg.get("best_score", 1.0) or 1.0),
        metadata=kg,
    )

    generated_ideas: list[dict] = []
    generation_parallel_num = (
        len(parallel_map)
        or (int(args.parallel_num) if args.parallel_num is not None else len(results))
        or 1
    )
    for result in results:
        generated_ideas.extend(
            idea_generation.generate_ideas_from_experiment(
                evo_dir,
                str(result["slug"]),
                parallel_num=max(1, generation_parallel_num),
                generation_round=args.round,
            )
        )
    idea_slugs = idea_generation.persist_ideas(evo_dir, generated_ideas, generation_round=args.round)
    generated_claims = claim_engine.generate_claims(evo_dir)
    kg = _read_kg(evo_dir)
    gaps = _run_gap_analysis(evo_dir, kg)
    packed = query_pack.build_query_pack(evo_dir, max_chars=args.max_chars)
    kg = _read_kg(evo_dir)
    round_commits = kg.get("round_commits") if isinstance(kg.get("round_commits"), dict) else {}
    round_commits[str(args.round)] = {
        "fingerprint": fingerprint,
        "parallel_indices": [result["parallel_index"] for result in results],
        "committed_at": _now_iso(),
    }
    kg["round_commits"] = round_commits
    kg = _sync_and_finalize(
        evo_dir,
        session=_session_from_kg(kg, evo_dir),
        baseline=kg.get("baseline_performance"),
        kernel_summary=kg.get("kernel_summary"),
        hw_params=kg.get("hw_params"),
        baseline_evidence=kg.get("baseline_evidence"),
        best_score=float(kg.get("best_score", 1.0) or 1.0),
        metadata=kg,
    )
    # The final sync is the point at which the finished round is durable in
    # knowledge_graph.json.  Keep the drift write after that persistence, just
    # like wm_ops.cmd_refine, and keep it best-effort so refine cannot fail on
    # a missing/corrupt runtime state file.
    _maybe_update_drift_status(evo_dir, kg)
    _maybe_mark_profiling_skipped(evo_dir, args.results_dir, results)
    drift_status = _read_drift_status(evo_dir)
    return {
        "mode": "kg",
        "round": args.round,
        "idempotent": False,
        "experiments": created_experiments,
        "generated_ideas": idea_slugs,
        "generated_claims": generated_claims,
        "gap_count": len(gaps),
        "query_pack_chars": len(packed),
        "best_score": kg.get("best_score", 1.0),
        "stagnation_count": kg.get("stagnation_count", 0),
        "stagnation_count_vs_base": kg.get("stagnation_count_vs_base", 0),
        "drift_status": drift_status,
        "counters": kg.get("counters", {}),
    }


def summary(evo_dir: str, max_chars: int = 1200) -> str:
    kg = _read_kg(evo_dir)
    stats = get_stats(evo_dir)
    session = kg.get("session") or {}
    lines = [
        "[Knowledge Graph Summary]",
        f"Session: {session.get('session_id', '?')} | "
        f"Rounds: {session.get('actual_rounds_completed', 0)}/{session.get('requested_rounds', 0)}",
        f"Best: {kg.get('best_score', 1.0)}x | "
        f"Stagnation: {kg.get('stagnation_count', 0)}/"
        f"{kg.get('stagnation_count_vs_base', 0)} | "
        f"Drift: {_read_drift_status(evo_dir)}",
        f"Supervisor: {kg.get('supervisor_used_count', 0)} used | "
        f"last round {kg.get('last_supervisor_round', -2)}",
        f"Ideas: {stats['total_ideas']} | Experiments: {stats['total_experiments']} | "
        f"Claims: {stats['total_claims']} | Open gaps: {stats['open_gaps']}",
    ]
    if kg.get("kernel_summary"):
        lines.append(f"Kernel: {kg['kernel_summary'][:260]}")
    query_path = Path(evo_dir) / "query_pack.md"
    if query_path.is_file():
        lines.append(f"Query pack: {query_path} ({len(query_path.read_text(encoding='utf-8'))} chars)")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max(0, max_chars - 3)] + "..."


def _supervisor_parent_context(
    evo_dir: str, round_num: int,
) -> tuple[str | None, str | None]:
    """Return the latest experiment and its idea for Supervisor provenance."""
    experiments = load_experiments(evo_dir)
    candidates = [
        exp for exp in experiments.values()
        if int(exp.get("round", 0) or 0) <= round_num
    ]
    if not candidates:
        return None, None

    def score(exp: dict) -> tuple[int, int, float]:
        verdict_rank = 1 if exp.get("verdict") in ("yes", "partial") else 0
        speedup = exp.get("speedup")
        try:
            numeric_speedup = float(speedup) if speedup is not None else 0.0
        except (TypeError, ValueError):
            numeric_speedup = 0.0
        return int(exp.get("round", 0) or 0), verdict_rank, numeric_speedup

    parent = max(candidates, key=score)
    parent_exp = str(parent.get("node_id") or "") or None
    parent_idea = str(parent.get("idea_id") or "") or None
    return parent_exp, parent_idea


def _supervisor_idea_from_node(
    node: dict[str, Any], trigger: str, round_num: int,
    parent_exp: str | None, parent_idea: str | None,
) -> dict[str, Any]:
    """Translate one MCTS-shaped Supervisor node into a KG idea payload."""
    description = str(node.get("description") or "Supervisor suggested direction")
    rationale = str(node.get("rationale") or "")
    title = f"Supervisor R{round_num}: {description[:100]}"
    raw_tags = node.get("strategy_combination", [])
    if isinstance(raw_tags, str):
        raw_tags = [part.strip() for part in raw_tags.split(",") if part.strip()]
    strategy_tags = [str(tag) for tag in raw_tags if str(tag).strip()]

    mode = str(node.get("mode") or "strategy_guided")
    if mode not in {"strategy_guided", "open_exploration", "profiling_driven", "workflow_guided"}:
        mode = "open_exploration" if not strategy_tags else "strategy_guided"
    # Supervisor's MCTS output has no workflow_ref; do not create an invalid
    # workflow_guided KG idea when the model used that legacy label casually.
    if mode == "workflow_guided":
        mode = "strategy_guided"

    optimization_type = str(node.get("optimization_type") or "algorithm")
    if optimization_type not in {
        "bandwidth", "tiling", "algorithm", "register_opt", "vf_fusion", "instruction_sched",
    }:
        optimization_type = idea_generation._infer_optimization_type(strategy_tags)

    try:
        difficulty = max(1, min(5, int(node.get("difficulty", 3))))
    except (TypeError, ValueError):
        difficulty = 3

    target_gaps = node.get("target_gaps", [])
    if not isinstance(target_gaps, list):
        target_gaps = []
    return {
        "title": title,
        "strategy_tags": strategy_tags,
        "optimization_type": optimization_type,
        "difficulty": difficulty,
        "mode": mode,
        "generation_lens": "supervisor_sourced",
        "target_gaps": [str(gap) for gap in target_gaps],
        "tags": ["supervisor", trigger],
        "parent_idea": parent_idea,
        "parent_experiment": parent_exp,
        "description": description,
        "thesis": description,
        "strategy_rationale": rationale,
        "generation_round": round_num,
    }


def record_supervisor(
    evo_dir: str, round_num: int, trigger: str, result: dict[str, Any],
    max_rounds: int | None = None,
) -> dict[str, Any]:
    """Consume one Supervisor JSON result and persist it in KG mode.

    The agent performs trigger-specific evidence checks; this command owns the
    shared cooldown/cap guard, counter mutation, KG idea conversion, graph
    provenance, and open-question recording so the agent never edits the JSON.
    """
    if detect_mode(evo_dir) != "kg":
        raise ValueError("kg_ops supervisor requires KG mode")
    if round_num < 1:
        raise ValueError("Supervisor round must be >= 1")
    if trigger not in _SUPERVISOR_TRIGGERS:
        raise ValueError(f"Invalid Supervisor trigger: {trigger}")
    if not isinstance(result, dict):
        raise ValueError("Supervisor result must be a JSON object")

    kg = _read_kg(evo_dir)
    used = int(kg.get("supervisor_used_count", 0) or 0)
    last_round = int(kg.get("last_supervisor_round", -2) or -2)
    session = kg.get("session") or {}
    cap = max_rounds if max_rounds is not None else int(session.get("requested_rounds", 0) or 0)

    if round_num - last_round < 2:
        return {
            "accepted": False,
            "blocked": "cooldown",
            "reason": f"round {round_num} - last_supervisor_round {last_round} < 2",
            "supervisor_used_count": used,
            "last_supervisor_round": last_round,
        }
    if used >= cap:
        return {
            "accepted": False,
            "blocked": "cap",
            "reason": f"supervisor_used_count {used} >= max_rounds {cap}",
            "supervisor_used_count": used,
            "last_supervisor_round": last_round,
        }

    verdict = str(result.get("verdict", "")).lower()
    if verdict not in {"continue", "terminate"}:
        raise ValueError("Supervisor verdict must be continue or terminate")
    analysis = result.get("analysis")
    if not isinstance(analysis, dict):
        analysis = {}
    raw_nodes = result.get("new_nodes", [])
    if not isinstance(raw_nodes, list):
        raise ValueError("Supervisor new_nodes must be an array")
    parent_exp, parent_idea = _supervisor_parent_context(evo_dir, round_num)
    ideas: list[dict[str, Any]] = []
    if verdict == "continue":
        for node in raw_nodes[:3]:
            if isinstance(node, dict):
                ideas.append(_supervisor_idea_from_node(
                    node, trigger, round_num, parent_exp, parent_idea,
                ))

    idea_slugs = idea_generation.persist_ideas(
        evo_dir, ideas, generation_round=round_num,
    )

    open_questions = kg.get("open_questions")
    if not isinstance(open_questions, list):
        open_questions = []
    bottleneck = analysis.get("bottleneck")
    if bottleneck or analysis.get("reasoning"):
        open_questions.append({
            "source": "supervisor",
            "round": round_num,
            "trigger": trigger,
            "bottleneck": str(bottleneck or ""),
            "question": str(bottleneck or analysis.get("reasoning") or ""),
        })

    kg["open_questions"] = open_questions
    kg["supervisor_used_count"] = used + 1
    kg["last_supervisor_round"] = round_num
    # Preserve the original MCTS asymmetry: only the GATE/stall branch may
    # terminate; a successful stall intervention also clears its counter.
    if trigger == "stall" and verdict == "continue" and idea_slugs:
        kg["stagnation_count"] = 0

    _write_json(Path(evo_dir) / "knowledge_graph.json", kg)
    kg = _sync_and_finalize(
        evo_dir,
        session=_session_from_kg(kg, evo_dir),
        baseline=kg.get("baseline_performance"),
        kernel_summary=kg.get("kernel_summary"),
        hw_params=kg.get("hw_params"),
        baseline_evidence=kg.get("baseline_evidence"),
        best_score=float(kg.get("best_score", 1.0) or 1.0),
    )
    return {
        "accepted": True,
        "trigger": trigger,
        "verdict": verdict,
        "generated_ideas": idea_slugs,
        "supervisor_used_count": kg.get("supervisor_used_count", used + 1),
        "last_supervisor_round": kg.get("last_supervisor_round", round_num),
        "open_questions": len(kg.get("open_questions", [])),
    }


def _entity_kind(ref: Any) -> str | None:
    if not isinstance(ref, str) or ":" not in ref:
        return None
    prefix, _ = ref.split(":", 1)
    return {"idea": "idea", "exp": "experiment", "claim": "claim", "gap": "gap"}.get(prefix)


def _entity_slug(ref: str) -> str:
    if ":" in ref:
        return ref.split(":", 1)[1]
    return ref


def _check_frontmatter_entities(evo_dir: str, kind: str, prefix: str) -> tuple[dict[str, dict], list[str]]:
    errors: list[str] = []
    directory = Path(evo_dir) / {"idea": "ideas", "experiment": "experiments", "claim": "claims"}[kind]
    entities: dict[str, dict] = {}
    seen_node_ids: set[str] = set()
    if not directory.is_dir():
        errors.append(f"missing entity directory: {directory}")
        return entities, errors
    for md_path in sorted(directory.glob("*.md")):
        fm = _parse_frontmatter(_read_text(md_path))
        if not fm:
            errors.append(f"{md_path}: missing or invalid frontmatter")
            continue
        if fm.get("type") != kind:
            errors.append(f"{md_path}: type={fm.get('type')!r}, expected {kind!r}")
        slug = md_path.stem
        node_id = fm.get("node_id")
        expected_id = f"{prefix}:{slug}"
        if node_id != expected_id:
            errors.append(f"{md_path}: node_id={node_id!r}, expected {expected_id!r}")
        if node_id in seen_node_ids:
            errors.append(f"duplicate entity node_id: {node_id}")
        seen_node_ids.add(node_id)
        entities[slug] = fm
    return entities, errors


def _validate_sidecars(evo_dir: str, experiments: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    directory = Path(evo_dir) / "experiments"
    for sidecar_path in sorted(directory.glob("*.json")):
        slug = sidecar_path.stem
        md_path = directory / f"{slug}.md"
        if not md_path.is_file():
            errors.append(f"orphan experiment sidecar: {sidecar_path}")
            continue
        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"invalid experiment sidecar JSON: {sidecar_path}")
            continue
        if not isinstance(data, dict):
            errors.append(f"experiment sidecar must be an object: {sidecar_path}")
            continue
        fm = experiments.get(slug, {})
        # Structured profiling fields intentionally live only in the sidecar.
        # If identity/frontmatter fields are present in a hand-written sidecar,
        # compare them so drift cannot hide behind the split representation.
        for field in ("type", "node_id", "idea_id", "verdict", "round", "parallel_index", "speedup"):
            if field in data and field in fm and data[field] != fm[field]:
                errors.append(
                    f"{sidecar_path}: sidecar {field}={data[field]!r} "
                    f"does not match frontmatter {fm[field]!r}"
                )
    for directory_name in ("ideas", "claims"):
        for sidecar_path in (Path(evo_dir) / directory_name).glob("*.json"):
            errors.append(f"unexpected sidecar in {directory_name}: {sidecar_path}")
    return errors


def _load_edges_strict(evo_dir: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Load every edge and retain parse failures for the integrity report.

    ``kg_wiki.load_edges`` is intentionally forgiving for normal consumers,
    but a validator must not silently discard a malformed JSONL record.
    """
    edges_path = Path(evo_dir) / "graph" / "edges.jsonl"
    if not edges_path.is_file():
        return [], []
    edges: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = edges_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"cannot read edge file {edges_path}: {exc}"]
    for line_no, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            edge = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"edge line {line_no}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(edge, dict):
            errors.append(f"edge line {line_no}: edge must be a JSON object")
            continue
        edges.append(edge)
    return edges, errors


def validate_kg(evo_dir: str) -> list[str]:
    """Return integrity errors without repairing the graph."""
    errors: list[str] = []
    if detect_mode(evo_dir) != "kg":
        return [f"not a KG session: {Path(evo_dir) / 'knowledge_graph.json'} is missing"]
    try:
        kg = _read_kg(evo_dir)
    except ValueError as exc:
        return [str(exc)]
    for field, default in (("supervisor_used_count", 0), ("last_supervisor_round", -2)):
        value = kg.get(field, default)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"knowledge_graph.json: {field} must be an integer")
    if not isinstance(kg.get("open_questions", []), list):
        errors.append("knowledge_graph.json: open_questions must be a list")
    ideas, idea_errors = _check_frontmatter_entities(evo_dir, "idea", "idea")
    experiments, experiment_errors = _check_frontmatter_entities(evo_dir, "experiment", "exp")
    claims, claim_errors = _check_frontmatter_entities(evo_dir, "claim", "claim")
    errors.extend(idea_errors + experiment_errors + claim_errors)
    errors.extend(_validate_sidecars(evo_dir, experiments))

    idea_ids = {f"idea:{slug}" for slug in ideas}
    experiment_ids = {f"exp:{slug}" for slug in experiments}
    claim_ids = {f"claim:{slug}" for slug in claims}
    gaps = parse_gap_map(evo_dir)
    gap_ids = {f"gap:{gap['id']}" for gap in gaps}
    open_gap_keys: dict[str, str] = {}
    for gap in gaps:
        key = str(gap.get("key") or "").strip().lower()
        if gap.get("status", "open") != "open" and not str(
            gap.get("closure_reason") or ""
        ).strip():
            errors.append(
                f"{gap.get('id')}: closed gap missing closure_reason"
            )
        if not key:
            continue  # legacy gap maps may not have stable identities yet
        if gap.get("status", "open") == "open" and key in open_gap_keys:
            errors.append(
                f"duplicate open gap key: {key} ({open_gap_keys[key]}, {gap.get('id')})"
            )
        elif gap.get("status", "open") == "open":
            open_gap_keys[key] = str(gap.get("id"))

    # Entity IDs are namespaced, but duplicate slugs still make human-facing
    # links and generated refs ambiguous.  Treat a cross-directory collision
    # as an integrity error as well as duplicate node_ids within a directory.
    slug_sets = {
        "ideas": set(ideas),
        "experiments": set(experiments),
        "claims": set(claims),
    }
    for left_name, left_slugs in slug_sets.items():
        for right_name, right_slugs in slug_sets.items():
            if left_name >= right_name:
                continue
            for slug in sorted(left_slugs & right_slugs):
                errors.append(f"duplicate entity slug across {left_name}/{right_name}: {slug}")

    for slug, experiment in experiments.items():
        verdict = experiment.get("verdict")
        if verdict not in {"yes", "partial", "no"}:
            errors.append(f"exp:{slug}: invalid verdict {verdict!r}")
        confidence = experiment.get("confidence")
        if confidence not in {"high", "medium", "low"}:
            errors.append(f"exp:{slug}: invalid confidence {confidence!r}")
        failure_type = experiment.get("failure_type")
        if failure_type not in {None, "impl_error", "strategy_infeasible", "profiling_lost"}:
            errors.append(f"exp:{slug}: invalid failure_type {failure_type!r}")
        try:
            raw_round = experiment.get("round", 0)
            raw_parallel_index = experiment.get("parallel_index", 0)
            if isinstance(raw_round, bool) or isinstance(raw_parallel_index, bool):
                raise ValueError
            round_num = int(raw_round or 0)
            parallel_index = int(raw_parallel_index or 0)
            if round_num < 0 or parallel_index < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"exp:{slug}: round/parallel_index must be non-negative integers")
        retry_count = experiment.get("retry_count", 0)
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
        ):
            errors.append(f"exp:{slug}: retry_count must be a non-negative integer")
        speedup = experiment.get("speedup")
        if verdict in {"yes", "partial"} and (
            not isinstance(speedup, (int, float))
            or isinstance(speedup, bool)
            or not math.isfinite(float(speedup))
            or float(speedup) <= 0
        ):
            errors.append(f"exp:{slug}: passed experiment requires a finite positive speedup")
        evaluation = experiment.get("evaluation")
        if evaluation is not None and not isinstance(evaluation, dict):
            errors.append(f"exp:{slug}: evaluation sidecar must be an object")
        elif isinstance(evaluation, dict):
            gating = evaluation.get("gating")
            gating_to_verdict = {
                "fully_passed": "yes",
                "partial_passed": "partial",
                "target_regression": "no",
                "generalization_regression": "no",
                "failed": "no",
            }
            if gating not in (None, "") and gating not in gating_to_verdict:
                errors.append(f"exp:{slug}: invalid evaluation.gating {gating!r}")
            elif gating in gating_to_verdict and verdict != gating_to_verdict[gating]:
                errors.append(
                    f"exp:{slug}: verdict={verdict!r} conflicts with evaluation.gating={gating!r}"
                )
            for flag in ("compilation_success", "precision_passed"):
                if flag in evaluation and not isinstance(evaluation[flag], bool):
                    errors.append(f"exp:{slug}: evaluation.{flag} must be boolean")
                if verdict in {"yes", "partial"} and evaluation.get(flag) is False:
                    errors.append(
                        f"exp:{slug}: passed verdict conflicts with evaluation.{flag}=false"
                    )
        idea_id = experiment.get("idea_id")
        resolved = f"idea:{_entity_slug(str(idea_id))}" if idea_id else ""
        if resolved not in idea_ids:
            errors.append(f"exp:{slug}: idea_id does not resolve: {idea_id!r}")
    for slug, claim in claims.items():
        for field in ("supporting_experiments", "refuting_experiments"):
            refs = claim.get(field, [])
            if not isinstance(refs, list):
                errors.append(f"claim:{slug}: {field} must be a list")
                continue
            for ref in refs:
                resolved = f"exp:{_entity_slug(str(ref))}"
                if resolved not in experiment_ids:
                    errors.append(f"claim:{slug}: {field} ref does not resolve: {ref!r}")

    for slug, idea in ideas.items():
        parent_idea_ref = idea.get("parent_idea")
        if parent_idea_ref not in (None, "", "null"):
            parent_id = f"idea:{_entity_slug(str(parent_idea_ref))}"
            chain: set[str] = {f"idea:{slug}"}
            cursor = parent_id
            while cursor in idea_ids:
                if cursor in chain:
                    errors.append(f"idea:{slug}: parent_idea lineage contains a cycle")
                    break
                chain.add(cursor)
                parent_slug = cursor.removeprefix("idea:")
                next_ref = ideas.get(parent_slug, {}).get("parent_idea")
                if next_ref in (None, "", "null"):
                    break
                cursor = f"idea:{_entity_slug(str(next_ref))}"
        for field, expected_kind, known_ids in (
            ("parent_idea", "idea", idea_ids),
            ("parent_experiment", "exp", experiment_ids),
        ):
            ref = idea.get(field)
            if ref in (None, "", "null"):
                continue
            resolved = f"{expected_kind}:{_entity_slug(str(ref))}"
            if resolved not in known_ids:
                errors.append(f"idea:{slug}: {field} does not resolve: {ref!r}")
        if idea.get("mode") != "workflow_guided":
            continue
        workflow_ref = idea.get("workflow_ref")
        if (
            not isinstance(workflow_ref, dict)
            or not str(workflow_ref.get("workflow_file") or workflow_ref.get("file") or "").strip()
            or not isinstance(workflow_ref.get("stage_order"), list)
            or not workflow_ref.get("stage_order")
        ):
            errors.append(
                f"idea:{slug}: workflow_guided requires workflow_ref.workflow_file and non-empty stage_order"
            )

    endpoint_sets = {
        "idea": idea_ids,
        "experiment": experiment_ids,
        "claim": claim_ids,
        "gap": gap_ids,
    }
    edges, edge_parse_errors = _load_edges_strict(evo_dir)
    errors.extend(edge_parse_errors)
    for line_no, edge in enumerate(edges, start=1):
        edge_type = edge.get("type")
        if edge_type not in _VALID_EDGE_TYPES:
            errors.append(f"edge line {line_no}: unknown type {edge_type!r}")
            continue
        from_id, to_id = edge.get("from"), edge.get("to")
        from_kind, to_kind = _entity_kind(from_id), _entity_kind(to_id)
        allowed_from, allowed_to = _EDGE_DIRECTIONS[edge_type]
        if from_kind not in allowed_from or to_kind not in allowed_to:
            errors.append(
                f"edge line {line_no}: direction {from_id!r} --{edge_type}--> {to_id!r} "
                f"must be {sorted(allowed_from)} -> {sorted(allowed_to)}"
            )
            continue
        from_set = endpoint_sets.get(from_kind, set())
        to_set = endpoint_sets.get(to_kind, set())
        # gap:Gx is a legitimate non-entity endpoint: gaps are records in
        # gap_map.md rather than files under an entity directory.  They still
        # must be declared in that map, so malformed/dangling gap IDs fail.
        if from_id not in from_set:
            errors.append(f"edge line {line_no}: unresolved from endpoint {from_id!r}")
        if to_id not in to_set:
            errors.append(f"edge line {line_no}: unresolved to endpoint {to_id!r}")

    counters = kg.get("counters")
    if not isinstance(counters, dict):
        errors.append("knowledge_graph.json: counters must be an object")
    else:
        expected_counters = {
            "total_ideas": len(ideas),
            "active_ideas": sum(1 for idea in ideas.values() if idea.get("stage") == "active"),
            "total_experiments": len(experiments),
            "total_claims": len(claims),
            "rounds_completed": max((_safe_int(exp.get("round", 0) or 0) for exp in experiments.values()), default=0),
        }
        for key, expected in expected_counters.items():
            if counters.get(key) != expected:
                errors.append(f"counter mismatch: {key}={counters.get(key)!r}, expected {expected}")

    registry = kg.get("idea_registry")
    if not isinstance(registry, dict):
        errors.append("knowledge_graph.json: idea_registry must be an object")
    elif set(registry) != set(ideas):
        errors.append(
            "idea_registry mismatch: "
            f"registered={sorted(registry)}, on_disk={sorted(ideas)}"
        )
    if isinstance(registry, dict):
        for slug in sorted(set(registry) & set(ideas)):
            if not isinstance(registry[slug], dict):
                errors.append(f"idea_registry[{slug}] must be an object")
                continue
            tested_rounds = [
                _safe_int(exp.get("round", 0) or 0)
                for exp in experiments.values()
                if str(exp.get("idea_id", "")).removeprefix("idea:") == slug
                and _safe_int(exp.get("round", 0) or 0) > 0
            ]
            expected_tested = min(tested_rounds) if tested_rounds else None
            if "round_tested" in registry[slug] and registry[slug].get("round_tested") != expected_tested:
                errors.append(
                    f"idea_registry[{slug}].round_tested={registry[slug].get('round_tested')!r}, "
                    f"expected {expected_tested!r}"
                )

    session = kg.get("session") or {}
    max_round = max((_safe_int(exp.get("round", 0) or 0) for exp in experiments.values()), default=0)
    actual_rounds = _safe_int(session.get("actual_rounds_completed", 0) or 0)
    if actual_rounds != max_round:
        errors.append(
            f"session.actual_rounds_completed={actual_rounds} does not match on-disk max round {max_round}"
        )
    # ``round_commits`` was added after the first KG/wiki releases.  A
    # pre-existing graph without the field is still readable/valid; new
    # refine calls create it before they can be considered committed.  Once a
    # graph advertises the field, however, every completed round must have a
    # durable fingerprint so a crash cannot silently rewrite history.
    round_commits = kg.get("round_commits")
    if round_commits is not None and not isinstance(round_commits, dict):
        errors.append("knowledge_graph.json: round_commits must be an object")
    elif isinstance(round_commits, dict):
        for round_num in range(1, actual_rounds + 1):
            commit = round_commits.get(str(round_num))
            if not isinstance(commit, dict) or not commit.get("fingerprint"):
                errors.append(
                    f"round {round_num}: missing durable round_commit; rerun kg_ops refine for recovery"
                )
                continue
            indices = commit.get("parallel_indices")
            if (
                not isinstance(indices, list)
                or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
                or len(indices) != len(set(indices))
            ):
                errors.append(f"round {round_num}: invalid committed parallel_indices")
                continue
            persisted_indices = sorted(
                _safe_int(exp.get("parallel_index", 0) or 0)
                for exp in experiments.values()
                if _safe_int(exp.get("round", 0) or 0) == round_num
            )
            if sorted(indices) != persisted_indices:
                errors.append(
                    f"round {round_num}: committed parallel_indices={sorted(indices)!r} "
                    f"do not match persisted experiments={persisted_indices!r}"
                )
    return errors


def _print_validation(evo_dir: str) -> int:
    errors = validate_kg(evo_dir)
    if errors:
        print("KG validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("KG validation PASSED: entities, edges, sidecars, and counters are consistent.")
    return 0


def _add_common_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("evo_dir_pos", nargs="?", help="Evolution directory (alternative to --evo-dir)")
    parser.add_argument("--evo-dir", dest="evo_dir", help="Evolution directory")


def _cmd_init(args: argparse.Namespace) -> None:
    print(json.dumps(init_session(args), ensure_ascii=False, indent=2))


def _cmd_refine(args: argparse.Namespace) -> None:
    print(json.dumps(refine_session(args), ensure_ascii=False, indent=2))


def _cmd_prepare_round(args: argparse.Namespace) -> None:
    print(json.dumps(prepare_round(args), ensure_ascii=False, indent=2))


def _cmd_supervisor(args: argparse.Namespace) -> None:
    result = _read_json_value(args.result, "--result")
    print(json.dumps(
        record_supervisor(
            _evo_dir_from_args(args), args.round, args.trigger, result,
            max_rounds=args.max_rounds,
        ),
        ensure_ascii=False,
        indent=2,
    ))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KG-mode evolution orchestrator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize and seed a KG session")
    _add_common_path_arguments(p_init)
    p_init.add_argument("--session-id")
    p_init.add_argument("--op-name", default=None)
    p_init.add_argument("--requested-rounds", type=int, default=1)
    p_init.add_argument("--parallel-num", type=int, default=3)
    p_init.add_argument("--start-time", default=None)
    p_init.add_argument("--kernel-summary", default="")
    p_init.add_argument("--kernel-summary-file", default=None)
    p_init.add_argument("--hw-params", default=None)
    p_init.add_argument("--kernel-traits", default=None)
    p_init.add_argument("--baseline-performance", default=None)
    p_init.add_argument("--baseline-evidence", default=None)
    p_init.add_argument("--baseline-eval", default=None)
    p_init.add_argument("--baseline-time-us", type=float, default=None)
    p_init.add_argument("--baseline-time-ms", type=float, default=None)
    p_init.add_argument("--baseline-speedup", type=float, default=None)
    p_init.add_argument("--max-chars", type=int, default=8000)
    p_init.set_defaults(func=_cmd_init)

    p_refine = sub.add_parser("refine", help="Commit one round of KG experiments")
    _add_common_path_arguments(p_refine)
    p_refine.add_argument("--round", type=int, required=True)
    p_refine.add_argument("--results", "--parallel-results", dest="results", default=None,
                          help="JSON array/object or JSON file with per-parallel results")
    p_refine.add_argument("--results-dir", default=None,
                          help="Directory containing parallel_*/evaluation_results.json")
    p_refine.add_argument("--parallel-map", default=None,
                          help="JSON object mapping parallel index to idea slug/ref")
    p_refine.add_argument(
        "--parallel-num", type=int, default=None,
        help="Expected parallel slots; inferred from durable parallel_map when omitted",
    )
    p_refine.add_argument("--max-chars", type=int, default=8000)
    p_refine.set_defaults(func=_cmd_refine)

    p_prepare = sub.add_parser(
        "prepare-round", help="Persist jury parallel_index→idea selection before dispatch"
    )
    _add_common_path_arguments(p_prepare)
    p_prepare.add_argument("--round", type=int, required=True)
    p_prepare.add_argument("--parallel-map", required=True,
                           help="JSON object mapping parallel index to idea slug/ref")
    p_prepare.add_argument(
        "--parallel-num", type=int, default=None,
        help="Expected parallel slots; inferred from parallel-map when omitted",
    )
    p_prepare.set_defaults(func=_cmd_prepare_round)

    p_supervisor = sub.add_parser(
        "supervisor", aliases=["record-supervisor"],
        help="Consume a Supervisor JSON result in KG mode",
    )
    _add_common_path_arguments(p_supervisor)
    p_supervisor.add_argument("--round", type=int, required=True)
    p_supervisor.add_argument("--trigger", choices=sorted(_SUPERVISOR_TRIGGERS), required=True)
    p_supervisor.add_argument(
        "--result", required=True,
        help="Supervisor JSON object or @path containing the result",
    )
    p_supervisor.add_argument(
        "--max-rounds", type=int, default=None,
        help="Hard Supervisor cap; defaults to session.requested_rounds",
    )
    p_supervisor.set_defaults(func=_cmd_supervisor)

    p_summary = sub.add_parser("summary", help="Print compact KG context")
    _add_common_path_arguments(p_summary)
    p_summary.add_argument("--max-chars", type=int, default=1200)
    p_summary.set_defaults(func=lambda args: print(summary(_evo_dir_from_args(args), args.max_chars)))

    p_validate = sub.add_parser("validate", help="Validate KG integrity")
    _add_common_path_arguments(p_validate)
    p_validate.set_defaults(func=lambda args: _print_validation(_evo_dir_from_args(args)))

    p_stats = sub.add_parser("stats", help="Show KG statistics")
    _add_common_path_arguments(p_stats)
    p_stats.set_defaults(func=lambda args: print(json.dumps(
        get_stats(_evo_dir_from_args(args)), ensure_ascii=False, indent=2)))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    try:
        result = args.func(args)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.error("kg_ops: ERROR — %s", exc)
        sys.exit(1)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
