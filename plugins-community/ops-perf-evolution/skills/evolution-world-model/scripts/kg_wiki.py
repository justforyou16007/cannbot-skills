#!/usr/bin/env python3
"""
Knowledge-Graph Wiki — Entity CRUD for the Idea-Exp-Claim evolution pipeline.

Adapted from ARIS research_wiki.py for the operator evolution domain. All
knowledge-graph mutations (idea/experiment/claim/edge/gap) go through this
single module so invariants are enforced in one place.

Usage:
    python3 kg_wiki.py init <kg_root>
    python3 kg_wiki.py upsert_idea <kg_root> --slug <slug> --title "..." \\
        --strategy-tags P1,P7 --optimization-type bandwidth --difficulty 3 \\
        --mode strategy_guided --generation-lens profiling_driven \\
        --generation-round 1 [--target-gaps G1,G3] [--thesis "..."]
    python3 kg_wiki.py add_experiment <kg_root> --slug <slug> --title "..." \\
        --idea-id idea:<slug> --verdict yes --speedup 2.3 \\
        --solution-ref round_1/parallel_0 --round 1 --parallel-index 0
    python3 kg_wiki.py add_claim <kg_root> --slug <slug> --name "..." \\
        --status verified --statement "..." --scope "..." \\
        --strategy-implications P1 --bottleneck-labels mte2_stall
    python3 kg_wiki.py add_edge <kg_root> --from <id> --to <id> --type <type>
    python3 kg_wiki.py sync <kg_root>
    python3 kg_wiki.py stats <kg_root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IDEA_STAGES = {"proposed", "active", "piloted", "archived"}
_IDEA_OUTCOMES = {"unknown", "pending", "negative", "mixed", "positive"}
_IDEA_MODES = {"strategy_guided", "open_exploration", "profiling_driven", "workflow_guided"}
_GENERATION_LENSES = {
    "profiling_driven", "method_transfer", "gap_analysis",
    "continuation", "open_exploration", "workflow_guided",
    "supervisor_sourced",
}
_OPTIMIZATION_TYPES = {
    "bandwidth", "tiling", "algorithm",
    "register_opt", "vf_fusion", "instruction_sched",
}

_EXPERIMENT_VERDICTS = {"yes", "partial", "no"}
_EXPERIMENT_CONFIDENCE = {"high", "medium", "low"}
_FAILURE_TYPES = {"impl_error", "strategy_infeasible"}

_CLAIM_STATUSES = {"drafted", "unproven", "verified", "refuted"}
_CLAIM_POLARITIES = {"positive", "negative"}

_EDGE_TYPES = {
    "tested_by",        # idea -> exp
    "supports",         # exp -> claim/idea
    "invalidates",      # exp -> claim/idea
    "addresses_gap",    # idea/claim -> gap
    "inspired_by",      # idea -> idea (cross-idea lineage)
    "derived_from",     # idea -> exp (new idea from experiment result)
    "depends_on",       # claim -> claim
    "refutes",          # claim -> claim
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaml_quote(val: str) -> str:
    """Quote a string for safe YAML frontmatter embedding."""
    if not val:
        return '""'
    # If it contains characters that might break YAML, wrap in quotes
    if any(c in val for c in (':', '#', '{', '}', '[', ']', ',', '&', '*',
                               '?', '|', '-', '<', '>', '=', '!', '%',
                               '@', '`', '"', "'", '\n')):
        escaped = val.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        return f'"{escaped}"'
    return val


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    original_text = text
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text).strip("-_")
    if text:
        return text[:80]
    # Preserve a stable, non-generic identity for titles written entirely in
    # scripts that are removed by the ASCII normalization above (for example,
    # Chinese titles).
    return hashlib.md5(original_text.encode("utf-8")).hexdigest()[:12]


def _read_text(path: Path) -> str:
    """Read file text, returning empty string if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML-ish frontmatter from markdown. Returns dict of key-value pairs."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    fm_block = text[4:end]
    result: dict[str, Any] = {}
    for line in fm_block.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(\w[\w_-]*)\s*:\s*(.*)$", line)
        if match:
            key, val = match.group(1), match.group(2).strip()
            # Parse list values: [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip().strip('"').strip("'")
                         for v in val[1:-1].split(",") if v.strip()]
                result[key] = items
            # Parse quoted strings
            elif val.startswith('"') and val.endswith('"'):
                result[key] = val[1:-1].replace('\\"', '"').replace('\\n', '\n')
            elif val == "null" or val == "~":
                result[key] = None
            else:
                # Try numeric
                try:
                    result[key] = float(val) if '.' in val else int(val)
                except ValueError:
                    result[key] = val
    return result


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

def append_log(kg_root: str, message: str) -> None:
    """Append a timestamped line to log.md."""
    root = Path(kg_root)
    log_path = root / "log.md"
    if not log_path.exists():
        log_path.write_text("# Evolution Knowledge Graph Log\n\n", encoding="utf-8")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"- `{_now_iso()}` — {message}\n")


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_kg(kg_root: str) -> None:
    """Initialize a knowledge graph directory structure."""
    root = Path(kg_root)
    for subdir in ("ideas", "experiments", "claims", "graph"):
        (root / subdir).mkdir(parents=True, exist_ok=True)

    # gap_map.md
    gap_path = root / "gap_map.md"
    if not gap_path.exists():
        gap_path.write_text(
            "# Gap Map\n\n_Knowledge gaps identified during evolution._\n",
            encoding="utf-8",
        )

    # log.md
    log_path = root / "log.md"
    if not log_path.exists():
        log_path.write_text("# Evolution Knowledge Graph Log\n\n", encoding="utf-8")

    # graph/edges.jsonl
    edges_path = root / "graph" / "edges.jsonl"
    if not edges_path.exists():
        edges_path.touch()

    # query_pack.md
    qp_path = root / "query_pack.md"
    if not qp_path.exists():
        qp_path.write_text("# Query Pack\n\n_Compressed evolution context._\n",
                          encoding="utf-8")

    # knowledge_graph.json — registry and session-level metadata.  Do not
    # overwrite an existing file: later initialization calls must preserve
    # accumulated evolution state.
    kg_json_path = root / "knowledge_graph.json"
    if not kg_json_path.exists():
        kg_json_path.write_text(
            json.dumps({
                "kernel_summary": "",
                "session": {
                    "session_id": "",
                    "start_time": "",
                    "requested_rounds": 0,
                    "actual_rounds_completed": 0,
                    "evo_dir": str(root),
                    "op_name": "",
                },
                "baseline_performance": {
                    "speedup": 1.0,
                    "time_ms": None,
                },
                "hw_params": None,
                "baseline_evidence": None,
                "best_score": 1.0,
                "stagnation_count": 0,
                "stagnation_count_vs_base": 0,
                "supervisor_used_count": 0,
                "last_supervisor_round": -2,
                "open_questions": [],
                "discovered_strategies": [],
                "idea_registry": {},
                "counters": {
                    "total_ideas": 0,
                    "active_ideas": 0,
                    "total_experiments": 0,
                    "total_claims": 0,
                    "rounds_completed": 0,
                },
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    append_log(kg_root, "init: knowledge graph initialized")
    logger.info("Knowledge graph initialized at %s", root)


# ---------------------------------------------------------------------------
# Edge operations
# ---------------------------------------------------------------------------

def load_edges(kg_root: str) -> list[dict]:
    """Load all edges from graph/edges.jsonl."""
    edges_path = Path(kg_root) / "graph" / "edges.jsonl"
    if not edges_path.exists():
        return []
    edges = []
    for line in _read_text(edges_path).strip().split("\n"):
        line = line.strip()
        if line:
            try:
                edges.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed edge line: %s", line[:80])
    return edges


def add_edge(kg_root: str, from_id: str, to_id: str, edge_type: str,
             evidence: str = "", *, dedup: bool = True) -> None:
    """Add a typed edge to graph/edges.jsonl."""
    if edge_type not in _EDGE_TYPES:
        raise ValueError(f"Unknown edge type '{edge_type}'. Valid: {sorted(_EDGE_TYPES)}")

    root = Path(kg_root)
    edges_path = root / "graph" / "edges.jsonl"
    edges_path.parent.mkdir(parents=True, exist_ok=True)

    if dedup:
        if any(
            edge.get("from") == from_id
            and edge.get("to") == to_id
            and edge.get("type") == edge_type
            for edge in load_edges(kg_root)
        ):
            logger.info("Edge already exists: %s --%s--> %s", from_id, edge_type, to_id)
            return

    edge = {
        "from": from_id,
        "to": to_id,
        "type": edge_type,
        "evidence": evidence,
        "added": _now_iso(),
    }

    with open(edges_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    logger.info("Edge added: %s --%s--> %s", from_id, edge_type, to_id)


# ---------------------------------------------------------------------------
# Idea operations
# ---------------------------------------------------------------------------

def _render_idea_page(
    slug: str, title: str, description: str,
    stage: str, outcome: str,
    strategy_tags: list[str], optimization_type: str,
    difficulty: int, mode: str, workflow_ref: dict | None,
    parent_idea: str | None, parent_experiment: str | None,
    generation_lens: str, generation_round: int,
    target_gaps: list[str], tags: list[str],
    thesis: str, strategy_rationale: str,
) -> str:
    """Render an ideas/<slug>.md page with YAML frontmatter."""
    lines = ["---"]
    lines.append("type: idea")
    lines.append(f"node_id: idea:{slug}")
    lines.append(f"title: {_yaml_quote(title)}")
    lines.append(f"stage: {stage}")
    lines.append(f"outcome: {outcome}")
    lines.append(f"strategy_tags: [{', '.join(strategy_tags)}]")
    lines.append(f"optimization_type: {optimization_type}")
    lines.append(f"difficulty: {difficulty}")
    lines.append(f"mode: {mode}")
    if workflow_ref:
        lines.append(f"workflow_ref: {json.dumps(workflow_ref, ensure_ascii=False)}")
    else:
        lines.append("workflow_ref: null")
    lines.append(f"parent_idea: {parent_idea or 'null'}")
    lines.append(f"parent_experiment: {parent_experiment or 'null'}")
    lines.append(f"generation_lens: {generation_lens}")
    lines.append(f"generation_round: {generation_round}")
    lines.append(f"target_gaps: [{', '.join(target_gaps)}]")
    lines.append(f"tags: [{', '.join(_yaml_quote(t) for t in tags)}]")
    lines.append(f"added: {_now_iso()}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**stage:** `{stage}`  ·  **outcome:** `{outcome}`  ·  "
                 f"**difficulty:** {difficulty}  ·  **lens:** {generation_lens}")
    if description.strip():
        lines.append("")
        lines.append(description.strip())
    lines.append("")
    lines.append("## Thesis")
    lines.append(thesis.strip() if thesis.strip()
                 else "_描述这个优化假设的核心思路和预期效果._")
    lines.append("")
    lines.append("## Strategy Rationale")
    lines.append(strategy_rationale.strip() if strategy_rationale.strip()
                 else "_列出每个策略 ID 及其在此 idea 中的作用._")
    lines.append("")
    lines.append("## Connections")
    lines.append("_Edges 记录在 graph/edges.jsonl; 此处为可读摘要._")
    lines.append("")
    return "\n".join(lines) + "\n"


def upsert_idea(
    kg_root: str, slug: str, title: str, *,
    strategy_tags: list[str] | None = None,
    optimization_type: str = "algorithm",
    difficulty: int = 3,
    mode: str = "strategy_guided",
    workflow_ref: dict | None = None,
    parent_idea: str | None = None,
    parent_experiment: str | None = None,
    generation_lens: str = "profiling_driven",
    generation_round: int = 0,
    target_gaps: list[str] | None = None,
    tags: list[str] | None = None,
    description: str = "",
    thesis: str = "",
    strategy_rationale: str = "",
    stage: str = "proposed",
    outcome: str = "pending",
    update_on_exist: bool = False,
) -> Path:
    """Create (or update) an ideas/<slug>.md entity.

    Returns the path to the created/updated file.
    """
    root = Path(kg_root)
    if not (root / "ideas").exists():
        raise RuntimeError(f"{root} is not an initialized KG (ideas/ missing)")

    if stage not in _IDEA_STAGES:
        raise ValueError(f"Invalid idea stage '{stage}'. Valid: {sorted(_IDEA_STAGES)}")
    if outcome not in _IDEA_OUTCOMES:
        raise ValueError(f"Invalid idea outcome '{outcome}'. Valid: {sorted(_IDEA_OUTCOMES)}")
    if mode not in _IDEA_MODES:
        raise ValueError(f"Invalid idea mode '{mode}'. Valid: {sorted(_IDEA_MODES)}")
    if generation_lens not in _GENERATION_LENSES:
        raise ValueError(f"Invalid lens '{generation_lens}'. Valid: {sorted(_GENERATION_LENSES)}")
    if optimization_type not in _OPTIMIZATION_TYPES:
        raise ValueError(f"Invalid opt type '{optimization_type}'. "
                         f"Valid: {sorted(_OPTIMIZATION_TYPES)}")
    if not 1 <= difficulty <= 5:
        raise ValueError(f"difficulty must be 1-5, got {difficulty}")
    if mode == "workflow_guided" and not workflow_ref:
        raise ValueError("workflow_guided mode requires workflow_ref")

    strategy_tags = strategy_tags or []
    target_gaps = target_gaps or []
    tags = tags or []
    slug = _slugify(slug or title)

    page_path = root / "ideas" / f"{slug}.md"
    if page_path.exists() and not update_on_exist:
        logger.info("Idea exists: %s (slug dedup) — skipping.", slug)
        return page_path
    was_update = page_path.exists()

    rendered = _render_idea_page(
        slug, title, description, stage, outcome,
        strategy_tags, optimization_type, difficulty, mode, workflow_ref,
        parent_idea, parent_experiment, generation_lens, generation_round,
        target_gaps, tags, thesis, strategy_rationale,
    )
    page_path.write_text(rendered, encoding="utf-8")

    # Wire edges
    if parent_idea:
        add_edge(kg_root, f"idea:{slug}", parent_idea, "inspired_by",
                 evidence=f"idea {slug} inspired by {parent_idea}")
    if parent_experiment:
        add_edge(kg_root, f"idea:{slug}", parent_experiment, "derived_from",
                 evidence=f"idea {slug} derived from experiment result")
    for gap in target_gaps:
        gap_id = gap if gap.startswith("gap:") else f"gap:{gap}"
        add_edge(kg_root, f"idea:{slug}", gap_id, "addresses_gap",
                 evidence=f"idea {slug} targets gap {gap}")

    action = "updated" if was_update else "added"
    append_log(kg_root, f"upsert_idea: {action} idea:{slug} "
               f"[stage={stage} outcome={outcome} lens={generation_lens}]")
    logger.info("Idea %s: %s [stage=%s outcome=%s]", action, page_path, stage, outcome)
    return page_path


def load_ideas(kg_root: str) -> dict[str, dict]:
    """Load all ideas from ideas/*.md, keyed by slug."""
    root = Path(kg_root)
    ideas_dir = root / "ideas"
    if not ideas_dir.exists():
        return {}
    ideas = {}
    for md_file in sorted(ideas_dir.glob("*.md")):
        fm = _parse_frontmatter(_read_text(md_file))
        if fm.get("type") == "idea":
            slug = md_file.stem
            fm["_path"] = str(md_file)
            ideas[slug] = fm
    return ideas


def get_banlist(kg_root: str) -> list[dict]:
    """Return ideas that should be banned from re-generation.

    Only includes ideas whose failure was due to strategy_infeasible
    (the strategy itself doesn't work for this operator). impl_error
    failures are NOT banned — the strategy might work with a correct
    implementation, so they go into a repair queue instead.
    """
    ideas = load_ideas(kg_root)
    experiments = load_experiments(kg_root)

    banned = []
    for idea in ideas.values():
        if idea.get("outcome") != "negative":
            continue
        # Check if the failure was strategy_infeasible (not impl_error)
        idea_nid = idea.get("node_id", "")
        is_infeasible = False
        for exp in experiments.values():
            if exp.get("idea_id") == idea_nid:
                if exp.get("failure_type") == "strategy_infeasible":
                    is_infeasible = True
                    break
        if is_infeasible:
            banned.append(idea)

    return banned


def update_idea_outcome(kg_root: str, slug: str, outcome: str,
                        stage: str | None = None) -> None:
    """Update an idea's outcome (and optionally stage) in place."""
    if outcome not in _IDEA_OUTCOMES:
        raise ValueError(f"Invalid outcome '{outcome}'")
    if stage and stage not in _IDEA_STAGES:
        raise ValueError(f"Invalid stage '{stage}'")

    root = Path(kg_root)
    page_path = root / "ideas" / f"{slug}.md"
    if not page_path.exists():
        raise FileNotFoundError(f"Idea not found: {slug}")

    text = page_path.read_text(encoding="utf-8")
    text = re.sub(r"^outcome:\s*\S+", f"outcome: {outcome}", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*outcome:\*\*\s*`\w+`",
                  f"**outcome:** `{outcome}`", text)
    if stage:
        text = re.sub(r"^stage:\s*\S+", f"stage: {stage}", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*stage:\*\*\s*`\w+`",
                      f"**stage:** `{stage}`", text)
    page_path.write_text(text, encoding="utf-8")
    append_log(kg_root, f"update_idea_outcome: idea:{slug} -> outcome={outcome}"
               + (f" stage={stage}" if stage else ""))


# ---------------------------------------------------------------------------
# Experiment operations
# ---------------------------------------------------------------------------

def _render_experiment_page(
    slug: str, title: str, idea_id: str,
    verdict: str, confidence: str, speedup: float | None,
    solution_ref: str, round_num: int, parallel_index: int,
    profiling_insight: dict | None,
    profiling_evidence: dict | None,
    facts: dict | None,
    diagnosis: dict | None,
    failure_type: str | None, failure_reason: str | None,
    retry_count: int,
    tags: list[str],
) -> str:
    """Render an experiments/<slug>.md page."""
    lines = ["---"]
    lines.append("type: experiment")
    lines.append(f"node_id: exp:{slug}")
    lines.append(f"title: {_yaml_quote(title)}")
    lines.append(f"idea_id: {_yaml_quote(idea_id)}")
    lines.append(f"verdict: {verdict}")
    lines.append(f"confidence: {confidence}")
    lines.append(f"speedup: {speedup if speedup is not None else 'null'}")
    lines.append(f"solution_ref: {_yaml_quote(solution_ref)}")
    lines.append(f"round: {round_num}")
    lines.append(f"parallel_index: {parallel_index}")
    lines.append(f"failure_type: {failure_type or 'null'}")
    lines.append(f"failure_reason: {_yaml_quote(failure_reason or '')}")
    lines.append(f"retry_count: {retry_count}")
    lines.append(f"tags: [{', '.join(_yaml_quote(t) for t in tags)}]")
    lines.append(f"added: {_now_iso()}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**verdict:** `{verdict}`  ·  **confidence:** `{confidence}`"
                 f"  ·  **speedup:** {speedup}")
    lines.append("")
    lines.append(f"**idea:** `{idea_id}`  ·  **solution:** `{solution_ref}`")
    lines.append("")

    if profiling_insight:
        lines.append("## Profiling Insight")
        lines.append(f"```json\n{json.dumps(profiling_insight, ensure_ascii=False, indent=2)}\n```")
        lines.append("")

    if profiling_evidence:
        lines.append("## Profiling Evidence")
        lines.append(f"```json\n{json.dumps(profiling_evidence, ensure_ascii=False, indent=2)}\n```")
        lines.append("")

    if facts:
        lines.append("## Facts")
        lines.append(f"```json\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n```")
        lines.append("")

    if diagnosis:
        lines.append("## Diagnosis")
        lines.append(f"```json\n{json.dumps(diagnosis, ensure_ascii=False, indent=2)}\n```")
        lines.append("")

    if failure_type:
        lines.append("## Failure Analysis")
        lines.append(f"- **type:** `{failure_type}`")
        lines.append(f"- **reason:** {failure_reason or '_undiagnosed_'}")
        lines.append(f"- **retry_count:** {retry_count}")
        lines.append("")

    lines.append("## Connections")
    lines.append("_Edges 记录在 graph/edges.jsonl; 此处为可读摘要._")
    lines.append("")
    return "\n".join(lines) + "\n"


def add_experiment(
    kg_root: str, slug: str, title: str, *,
    idea_id: str,
    verdict: str,
    confidence: str = "medium",
    speedup: float | None = None,
    solution_ref: str = "",
    round_num: int = 0,
    parallel_index: int = 0,
    profiling_insight: dict | None = None,
    profiling_evidence: dict | None = None,
    facts: dict | None = None,
    diagnosis: dict | None = None,
    failure_type: str | None = None,
    failure_reason: str | None = None,
    retry_count: int = 0,
    tags: list[str] | None = None,
    update_on_exist: bool = False,
) -> Path:
    """Create (or update) an experiments/<slug>.md entity."""
    root = Path(kg_root)
    if not (root / "experiments").exists():
        raise RuntimeError(f"{root} is not an initialized KG (experiments/ missing)")

    if verdict not in _EXPERIMENT_VERDICTS:
        raise ValueError(f"Invalid verdict '{verdict}'. Valid: {sorted(_EXPERIMENT_VERDICTS)}")
    if confidence not in _EXPERIMENT_CONFIDENCE:
        raise ValueError(f"Invalid confidence '{confidence}'")
    if failure_type and failure_type not in _FAILURE_TYPES:
        raise ValueError(f"Invalid failure_type '{failure_type}'")

    tags = tags or []
    slug = _slugify(slug or title)

    page_path = root / "experiments" / f"{slug}.md"
    if page_path.exists() and not update_on_exist:
        logger.info("Experiment exists: %s — skipping.", slug)
        return page_path
    was_update = page_path.exists()

    # Write sidecar JSON for structured data that doesn't fit in frontmatter
    # (profiling_insight, profiling_evidence, facts, diagnosis).
    # This ensures load_experiments() can round-trip all structured fields.
    sidecar_path = root / "experiments" / f"{slug}.json"
    old_sidecar: dict = {}
    if sidecar_path.exists():
        try:
            loaded_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(loaded_sidecar, dict):
                old_sidecar = loaded_sidecar
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load existing sidecar: %s", sidecar_path)

    # An update may omit structured evidence.  In that case retain the old
    # value; an explicitly supplied value still replaces it.
    if profiling_insight is None:
        profiling_insight = old_sidecar.get("profiling_insight")
    if profiling_evidence is None:
        profiling_evidence = old_sidecar.get("profiling_evidence")
    if facts is None:
        facts = old_sidecar.get("facts")
    if diagnosis is None:
        diagnosis = old_sidecar.get("diagnosis")

    # Re-render after merging the old values so the markdown page and sidecar
    # remain consistent on update.
    rendered = _render_experiment_page(
        slug, title, idea_id, verdict, confidence, speedup,
        solution_ref, round_num, parallel_index,
        profiling_insight, profiling_evidence, facts, diagnosis,
        failure_type, failure_reason, retry_count, tags,
    )
    page_path.write_text(rendered, encoding="utf-8")

    sidecar: dict = {}
    if profiling_insight is not None:
        sidecar["profiling_insight"] = profiling_insight
    if profiling_evidence is not None:
        sidecar["profiling_evidence"] = profiling_evidence
    if facts is not None:
        sidecar["facts"] = facts
    if diagnosis is not None:
        sidecar["diagnosis"] = diagnosis
    if sidecar:
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif sidecar_path.exists():
        # Clean up stale sidecar if no structured data
        sidecar_path.unlink(missing_ok=True)

    # Wire edges: idea --tested_by--> experiment.
    add_edge(kg_root, idea_id, f"exp:{slug}", "tested_by",
             evidence=f"experiment {slug} tests {idea_id}")

    action = "updated" if was_update else "added"
    append_log(kg_root, f"add_experiment: {action} exp:{slug} "
               f"[verdict={verdict} speedup={speedup}]")
    logger.info("Experiment %s: %s [verdict=%s speedup=%s]",
                action, page_path, verdict, speedup)
    return page_path


def load_experiments(kg_root: str) -> dict[str, dict]:
    """Load all experiments from experiments/*.md + sidecar .json, keyed by slug."""
    root = Path(kg_root)
    exp_dir = root / "experiments"
    if not exp_dir.exists():
        return {}
    experiments = {}
    for md_file in sorted(exp_dir.glob("*.md")):
        fm = _parse_frontmatter(_read_text(md_file))
        if fm.get("type") == "experiment":
            slug = md_file.stem
            fm["_path"] = str(md_file)
            # Merge sidecar JSON (profiling_insight, profiling_evidence, facts, diagnosis)
            sidecar_path = exp_dir / f"{slug}.json"
            if sidecar_path.exists():
                try:
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    fm.update(sidecar)
                except (json.JSONDecodeError, OSError):
                    logger.warning("Failed to load sidecar: %s", sidecar_path)
            experiments[slug] = fm
    return experiments


# ---------------------------------------------------------------------------
# Claim operations
# ---------------------------------------------------------------------------

def _render_claim_page(
    slug: str, name: str, status: str,
    statement: str, scope: str,
    supporting_experiments: list[str],
    refuting_experiments: list[str],
    strategy_implications: list[str],
    bottleneck_labels: list[str],
    confidence: float,
    polarity: str = "positive",
) -> str:
    """Render a claims/<slug>.md page."""
    lines = ["---"]
    lines.append("type: claim")
    lines.append(f"node_id: claim:{slug}")
    lines.append(f"name: {_yaml_quote(name)}")
    lines.append(f"status: {status}")
    lines.append(f"polarity: {polarity}")
    lines.append(f"supporting_experiments: [{', '.join(supporting_experiments)}]")
    lines.append(f"refuting_experiments: [{', '.join(refuting_experiments)}]")
    lines.append(f"strategy_implications: [{', '.join(strategy_implications)}]")
    lines.append(f"bottleneck_labels: [{', '.join(bottleneck_labels)}]")
    lines.append(f"confidence: {confidence}")
    lines.append(f"added: {_now_iso()}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"**status:** `{status}`  ·  **polarity:** `{polarity}`"
                 f"  ·  **confidence:** {confidence:.2f}")
    lines.append("")
    lines.append("## Statement")
    lines.append(statement.strip() if statement.strip()
                 else "_TODO: 形式化的 claim 描述._")
    lines.append("")
    lines.append("## Scope")
    lines.append(scope.strip() if scope.strip()
                 else "_TODO: 此 claim 的适用范围和限制._")
    lines.append("")
    lines.append("## Evidence Chain")
    if supporting_experiments:
        lines.append("**Supporting experiments:**")
        for exp_id in supporting_experiments:
            lines.append(f"- `{exp_id}`")
    if refuting_experiments:
        lines.append("**Refuting experiments:**")
        for exp_id in refuting_experiments:
            lines.append(f"- `{exp_id}`")
    if not supporting_experiments and not refuting_experiments:
        lines.append("_No experiments linked yet._")
    lines.append("")
    lines.append("## Connections")
    lines.append("_Edges 记录在 graph/edges.jsonl; 此处为可读摘要._")
    lines.append("")
    return "\n".join(lines) + "\n"


def add_claim(
    kg_root: str, slug: str, name: str, *,
    status: str = "drafted",
    polarity: str = "positive",
    statement: str = "",
    scope: str = "",
    supporting_experiments: list[str] | None = None,
    refuting_experiments: list[str] | None = None,
    strategy_implications: list[str] | None = None,
    bottleneck_labels: list[str] | None = None,
    confidence: float = 0.0,
    update_on_exist: bool = False,
) -> Path:
    """Create (or update) a claims/<slug>.md entity."""
    if polarity not in _CLAIM_POLARITIES:
        raise ValueError(
            f"Invalid claim polarity '{polarity}'. "
            f"Valid: {sorted(_CLAIM_POLARITIES)}"
        )

    root = Path(kg_root)
    if not (root / "claims").exists():
        raise RuntimeError(f"{root} is not an initialized KG (claims/ missing)")

    if status not in _CLAIM_STATUSES:
        raise ValueError(f"Invalid claim status '{status}'. Valid: {sorted(_CLAIM_STATUSES)}")

    supporting_experiments = supporting_experiments or []
    refuting_experiments = refuting_experiments or []
    strategy_implications = strategy_implications or []
    bottleneck_labels = bottleneck_labels or []
    slug = _slugify(slug or name)

    page_path = root / "claims" / f"{slug}.md"
    if page_path.exists() and not update_on_exist:
        logger.info("Claim exists: %s — skipping.", slug)
        return page_path
    was_update = page_path.exists()

    rendered = _render_claim_page(
        slug, name, status, statement, scope,
        supporting_experiments, refuting_experiments,
        strategy_implications, bottleneck_labels, confidence,
        polarity=polarity,
    )
    page_path.write_text(rendered, encoding="utf-8")

    # Wire edges: exp --supports/invalidates--> claim
    for exp_id in supporting_experiments:
        add_edge(kg_root, exp_id, f"claim:{slug}", "supports",
                 evidence=f"experiment supports claim {slug}")
    for exp_id in refuting_experiments:
        add_edge(kg_root, exp_id, f"claim:{slug}", "invalidates",
                 evidence=f"experiment invalidates claim {slug}")

    action = "updated" if was_update else "added"
    append_log(kg_root, f"add_claim: {action} claim:{slug} [status={status}]")
    logger.info("Claim %s: %s [status=%s]", action, page_path, status)
    return page_path


def load_claims(kg_root: str) -> dict[str, dict]:
    """Load all claims from claims/*.md, keyed by slug."""
    root = Path(kg_root)
    claims_dir = root / "claims"
    if not claims_dir.exists():
        return {}
    claims = {}
    for md_file in sorted(claims_dir.glob("*.md")):
        fm = _parse_frontmatter(_read_text(md_file))
        if fm.get("type") == "claim":
            slug = md_file.stem
            fm["_path"] = str(md_file)
            claims[slug] = fm
    return claims


# ---------------------------------------------------------------------------
# Knowledge graph synchronization
# ---------------------------------------------------------------------------

def _default_session(kg_root: str) -> dict[str, Any]:
    """Return a schema-shaped placeholder session for a new KG."""
    return {
        "session_id": "",
        "start_time": "",
        "requested_rounds": 0,
        "actual_rounds_completed": 0,
        "evo_dir": str(Path(kg_root)),
        "op_name": "",
    }


def sync_knowledge_graph(
    kg_root: str,
    session: dict | None = None,
    baseline: dict | None = None,
    best_score: float | None = None,
) -> dict[str, Any]:
    """从磁盘实体重新同步 knowledge_graph.json。"""
    root = Path(kg_root)
    kg_json_path = root / "knowledge_graph.json"
    kg_data: dict[str, Any] = {}

    if kg_json_path.exists():
        try:
            loaded = json.loads(kg_json_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                kg_data = loaded
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load existing KG JSON: %s", kg_json_path)

    if session is not None:
        kg_data["session"] = session
    elif not isinstance(kg_data.get("session"), dict):
        kg_data["session"] = _default_session(kg_root)

    if baseline is not None:
        kg_data["baseline_performance"] = baseline
    elif not isinstance(kg_data.get("baseline_performance"), dict):
        kg_data["baseline_performance"] = {"speedup": 1.0, "time_ms": None}

    # Keep schema fields that are not derived from entity files intact while
    # making a newly created/salvaged KG usable by downstream consumers.
    kg_data.setdefault("kernel_summary", "")
    kg_data.setdefault("hw_params", None)
    kg_data.setdefault("baseline_evidence", None)
    kg_data.setdefault("stagnation_count", 0)
    kg_data.setdefault("stagnation_count_vs_base", 0)
    kg_data.setdefault("supervisor_used_count", 0)
    kg_data.setdefault("last_supervisor_round", -2)
    kg_data.setdefault("open_questions", [])
    kg_data.setdefault("discovered_strategies", [])
    kg_data.setdefault("best_score", 1.0)
    if best_score is not None:
        kg_data["best_score"] = best_score

    ideas = load_ideas(kg_root)
    experiments = load_experiments(kg_root)
    claims = load_claims(kg_root)

    idea_registry: dict[str, dict[str, Any]] = {}
    for slug, idea in ideas.items():
        idea_registry[slug] = {
            "stage": idea.get("stage", "proposed"),
            "outcome": idea.get("outcome", "pending"),
            "generation_lens": idea.get("generation_lens", ""),
            "round_created": idea.get("generation_round", 0),
            "round_tested": None,
        }

    rounds_completed = max(
        (int(exp.get("round", 0) or 0) for exp in experiments.values()),
        default=0,
    )

    kg_data["idea_registry"] = idea_registry
    kg_data["counters"] = {
        "total_ideas": len(ideas),
        "active_ideas": sum(
            1 for idea in ideas.values() if idea.get("stage") == "active"
        ),
        "total_experiments": len(experiments),
        "total_claims": len(claims),
        "rounds_completed": rounds_completed,
    }

    root.mkdir(parents=True, exist_ok=True)
    kg_json_path.write_text(
        json.dumps(kg_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    append_log(
        kg_root,
        "sync_knowledge_graph: "
        f"{len(ideas)} ideas, {len(experiments)} experiments, {len(claims)} claims",
    )
    return kg_data


# ---------------------------------------------------------------------------
# Gap Map operations
# ---------------------------------------------------------------------------

def _legacy_gap_closure_reason(status: str, evidence: str) -> str:
    """Recover closure provenance from pre-``closure_reason`` gap maps.

    Older writers appended ``closed by ...`` to Evidence.  Keep that form
    readable.  If no legacy reason exists, leave the field empty so KG
    validation can flag the record instead of inventing provenance.
    """
    if status == "open":
        return ""
    evidence = str(evidence or "").strip()
    match = re.search(r"(?:^|;\s*)(closed by\b.+)$", evidence, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if evidence:
        return f"legacy closure evidence: {evidence}"
    return ""


def parse_gap_map(kg_root: str) -> list[dict]:
    """Parse gap_map.md into structured gap entries."""
    root = Path(kg_root)
    gap_path = root / "gap_map.md"
    if not gap_path.exists():
        return []

    text = _read_text(gap_path)
    gaps = []
    current_gap: dict | None = None

    for line in text.split("\n"):
        # Gap header: ## G1: description  or  ### G1: description
        m = re.match(r"^#{2,3}\s+(G\d+):\s*(.+)$", line)
        if m:
            if current_gap:
                gaps.append(current_gap)
            current_gap = {
                "id": m.group(1),
                "description": m.group(2).strip(),
                "status": "open",
                "key": "",
                "kind": "",
                "subject": "",
                "evidence": "",
                "closure_reason": "",
                "suggested_lenses": [],
                "priority": 1.0,
            }
            continue

        if current_gap:
            sl = line.strip()
            if sl.startswith("- Status:") or sl.startswith("- **Status**:"):
                status = sl.split(":", 1)[1].strip().lower()
                current_gap["status"] = status
            elif sl.startswith("- Key:") or sl.startswith("- **Key**:"):
                current_gap["key"] = sl.split(":", 1)[1].strip()
            elif sl.startswith("- Kind:") or sl.startswith("- **Kind**:"):
                current_gap["kind"] = sl.split(":", 1)[1].strip()
            elif sl.startswith("- Subject:") or sl.startswith("- **Subject**:"):
                current_gap["subject"] = sl.split(":", 1)[1].strip()
            elif sl.startswith("- Evidence:") or sl.startswith("- **Evidence**:"):
                current_gap["evidence"] = sl.split(":", 1)[1].strip()
            elif (sl.startswith("- Closure reason:")
                  or sl.startswith("- **Closure reason**:")):
                current_gap["closure_reason"] = sl.split(":", 1)[1].strip()
            elif sl.startswith("- Suggested lenses:") or sl.startswith("- **Suggested lenses**:"):
                lenses_str = sl.split(":", 1)[1].strip()
                current_gap["suggested_lenses"] = [
                    l.strip() for l in lenses_str.split(",") if l.strip()
                ]
            elif sl.startswith("- Priority:") or sl.startswith("- **Priority**:"):
                try:
                    current_gap["priority"] = float(sl.split(":", 1)[1].strip())
                except ValueError:
                    pass

    if current_gap:
        gaps.append(current_gap)

    for gap in gaps:
        if gap.get("status", "open") != "open" and not str(
            gap.get("closure_reason") or ""
        ).strip():
            gap["closure_reason"] = _legacy_gap_closure_reason(
                str(gap.get("status", "closed")),
                str(gap.get("evidence") or ""),
            )

    return gaps


def write_gap_map(kg_root: str, gaps: list[dict]) -> None:
    """Write structured gap entries back to gap_map.md."""
    root = Path(kg_root)
    gap_path = root / "gap_map.md"

    lines = ["# Gap Map\n"]

    # Split into open and closed
    open_gaps = [g for g in gaps if g.get("status", "open") == "open"]
    closed_gaps = [g for g in gaps if g.get("status", "open") != "open"]

    if open_gaps:
        lines.append("## Active Gaps\n")
        for gap in open_gaps:
            lines.append(f"### {gap['id']}: {gap['description']}")
            lines.append(f"- **Status**: {gap.get('status', 'open')}")
            if gap.get("key"):
                lines.append(f"- **Key**: {gap['key']}")
            if gap.get("kind"):
                lines.append(f"- **Kind**: {gap['kind']}")
            if gap.get("subject"):
                lines.append(f"- **Subject**: {gap['subject']}")
            if gap.get("evidence"):
                lines.append(f"- **Evidence**: {gap['evidence']}")
            if gap.get("closure_reason"):
                lines.append(f"- **Closure reason**: {gap['closure_reason']}")
            if gap.get("suggested_lenses"):
                lines.append(f"- **Suggested lenses**: {', '.join(gap['suggested_lenses'])}")
            if gap.get("priority", 1.0) != 1.0:
                lines.append(f"- **Priority**: {gap['priority']}")
            lines.append("")

    if closed_gaps:
        lines.append("## Closed Gaps\n")
        for gap in closed_gaps:
            lines.append(f"### {gap['id']}: {gap['description']}")
            lines.append(f"- **Status**: {gap.get('status', 'closed')}")
            if gap.get("key"):
                lines.append(f"- **Key**: {gap['key']}")
            if gap.get("kind"):
                lines.append(f"- **Kind**: {gap['kind']}")
            if gap.get("subject"):
                lines.append(f"- **Subject**: {gap['subject']}")
            if gap.get("evidence"):
                lines.append(f"- **Evidence**: {gap['evidence']}")
            if gap.get("closure_reason"):
                lines.append(f"- **Closure reason**: {gap['closure_reason']}")
            lines.append("")

    gap_path.write_text("\n".join(lines), encoding="utf-8")
    append_log(kg_root, f"write_gap_map: {len(open_gaps)} open, {len(closed_gaps)} closed")


def next_gap_id(kg_root: str) -> str:
    """Return the next available gap ID (G1, G2, ...)."""
    gaps = parse_gap_map(kg_root)
    if not gaps:
        return "G1"
    max_num = max(int(g["id"][1:]) for g in gaps if re.match(r"G\d+", g["id"]))
    return f"G{max_num + 1}"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_stats(kg_root: str) -> dict[str, Any]:
    """Return summary statistics for the knowledge graph."""
    ideas = load_ideas(kg_root)
    experiments = load_experiments(kg_root)
    claims = load_claims(kg_root)
    gaps = parse_gap_map(kg_root)
    edges = load_edges(kg_root)

    idea_by_stage = {}
    idea_by_outcome = {}
    for idea in ideas.values():
        s = idea.get("stage", "unknown")
        idea_by_stage[s] = idea_by_stage.get(s, 0) + 1
        o = idea.get("outcome", "unknown")
        idea_by_outcome[o] = idea_by_outcome.get(o, 0) + 1

    exp_by_verdict = {}
    for exp in experiments.values():
        v = exp.get("verdict", "unknown")
        exp_by_verdict[v] = exp_by_verdict.get(v, 0) + 1

    return {
        "total_ideas": len(ideas),
        "ideas_by_stage": idea_by_stage,
        "ideas_by_outcome": idea_by_outcome,
        "total_experiments": len(experiments),
        "experiments_by_verdict": exp_by_verdict,
        "total_claims": len(claims),
        "total_gaps": len(gaps),
        "open_gaps": sum(1 for g in gaps if g.get("status", "open") == "open"),
        "total_edges": len(edges),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    init_kg(args.kg_root)


def cmd_upsert_idea(args: argparse.Namespace) -> None:
    strategy_tags = [s.strip() for s in args.strategy_tags.split(",") if s.strip()] \
        if args.strategy_tags else []
    target_gaps = [g.strip() for g in args.target_gaps.split(",") if g.strip()] \
        if args.target_gaps else []
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else []

    upsert_idea(
        args.kg_root, args.slug, args.title,
        strategy_tags=strategy_tags,
        optimization_type=args.optimization_type,
        difficulty=args.difficulty,
        mode=args.mode,
        generation_lens=args.generation_lens,
        generation_round=args.generation_round,
        target_gaps=target_gaps,
        tags=tags,
        description=args.description or "",
        thesis=args.thesis or "",
        strategy_rationale=args.strategy_rationale or "",
        stage=args.stage,
        outcome=args.outcome,
        parent_idea=args.parent_idea,
        parent_experiment=args.parent_experiment,
        update_on_exist=args.update_on_exist,
    )


def cmd_add_experiment(args: argparse.Namespace) -> None:
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] \
        if args.tags else []

    def _parse_json_arg(val: str | None) -> dict | None:
        if not val:
            return None
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            # Try as file path
            try:
                return json.loads(Path(val).read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning("Could not parse JSON arg: %s", val[:80] if val else "")
                return None

    add_experiment(
        args.kg_root, args.slug, args.title,
        idea_id=args.idea_id,
        verdict=args.verdict,
        confidence=args.confidence,
        speedup=args.speedup,
        solution_ref=args.solution_ref or "",
        round_num=args.round,
        parallel_index=args.parallel_index,
        profiling_insight=_parse_json_arg(args.profiling_insight),
        profiling_evidence=_parse_json_arg(args.profiling_evidence),
        facts=_parse_json_arg(args.facts),
        diagnosis=_parse_json_arg(args.diagnosis),
        failure_type=args.failure_type,
        failure_reason=args.failure_reason,
        retry_count=args.retry_count,
        tags=tags,
        update_on_exist=args.update_on_exist,
    )


def cmd_add_claim(args: argparse.Namespace) -> None:
    supporting = [s.strip() for s in args.supporting_experiments.split(",") if s.strip()] \
        if args.supporting_experiments else []
    refuting = [s.strip() for s in args.refuting_experiments.split(",") if s.strip()] \
        if args.refuting_experiments else []
    strat_impl = [s.strip() for s in args.strategy_implications.split(",") if s.strip()] \
        if args.strategy_implications else []
    bn_labels = [s.strip() for s in args.bottleneck_labels.split(",") if s.strip()] \
        if args.bottleneck_labels else []

    add_claim(
        args.kg_root, args.slug, args.name,
        status=args.status,
        polarity=args.polarity,
        statement=args.statement or "",
        scope=args.scope or "",
        supporting_experiments=supporting,
        refuting_experiments=refuting,
        strategy_implications=strat_impl,
        bottleneck_labels=bn_labels,
        confidence=args.confidence,
        update_on_exist=args.update_on_exist,
    )


def cmd_add_edge(args: argparse.Namespace) -> None:
    add_edge(args.kg_root, getattr(args, "from"), args.to, args.type,
             evidence=args.evidence or "")


def cmd_stats(args: argparse.Namespace) -> None:
    stats = get_stats(args.kg_root)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def cmd_update_idea_outcome(args: argparse.Namespace) -> None:
    update_idea_outcome(args.kg_root, args.slug, args.outcome,
                        stage=args.stage)


def cmd_sync(args: argparse.Namespace) -> None:
    def _parse_json_arg(value: str | None) -> dict | None:
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(Path(value).read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                raise ValueError(f"Could not parse JSON argument: {value[:80]}")
        if not isinstance(parsed, dict):
            raise ValueError("sync JSON arguments must be objects")
        return parsed

    kg_data = sync_knowledge_graph(
        args.kg_root,
        session=_parse_json_arg(args.session),
        baseline=_parse_json_arg(args.baseline),
        best_score=args.best_score,
    )
    print(json.dumps({
        "synced": True,
        "counters": kg_data.get("counters", {}),
    }, indent=2, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knowledge-Graph Wiki CLI for evolution pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize KG directory structure")
    p_init.add_argument("kg_root")
    p_init.set_defaults(func=cmd_init)

    # upsert_idea
    p_idea = subparsers.add_parser("upsert_idea", help="Create/update an idea")
    p_idea.add_argument("kg_root")
    p_idea.add_argument("--slug", required=True)
    p_idea.add_argument("--title", required=True)
    p_idea.add_argument("--strategy-tags", default="")
    p_idea.add_argument("--optimization-type", default="algorithm")
    p_idea.add_argument("--difficulty", type=int, default=3)
    p_idea.add_argument("--mode", default="strategy_guided")
    p_idea.add_argument("--generation-lens", default="profiling_driven")
    p_idea.add_argument("--generation-round", type=int, default=0)
    p_idea.add_argument("--target-gaps", default="")
    p_idea.add_argument("--tags", default="")
    p_idea.add_argument("--description", default="")
    p_idea.add_argument("--thesis", default="")
    p_idea.add_argument("--strategy-rationale", default="")
    p_idea.add_argument("--stage", default="proposed")
    p_idea.add_argument("--outcome", default="pending")
    p_idea.add_argument("--parent-idea", default=None)
    p_idea.add_argument("--parent-experiment", default=None)
    p_idea.add_argument("--update-on-exist", action="store_true")
    p_idea.set_defaults(func=cmd_upsert_idea)

    # add_experiment
    p_exp = subparsers.add_parser("add_experiment", help="Create/update an experiment")
    p_exp.add_argument("kg_root")
    p_exp.add_argument("--slug", required=True)
    p_exp.add_argument("--title", required=True)
    p_exp.add_argument("--idea-id", required=True)
    p_exp.add_argument("--verdict", required=True)
    p_exp.add_argument("--confidence", default="medium")
    p_exp.add_argument("--speedup", type=float, default=None)
    p_exp.add_argument("--solution-ref", default="")
    p_exp.add_argument("--round", type=int, default=0)
    p_exp.add_argument("--parallel-index", type=int, default=0)
    p_exp.add_argument("--profiling-insight", default=None,
                       help="JSON string or file path for profiling_insight")
    p_exp.add_argument("--profiling-evidence", default=None,
                       help="JSON string or file path for profiling_evidence")
    p_exp.add_argument("--facts", default=None,
                       help="JSON string or file path for extracted facts")
    p_exp.add_argument("--diagnosis", default=None,
                       help="JSON string or file path for diagnosis")
    p_exp.add_argument("--failure-type", default=None)
    p_exp.add_argument("--failure-reason", default=None)
    p_exp.add_argument("--retry-count", type=int, default=0)
    p_exp.add_argument("--tags", default="")
    p_exp.add_argument("--update-on-exist", action="store_true")
    p_exp.set_defaults(func=cmd_add_experiment)

    # add_claim
    p_claim = subparsers.add_parser("add_claim", help="Create/update a claim")
    p_claim.add_argument("kg_root")
    p_claim.add_argument("--slug", required=True)
    p_claim.add_argument("--name", required=True)
    p_claim.add_argument("--status", default="drafted")
    p_claim.add_argument("--polarity", default="positive",
                         help="Claim polarity: positive or negative")
    p_claim.add_argument("--statement", default="")
    p_claim.add_argument("--scope", default="")
    p_claim.add_argument("--supporting-experiments", default="")
    p_claim.add_argument("--refuting-experiments", default="")
    p_claim.add_argument("--strategy-implications", default="")
    p_claim.add_argument("--bottleneck-labels", default="")
    p_claim.add_argument("--confidence", type=float, default=0.0)
    p_claim.add_argument("--update-on-exist", action="store_true")
    p_claim.set_defaults(func=cmd_add_claim)

    # add_edge
    p_edge = subparsers.add_parser("add_edge", help="Add a typed edge")
    p_edge.add_argument("kg_root")
    p_edge.add_argument("--from", required=True)
    p_edge.add_argument("--to", required=True)
    p_edge.add_argument("--type", required=True)
    p_edge.add_argument("--evidence", default="")
    p_edge.set_defaults(func=cmd_add_edge)

    # update_idea_outcome
    p_uio = subparsers.add_parser("update_idea_outcome",
                                   help="Update idea outcome/stage in place")
    p_uio.add_argument("kg_root")
    p_uio.add_argument("--slug", required=True)
    p_uio.add_argument("--outcome", required=True)
    p_uio.add_argument("--stage", default=None)
    p_uio.set_defaults(func=cmd_update_idea_outcome)

    # sync
    p_sync = subparsers.add_parser(
        "sync", help="Rebuild KG registry and counters from entity files"
    )
    p_sync.add_argument("kg_root")
    p_sync.add_argument("--session", default=None,
                        help="Session JSON string or file path")
    p_sync.add_argument("--baseline", default=None,
                        help="Baseline performance JSON string or file path")
    p_sync.add_argument("--best-score", type=float, default=None)
    p_sync.set_defaults(func=cmd_sync)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show KG statistics")
    p_stats.add_argument("kg_root")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()
    result = args.func(args)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()
