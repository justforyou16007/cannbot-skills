"""Regression tests for the KG/wiki orchestration boundary.

These tests intentionally exercise the CLI, because the bugs fixed in this
area were mostly protocol/round-trip failures rather than isolated helpers.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
KG_OPS = SCRIPTS / "kg_ops.py"


def _run(tmp_path: Path, *args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(KG_OPS), *args],
        cwd=ROOT.parents[2],
        text=True,
        capture_output=True,
    )
    if proc.returncode:
        raise AssertionError(
            f"kg_ops failed ({proc.returncode}):\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - useful failure context
        raise AssertionError(f"kg_ops did not return JSON: {proc.stdout!r}\n{proc.stderr}") from exc


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _init(tmp_path: Path, parallel: int = 2, kernel_traits: dict | None = None) -> dict:
    baseline_eval = json.dumps({
        "baseline_time_us": 100.0,
        "avg_speedup": 2.0,
        "evolved": {
            "pipeline": {
                "aiv_mte2_ratio": 0.80,
                "aiv_vec_ratio": 0.20,
                "aiv_scalar_ratio": 0.01,
            },
        },
    })
    args = [
        "init",
        str(tmp_path),
        "--op-name",
        "demo",
        "--requested-rounds",
        "2",
        "--parallel-num",
        str(parallel),
        "--baseline-eval",
        baseline_eval,
    ]
    if kernel_traits is not None:
        args.extend(["--kernel-traits", json.dumps(kernel_traits, ensure_ascii=False)])
    return _run(
        tmp_path,
        *args,
    )


def _kg(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "knowledge_graph.json").read_text(encoding="utf-8"))


def test_mode_predicate_and_flat_baseline_are_stable(tmp_path: Path):
    kg_ops = _load_module("kg_ops_flow_test", KG_OPS)
    assert kg_ops.detect_mode(str(tmp_path)) == "kg"

    result = _init(tmp_path)
    data = _kg(tmp_path)
    assert data["baseline_performance"]["speedup"] == pytest.approx(2.0)
    assert data["baseline_performance"]["time_us"] == pytest.approx(100.0)
    assert data["baseline_evidence"]["bottleneck_type"] == "mte2_stall"
    idea_count = len(list((tmp_path / "ideas").glob("*.md")))
    assert result["resumed"] is False

    resumed = _init(tmp_path)
    assert resumed["resumed"] is True
    assert len(list((tmp_path / "ideas").glob("*.md"))) == idea_count
    assert _kg(tmp_path)["best_score"] == pytest.approx(2.0)

    (tmp_path / "world_model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting evolution stores"):
        kg_ops.detect_mode(str(tmp_path))


def test_init_persists_workflow_guided_branch_from_router_result(tmp_path: Path):
    _init(
        tmp_path,
        parallel=1,
        kernel_traits={
            "workflow_ref": {
                "family": "conv",
                "workflow_file": "scenario-routing/convolution.md",
                "branch": "C-6",
                "branch_condition": "conv3d 三维滑窗",
                "traits": [],
                "stage_order": ["_common-prerequisites.md C1-C5", "C-6 branch", "GEMM 收尾"],
                "strategy_tags": ["P128", "P129"],
            },
        },
    )
    wiki = _load_module("kg_wiki_workflow_test", SCRIPTS / "kg_wiki.py")
    ideas = wiki.load_ideas(str(tmp_path))
    guided = [idea for idea in ideas.values() if idea.get("mode") == "workflow_guided"]
    assert len(guided) == 1
    assert guided[0]["workflow_ref"]["branch"] == "C-6"
    assert guided[0]["strategy_tags"] == ["P128", "P129"]


def test_workflow_ref_file_alias_is_canonicalized(tmp_path: Path):
    wiki = _load_module("kg_wiki_workflow_alias_test", SCRIPTS / "kg_wiki.py")
    wiki.init_kg(str(tmp_path))
    wiki.upsert_idea(
        str(tmp_path),
        "wf-alias",
        "Workflow alias",
        mode="workflow_guided",
        workflow_ref={"file": "family.md", "stage_order": [" stage_1 ", 2]},
        generation_lens="workflow_guided",
    )
    idea = wiki.load_ideas(str(tmp_path))["wf-alias"]
    assert idea["workflow_ref"]["workflow_file"] == "family.md"
    assert idea["workflow_ref"]["stage_order"] == ["stage_1", "2"]


def test_workflow_guided_continuation_keeps_workflow_provenance(tmp_path: Path):
    _init(
        tmp_path,
        parallel=1,
        kernel_traits={
            "workflow_ref": {
                "family": "conv",
                "workflow_file": "convolution.md",
                "branch": "C-1",
                "stage_order": ["stage_1", "stage_2"],
                "strategy_tags": ["P128"],
            },
        },
    )
    wiki = _load_module("kg_wiki_continuation_test", SCRIPTS / "kg_wiki.py")
    generation = _load_module("idea_generation_continuation_test", SCRIPTS / "idea_generation.py")
    guided_slug = next(
        slug for slug, idea in wiki.load_ideas(str(tmp_path)).items()
        if idea.get("mode") == "workflow_guided"
    )
    wiki.add_experiment(
        str(tmp_path),
        "guided-success",
        "Guided success",
        idea_id=f"idea:{guided_slug}",
        verdict="yes",
        speedup=1.2,
        profiling_evidence={"bottleneck_type": "mte2_stall"},
    )
    children = generation.continuation_lens(str(tmp_path), "guided-success", max_ideas=2)
    assert children
    assert all(child["mode"] == "workflow_guided" for child in children)
    assert all(child["workflow_ref"]["workflow_file"] == "convolution.md" for child in children)


def test_materialize_preserves_metadata_and_prepare_infers_parallel_count(tmp_path: Path):
    _init(tmp_path, parallel=1)
    wiki = _load_module("kg_wiki_flow_test", SCRIPTS / "kg_wiki.py")
    parent_slug = next(iter(wiki.load_ideas(str(tmp_path))))
    wiki.upsert_idea(
        str(tmp_path),
        "jury-open",
        "开放探索方案：脚手架",
        strategy_tags=[],
        mode="open_exploration",
        generation_lens="open_exploration",
        target_gaps=["gap:G7"],
        tags=["open"],
        workflow_ref={"workflow_file": "family.md", "stage_order": ["stage_1"]},
        parent_idea=f"idea:{parent_slug}",
        description="保留这段 jury 背景说明。",
        strategy_rationale="保留原始策略依据。",
        thesis="短占位",
    )
    thesis = "通过固定工作流阶段逐项验证数据布局、搬运粒度和尾块边界，并在每个阶段记录可复现实验与跳过理由，确保该探索假设可以被实现、重复、比较和审计，同时明确资源约束、失败回退与后续迭代的判定条件。"
    wiki.materialize_idea(
        str(tmp_path),
        "jury-open",
        title="按阶段重排搬运与计算流水",
        thesis=thesis,
    )
    idea = wiki.load_ideas(str(tmp_path))["jury-open"]
    assert idea["title"] == "按阶段重排搬运与计算流水"
    assert idea["target_gaps"] == ["gap:G7"]
    assert idea["workflow_ref"]["stage_order"] == ["stage_1"]
    page = (tmp_path / "ideas" / "jury-open.md").read_text(encoding="utf-8")
    assert "保留这段 jury 背景说明。" in page
    assert "保留原始策略依据。" in page

    prepared = _run(
        tmp_path,
        "prepare-round",
        str(tmp_path),
        "--round",
        "1",
        "--parallel-map",
        '{"0":"idea:jury-open"}',
    )
    assert prepared["parallel_map"] == {"0": "idea:jury-open"}
    assert json.loads(
        (tmp_path / "round_1" / "parallel_map.json").read_text(encoding="utf-8")
    ) == {"0": "idea:jury-open"}


def test_refine_is_baseline_anchored_and_idempotent(tmp_path: Path):
    _init(tmp_path, parallel=1)
    wiki = _load_module("kg_wiki_refine_test", SCRIPTS / "kg_wiki.py")
    idea = next(iter(wiki.load_ideas(str(tmp_path))))
    result = json.dumps([{
        "parallel_index": 0,
        "idea_id": f"idea:{idea}",
        "verdict": "yes",
        "speedup": 1.5,
        "profiling_insight": {"bottleneck": "mte2_stall"},
        "compilation_success": True,
        "precision_passed": True,
    }])
    first = _run(
        tmp_path,
        "refine",
        str(tmp_path),
        "--round",
        "1",
        "--results",
        result,
        "--parallel-map",
        json.dumps({"0": f"idea:{idea}"}),
    )
    assert first["idempotent"] is False
    assert _kg(tmp_path)["best_score"] == pytest.approx(2.0)
    assert len(list((tmp_path / "experiments").glob("*.md"))) == 1

    second = _run(
        tmp_path,
        "refine",
        str(tmp_path),
        "--round",
        "1",
        "--results",
        result,
        "--parallel-map",
        json.dumps({"0": f"idea:{idea}"}),
    )
    assert second["idempotent"] is True
    assert len(list((tmp_path / "experiments").glob("*.md"))) == 1
    validated = subprocess.run(
        [sys.executable, str(KG_OPS), "validate", str(tmp_path)],
        cwd=ROOT.parents[2],
        text=True,
        capture_output=True,
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr


def test_refine_recovers_terminal_marker_for_missing_parallel(tmp_path: Path):
    _init(tmp_path, parallel=2)
    wiki = _load_module("kg_wiki_marker_test", SCRIPTS / "kg_wiki.py")
    ideas = [
        slug for slug, idea in wiki.load_ideas(str(tmp_path)).items()
        if idea.get("mode") != "open_exploration"
    ][:2]
    assert len(ideas) == 2
    round_dir = tmp_path / "round_1" / "parallel_0"
    round_dir.mkdir(parents=True)
    (round_dir / ".subagent_exit_status").write_text(
        json.dumps({"exit_status": "timeout"}), encoding="utf-8"
    )
    mapping = json.dumps({"0": f"idea:{ideas[0]}", "1": f"idea:{ideas[1]}"})
    result = _run(
        tmp_path,
        "refine",
        str(tmp_path),
        "--round",
        "1",
        "--results-dir",
        str(tmp_path / "round_1"),
        "--parallel-map",
        mapping,
    )
    assert len(result["experiments"]) == 2
    assert all(exp.startswith("exp:") for exp in result["experiments"])
    assert len(list((tmp_path / "experiments").glob("*.md"))) == 2


def test_lingxi_score_scale_and_report_fallback_ignore_failed_nodes():
    kg_ops = _load_module("kg_ops_scale_test", KG_OPS)
    assert kg_ops._normalise_parallel_map({"parallel_01": "idea:x"}) == {"1": "idea:x"}
    with pytest.raises(ValueError, match="duplicate parallel_map index"):
        kg_ops._normalise_parallel_map({"01": "idea:x", "1": "idea:y"})

    contradictory = kg_ops._normalise_result(
        {
            "parallel_index": 0,
            "verdict": "yes",
            "compilation_success": False,
            "precision_passed": True,
            "speedup": 1.1,
            "profiling_insight": {"bottleneck": "mte2_stall"},
        },
        1,
        {"0": "idea:x"},
    )
    assert contradictory["verdict"] == "no"
    assert contradictory["failure_type"] == "impl_error"

    normalized = kg_ops._normalise_result(
        {
            "parallel_index": 0,
            "comparison": {
                "speedup": 100.0,
                "ref_base": "pytorch",
                "base_time_ms": 2.0,
                "gen_time_ms": 1.0,
                "compilation_success": True,
                "precision_passed": True,
            },
            "profiling_insight": {"bottleneck": "mte2_stall"},
        },
        1,
        {"0": "idea:x"},
    )
    assert normalized["speedup"] == pytest.approx(2.0)

    invalid = kg_ops._normalise_result(
        {
            "parallel_index": 0,
            "comparison": {
                "speedup": 100.0,
                "ref_base": "parent_kernel",
                "base_time_ms": 2.0,
                "gen_time_ms": 1.0,
                "compilation_success": True,
                "precision_passed": True,
            },
            "profiling_insight": {"bottleneck": "mte2_stall"},
        },
        1,
        {"0": "idea:x"},
    )
    assert invalid["verdict"] == "no"
    assert invalid["failure_type"] == "impl_error"

    timeout = kg_ops._normalise_result(
        {
            "parallel_index": 0,
            "verdict": "no",
            "failure_type": "timeout",
            "failure_reason": "partial timed out",
        },
        1,
        {"0": "idea:x"},
    )
    assert timeout["failure_type"] == "impl_error"
    assert "original_failure_type=timeout" in timeout["failure_reason"]

    report = _load_module(
        "generate_report_flow_test", ROOT.parent / "evolution-report" / "scripts" / "generate_report.py"
    )
    path, score = report._find_path_by_highest_score(
        {
            "root": {"children": ["exp:bad", "exp:good"]},
            "exp:bad": {"score": 99.0, "status": "failed", "verdict": "no", "children": []},
            "exp:good": {"score": 1.1, "status": "passed", "verdict": "yes", "children": []},
        },
        {"exp:bad": "root", "exp:good": "root"},
    )
    assert path == ["root", "exp:good"]
    assert score == pytest.approx(1.1)


def test_result_slot_and_gating_normalization_are_authoritative():
    kg_ops = _load_module("kg_ops_result_shape_test", KG_OPS)
    normalized = kg_ops._normalise_result(
        {
            "parallel_index": "parallel_01",
            "gating": " FULLY_PASSED ",
            "speedup": 1.25,
            "profiling_insight": {"bottleneck": "mte2_stall"},
        },
        1,
        {"1": "idea:x"},
    )
    assert normalized["parallel_index"] == 1
    assert normalized["verdict"] == "yes"
    assert normalized["evaluation"]["gating"] == "fully_passed"
    assert normalized["evaluation"]["compilation_success"] is True
    assert normalized["evaluation"]["precision_passed"] is True

    guarded = kg_ops._normalise_result(
        {
            "parallel_index": 0,
            "gating": "fully_passed",
            "speedup": 1.25,
            "profiling_insight": {"bottleneck": "mte2_stall"},
        },
        1,
        {"0": "idea:x"},
    )
    kg_ops._apply_full_case_guard(guarded, 2)
    assert guarded["verdict"] == "no"
    assert guarded["failure_type"] == "impl_error"
    assert guarded["evaluation"]["gating"] == "failed"

    leading_zero = kg_ops._normalise_result(
        {
            "parallel_index": "parallel_01",
            "idea_id": "idea:x",
            "verdict": "no",
            "failure_type": "timeout",
        },
        1,
        {"1": "idea:x"},
    )
    assert leading_zero["parallel_index"] == 1


def test_results_dir_rejects_embedded_stale_parallel_index(tmp_path: Path):
    kg_ops = _load_module("kg_ops_result_dir_test", KG_OPS)
    parallel = tmp_path / "parallel_0"
    parallel.mkdir()
    (parallel / "evaluation_results.json").write_text(
        json.dumps({"parallel_index": 1, "idea_id": "idea:x"}),
        encoding="utf-8",
    )
    args = type("Args", (), {
        "results": None,
        "parallel_results": None,
        "results_dir": str(tmp_path),
    })()
    with pytest.raises(ValueError, match="conflicts with directory parallel_0"):
        kg_ops._load_result_records(args)


def test_persist_ideas_does_not_reuse_slug_when_thesis_changes(tmp_path: Path):
    _init(tmp_path, parallel=1)
    idea_generation = _load_module(
        "idea_generation_identity_test", SCRIPTS / "idea_generation.py"
    )
    first = {
        "title": "同名候选",
        "strategy_tags": ["P1"],
        "optimization_type": "tiling",
        "difficulty": 2,
        "mode": "strategy_guided",
        "generation_lens": "profiling_driven",
        "thesis": "第一版假设：通过重排搬运与计算顺序降低等待，并记录完整的边界、资源和回退判定，确保实现可复现且能与基线进行公平比较。",
    }
    second = dict(first)
    second["thesis"] = "第二版假设：改用不同的流水切分策略，显式比较同步点、尾块和寄存器压力，并记录失败回退条件，避免把同名但不同证据的方案覆盖。"
    first_slug = idea_generation.persist_ideas(str(tmp_path), [first], generation_round=1)[0]
    assert idea_generation.persist_ideas(str(tmp_path), [first], generation_round=1) == [first_slug]
    second_slug = idea_generation.persist_ideas(str(tmp_path), [second], generation_round=1)[0]
    assert first_slug != second_slug
    assert len(list((tmp_path / "ideas").glob("*.md"))) >= 3
