#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""hwkb.py 的单元测试。

知识库有两道闸门，测试分别盯住它们：

  准入闸   没有目标架构上的探针实测背书，卡片不得入库。
           这道闸要是漏了，知识库就退化成「agent 的主观笔记合集」。

  边界闸   卡片不得留下参考实现的痕迹。这道闸要是漏了，知识就绑死在
           某个参考上，跨算子复用无从谈起 —— 而跨算子复用正是这个
           知识库存在的全部理由。

运行: python3 -m pytest plugins-community/forge-kernel/tests/ -v
"""

import json
import sys
import textwrap
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills" / "forge-hwkb"
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import hwkb  # noqa: E402

SHIPPED_CARD = SKILL_ROOT / "references" / "dav2201" / "R-2201-001.md"


def make_card(**overrides) -> str:
    """造一张合法卡片，overrides 用来定点改坏某个字段。

    传 None 表示删除该 frontmatter 字段；body 用来替换正文。
    """
    meta = {
        "id": "R-2201-900",
        "arch": "DAV_2201",
        "mechanism": "测试用机制",
        "symptom": "测试用症状",
        "symptom_channel": "evaluation_results.json:baseline.pipeline.mte2_pct",
        "operator_families": [],
        "provenance": [{"kind": "public_doc", "ref": "ops/npu-arch/SKILL.md"}],
        "certified_by": {"job": "job 1", "arch": "DAV_2201", "delta": "-10%"},
        "status": "active",
    }
    body = overrides.pop("body", None)
    for key, value in overrides.items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value

    if body is None:
        body = textwrap.dedent("""
            ## 机制
            机制说明。

            ## 触发症状
            症状说明。

            ## 最小可运行例
            例子说明。

            ## 已知坑
            坑位说明。
        """)

    import yaml
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True) + "---\n" + body


def codes(issues):
    return {i.code for i in issues}


def parse(text):
    return hwkb.parse_card(text)


# ---------------------------------------------------------------------------
# 基线
# ---------------------------------------------------------------------------

def test_shipped_card_passes_admission():
    """随包卡片必须合规 —— 它是 agent 写卡片时的模板。"""
    meta, body = hwkb.load_card(SHIPPED_CARD)
    assert hwkb.check_admission(meta, body) == []


def test_shipped_card_passes_boundary():
    meta, body = hwkb.load_card(SHIPPED_CARD)
    assert hwkb.check_no_code_boundary(meta, body, []) == []


def test_synthetic_card_is_valid():
    meta, body = parse(make_card())
    assert hwkb.check_admission(meta, body) == []


# ---------------------------------------------------------------------------
# 卡片解析
# ---------------------------------------------------------------------------

def test_missing_frontmatter_rejected():
    with pytest.raises(ValueError, match="frontmatter"):
        parse("## 机制\n没有头部。\n")


def test_malformed_yaml_rejected():
    with pytest.raises(ValueError, match="YAML"):
        parse("---\nid: [未闭合\n---\n正文\n")


def test_non_mapping_frontmatter_rejected():
    with pytest.raises(ValueError, match="键值映射"):
        parse("---\n- 列表不是映射\n---\n正文\n")


# ---------------------------------------------------------------------------
# 准入闸：探针认证
# ---------------------------------------------------------------------------

def test_admission_rejects_uncertified_card():
    """核心用例：没有实测背书一律拒收。"""
    meta, body = parse(make_card(certified_by=None))
    assert "A-CERT" in codes(hwkb.check_admission(meta, body))


def test_admission_rejects_certification_without_job():
    meta, body = parse(make_card(
        certified_by={"arch": "DAV_2201", "delta": "-10%"}))
    assert "A-CERT-JOB" in codes(hwkb.check_admission(meta, body))


def test_admission_rejects_certification_without_delta():
    """没有实测变化量就无法区分真机制和巧合加速。"""
    meta, body = parse(make_card(
        certified_by={"job": "job 1", "arch": "DAV_2201"}))
    assert "A-CERT-DELTA" in codes(hwkb.check_admission(meta, body))


def test_admission_rejects_cross_arch_certification():
    """在 3510 上测出来的结论不能挂到 2201 的卡片上。"""
    meta, body = parse(make_card(
        certified_by={"job": "job 1", "arch": "DAV_3510", "delta": "-10%"}))
    issues = hwkb.check_admission(meta, body)
    assert "A-CERT-ARCH" in codes(issues)
    assert any("不一致" in i.message for i in issues)


def test_admission_rejects_unknown_arch():
    meta, body = parse(make_card(arch="DAV_9999"))
    assert "A-ARCH" in codes(hwkb.check_admission(meta, body))


# ---------------------------------------------------------------------------
# 准入闸：必填字段与出处
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field",
                         ["id", "arch", "mechanism", "symptom", "symptom_channel"])
def test_admission_requires_core_fields(field):
    meta, body = parse(make_card(**{field: None}))
    assert "A-FIELD" in codes(hwkb.check_admission(meta, body))


def test_admission_requires_provenance():
    meta, body = parse(make_card(provenance=None))
    assert "A-PROV" in codes(hwkb.check_admission(meta, body))


def test_admission_rejects_disallowed_provenance_kind():
    """「我觉得」不是出处。"""
    meta, body = parse(make_card(
        provenance=[{"kind": "agent_opinion", "ref": "凭经验"}]))
    assert "A-PROV-KIND" in codes(hwkb.check_admission(meta, body))


@pytest.mark.parametrize("kind", ["probe_verdict", "public_doc", "public_api"])
def test_admission_accepts_allowed_provenance_kinds(kind):
    meta, body = parse(make_card(provenance=[{"kind": kind, "ref": "某处"}]))
    assert "A-PROV-KIND" not in codes(hwkb.check_admission(meta, body))


# ---------------------------------------------------------------------------
# 准入闸：正文四小节
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("section", ["机制", "触发症状", "最小可运行例", "已知坑"])
def test_admission_requires_all_four_sections(section):
    full = {
        "机制": "机制说明。",
        "触发症状": "症状说明。",
        "最小可运行例": "例子说明。",
        "已知坑": "坑位说明。",
    }
    del full[section]
    body = "\n".join(f"## {k}\n{v}\n" for k, v in full.items())
    meta, parsed_body = parse(make_card(body=body))
    issues = hwkb.check_admission(meta, parsed_body)
    assert "A-SECTION" in codes(issues)
    assert any(section in i.message for i in issues)


def test_admission_rejects_empty_section():
    body = textwrap.dedent("""
        ## 机制

        ## 触发症状
        症状说明。

        ## 最小可运行例
        例子说明。

        ## 已知坑
        坑位说明。
    """)
    meta, parsed_body = parse(make_card(body=body))
    assert "A-SECTION" in codes(hwkb.check_admission(meta, parsed_body))


# ---------------------------------------------------------------------------
# 边界闸：no-code
# ---------------------------------------------------------------------------

def test_lint_catches_denylisted_token_in_body():
    """参考实现的内部符号名不得进入可复用经验。"""
    body = textwrap.dedent("""
        ## 机制
        照搬 SecretRefKernel 的做法即可。

        ## 触发症状
        症状。

        ## 最小可运行例
        例子。

        ## 已知坑
        坑。
    """)
    meta, parsed_body = parse(make_card(body=body))
    issues = hwkb.check_no_code_boundary(meta, parsed_body, ["SecretRefKernel"])
    assert "L-DENY" in codes(issues)


def test_lint_is_case_insensitive():
    body = "## 机制\n参考 secretrefkernel。\n\n## 触发症状\n症状。\n\n## 最小可运行例\n例。\n\n## 已知坑\n坑。\n"
    meta, parsed_body = parse(make_card(body=body))
    assert "L-DENY" in codes(
        hwkb.check_no_code_boundary(meta, parsed_body, ["SecretRefKernel"]))


def test_lint_catches_denylisted_token_in_frontmatter():
    """头部字段同样受约束，不能把专有名词藏进 mechanism 里。"""
    meta, body = parse(make_card(mechanism="SecretRefKernel 的流水编排"))
    assert "L-DENY" in codes(
        hwkb.check_no_code_boundary(meta, body, ["SecretRefKernel"]))


def test_lint_catches_source_path_in_provenance():
    """出处指向源码文件，说明引用了不该读的东西。"""
    meta, body = parse(make_card(
        provenance=[{"kind": "probe_verdict", "ref": "ref_impl/attention_kernel.cpp:88"}]))
    assert "L-SRC" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_lint_allows_doc_path_in_public_doc():
    """指向公开文档的 .md 路径是正当出处，不该误伤。"""
    meta, body = parse(make_card(
        provenance=[{"kind": "public_doc", "ref": "ops/npu-arch/SKILL.md"}]))
    assert "L-SRC" not in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_lint_passes_with_empty_denylist():
    meta, body = parse(make_card())
    assert hwkb.check_no_code_boundary(meta, body, []) == []


# ---------------------------------------------------------------------------
# denylist 读取
# ---------------------------------------------------------------------------

def test_denylist_parsing(tmp_path):
    (tmp_path / hwkb.DENYLIST_FILENAME).write_text(
        "# 注释行\n"
        "SecretRefKernel\n"
        "\n"
        "  AnotherSymbol   # 行尾注释\n",
        encoding="utf-8",
    )
    assert hwkb.load_denylist(tmp_path) == ["SecretRefKernel", "AnotherSymbol"]


def test_denylist_missing_file_is_empty(tmp_path):
    assert hwkb.load_denylist(tmp_path) == []


# ---------------------------------------------------------------------------
# 死路表
# ---------------------------------------------------------------------------

def test_nogo_roundtrip(tmp_path):
    import argparse
    args = argparse.Namespace(
        kb_root=str(tmp_path), symptom="MTE2 占比高", tried="加大 double buffer 深度",
        evidence="[MEASURED, job 9] 无改善，UB 被挤爆", arch="DAV_2201",
        operator="layer_norm", as_json=False,
    )
    assert hwkb.cmd_nogo_add(args) == 0

    check = argparse.Namespace(
        kb_root=str(tmp_path), symptom="MTE2", arch="DAV_2201", as_json=False)
    # 命中死路返回 1，提示调用方避开
    assert hwkb.cmd_nogo_check(check) == 1

    miss = argparse.Namespace(
        kb_root=str(tmp_path), symptom="完全不相干的症状", arch=None, as_json=False)
    assert hwkb.cmd_nogo_check(miss) == 0


def test_nogo_requires_evidence(tmp_path):
    """未经测量的放弃不构成死路 —— 否则死路表会把没试过的方向也堵死。"""
    import argparse
    args = argparse.Namespace(
        kb_root=str(tmp_path), symptom="某症状", tried="某方向",
        evidence="   ", arch=None, operator="", as_json=False,
    )
    assert hwkb.cmd_nogo_add(args) == 2


def test_nogo_filters_by_arch(tmp_path):
    import argparse
    hwkb.cmd_nogo_add(argparse.Namespace(
        kb_root=str(tmp_path), symptom="共同症状", tried="方向 A",
        evidence="[MEASURED, job 1]", arch="DAV_2201", operator="", as_json=False))

    other = argparse.Namespace(
        kb_root=str(tmp_path), symptom="共同症状", arch="DAV_3510", as_json=False)
    assert hwkb.cmd_nogo_check(other) == 0  # 架构不同，不该命中


def test_nogo_corrupt_file_degrades_gracefully(tmp_path):
    (tmp_path / hwkb.NOGO_FILENAME).write_text("不是 JSON", encoding="utf-8")
    assert hwkb.load_nogo(tmp_path) == []


# ---------------------------------------------------------------------------
# 入库命令
# ---------------------------------------------------------------------------

def test_add_rejects_invalid_card(tmp_path):
    import argparse
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(certified_by=None), encoding="utf-8")
    args = argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=False, as_json=False)
    assert hwkb.cmd_add(args) == 1
    # 拒绝入库就不该留下文件
    assert not (tmp_path / "dav2201" / "R-2201-900.md").exists()


def test_add_accepts_valid_card(tmp_path):
    import argparse
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(), encoding="utf-8")
    args = argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=False, as_json=False)
    assert hwkb.cmd_add(args) == 0
    assert (tmp_path / "dav2201" / "R-2201-900.md").exists()


def test_add_refuses_overwrite_without_force(tmp_path):
    import argparse
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(), encoding="utf-8")
    args = argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=False, as_json=False)
    assert hwkb.cmd_add(args) == 0
    assert hwkb.cmd_add(args) == 2  # 第二次应拒绝

    forced = argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=True, as_json=False)
    assert hwkb.cmd_add(forced) == 0


def test_add_blocked_by_denylist(tmp_path):
    """边界闸在入库路径上同样生效，不只是 lint 命令里。"""
    import argparse
    (tmp_path / hwkb.DENYLIST_FILENAME).write_text("SecretRefKernel\n", encoding="utf-8")
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(mechanism="SecretRefKernel 的编排"), encoding="utf-8")
    args = argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=False, as_json=False)
    assert hwkb.cmd_add(args) == 1


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def _query(tmp_path, **kw):
    import argparse
    base = dict(kb_root=str(tmp_path), symptom="", arch=None, family="",
                include_retired=False, as_json=True)
    base.update(kw)
    return hwkb.cmd_query(argparse.Namespace(**base))


def _seed(tmp_path, **overrides):
    import argparse
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(**overrides), encoding="utf-8")
    hwkb.cmd_add(argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=True, as_json=False))


def test_query_hits_by_symptom(tmp_path, capsys):
    _seed(tmp_path, symptom="MTE2 占比过高")
    capsys.readouterr()  # 丢掉入库时的输出，只解析 query 的 JSON
    assert _query(tmp_path, symptom="MTE2") == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1


def test_query_miss_returns_nonzero(tmp_path):
    _seed(tmp_path, symptom="MTE2 占比过高")
    assert _query(tmp_path, symptom="完全不相干") == 1


def test_query_empty_families_match_any(tmp_path, capsys):
    """operator_families 为空表示算子无关，应对任何族命中 —— 这正是跨算子复用。"""
    _seed(tmp_path, operator_families=[])
    capsys.readouterr()  # 丢掉入库时的输出
    assert _query(tmp_path, family="matmul") == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1


def test_query_respects_family_filter(tmp_path):
    _seed(tmp_path, operator_families=["reduction"])
    assert _query(tmp_path, family="matmul") == 1  # 未命中


def test_query_skips_retired_by_default(tmp_path):
    _seed(tmp_path, status="retired")
    assert _query(tmp_path) == 1
    assert _query(tmp_path, include_retired=True) == 0
