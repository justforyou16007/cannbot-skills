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
        "provenance_archive": "provenance/R-2201-900.md",
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


def _load_seal_reference():
    """按路径装载 seal_reference.py。

    它在另一个技能的 scripts 下，不在 sys.path 上 —— 而这几条用例要查的
    恰恰是**跨技能的那根接线**，所以必须装真货，不能用替身。
    """
    import importlib.util  # noqa: PLC0415
    path = (Path(__file__).resolve().parent.parent / "skills"
            / "forge-campaign" / "scripts" / "seal_reference.py")
    spec = importlib.util.spec_from_file_location("seal_reference", path)
    assert spec is not None and spec.loader is not None, f"装载不了 {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_archive(kb_root, card_id="R-2201-900"):
    """给卡片准备它的人类专用归档。

    `add` 要求指针指向的文件真的存在 —— 指向空处的指针和没有指针一样。
    所以每个走 add / lint 的用例都得先把这个文件放好，正如真实流程里
    agent 写卡片之前要先把参考侧细节写进归档。
    """
    directory = Path(kb_root) / hwkb.PROVENANCE_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{card_id}.md").write_text("测试归档", encoding="utf-8")
    return directory / f"{card_id}.md"


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
        operator="layer_norm", input_shape="", as_json=False,
        retry_precondition="常驻块缩小、UB 余量 > 64KB 时重试",
    )
    assert hwkb.cmd_nogo_add(args) == 0

    check = argparse.Namespace(
        kb_root=str(tmp_path), symptom="MTE2", arch="DAV_2201",
        operator="layer_norm", input_shape="", as_json=False)
    # 同算子、同形状的命中有约束力，返回 1 让调用方避开
    assert hwkb.cmd_nogo_check(check) == 1

    miss = argparse.Namespace(
        kb_root=str(tmp_path), symptom="完全不相干的症状", arch=None,
        operator="", input_shape="", as_json=False)
    assert hwkb.cmd_nogo_check(miss) == 0


def test_nogo_requires_evidence(tmp_path):
    """未经测量的放弃不构成死路 —— 否则死路表会把没试过的方向也堵死。"""
    import argparse
    args = argparse.Namespace(
        kb_root=str(tmp_path), symptom="某症状", tried="某方向",
        evidence="   ", arch="DAV_2201", operator="", input_shape="",
        retry_precondition="换形状后重试", as_json=False,
    )
    assert hwkb.cmd_nogo_add(args) == 2


def test_nogo_filters_by_arch(tmp_path):
    import argparse
    hwkb.cmd_nogo_add(argparse.Namespace(
        kb_root=str(tmp_path), symptom="共同症状", tried="方向 A",
        evidence="[MEASURED, job 1]", arch="DAV_2201", operator="",
        input_shape="", retry_precondition="换架构后重测", as_json=False))

    other = argparse.Namespace(
        kb_root=str(tmp_path), symptom="共同症状", arch="DAV_3510",
        operator="", input_shape="", as_json=False)
    assert hwkb.cmd_nogo_check(other) == 0  # 架构不同，不该命中


def test_nogo_corrupt_file_is_refused_not_silently_emptied(tmp_path):
    """损坏的死路表必须报错，不能退回空表。

    这条用例原先断言「优雅降级为空表」。改掉的理由是那条降级路径紧接着就会
    毁掉数据：`nogo add` 的流程是「读出全部条目 → 追加一条 → 整份写回」，
    读成空表意味着下一步写入把整张死路表覆盖成只有新加的那一条。

    死路表的作用是防止后续 campaign 重走已经证伪的方向。它被悄悄清空之后，
    那些方向会被一条条重试，每条都要花掉真实的评测预算 —— 而没有任何
    报错提示发生过这件事。文件损坏本身不罕见：写入时被 Ctrl-C 打断就会
    留下一个截断的 JSON。
    """
    (tmp_path / hwkb.NOGO_FILENAME).write_text("不是 JSON", encoding="utf-8")
    with pytest.raises(hwkb.NogoError, match="不是合法 JSON"):
        hwkb.load_nogo(tmp_path)


def test_nogo_check_refuses_instead_of_reporting_no_hit(tmp_path):
    """表读不出来时不能回答「没试过，可以尝试」。

    调用方正是据此决定要不要花预算重试，所以这里给出肯定的放行结论比
    报错危险得多。
    """
    import argparse  # noqa: PLC0415

    (tmp_path / hwkb.NOGO_FILENAME).write_text('{"entries": [', encoding="utf-8")
    args = argparse.Namespace(kb_root=str(tmp_path), symptom="MTE2 偏高",
                              arch=None, operator="", input_shape="",
                              as_json=False)
    assert hwkb.cmd_nogo_check(args) == 2


def test_nogo_missing_file_is_still_an_empty_list(tmp_path):
    """文件不存在是正常起点，不是错误 —— 别把这两件事混起来。"""
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
    seed_archive(tmp_path)
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(), encoding="utf-8")
    args = argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=False, as_json=False)
    assert hwkb.cmd_add(args) == 0
    assert (tmp_path / "dav2201" / "R-2201-900.md").exists()


def test_add_refuses_overwrite_without_force(tmp_path):
    import argparse
    seed_archive(tmp_path)
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
    seed_archive(tmp_path)
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


# ---------------------------------------------------------------------------
# 卡片 lint 的名词匹配同样要带词边界
# ---------------------------------------------------------------------------

def test_lint_denylist_does_not_match_longer_identifier():
    """清单里的 DataCopyEx 不应命中正文里的 DataCopyExtParams。"""
    card = (
        "---\n"
        "id: R-2201-902\n"
        "arch: DAV_2201\n"
        "mechanism: 用 DataCopyExtParams 描述非连续搬运\n"
        "symptom: MTE2 偏高\n"
        "symptom_channel: evaluation_results.json:baseline.pipeline.mte2_pct\n"
        "---\n"
        "## 机制\n按 DataCopyPadExtParams 补边。\n"
    )
    meta, body = hwkb.parse_card(card)
    issues = hwkb.check_no_code_boundary(
        meta, body, ["DataCopyEx", "DataCopyPad"])
    assert "L-DENY" not in {i.code for i in issues}


def test_lint_denylist_matches_whole_word_case_insensitively():
    """带词边界之后，大小写不同的整词命中仍要报。"""
    card = (
        "---\n"
        "id: R-2201-903\n"
        "arch: DAV_2201\n"
        "mechanism: 照搬 refsecretcore 的编排\n"
        "symptom: MTE2 偏高\n"
        "symptom_channel: evaluation_results.json:baseline.pipeline.mte2_pct\n"
        "---\n"
        "## 机制\n按参考做。\n"
    )
    meta, body = hwkb.parse_card(card)
    issues = hwkb.check_no_code_boundary(meta, body, ["RefSecretCore"])
    assert "L-DENY" in {i.code for i in issues}


# ---------------------------------------------------------------------------
# L-BANNED：禁用词
#
# denylist 拦的是参考的**私有符号名**。「oracle」「参考实现」不是任何源码
# 里的标识符，所以 denylist 永远抓不到 —— 但它们一样把卡片钉死在一次
# 具体的逆向上。这是另一条规则，不是同一条的扩展。
# ---------------------------------------------------------------------------

def test_banned_term_in_body_is_caught():
    meta, body = parse(make_card(body="## 机制\n比参考实现慢 20%。\n"))
    assert "L-BANNED" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_banned_term_in_frontmatter_is_caught():
    meta, body = parse(make_card(mechanism="复刻 oracle 的搬运编排"))
    assert "L-BANNED" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_banned_term_followed_by_cjk_is_still_caught():
    """这条是整组规则的理由所在。

    Python 的 `\\w`（因而 `\\b`）把 CJK 当词字符，所以 `\\boracle\\b` 对
    「oracle的实现」**不匹配** —— `e` 和 `的` 都是词字符，中间没有边界。
    而卡片正文强制中文，「英文词紧跟一个汉字」是默认情形而不是边角情形：
    用 `\\b` 写这条规则，等于它基本不生效。
    """
    meta, body = parse(make_card(body="## 机制\noracle的编排是两级流水。\n"))
    assert "L-BANNED" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_banned_term_as_part_of_a_longer_identifier_is_not_caught():
    """`oracle` 不该吃掉 `oracle_free_path` —— 误伤会让人把整条规则关掉。"""
    meta, body = parse(make_card(body="## 机制\n走 oracle_free_path 分支。\n"))
    assert "L-BANNED" not in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_clean_card_has_no_banned_term():
    meta, body = parse(make_card())
    assert "L-BANNED" not in codes(hwkb.check_no_code_boundary(meta, body, []))


# ---------------------------------------------------------------------------
# L-NARRATIVE：卡片不是 campaign 日记
# ---------------------------------------------------------------------------

def test_date_in_body_is_caught():
    meta, body = parse(make_card(body="## 机制\n2026-11-03 那次测出来是这样。\n"))
    assert "L-NARRATIVE" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_date_right_after_cjk_is_caught():
    """同 ASCII 边界那条理由：`\\b\\d{4}` 对「于2026-11-03」不匹配。"""
    meta, body = parse(make_card(body="## 机制\n于2026-11-03复现。\n"))
    assert "L-NARRATIVE" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_round_number_is_caught():
    meta, body = parse(make_card(body="## 机制\n第 3 轮才发现这件事。\n"))
    assert "L-NARRATIVE" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_iteration_number_is_caught():
    meta, body = parse(make_card(body="## 机制\nFixed at iter 7.\n"))
    assert "L-NARRATIVE" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_commit_sha_is_caught():
    meta, body = parse(make_card(body="## 机制\n见 a3f91bc 那次提交。\n"))
    assert "L-NARRATIVE" in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_plain_number_is_not_mistaken_for_a_sha(tmp_path):
    """`8388608` 全是数字，落在 [0-9a-f] 里但它是个数。

    这类误报的代价不是多一条告警，而是整条规则被豁免掉 —— 那之后真 sha
    也不拦了。所以宁可漏掉纯数字的 sha。
    """
    meta, body = parse(make_card(body="## 机制\n理论最小搬运 8388608 字节。\n"))
    assert "L-NARRATIVE" not in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_hex_looking_word_is_not_mistaken_for_a_sha():
    """`defaced` 正好只用 a-f，但它是个英文词。"""
    assert not hwkb._looks_like_sha("defaced")
    assert not hwkb._looks_like_sha("8388608")
    assert hwkb._looks_like_sha("a3f91bc")


def test_certification_pointer_may_contain_a_date():
    """叙事规则管结论的表述，不管证据的指针。

    `certified_by` 的职责恰恰是指向一次具体的测量，写得越具体越好。
    把它也扫一遍，等于逼着人把唯一能复核的那个指针模糊掉。
    """
    meta, body = parse(make_card(
        certified_by={"job": "2026-11-03/job-7", "arch": "DAV_2201",
                      "delta": "-10%"}))
    assert "L-NARRATIVE" not in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_operator_name_is_caught_when_the_list_is_available():
    meta, body = parse(make_card(body="## 机制\n在 add_layer_norm 上省掉一次搬运。\n"))
    issues = hwkb.check_no_code_boundary(meta, body, [], ["add_layer_norm"])
    assert "L-NARRATIVE" in codes(issues)


def test_operator_name_rule_is_off_without_the_list():
    """半条规则靠外部文件驱动 —— 文件不在时它是空操作，这本身没问题。

    有问题的是「空操作 + 一句通过」，所以 _gate_status 会把这件事带出来，
    由 test_lint_names_the_gate_that_is_not_in_effect 盯住。
    """
    meta, body = parse(make_card(body="## 机制\n在 add_layer_norm 上省掉一次搬运。\n"))
    assert "L-NARRATIVE" not in codes(hwkb.check_no_code_boundary(meta, body, []))


def test_operator_names_file_round_trips(tmp_path):
    (tmp_path / hwkb.OPERATOR_NAMES_FILENAME).write_text(
        "# 注释\nadd_layer_norm\n\nrms_norm  # 行内注释\n", encoding="utf-8")
    assert hwkb.load_operator_names(tmp_path) == ["add_layer_norm", "rms_norm"]
    assert hwkb.operator_names_status(
        tmp_path, hwkb.load_operator_names(tmp_path)) is None


def test_operator_names_status_reports_when_the_half_rule_is_off(tmp_path):
    status = hwkb.operator_names_status(tmp_path, [])
    assert status is not None
    assert "未生效" in status
    # 另外三条不依赖文件，这件事要说清楚 —— 否则读的人以为整条规则都没了
    assert "始终生效" in status


def test_seal_and_hwkb_agree_on_the_operator_names_filename():
    """写方和读方拼的是同一个文件名。

    拼错不会报错，只会让 L-NARRATIVE 的算子名那半条静默失效 —— 和
    L-DENY 当初失效的方式一模一样（seal 写一处、hwkb 读另一处）。
    """
    module = _load_seal_reference()
    assert module.OPERATOR_NAMES_FILENAME == hwkb.OPERATOR_NAMES_FILENAME


def test_seal_writes_the_operator_name_into_the_kb(tmp_path):
    """封存那一步要把算子名送到 hwkb 读的那个位置。"""
    module = _load_seal_reference()
    artifact = tmp_path / "ref"
    artifact.mkdir()
    (artifact / "kernel.cpp").write_text("void RefSecretCore() {}\n",
                                         encoding="utf-8")
    kb_root = tmp_path / "kb"
    module.seal(artifact, "add_layer_norm", tmp_path / "work",
                use_public_api_filter=False, kb_root=kb_root)
    assert hwkb.load_operator_names(kb_root) == ["add_layer_norm"]


# ---------------------------------------------------------------------------
# A-ARCHIVE：人类专用归档
# ---------------------------------------------------------------------------

def test_missing_provenance_archive_is_rejected():
    meta, body = parse(make_card(provenance_archive=None))
    assert "A-ARCHIVE" in codes(hwkb.check_admission(meta, body))


def test_archive_outside_the_provenance_dir_is_rejected():
    """别处没有 deny 规则保护，等于把参考细节放回 agent 的读集。"""
    meta, body = parse(make_card(provenance_archive="notes/R-2201-900.md"))
    assert "A-ARCHIVE" in codes(hwkb.check_admission(meta, body))


def test_archive_with_a_traversal_is_rejected():
    meta, body = parse(make_card(
        provenance_archive="provenance/../../etc/passwd"))
    assert "A-ARCHIVE" in codes(hwkb.check_admission(meta, body))


def test_archive_must_match_the_card_id():
    """一卡一档。共用一份归档时，退役一张卡就会带走另一张的证据。"""
    meta, body = parse(make_card(provenance_archive="provenance/R-2201-001.md"))
    issues = hwkb.check_admission(meta, body)
    assert any("一卡一档" in i.message for i in issues)


def test_archive_pointer_into_thin_air_is_rejected(tmp_path):
    """指针指向空处和没有指针一样。"""
    meta, body = parse(make_card())
    assert "A-ARCHIVE" in codes(hwkb.check_admission(meta, body, tmp_path))


def test_archive_pointer_passes_once_the_file_exists(tmp_path):
    seed_archive(tmp_path)
    meta, body = parse(make_card())
    assert "A-ARCHIVE" not in codes(hwkb.check_admission(meta, body, tmp_path))


def test_the_shipped_card_has_its_archive():
    """发布的那张卡片自己要满足这条规则，否则规则只是写给别人的。"""
    meta, _ = hwkb.load_card(SHIPPED_CARD)
    kb_root = SKILL_ROOT / "references"
    assert (kb_root / str(meta["provenance_archive"])).is_file()


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

def test_add_rewrites_the_index(tmp_path):
    import argparse
    seed_archive(tmp_path)
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(symptom="MTE2 占比过高"), encoding="utf-8")
    assert hwkb.cmd_add(argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=False,
        as_json=False)) == 0
    index = (tmp_path / "dav2201" / "INDEX.md").read_text(encoding="utf-8")
    assert "MTE2 占比过高" in index
    assert "R-2201-900.md" in index


def test_index_rewrite_keeps_hand_written_text(tmp_path):
    """自动段之外的内容不许动。

    INDEX.md 里的导读是人和 agent 都要读的，生成器的需要不是删掉它的理由。
    """
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    (arch_dir / "INDEX.md").write_text(
        "# 导读\n\n这段话必须活下来。\n", encoding="utf-8")
    hwkb.write_index(tmp_path, "DAV_2201")
    text = (arch_dir / "INDEX.md").read_text(encoding="utf-8")
    assert "这段话必须活下来。" in text
    assert hwkb.INDEX_BEGIN in text

    # 再写一次不该叠加第二份自动段
    hwkb.write_index(tmp_path, "DAV_2201")
    text = (arch_dir / "INDEX.md").read_text(encoding="utf-8")
    assert text.count(hwkb.INDEX_BEGIN) == 1


def test_card_dropped_into_the_directory_is_reported(tmp_path):
    """没走 add 的卡片，索引里没有它 —— 那张密集表是 agent 每轮读到的东西。"""
    import argparse
    seed_archive(tmp_path)
    draft = tmp_path / "draft.md"
    draft.write_text(make_card(), encoding="utf-8")
    hwkb.cmd_add(argparse.Namespace(
        kb_root=str(tmp_path), card=str(draft), force=False, as_json=False))

    seed_archive(tmp_path, "R-2201-901")
    (tmp_path / "dav2201" / "R-2201-901.md").write_text(
        make_card(id="R-2201-901",
                  provenance_archive="provenance/R-2201-901.md"),
        encoding="utf-8")
    assert "L-INDEX" in codes(hwkb.check_index_fresh(tmp_path, "DAV_2201"))


def test_empty_arch_dir_needs_no_index(tmp_path):
    """知识库的正常起点不该是红的 —— 在这里响一次，规则就会被整条关掉。"""
    (tmp_path / "dav2201").mkdir(parents=True)
    assert hwkb.check_index_fresh(tmp_path, "DAV_2201") == []


# ---------------------------------------------------------------------------
# 隔离区
# ---------------------------------------------------------------------------

def _lint(tmp_path, quarantine=True):
    import argparse  # noqa: PLC0415
    return hwkb.cmd_lint(argparse.Namespace(
        kb_root=str(tmp_path), card=None, arch=None, quarantine=quarantine,
        as_json=False))


def test_first_lint_failure_leaves_the_card_in_place(tmp_path):
    """第一次失败往往是形式问题 —— 少个小节、kind 写错、归档指针拼错。

    这些改一行就好，而隔离会让这张卡片离开 agent 的视野，于是那条真经验
    也跟着离开，直到有人想起来去 quarantine 目录里翻。
    """
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    seed_archive(tmp_path)
    card = arch_dir / "R-2201-900.md"
    card.write_text(make_card(body="## 机制\n比参考实现快。\n"),
                    encoding="utf-8")

    assert _lint(tmp_path) == 1
    assert card.exists()                       # 留在原处
    meta, _ = hwkb.load_card(card)
    assert meta[hwkb.LINT_FAILURES_FIELD] == 1


def test_lint_quarantine_moves_the_card_out_of_the_readable_tree(tmp_path):
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    seed_archive(tmp_path)
    (arch_dir / "R-2201-900.md").write_text(
        make_card(body="## 机制\n比参考实现快。\n"), encoding="utf-8")

    # 第一次只记账，第二次才隔离
    assert _lint(tmp_path) == 1
    assert _lint(tmp_path) == 1
    # 移走，不是删掉 —— 越界的卡片里往往有真经验，只是写法不合规
    assert not (arch_dir / "R-2201-900.md").exists()
    assert (tmp_path / hwkb.QUARANTINE_DIRNAME / "dav2201"
            / "R-2201-900.md").is_file()
    # 移走之后 query 不再供给它
    assert [p.name for p, _, _ in hwkb.iter_cards(tmp_path, "DAV_2201")] == []


def test_failure_count_survives_across_processes(tmp_path):
    """计数记在 frontmatter 而不是内存里。

    记在内存里的话，每次 lint 都是第一次，「返修两轮再隔离」这条规则
    等于不存在。
    """
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    seed_archive(tmp_path)
    card = arch_dir / "R-2201-900.md"
    card.write_text(make_card(body="## 机制\n比参考实现快。\n"),
                    encoding="utf-8")
    _lint(tmp_path)
    text = card.read_text(encoding="utf-8")
    assert hwkb.LINT_FAILURES_FIELD in text     # 落盘了


def test_unparseable_card_is_quarantined_immediately(tmp_path):
    """frontmatter 读不动时没处记计数，而这类卡片最常见的成因恰恰是
    正文里粘了大段源码把 frontmatter 撑坏了。"""
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    (arch_dir / "R-2201-901.md").write_text(
        "这不是 frontmatter\nvoid Kernel() {}\n", encoding="utf-8")
    assert _lint(tmp_path) == 1
    assert not (arch_dir / "R-2201-901.md").exists()


def test_passing_lint_clears_the_failure_count(tmp_path):
    """计数衡量的是「连续不过检」，不是「一辈子不过检过几次」。

    不清的话，一张修好又改坏的卡片会带着历史计数直接被隔离 —— 而它这次
    只错了一处。
    """
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    seed_archive(tmp_path)
    card = arch_dir / "R-2201-900.md"
    card.write_text(make_card(body="## 机制\n比参考实现快。\n"),
                    encoding="utf-8")
    _lint(tmp_path)                                  # 记一次
    hwkb.write_index(tmp_path, "DAV_2201")
    card.write_text(make_card(), encoding="utf-8")   # 修好
    _lint(tmp_path)
    meta, _ = hwkb.load_card(card)
    assert hwkb.LINT_FAILURES_FIELD not in meta


def test_lint_without_quarantine_leaves_the_card_in_place(tmp_path):
    import argparse
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    seed_archive(tmp_path)
    (arch_dir / "R-2201-900.md").write_text(
        make_card(body="## 机制\n比参考实现快。\n"), encoding="utf-8")
    assert hwkb.cmd_lint(argparse.Namespace(
        kb_root=str(tmp_path), card=None, arch=None, quarantine=False,
        as_json=False)) == 1
    assert (arch_dir / "R-2201-900.md").exists()


def test_lint_all_also_runs_admission(tmp_path):
    """准入只在 add 那一刻跑过一次，而卡片此后会被手改、被合并、被直接丢进目录。

    只查边界的 lint 对这些情况会说「通过」，而那句话并不包含「这张卡片
    仍然带着可核验的认证」。
    """
    import argparse
    arch_dir = tmp_path / "dav2201"
    arch_dir.mkdir(parents=True)
    (arch_dir / "R-2201-900.md").write_text(
        make_card(certified_by=None), encoding="utf-8")
    assert hwkb.cmd_lint(argparse.Namespace(
        kb_root=str(tmp_path), card=None, arch=None, quarantine=False,
        as_json=False)) == 1


# ---------------------------------------------------------------------------
# 死路表：重试前置条件表，不是禁令表
# ---------------------------------------------------------------------------

def _nogo_add(tmp_path, **kw):
    import argparse
    base = dict(kb_root=str(tmp_path), symptom="MTE2 占比高",
                tried="加大 double buffer 深度",
                evidence="[MEASURED, job 9] 无改善，UB 被挤爆",
                arch="DAV_2201", operator="layer_norm",
                input_shape="B=8,S=4096,H=1024",
                retry_precondition="常驻块缩小、UB 余量 > 64KB 时重试",
                as_json=False)
    base.update(kw)
    return hwkb.cmd_nogo_add(argparse.Namespace(**base))


def _nogo_check(tmp_path, **kw):
    import argparse
    base = dict(kb_root=str(tmp_path), symptom="MTE2", arch="DAV_2201",
                operator="layer_norm", input_shape="B=8,S=4096,H=1024",
                as_json=False)
    base.update(kw)
    return hwkb.cmd_nogo_check(argparse.Namespace(**base))


def test_nogo_add_requires_a_retry_precondition(tmp_path):
    """只说「别试了」的记录会把一个条件变了就该重试的方向永久封死。"""
    assert _nogo_add(tmp_path, retry_precondition="  ") == 2


def test_nogo_hit_under_the_same_conditions_is_binding(tmp_path):
    assert _nogo_add(tmp_path) == 0
    assert _nogo_check(tmp_path) == 1


def test_nogo_hit_on_another_shape_is_advisory(tmp_path, capsys):
    """双缓冲在 H=1024 下挤爆 UB，在 H=256 下不会。

    一律当禁令用，会把一堆本该成立的方向永久封死，而且封得毫无痕迹 ——
    调用方只看到 exit 1。
    """
    assert _nogo_add(tmp_path) == 0
    capsys.readouterr()
    assert _nogo_check(tmp_path, input_shape="B=8,S=4096,H=256") == 0
    out = capsys.readouterr().out
    assert "不构成禁令" in out
    assert "常驻块缩小" in out      # 前置条件要一起打出来


def test_nogo_hit_on_another_operator_is_advisory(tmp_path):
    assert _nogo_add(tmp_path) == 0
    assert _nogo_check(tmp_path, operator="rms_norm") == 0


def test_nogo_entry_without_an_operator_binds_anyone(tmp_path):
    """没记算子 = 算子无关的硬件结论，对谁都成立。"""
    assert _nogo_add(tmp_path, operator="", input_shape="") == 0
    assert _nogo_check(tmp_path, operator="rms_norm", input_shape="任意") == 1


def test_nogo_without_a_shape_in_the_query_is_not_binding(tmp_path, capsys):
    """条目记了形状而这次没给，就无从判断是不是同一个条件 —— 不该当禁令。"""
    assert _nogo_add(tmp_path) == 0
    capsys.readouterr()
    assert _nogo_check(tmp_path, input_shape="") == 0


def test_nogo_json_separates_binding_from_advisory(tmp_path, capsys):
    assert _nogo_add(tmp_path) == 0
    capsys.readouterr()
    _nogo_check(tmp_path, input_shape="B=8,S=4096,H=256", as_json=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["binding"] == []
    assert len(doc["advisory"]) == 1
    assert "形状" in doc["advisory"][0]["why_not_binding"]
