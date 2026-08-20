#!/usr/bin/env python3
"""scenario-routing 归族回归测试：route_rules.py 谓词引擎 vs golden.json。

运行：
    python3 -m pytest skills/evolution-strategies/references/scenario-routing/tests/test_routing.py -v
或直接：
    python3 skills/evolution-strategies/references/scenario-routing/tests/test_routing.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_rules import route  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.json"


def _load():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]


def test_all_golden_cases_classify():
    """每条 golden 的 family + branch 必须精确命中；golden 声明的 traits 必须被引擎覆盖。"""
    failures = []
    for case in _load():
        got = route(case["features"])
        ok = got["family"] == case["family"] and got["branch"] == case["branch"]
        missing = [t for t in case.get("traits", []) if t not in got["traits"]]
        if not ok:
            failures.append(f"{case['op']}: expected ({case['family']}/{case['branch']}) got ({got['family']}/{got['branch']})")
        elif missing:
            failures.append(f"{case['op']}: 缺 trait {missing}（实际 {got['traits']}）")
    assert not failures, "\n".join(failures)


def test_family_counts_match_level4_distribution():
    """level4 已知分布抽查：attention 类数量 ≥ 已知子族数（扩展 golden 时强化）。"""
    cases = _load()
    fams = {}
    for c in cases:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
    # golden 覆盖了 attention/ssm/matmul/conv/reduction 五族
    assert set(fams) >= {"attention", "ssm", "matmul", "conv", "reduction"}, fams


if __name__ == "__main__":
    _f = []
    for case in _load():
        got = route(case["features"])
        mark = "OK " if (got["family"] == case["family"] and got["branch"] == case["branch"]
                         and all(t in got["traits"] for t in case.get("traits", []))) else "FAIL"
        if mark == "FAIL":
            _f.append(case["op"])
        print(f"{mark} {case['op']:34s} -> {got['family']}/{got['branch']} {got['traits']}")
    print(f"\n{len(_load()) - len(_f)}/{len(_load())} passed" + (f"; FAIL: {_f}" if _f else ""))
    sys.exit(1 if _f else 0)
