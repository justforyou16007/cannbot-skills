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
"""手段定价与消融：在投入之前先知道这条路最多值多少。

论文 §3.3 说 arch agent 在内层停滞时重新进入，「给手段定价、跑裁决探针
以分离单项效应与耦合效应」。Appendix I 的案例把这件事说得很具体：
单独去掉输出写回这一项，拿到 6.0% —— 据此判断瓶颈在 kernel 的边界上，
于是重开了 M0 的第一个决策。

定价给的是上界，这才是它的价值
--------------------------------
把某个环节**整个删掉**（哪怕结果因此是错的）所省下的时间，就是优化这个
环节能拿到的**收益上限**。删掉输出写回只省 6%，那么无论怎样优化写回，
最多也就 6%。

这个数字的用处是**决定不做什么**。优化循环里最贵的不是做错方向，而是
在一个天花板只有 2% 的方向上投入三轮迭代。先定价，就能把这类投入提前
砍掉 —— 代价只是一次测量。

探针变体是错的，这是设计使然
------------------------------
删掉输出写回的 kernel 算出来的结果不对。它不是候选，不能上线，也不该
进评测流程的接受判定。本工具会把所有变体标为 probe_only，防止它们被
误当成可交付产物。

耦合效应
--------
两个手段各自省 5%，一起上却只省 6% —— 说明它们在抢同一份时间，是耦合的；
真实可得是 6% 而不是 10%。反过来，一起上省了 14%，说明有协同。
不做这个分离，就会按 5+5=10 去规划，然后发现对不上。

用法:
    # 把一次测量登记进来
    python3 lever_probe.py add --store ./forge_work/levers.json \\
        --baseline-us 456.5 --noise-floor 2.3
    python3 lever_probe.py add --store ./forge_work/levers.json \\
        --lever drop_output_store --time-us 429.1
    python3 lever_probe.py add --store ./forge_work/levers.json \\
        --lever drop_output_store --lever drop_mask --time-us 420.0

    # 出定价表与耦合分析
    python3 lever_probe.py analyze --store ./forge_work/levers.json

退出码: 0 = 分析完成; 1 = 数据不足; 2 = 用法或 IO 错误
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCHEMA_VERSION = "1.0"

#: 判定取保守：差值落在噪声底以内一律算「测不出来」。
#: 浮点比较同样要带容差，理由与 campaign.py 的接受判据相同。
FLOAT_TOL = 1e-9


class LeverError(Exception):
    """数据不足或不一致。"""


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------

def new_store(baseline_us: float, noise_floor_pct: float,
              label: str = "baseline",
              variant_hash: str = "") -> Dict[str, Any]:
    if baseline_us <= 0:
        raise LeverError("基线耗时必须为正")
    if noise_floor_pct < 0:
        raise LeverError("噪声底不能为负")
    return {
        "schema_version": SCHEMA_VERSION,
        # 基线的代码指纹。给了它，之后每个变体都能被核对「你真的改过吗」——
        # 没改过的变体测出来的差值全是噪声，而那个差值会被当成价格用。
        "baseline": {"label": label, "time_us": baseline_us,
                     "variant_hash": variant_hash},
        "noise_floor_pct": noise_floor_pct,
        "probe_only": True,
        "warning": ("本文件记录的变体是**定价探针**，多数会产出错误结果"
                    "（例如删掉输出写回）。它们不是候选，不得上线，"
                    "也不得进入接受判定"),
        "variants": [],
    }


def load_store(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LeverError(f"无法读取 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LeverError(f"{path} 不是合法 JSON: {exc}") from exc
    if not isinstance(doc, dict) or "baseline" not in doc:
        raise LeverError(f"{path} 不是有效的定价记录；先用 add --baseline-us 建立")
    return doc


def save_store(path: Path, store: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise LeverError(f"无法写入 {path}: {exc}") from exc


#: 手段的六种分类。
#:
#: 单项定价（「只删这一项」）只能回答「它单独值多少」。而优化循环实际要问
#: 的是「全上之后，再动它还有没有收益」—— 这两个问题的答案可以相反。
#:
#: 关键的一类是 `coupled`：**单独删掉省不出时间，但别人依赖它**。只有单项
#: 腿时，它的读数落在噪声里 → 被判「测不出来」→ 进 `not_worth_pursuing`
#: → 工具主动告诉你不要在一个承重的方向上投入。这不是漏报，是反向报错。
#:
#: 分类要两条腿的读数才能给：
#:   单项增益   只删这一项省下多少       → 它单独值多少
#:   边际增益   全删 − 全删但留下这一项   → 别人都在的情况下它还值多少
LEVER_CLASSES = {
    "independent": "单独有用，全上时也仍有用 —— 可以独立规划",
    "redundant": "单独有用，但别人已经吃掉了它的收益 —— 不要重复投入",
    "coupled": "单独看不出收益，但别人依赖它 —— **必须保留**，"
               "而且不能据单项读数把它判成无用",
    "inert": "两条腿都看不出收益 —— 真的没有可测量的空间",
    "harmful": "删掉它反而更慢 —— 它的存在改善了别的东西",
    "unknown": "缺一条腿的测量，判不了",
}


def _gain_pct(baseline_us: float, time_us: float) -> float:
    """去掉某环节后省下的时间，占基线的百分比。"""
    return (baseline_us - time_us) / baseline_us * 100.0


def _significant(value_pct: float, noise_pct: float) -> bool:
    """差值是否超出噪声。边界上算不超出，理由同 campaign 的接受判据。"""
    if math.isclose(abs(value_pct), noise_pct, rel_tol=1e-9, abs_tol=FLOAT_TOL):
        return False
    return abs(value_pct) > noise_pct

def _variant_key(levers: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(levers)))


def classify_levers(store: Dict[str, Any]) -> Dict[str, Any]:
    """两条腿合起来给分类。

    `universe` 取全部变体里出现过的手段的并集 —— 也就是「这次定价一共考虑
    了哪些环节」。边际读数需要两个变体：

      V_all     手段集恰好等于 universe（「全删」）
      V_minus_L 手段集等于 universe − {L}（「全删但留下 L」）

    两者都在才算得出 L 的边际增益。缺哪个就说哪个缺，不拿单项读数顶替 ——
    顶替正是 `coupled` 被判成 `inert` 的那条路。
    """
    baseline_us = float(store["baseline"]["time_us"])
    noise = float(store["noise_floor_pct"])
    variants = store.get("variants", [])

    by_key: Dict[Tuple[str, ...], float] = {}
    universe: Set[str] = set()
    for entry in variants:
        key = _variant_key(entry["levers"])
        by_key[key] = float(entry["time_us"])
        universe |= set(key)

    all_key = tuple(sorted(universe))
    all_gain = (_gain_pct(baseline_us, by_key[all_key])
                if all_key in by_key else None)

    out: Dict[str, Any] = {}
    for lever in sorted(universe):
        single_key = (lever,)
        single_gain = (_gain_pct(baseline_us, by_key[single_key])
                       if single_key in by_key else None)

        minus_key = tuple(sorted(universe - {lever}))
        minus_gain = (_gain_pct(baseline_us, by_key[minus_key])
                      if minus_key in by_key else None)
        marginal = (all_gain - minus_gain
                    if all_gain is not None and minus_gain is not None
                    else None)

        if single_gain is not None and single_gain < 0 \
                and _significant(single_gain, noise):
            klass = "harmful"
        elif single_gain is None or marginal is None:
            klass = "unknown"
        else:
            single_ok = single_gain > 0 and _significant(single_gain, noise)
            marginal_ok = marginal > 0 and _significant(marginal, noise)
            if single_ok and marginal_ok:
                klass = "independent"
            elif single_ok:
                klass = "redundant"
            elif marginal_ok:
                klass = "coupled"
            else:
                klass = "inert"

        missing: List[str] = []
        if single_gain is None:
            missing.append(f"单项测量（--lever {lever}）")
        if all_gain is None:
            missing.append("全删测量（" + " ".join(
                f"--lever {x}" for x in all_key) + "）")
        if minus_gain is None and all_gain is not None:
            missing.append("全删但留下它的测量（" + " ".join(
                f"--lever {x}" for x in minus_key) + "）")

        out[lever] = {
            "lever": lever,
            "class": klass,
            "meaning": LEVER_CLASSES[klass],
            "single_pct": single_gain,
            "marginal_pct": marginal,
            "missing_measurements": missing,
        }
    return {"universe": all_key, "all_pct": all_gain, "levers": out}


def add_variant(store: Dict[str, Any], levers: List[str],
                time_us: float, note: str = "",
                variant_hash: str = "") -> Dict[str, Any]:
    if not levers:
        raise LeverError("变体必须至少关联一个手段（--lever）")
    if time_us <= 0:
        raise LeverError("耗时必须为正")

    # 变体指纹必须与基线不同。
    #
    # 一个「删掉了某个环节」的变体，如果代码和基线一模一样，那它根本没被
    # 改过 —— 测出来的差值全部是噪声。而那个差值会被当成这个环节的**价格**，
    # 据此决定要不要在它上面投几轮迭代。
    #
    # 这件事在流程里很容易发生：改动写在一个没被编译进去的分支里、改完忘了
    # 重新装 OPP、或者改的是另一个 shape 的路径。三种情形都会得到一个形状
    # 完全正常的结果文件。
    baseline_hash = str(store["baseline"].get("variant_hash") or "")
    if variant_hash and baseline_hash and variant_hash == baseline_hash:
        raise LeverError(
            f"变体指纹与基线相同（{variant_hash[:12]}）—— 这个变体没有被真正改过。\n"
            "  那么测出来的差值全部是噪声，而它会被当成这个环节的价格，"
            "据此决定要不要\n"
            "  在它上面投几轮迭代。\n"
            "  常见原因：改动写在没被编译进去的分支里、改完忘了重装 OPP、"
            "改的是另一个 shape 的路径。\n"
            "  三种情形都会给出一个形状完全正常的结果文件。")

    key = _variant_key(levers)
    for existing in store["variants"]:
        if _variant_key(existing["levers"]) == key:
            existing["time_us"] = time_us
            existing["note"] = note
            if variant_hash:
                existing["variant_hash"] = variant_hash
            return existing

    entry: Dict[str, Any] = {
        "levers": sorted(set(levers)), "time_us": time_us, "note": note}
    if variant_hash:
        entry["variant_hash"] = variant_hash
    store["variants"].append(entry)
    return entry


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------


def price_levers(store: Dict[str, Any]) -> Dict[str, Any]:
    """给每个单项手段定价，并分析成对耦合。"""
    baseline_us = float(store["baseline"]["time_us"])
    noise = float(store["noise_floor_pct"])
    variants = store.get("variants", [])

    if not variants:
        raise LeverError(
            "没有任何变体可分析。先登记至少一个单项手段的测量：\n"
            "  lever_probe.py add --store <file> --lever <名字> --time-us <耗时>"
        )

    # 单项：只含一个手段的变体
    singles: Dict[str, Dict[str, Any]] = {}
    for entry in variants:
        levers = entry["levers"]
        if len(levers) != 1:
            continue
        name = levers[0]
        gain = _gain_pct(baseline_us, float(entry["time_us"]))
        significant = _significant(gain, noise)
        if gain > 0 and significant:
            verdict = "有上限可追"
            meaning = (f"把这一项整个删掉最多省 {gain:.2f}%。"
                       f"优化它的收益**不会超过**这个数，据此决定值不值得投入")
        elif gain > 0:
            verdict = "测不出来"
            meaning = (f"删掉它只省 {gain:.2f}%，落在噪声底 {noise:.2f}% 以内。"
                       "这个方向没有可测量的空间，不要在这里投迭代")
        else:
            verdict = "反而更慢"
            meaning = (f"删掉它反而慢了 {-gain:.2f}%。通常意味着它的存在"
                       "改善了别的东西（如对齐、缓存局部性），或这次测量有问题")
        singles[name] = {
            "lever": name, "time_us": float(entry["time_us"]),
            "gain_pct": gain, "significant": significant,
            "verdict": verdict, "meaning": meaning,
            "note": entry.get("note", ""),
        }

    # 成对：同时含两个手段的变体，且两项各自都有单项测量
    couplings: List[Dict[str, Any]] = []
    for entry in variants:
        levers = sorted(set(entry["levers"]))
        if len(levers) != 2:
            continue
        a, b = levers
        if a not in singles or b not in singles:
            couplings.append({
                "levers": levers, "status": "incomplete",
                "meaning": (f"缺少 {a if a not in singles else b} 的单项测量，"
                            "无法分离耦合效应。先补单项再分析"),
            })
            continue

        observed = _gain_pct(baseline_us, float(entry["time_us"]))
        additive = singles[a]["gain_pct"] + singles[b]["gain_pct"]
        interaction = observed - additive

        if not _significant(interaction, noise):
            status = "独立"
            meaning = (f"两项合计 {observed:.2f}%，与各自相加 {additive:.2f}% "
                       "的差落在噪声内，可按相互独立规划")
        elif interaction < 0:
            status = "耦合抵消"
            meaning = (f"合计只有 {observed:.2f}%，低于相加的 {additive:.2f}%。"
                       "两项在抢同一份时间，真实可得是 "
                       f"{observed:.2f}% 而不是 {additive:.2f}% —— "
                       "按相加去规划会对不上账")
        else:
            status = "协同"
            meaning = (f"合计 {observed:.2f}% 高于相加的 {additive:.2f}%。"
                       "一起做比分开做更划算，应作为原子组同批落地")

        couplings.append({
            "levers": levers, "status": status,
            "observed_pct": observed, "additive_pct": additive,
            "interaction_pct": interaction, "meaning": meaning,
        })

    ranked = sorted(singles.values(), key=lambda d: -d["gain_pct"])
    actionable = [s for s in ranked if s["significant"] and s["gain_pct"] > 0]

    # 定价的主要产出之一是「别做什么」
    dead = [s["lever"] for s in ranked
            if s["gain_pct"] <= 0 or not s["significant"]]

    # 分类要两条腿，所以在这里合进来。
    #
    # not_worth_pursuing 据分类过滤：一个 coupled 的手段（单独看不出收益、
    # 但别人依赖它）此前会落进这张「不要再投」的名单 —— 那不是漏报，是
    # **反向报错**：工具主动告诉你不要在一个承重的方向上投入。
    classes = classify_levers(store)
    dead = [name for name in dead
            if classes["levers"].get(name, {}).get("class") != "coupled"]
    coupled = [name for name, info in classes["levers"].items()
               if info["class"] == "coupled"]

    return {
        "baseline_us": baseline_us,
        "noise_floor_pct": noise,
        "levers": ranked,
        "couplings": couplings,
        "classification": classes,
        "ceiling_pct": actionable[0]["gain_pct"] if actionable else 0.0,
        "actionable": [s["lever"] for s in actionable],
        "not_worth_pursuing": dead,
        "must_keep": coupled,
        "probe_only": True,
    }


def missing_probes(store: Dict[str, Any]) -> List[str]:
    """指出还缺哪些测量才能把耦合分析补全。"""
    singles = {v["levers"][0] for v in store.get("variants", [])
               if len(v["levers"]) == 1}
    pairs = {tuple(sorted(set(v["levers"]))) for v in store.get("variants", [])
             if len(set(v["levers"])) == 2}

    out: List[str] = []
    for a, b in itertools.combinations(sorted(singles), 2):
        if (a, b) not in pairs:
            out.append(f"{a} + {b}（两项的联合测量，用于分离耦合效应）")
    return out


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> int:
    path = Path(args.store)
    try:
        if args.baseline_us is not None:
            if path.exists() and not args.force:
                print(f"错误: {path} 已存在；要重建请加 --force", file=sys.stderr)
                return 2
            if args.noise_floor is None:
                print("错误: 建立基线时必须给 --noise-floor"
                      "（用 noise_floor.py 量出来的那个数）", file=sys.stderr)
                return 2
            store = new_store(args.baseline_us, args.noise_floor, args.label,
                              variant_hash=args.variant_hash or "")
            save_store(path, store)
            print(f"已建立基线 {args.baseline_us} us，噪声底 {args.noise_floor}%")
            print(f"提醒: {store['warning']}")
            return 0

        store = load_store(path)
        if args.time_us is None:
            print("错误: 登记变体需要 --time-us", file=sys.stderr)
            return 2
        entry = add_variant(store, list(args.lever or []),
                            args.time_us, args.note,
                            variant_hash=args.variant_hash or "")
        save_store(path, store)
        gain = _gain_pct(float(store["baseline"]["time_us"]), args.time_us)
        print(f"已登记 {'+'.join(entry['levers'])}: "
              f"{args.time_us} us（{gain:+.2f}%）")
        return 0
    except LeverError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        store = load_store(Path(args.store))
        report = price_levers(store)
        pending = missing_probes(store)
    except LeverError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        report["missing_probes"] = pending
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"基线 {report['baseline_us']} us，噪声底 {report['noise_floor_pct']:.2f}%")
    print()
    print("— 单项定价（删掉它最多能省多少）—")
    for item in report["levers"]:
        print(f"  {item['lever']:<24} {item['gain_pct']:+7.2f}%  [{item['verdict']}]")
        print(f"      {item['meaning']}")
    print()

    if report["couplings"]:
        print("— 耦合分析 —")
        for item in report["couplings"]:
            print(f"  {' + '.join(item['levers'])}: {item['status']}")
            print(f"      {item['meaning']}")
        print()

    classes = report["classification"]
    if classes["levers"]:
        print("— 两条腿合起来的分类 —")
        if classes["all_pct"] is None:
            print(f"  （缺「全删」那次测量，边际一列判不了。补：add "
                  + " ".join(f"--lever {x}" for x in classes["universe"]) + "）")
        for name in sorted(classes["levers"]):
            info = classes["levers"][name]
            single = ("   —  " if info["single_pct"] is None
                      else f"{info['single_pct']:+6.2f}%")
            marginal = ("   —  " if info["marginal_pct"] is None
                        else f"{info['marginal_pct']:+6.2f}%")
            print(f"  {name:<24} 单项 {single}  边际 {marginal}  "
                  f"[{info['class']}]")
            print(f"      {info['meaning']}")
            for miss in info["missing_measurements"]:
                print(f"      还缺: {miss}")
        print()

    print("— 结论 —")
    if report["must_keep"]:
        print(f"  必须保留: {', '.join(report['must_keep'])}")
        print("      单独删掉省不出时间，但别人依赖它。只看单项读数会把它")
        print("      判成「测不出来」并列进「不要再投」—— 那不是漏报，")
        print("      是工具主动告诉你不要在一个承重的方向上投入。")
    if report["actionable"]:
        print(f"  值得投入: {', '.join(report['actionable'])}")
        print(f"  最高上限: {report['ceiling_pct']:.2f}%"
              "（这是优化该项能拿到的收益上界，不是预期收益）")
    else:
        print("  没有任何单项的收益超出噪声底。")
        print("  这本身是个结论：当前架构下这些手段都没有空间，")
        print("  应当回到 M0 重新考虑架构，而不是继续在这一层找。")
    if report["not_worth_pursuing"]:
        print(f"  不要再投: {', '.join(report['not_worth_pursuing'])}")

    if pending:
        print()
        print("— 还缺的测量 —")
        for item in pending:
            print(f"  · {item}")

    print()
    print("注意: 以上变体多数会产出错误结果，它们是定价探针而非候选，")
    print("      不得上线，也不得进入接受判定。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="手段定价与消融：在投入之前先知道这条路最多值多少",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="登记基线或一次变体测量")
    p_add.add_argument("--store", required=True)
    p_add.add_argument("--baseline-us", type=float, default=None,
                       help="给了就是建立基线")
    p_add.add_argument("--noise-floor", type=float, default=None,
                       help="实测噪声底，建立基线时必填")
    p_add.add_argument("--label", default="baseline")
    p_add.add_argument("--lever", action="append", default=[],
                       help="本变体关掉的手段，可重复给以表示组合")
    p_add.add_argument("--time-us", type=float, default=None)
    p_add.add_argument("--note", default="")
    p_add.add_argument("--variant-hash", default=None,
                       help="这个变体的代码指纹（campaign.py 的 "
                            "directory_fingerprint，或 git sha）。"
                            "与基线相同时拒绝登记 —— 没改过的变体测出来的"
                            "差值全是噪声，而它会被当成这个环节的价格")
    p_add.add_argument("--force", action="store_true")
    p_add.set_defaults(func=cmd_add)

    p_an = sub.add_parser("analyze", help="出定价表与耦合分析")
    p_an.add_argument("--store", required=True)
    p_an.set_defaults(func=cmd_analyze)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
