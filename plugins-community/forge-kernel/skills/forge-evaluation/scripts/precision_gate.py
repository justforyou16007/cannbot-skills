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
"""精度判据：候选离数学真值多远，对比参考自己离真值多远。

为什么不能用容差比对
--------------------
现在的做法是 `allclose(候选, baseline, rtol=1e-3, atol=1e-3)`。这个判据
**分不出两件完全不同的事**：

  诚实实现   在参考舍入的地方舍入，只是归约顺序不同
  降精度换速度   用 bf16 累加、tf32 相乘、把 fp32 中间量砍成 fp16

1e-3 的容差下两者都过。把容差收紧会把前一类误杀 —— 归约顺序不同本来就会
产生这个量级的差。区分它们要换一个量：**对高精度真值的误差比**。

  truth    参考语义在 float64 下重算的结果
  e_ref    参考自己的输出（原生 dtype）相对 truth 的相对 L2 误差
  e_cand   候选输出在同一组输入上的同一个量
  判失败   e_cand > K · e_ref + max(FLOOR_ULPS · u(输出 dtype), FLOOR_REL)

地板项是必须的：参考本身精确时（一次拷贝、一次 gather，e_ref ≈ 0）
没有地板会把每一个「舍入了一次」的诚实实现全判失败。

还有一件事：锚点不会随代际漂移
------------------------------
容差比对的锚是 baseline 的输出。而 `promote` 之后 baseline 变成上一代产物，
于是锚点跟着移动 —— **每代允许再漂 1e-3，N 代之后与数学真值的距离无界，
而每一代都显示「精度通过」**。

误差比的锚是 float64 真值，它由参考**语义**算出来，与代际无关。所以这条
判据同时修掉了自我提升链上的精度漂移。

需要什么
--------
一个能在 float64 下跑的参考语义。我们的 baseline 是装好的 OPP 二进制，
它跑不了 float64 —— 所以 `call_spec` 要给一个 `torch_reference`：
算子语义的 PyTorch 表达（表达式字符串或 `模块:函数` 形式的可调用对象）。

拿不到它时**报告 unavailable，不是报告通过**。精度判据缺失时静默放行，
等于把「没查」说成「查过且通过」—— 而算错的 kernel 再快也无意义。

常数的出处
----------
`K_RATIO = 3` 与 `FLOOR_REL = 2^-17` 取自论文实现的同名判据。注意它那组
常数是在 B200 上、跨 14 个问题 1956 个读数标定出来的 —— **换到昇腾需要
重新标定**。`FLOOR_ULPS` 我们没有读到上游的取值，默认取 1.0（输出 dtype
的一个舍入单位），同样需要标定。在标定之前，这条判据的**方向**是对的
（它能分出降精度），但**阈值**只是一个起点。

用法:
    # 从评测产物算误差比（需要 torch）
    python3 precision_gate.py --inputs ./inputs/shape_s0 \\
        --baseline-output ./baseline_profiling/shape_s0/baseline_output_s0.pt \\
        --candidate-output ./evolved_profiling/shape_s0/evolved_output_s0.pt \\
        --torch-reference "torch.nn.functional.layer_norm(a + b, (H,))" \\
        --output ./forge_work/precision.json

    # 只看判据（不需要 torch，用于标定与自检）
    python3 precision_gate.py --check-criterion --e-ref 1e-7 --e-cand 2e-6 \\
        --dtype float32

退出码: 0 = 通过; 1 = 不通过; 2 = 用法错误或判不了（含 unavailable）
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: 候选的误差相对参考误差允许放大多少倍。
#:
#: 取自论文实现的 K_RATIO。它的标定结论是：诚实实现（参考在 fp32 下重跑、
#: 换一种归约顺序、仓里最好的榜单解）最坏只到限额的 0.25，而用 bf16/fp16
#: 累加、tf32 矩阵乘、fp8 量化输入的同一批参考，在它移动过的 745 个读数里
#: 被抓住 643 个 —— 其中 194 个是官方容差放过的。
K_RATIO = 3.0

#: 相对地板。参考精确时（e_ref ≈ 0）不设地板会把每个舍入一次的诚实实现误杀。
#: 2^-17 ≈ 7.6e-6，取自论文实现的 FLOOR_REL。
FLOOR_REL = 2.0 ** -17

#: 地板的另一半：输出 dtype 的多少个舍入单位。
#:
#: **这个值我们没有读到上游的取值**，默认取 1.0（一个舍入单位）。
#: 和 K_RATIO、FLOOR_REL 一样，换到昇腾都需要重新标定 —— 标定方法见
#: 下面 calibration_note()。
FLOOR_ULPS = 1.0

#: 各 dtype 的舍入单位 u = 2^-(有效位数)。
#:
#: 用于把地板表达成「输出 dtype 自己的精度」而不是一个绝对数：一个 fp16
#: 输出本来就只有 11 位有效位，拿 fp32 的地板去卡它没有意义。
UNIT_ROUNDOFF = {
    "float64": 2.0 ** -53,
    "float32": 2.0 ** -24,
    "bfloat16": 2.0 ** -8,
    "float16": 2.0 ** -11,
}


class PrecisionError(Exception):
    """精度判据无法执行。"""


# ---------------------------------------------------------------------------
# 判据本身（纯算术，不依赖 torch）
# ---------------------------------------------------------------------------

def unit_roundoff(dtype_name: str) -> float:
    """输出 dtype 的舍入单位。认不出来时按 float32 估，并由调用方记下这件事。"""
    key = str(dtype_name).replace("torch.", "").strip()
    return UNIT_ROUNDOFF.get(key, UNIT_ROUNDOFF["float32"])


def error_floor(dtype_name: str, floor_ulps: float = FLOOR_ULPS,
                floor_rel: float = FLOOR_REL) -> float:
    """误差地板：输出 dtype 的若干个舍入单位，且不低于相对地板。"""
    return max(floor_ulps * unit_roundoff(dtype_name), floor_rel)


def judge_errors(e_ref: Optional[float], e_cand: Optional[float],
                 dtype_name: str, k_ratio: float = K_RATIO,
                 floor_ulps: float = FLOOR_ULPS,
                 floor_rel: float = FLOOR_REL) -> Dict[str, Any]:
    """误差比判据。这是整个脚本里唯一决定对错的地方。

    三态，**不含「缺失即通过」**：
      pass         e_cand 在限额内
      fail         超限 —— 候选的误差比参考大得不像是舍入顺序造成的
      unavailable  算不出 e_ref 或 e_cand（比如没有参考语义、truth 跑不出来）
    """
    floor = error_floor(dtype_name, floor_ulps, floor_rel)
    if e_cand is None or e_ref is None:
        return {
            "verdict": "unavailable",
            "e_ref": e_ref, "e_cand": e_cand,
            "limit": None, "floor": floor, "k_ratio": k_ratio,
            "dtype": dtype_name,
            "reason": ("算不出误差比：缺少 "
                       + ("参考侧误差" if e_ref is None else "")
                       + ("、" if e_ref is None and e_cand is None else "")
                       + ("候选侧误差" if e_cand is None else "")
                       + "。缺失**不等于通过** —— 算错的 kernel 再快也无意义"),
        }
    if not math.isfinite(e_cand):
        return {
            "verdict": "fail",
            "e_ref": e_ref, "e_cand": e_cand,
            "limit": k_ratio * e_ref + floor, "floor": floor,
            "k_ratio": k_ratio, "dtype": dtype_name,
            "reason": ("候选侧误差是 inf/nan —— 真值有限而候选不是，"
                       "说明中间量溢出了（典型是 fp16 中间量）。"
                       "这是精度失败，不是数值噪声"),
        }
    limit = k_ratio * e_ref + floor
    headroom = (e_cand / limit) if limit > 0 else float("inf")
    return {
        "verdict": "pass" if e_cand <= limit else "fail",
        "e_ref": e_ref, "e_cand": e_cand,
        "limit": limit, "floor": floor, "k_ratio": k_ratio,
        "dtype": dtype_name,
        "headroom_ratio": round(headroom, 4),
        "reason": (
            f"候选相对误差 {e_cand:.3e}，参考 {e_ref:.3e}，"
            f"限额 {limit:.3e}（= {k_ratio}×参考 + 地板 {floor:.3e}）"
            + ("" if e_cand <= limit else
               f"。超限 {headroom:.2f} 倍 —— 这个幅度不像是归约顺序不同"
               "造成的，更像是某处降了精度（bf16 累加 / tf32 相乘 / "
               "fp32 中间量砍成 fp16）")),
    }


def calibration_note() -> str:
    """标定方法。常数没标定之前要把这段话带在报告里。"""
    return (
        f"K_RATIO={K_RATIO}、FLOOR_REL={FLOOR_REL:.3e} 取自论文实现，"
        f"FLOOR_ULPS={FLOOR_ULPS} 是本脚本的默认值（未读到上游取值）。"
        "三者都在 B200 上标定或未标定，**换到昇腾需要重新标定**："
        "做法是拿一批**诚实**实现（参考在 fp32 下重跑、换归约顺序、"
        "已知正确的历史版本）测出 e_cand/limit 的分布，再拿一批**故意降精度**的"
        "（bf16 累加、fp16 中间量）测同一个量，K 取能把两组分开的值。"
        "标定之前这条判据的方向是对的，阈值只是起点。")


# ---------------------------------------------------------------------------
# torch 管道（装载张量、升精度、算相对 L2）
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise PrecisionError(
            "需要 torch 才能装载评测产物并算误差。"
            "只想检查判据本身时用 --check-criterion（不需要 torch）") from exc
    return torch


def relative_l2(actual, truth) -> Optional[float]:
    """相对 L2 误差 ‖y − t‖ / ‖t‖。

    用相对 L2 而不是最大绝对差：降精度算术会让**每个元素**都动，
    范数对这种整体偏移敏感而对单点噪声不敏感。而最大绝对差正相反 ——
    它会被一个离群点主导，于是一次真实的整体降精度可能读起来还不如
    一个孤立的舍入差。
    """
    torch = _require_torch()
    a = actual.detach().to(torch.float64).flatten()
    t = truth.detach().to(torch.float64).flatten()
    if a.numel() != t.numel():
        return None
    denominator = torch.linalg.vector_norm(t).item()
    if denominator == 0.0:
        # 真值全零：改用绝对范数，否则相对误差无定义
        return torch.linalg.vector_norm(a - t).item()
    if not math.isfinite(denominator):
        return None
    value = torch.linalg.vector_norm(a - t).item() / denominator
    return value


def upcast_inputs(tensors: List[Any]) -> List[Any]:
    """把浮点输入升到 float64，整型与 fp8/fp4 原样保留。

    整型本来就精确；fp8/fp4 在 float64 里也是精确表示，升了没有意义。
    """
    torch = _require_torch()
    out = []
    for item in tensors:
        if hasattr(item, "dtype") and item.dtype.is_floating_point:
            out.append(item.to(torch.float64))
        else:
            out.append(item)
    return out


def load_tensor_list(path: Path) -> List[Any]:
    torch = _require_torch()
    try:
        obj = torch.load(str(path), weights_only=False, map_location="cpu")
    except OSError as exc:
        raise PrecisionError(f"无法装载 {path}: {exc}") from exc
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise PrecisionError(f"{path} 里不是张量也不是张量序列")


def load_inputs(input_dir: Path) -> Tuple[List[Any], Dict[str, Any]]:
    """按 evaluate_ops_direct 的落盘约定装载输入。"""
    torch = _require_torch()
    args: List[Any] = []
    index = 0
    while True:
        path = input_dir / f"input_{index}.pt"
        if not path.is_file():
            break
        args.append(torch.load(str(path), weights_only=True,
                               map_location="cpu"))
        index += 1
    kwargs: Dict[str, Any] = {}
    for path in sorted(input_dir.glob("kwarg_*.pt")):
        name = path.stem[len("kwarg_"):]
        kwargs[name] = torch.load(str(path), weights_only=True,
                                  map_location="cpu")
    if not args and not kwargs:
        raise PrecisionError(
            f"{input_dir} 下没有 input_*.pt / kwarg_*.pt。"
            "这些文件由 evaluate_ops_direct 生成输入时落盘")
    return args, kwargs


def run_reference(spec: str, args: List[Any],
                  kwargs: Dict[str, Any]) -> List[Any]:
    """在 float64 下跑参考语义。

    两种写法：
      模块:函数    形如 `mypkg.ref:layer_norm`，收 (args, kwargs)
      表达式       形如 `torch.nn.functional.layer_norm(a + b, (H,))`，
                   其中 a/b/... 绑定到 input_0/input_1/...，kwargs 按名绑定
    """
    torch = _require_torch()
    upcast_args = upcast_inputs(args)
    upcast_kwargs = {k: (v.to(torch.float64)
                         if hasattr(v, "dtype") and v.dtype.is_floating_point
                         else v)
                     for k, v in kwargs.items()}

    if ":" in spec and "(" not in spec.split(":", 1)[0]:
        module_name, _, func_name = spec.partition(":")
        import importlib  # noqa: PLC0415
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise PrecisionError(
                f"导入参考语义模块 {module_name} 失败: {exc}") from exc
        func = getattr(module, func_name, None)
        if func is None:
            raise PrecisionError(f"{module_name} 里没有 {func_name}")
        result = func(*upcast_args, **upcast_kwargs)
    else:
        names = "abcdefghijklmnop"
        env: Dict[str, Any] = {"torch": torch}
        for i, value in enumerate(upcast_args):
            env[f"input_{i}"] = value
            if i < len(names):
                env[names[i]] = value
        env.update(upcast_kwargs)
        try:
            # 参考语义是**我们自己**在 call_spec 里写的表达式，不是外部输入；
            # 它和算子定义一样属于任务描述的一部分。
            result = eval(spec, {"__builtins__": {}}, env)  # noqa: S307
        except Exception as exc:  # noqa: BLE001
            raise PrecisionError(
                f"参考语义求值失败: {exc}\n"
                f"  表达式: {spec}\n"
                f"  可用绑定: {sorted(k for k in env if k != 'torch')}") from exc

    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (list, tuple)):
        return list(result)
    raise PrecisionError("参考语义返回的不是张量也不是张量序列")


def evaluate(input_dir: Path, baseline_output: Path, candidate_output: Path,
             torch_reference: Optional[str],
             k_ratio: float = K_RATIO,
             floor_ulps: float = FLOOR_ULPS) -> Dict[str, Any]:
    """算出误差比并给出裁决。"""
    if not torch_reference:
        return {
            "schema_version": "1.0",
            "verdict": "unavailable",
            "reason": ("没有参考语义（call_spec 的 torch_reference）。"
                       "误差比需要一个能在 float64 下跑的参考语义 —— "
                       "装好的 OPP 二进制跑不了 float64。\n"
                       "  缺失不等于通过：这一轮的精度**没有被判过**，"
                       "而容差比对分不出「降精度换速度」。"),
            "calibration": calibration_note(),
        }

    args, kwargs = load_inputs(input_dir)
    truth = run_reference(torch_reference, args, kwargs)
    reference = load_tensor_list(baseline_output)
    candidate = load_tensor_list(candidate_output)

    per_output: List[Dict[str, Any]] = []
    worst: Optional[Dict[str, Any]] = None
    for index, truth_item in enumerate(truth):
        ref_item = reference[index] if index < len(reference) else None
        cand_item = candidate[index] if index < len(candidate) else None
        dtype_name = (str(cand_item.dtype) if cand_item is not None
                      else str(truth_item.dtype))
        e_ref = (relative_l2(ref_item, truth_item)
                 if ref_item is not None else None)
        e_cand = (relative_l2(cand_item, truth_item)
                  if cand_item is not None else None)
        item = judge_errors(e_ref, e_cand, dtype_name, k_ratio, floor_ulps)
        item["output_index"] = index
        per_output.append(item)
        rank = {"fail": 2, "unavailable": 1, "pass": 0}
        if worst is None or rank[item["verdict"]] > rank[worst["verdict"]]:
            worst = item

    assert worst is not None
    return {
        "schema_version": "1.0",
        # 取最坏的那一个输出：一个输出算错了，整个算子就是错的。
        "verdict": worst["verdict"],
        "reason": worst["reason"],
        "outputs": per_output,
        "anchor": "float64 真值（由参考语义重算，不随代际移动）",
        "torch_reference": torch_reference,
        "calibration": calibration_note(),
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行。独立出来是为了让文档命令门禁能拿到它。"""
    parser = argparse.ArgumentParser(
        description="精度判据：对高精度真值的误差比，而不是对 baseline 的容差")
    parser.add_argument("--inputs", type=Path, default=None,
                        help="该 shape 的输入目录（含 input_*.pt）")
    parser.add_argument("--baseline-output", type=Path, default=None)
    parser.add_argument("--candidate-output", type=Path, default=None)
    parser.add_argument("--torch-reference", default=None,
                        help="参考语义：`模块:函数` 或 PyTorch 表达式"
                             "（input_0/a、input_1/b… 为绑定）")
    parser.add_argument("--k-ratio", type=float, default=K_RATIO,
                        help=f"候选误差相对参考误差的允许倍数（默认 {K_RATIO}）")
    parser.add_argument("--floor-ulps", type=float, default=FLOOR_ULPS,
                        help=f"地板取输出 dtype 的多少个舍入单位"
                             f"（默认 {FLOOR_ULPS}）")
    parser.add_argument("--check-criterion", action="store_true",
                        help="只检查判据本身，不装载张量（不需要 torch）")
    parser.add_argument("--e-ref", type=float, default=None,
                        help="配合 --check-criterion：参考侧相对误差")
    parser.add_argument("--e-cand", type=float, default=None,
                        help="配合 --check-criterion：候选侧相对误差")
    parser.add_argument("--dtype", default="float32",
                        help="配合 --check-criterion：输出 dtype")
    parser.add_argument("--output", type=Path, default=None,
                        help="把报告写到文件，供 campaign.py record 引用")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.check_criterion:
        report = judge_errors(args.e_ref, args.e_cand, args.dtype,
                              args.k_ratio, args.floor_ulps)
        report["calibration"] = calibration_note()
    else:
        missing = [name for name, value in (
            ("--inputs", args.inputs),
            ("--baseline-output", args.baseline_output),
            ("--candidate-output", args.candidate_output)) if value is None]
        if missing:
            print(f"错误: 缺少 {missing}（或用 --check-criterion 只看判据）",
                  file=sys.stderr)
            return 2
        try:
            report = evaluate(args.inputs, args.baseline_output,
                              args.candidate_output, args.torch_reference,
                              args.k_ratio, args.floor_ulps)
        except PrecisionError as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 2

    if args.output:
        try:
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError as exc:
            print(f"错误: 无法写入 {args.output}: {exc}", file=sys.stderr)
            return 2

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        marks = {"pass": "通过", "fail": "不通过", "unavailable": "判不了"}
        print(f"精度判据  {marks[report['verdict']]}")
        print(f"  {report['reason']}")
        print(f"\n锚点      {report.get('anchor', 'float64 真值')}")
        print(f"标定      {report['calibration']}")

    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
