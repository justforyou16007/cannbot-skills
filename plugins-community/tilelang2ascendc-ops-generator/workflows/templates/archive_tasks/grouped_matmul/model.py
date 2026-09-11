#!/usr/bin/env python3
"""GroupedMatmul 样例参考实现（torch golden）。

语义（cann-bench level3/grouped_matmul）：
对每个 expert g ∈ [0, E)，按 group_list（cumsum）取 token 行 rows_g=[c_{g-1}, c_g)：
    y[rows_g] = x[rows_g] @ weight[g] (+ bias[g])
x: [M, K]；weight: [E, K, N]（transpose_weight=false）或 [E, N, K]（true）；
bias: [E, N] 可选；y: [M, N]。空组（c_g == c_{g-1}）跳过。
split_item=0/1 → List[Tensor] 长度 E；2/3 → List 长度 1（[M, N]）。

数据生成忠实 kernel_eval.data.data_generator 语义：
- 数值区间 [lo, hi]：均匀分布；lo==hi → 常量填充
- [nan, nan]：均匀 [-1,1] 底 + 50% NaN mask
- [-inf, inf]：均匀 [-1,1] 底 + 前 5% -inf、后 5% +inf
"""
import json
import math
import os

import torch
# fp32 golden 必须走精确路径（同 conv2d/56 线）：默认 HF32 只有 11bit 尾数，
# 会把"真 fp32 kernel vs HF32 golden"的不对称变成系统性假警报
import torch_npu  # noqa: F401
torch.npu.matmul.allow_hf32 = False
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor = None, group_list=None,
                split_item: int = 0, transpose_weight: bool = False):
        M, K = x.shape
        E = weight.shape[0]
        N = weight.shape[1] if transpose_weight else weight.shape[2]

        if isinstance(group_list, torch.Tensor):
            ends = group_list.to(torch.int64).tolist()
        else:
            ends = list(group_list)
        starts = [0] + ends[:-1]

        y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
        x_f = x.float()
        for g in range(E):
            s, e = starts[g], ends[g]
            if s == e:
                continue
            w_g = weight[g].float()
            if transpose_weight:
                mm = torch.matmul(x_f[s:e], w_g.transpose(-2, -1))
            else:
                mm = torch.matmul(x_f[s:e], w_g)
            if bias is not None:
                mm = mm + bias[g].float().unsqueeze(0)
            y[s:e] = mm.to(x.dtype)

        if split_item in (0, 1):
            return [y[starts[g]:ends[g]] for g in range(E)]
        return [y]


def _gen_tensor(shape, dtype, value_range):
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[dtype]
    if value_range is None:
        return torch.rand(shape, dtype=dt)

    def to_f(v):
        if isinstance(v, str):
            return {"inf": float("inf"), "-inf": float("-inf"),
                    "nan": float("nan")}[v]
        return float(v)

    lo, hi = to_f(value_range[0]), to_f(value_range[1])
    if math.isnan(lo) and math.isnan(hi):
        t = torch.rand(shape, dtype=torch.float32) * 2 - 1
        t[torch.rand(shape, dtype=torch.float32) < 0.5] = float("nan")
        return t.to(dt)
    if lo == float("-inf") or hi == float("inf"):
        t = torch.rand(shape, dtype=torch.float32) * 2 - 1
        flat = t.flatten()
        n = max(1, flat.numel() // 20)
        flat[:n] = float("-inf")
        flat[-n:] = float("inf")
        return t.to(dt)
    if lo == hi:
        return torch.full(shape, lo, dtype=dt)
    return (torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo).to(dt)


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "grouped_matmul.jsonl")
    input_groups = []
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inputs = {i["name"]: i for i in json.loads(line)["inputs"]}
            x = _gen_tensor(inputs["x"]["shape"], inputs["x"]["dtype"],
                            inputs["x"].get("value_range"))
            w = _gen_tensor(inputs["weight"]["shape"],
                            inputs["weight"]["dtype"],
                            inputs["weight"].get("value_range"))
            bias = None
            if "bias" in inputs:
                bias = _gen_tensor(inputs["bias"]["shape"],
                                   inputs["bias"]["dtype"],
                                   inputs["bias"].get("value_range"))
            group_list = inputs["group_list"]["value"]
            split_item = inputs.get("split_item", {}).get("value", 0)
            transpose_weight = inputs.get("transpose_weight", {}).get(
                "value", False)
            input_groups.append([x, w, bias, group_list, split_item,
                                 transpose_weight])
    return input_groups


def get_init_inputs():
    return []
