---
id: P102
bottlenecks: [scalar_compute, compute_bound]
op_families: [cv_fusion, matmul, attention, special]
complexity: L1
conflicts_with: []
synergizes_with: [P29, P46, P87, P92]
requires: []
has_preconditions: true
has_playbook: true
quantified_gain:
  - shape: "GDR B=1,T=64,Hk=4,Hv=4,fp16"
    baseline_us: 3650.0
    optimized_us: 261.0
    speedup: 14.0
    source: "gated_delta_rule_cv rewrite (scalar+VEC → MIX_AIC_1_1)"
  - shape: "GDR geomean over T=64..4096"
    baseline_us: 1.0
    optimized_us: 1.0
    speedup: 8.85
    source: "gated_delta_rule_cv vs PyTorch reference, wall-clock 20 iters"
---

# P102: 标量/向量模拟矩阵运算 → MIX CV 融合重写

## 核心思想

当 kernel 用**标量循环或纯 Vector 指令模拟矩阵运算**（dot product、matmul、三角求解、链式矩阵乘加）时，P94/P95 的向量化收益会封顶——Vector 单元本就不是为矩阵运算设计的。此时唯一数量级收益的路径是**架构级重写为 MIX CV 融合**：Cube（AIC）承担全部 matmul（Mmad），Vector（AIV）承担 elementwise/decay/mask/状态更新，中间结果经 GM workspace 交接，`SyncAll<false>()` 做全局 barrier。

实证（gated_delta_rule，910B2C）：scalar/VEC 实现优化到 3.3x 后仍比参考慢 17x（scalar 做矩阵运算是原罪）→ MIX_AIC_1_1 重写后 **geomean 8.85x vs PyTorch 参考**，单 stage 对比：AIV 标量三角求解 98µs/chunk → Cube Neumann 链 11 个 Mmad（见 P103）。

## 适用判定（何时选这张卡而不是 P94/P95）

```
kernel 内层循环是以下形态之一？
  ├─ for(i) for(j) acc += A[i][k] * B[k][j]        (标量 matmul/dot)
  ├─ for(i) for(j) for(k) ...                      (嵌套标量乘加，维度 ≥ 64)
  ├─ for(i) for(j<i) solve/linsolve 递推           (三角求解、前向替换)
  └─ Vector 指令拼出的"伪 matmul"（Mul+ReduceSum 逐行）
    │
    ├─ 是 → 矩阵维度 ≥ 64×64？
    │   ├─ 是 → **选 P102（CV 融合重写）**，P94/P95 只做收尾的 elementwise 部分
    │   └─ 否（< 64）→ P94/P95 向量化即可，Cube 启动开销不划算
    └─ 否（纯 elementwise/归约）→ P94/P95
```

## 执行入口（SOP 在 Playbook，调试手册在 references）

- **完整实操 SOP（Step 0-6，强制执行）**：`playbooks/P102_scalar_matmul_to_cv_fusion_rewrite.md`
  - Step 0：必读 mix-cv-fusion-guide.md
  - Step 1-2：定位 + 改造计划表
  - Step 3：**Phase 0 纯 AIC 单路径 derisk（禁止跳过直接写 MIX）** + canonical template + 自检
  - Step 4：MIX 骨架 + 配对探针（五条铁律）
  - Step 5：逐 stage 增量 + 端到端验证
  - Step 6：失败归因纪律（R1-R6 分类 + "平台不支持"结论的证据标准）
- **MIX 机制与 hang 排查手册**：`references/mix-cv-fusion-guide.md`（R1-R6 根因分类、四阶段流程、双缓冲决策矩阵、调试工具箱）

## 关键约束（节点描述必带）

- **四阶段不可跳步**：AIC_ONLY 单 matmul 验证 → MIX 骨架 → 配对探针 → 逐 stage 增量。直接写全双工 MIX 是最高频失败模式。
- **hang/NaN ≠ 平台不支持**：MIX 在 910B2C 已实证可行（8.85x）。下"环境不支持"结论前必须满足 playbook Step 6 的四条证据标准（最小探针失败 + 清 build 缓存 + 换干净 NPU + 具体报错原文）。
- **build 缓存陷阱**：改代码必须 `rm -rf build`，否则"改了没生效"造成假诊断。
- **单 chunk 不算验证**：状态依赖路径必须 T≥128 多 chunk 用例。

## 常见陷阱

⚠️ **改了两样东西（launch type + 参数）时，hang 不一定归因于最后改的那个**——AIC_ONLY 编译运行正常 ⇒ kernel 逻辑无问题，launch 模式是病根，不要回退参数空转
⚠️ **build 缓存**：setup.py/构建脚本检测到 .so 存在会跳过 cmake——**改代码必须 `rm -rf build`**，否则"改了没生效"造成假诊断
⚠️ **带病 NPU**：被前一次 hang kernel 污染的设备会让 cross-core barrier 失效，后续输出非确定性 NaN（位置逐 run 变化）——同 binary 换一块干净 NPU PASS ⇒ 卡问题，与代码无关
⚠️ **单 chunk 不算验证**：状态依赖路径（S 更新、跨 chunk scale）在单 chunk 下恰好正确，必须 T≥128 多 chunk 用例
⚠️ **诊断前先 torch 算法仿真**：纯 torch 复刻 kernel 的分解公式再对参考，不花编译时间就能抓公式错误

## 代码搜索关键词

```bash
# 检测标量/向量模拟矩阵运算（P102 适用信号）
grep -nE "for\s*\(.*\+\+.*\)\s*\{?\s*for\s*\(" kernel/*.cpp          # 嵌套标量循环
grep -nE "reinterpret_cast.*GetPhyAddr|\.GetPhyAddr\(\)" kernel/*.cpp  # raw pointer 标量计算
grep -nE "Mul\(.*\).*ReduceSum|ReduceSum" kernel/*.cpp                  # Vector 拼伪 matmul

# 检测是否已是 MIX 架构（已重写则不适用）
grep -nE "KERNEL_TYPE_MIX|ASCEND_IS_AIC|Mmad" kernel/*.cpp
```

## 来源

- gated_delta_rule CV 融合重写（a5_ops_slim/workspace/gated_delta_rule_cv）：scalar/VEC 3.3x 且比参考慢 17x → MIX_AIC_1_1 重写 geomean **8.85x**（T=64 时 14.0x），精度全 shape PASS（MERE < 9.766e-04）。方法论见该目录 docs/MIX_DEBUG_WORKFLOW.md
