---
name: ops-profiling
description: NPU 性能采集与分析，融合 msprof 算子级瓶颈定位与 kernel-level 对比测试，用于采集算子性能数据、对比自定义算子 vs 标杆加速比、定位性能瓶颈并给出优化建议。当用户在算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU profiling"、"性能对比"、"加速比"等场景时触发。
---

# 上板性能采集与调优

在真实 NPU 上采集算子性能数据，系统化解读指标文件，判定性能是否达标，定位瓶颈类型，并给出可操作的优化建议。

本技能基于 **msprof** 工具链，统一入口为两个脚本：
- **`msprof_profile_run.sh`** — 性能采集（标准采集 / 对比测试 / 快速对比）
- **`msprof_perf_summary.py`** — 结果解析（瓶颈分析 / 对比报告生成）

| 工具 | 流程文档 | 用途 |
|------|----------|------|
| **`msprof_profile_run.sh`** | [`references/msprof-guide.md`](references/msprof-guide.md) | 统一采集入口：标准采集、对比测试、快速对比 |
| **`msprof_perf_summary.py`** | [`references/msprof-guide.md`](references/msprof-guide.md) | 统一解析入口：瓶颈分析、对比报告生成 |
| **`perf_summary.py`** | [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | msprof op 归档（需 msopprof） |

---

## 使用方式

### 1. 标准采集模式（深度瓶颈分析）

对单个可执行文件采集 7 组 aic-metrics + sample-based：

```bash
bash scripts/msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
    ./matmul_tutorial_mxfp4_pingpong 8448 4096 4096

# 解析结果
python3 scripts/msprof_perf_summary.py ./msprof_output/PROF_GROUP_* <ops_dir>
```

### 2. 对比测试模式（kernel-level 加速比）

对算子目录下的 `model.py` vs `model_new_ascendc.py` 做对比测试：

```bash
bash scripts/msprof_profile_run.sh --compare --output-dir=./output/GELU --warm-up=3 --device=0
```

输出：`performance.json` + `performance.log` + `perf_report.md`

### 3. 快速模式（单卡批量，一次 msprof 采全部 case）

对算子目录做快速对比测试，不采集 7 个 aic-metrics，**单卡单进程批量**：

- **asc**：`_measure_asc_batch_quick` 给每个 case 预建一个 `torch.npu.Stream()`，
  `with torch.npu.stream(_s):` 内跑该 case，一个进程顺序跑完全部 case，**只启动 1 次 msprof**；
  解析 task_time.csv 按 stream_id 归组还原每 case 耗时。失败时逐 case 回退。
- **ref**：`_measure_batch_reference_quick_cached` 批量 + reference-cache（model.py 不变时只测一次）。
- 两段测量均持 `_device_flock` 与同卡其他进化 lane 串行协调。
- 解析必须包含 `MIX_AIC/MIX_AIV/AI_CORE` 等计算 kernel 类型，否则 aicore 时长欠采集约 2.6-3x。

```bash
bash scripts/msprof_profile_run.sh --quick --output-dir=./output/GELU --warm-up=3 --device=0
```

输出：`performance.json`（含 `asc_mode: batch|per_case`）+ `performance.log` + `perf_report.md`

> **关于 torch.npu.stream 对 asc 生效的机制**：AscendC 自定义算子经 TORCH_LIBRARY 注册，
> host 侧启动宏 `EXEC_KERNEL_CMD`（kernel/utils/torch_kernel_helper.h）用
> `c10_npu::getCurrentNPUStream()` 绑定启动流，故 `with torch.npu.stream(_s):` 内调用 op，
> kernel 落在 `_s` 上，task_time.csv 可按 stream_id 归组到每个 case。

---

## 输入目录结构（对比/快速模式）

**传统模式（端到端自动开发）：**
```
{output_dir}/
├── model.py              # 参考 PyTorch 实现
├── model_new_ascendc.py  # AscendC 实现
├── <op_name>.json        # 测试用例（JSON Lines）
└── kernel/               # AscendC kernel + whl 包
```

**JSONL 模式（ascend-kernel 工程结构）：**
```
csrc/ops/<op>/test/
├── <op>_perf_cases.jsonl   # JSONL 用例
├── model.py                # 参考实现
├── model_new_ascendc.py    # AscendC 实现
└── kernel/                 # kernel 工程
```

### 用例加载优先级

1. **优先**：`<op>_perf_cases.jsonl`（标准 JSONL 命名）
2. **回退**：任意 `*.jsonl` 文件
3. **回退**：任意 `*.json` 文件（非 `.bak`，JSON Lines 格式）
4. **回退**：从 `model.py` 的 `get_input_groups()` 加载

---

## 输出格式（对比模式）

Markdown 报告包含：
- **对比表**：`Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比`
- **全量汇总**：用例数、平均加速比、自定义/标杆更优条数
- **按数据类型汇总**：分 dtype 的统计
- **简短分析**：整体趋势结论
- **深度瓶颈分析入口**：提供 msprof 深度分析命令

额外输出：
- `performance.json` — 结构化数据（含 geomean/mean/median/min/max 加速比）
- `performance.log` — 打屏日志

---

## 深度分析：msprof / msprof op

当性能不达标或需要根因分析时，使用深度分析流程。

### 选用哪个工具：决策树

1. **用户显式指定工具**
   - 指定 `msprof op` / msopprof → 加载 [`references/msprof-op-guide.md`](references/msprof-op-guide.md)
   - 指定 `msprof` → 加载 [`references/msprof-guide.md`](references/msprof-guide.md)

2. **用户未指定** — 探测环境：
   - 仅 `msopprof` 可用 → [`references/msprof-op-guide.md`](references/msprof-op-guide.md)
   - 仅 `msprof` 可用 → [`references/msprof-guide.md`](references/msprof-guide.md)
   - 两者皆可用 → 须向用户确认或按项目约定选用其一
   - 两者皆不可用 → 报错，提示检查 CANN / `ASCEND_HOME` 安装

> **需要指令级流水线时序图？** 上板采集（msprof / msopprof）只有 PMU 聚合计数器，无逐指令时序。若目标是查看 MTE2/Cube/Vec 流水线时序图、CV overlap、DoubleBuffer 效果或代码行映射，需走仿真模式 — 见 **ops-simulator** skill（`ops/ops-simulator/references/msopprof-simulator-guide.md`）。

### 参考资源

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | `msprof op`：构建 / 采集 / 归档 / 判定 / 瓶颈 / 回归 | 选用 `msprof op` 时 |
| [`references/msprof-guide.md`](references/msprof-guide.md) | `msprof`：构建 / 采集 / 归档 / **主 Bound 判定** / 瓶颈 | 选用 `msprof` 时 |
| [`references/csv_fields_reference.md`](references/csv_fields_reference.md) | CSV 字段定义与阈值 | 理解指标含义时 |
| [`references/optimization_quickref.md`](references/optimization_quickref.md) | 瓶颈类型与优化方法 | 定位瓶颈后 |

---

## 适用场景总览

| 场景 | 推荐命令 | 说明 |
|------|----------|------|
| 算子开发完成后的性能验收 | `msprof_profile_run.sh --compare` | 快速对比自定义算子 vs 标杆 |
| 性能问题定位 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 深度瓶颈分析 |
| 优化效果验证 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 对比优化前后的归档数据 |
| 算子生成阶段快速测试 | `msprof_profile_run.sh --quick` | 快速对比测试（1 轮采集，只获取时间） |
| Agent team 测试阶段 | `--quick` / `--compare` + 标准采集 | 先快速对比，不达标再深度分析 |
| 端到端自动开发 Phase 5 | `msprof_profile_run.sh --quick` | 集成到 tilelang2ascendc-ops-generator 流程（只获取加速比） |
| 进化优化前基线测试 | `msprof_profile_run.sh --quick` | 快速获取基线加速比 |
