# msopprof simulator 仿真采集（流水线时序图）

> 目标：在 **MindStudio Insight** 中查看算子的 **指令级流水线时序图**（MTE2/Cube/Vec/Scalar 各 pipe 的逐指令执行时序）。
> 适用于 Agent 生成的算子（TileLang 路径：torch extension 形态的 kernel 工程，含 `model_new_ascendc.py` + `kernel/`）。

---

## 什么时候必须用仿真模式

| 采集方式 | 数据内容 | MindStudio 中能否看流水线时序图 |
|---------|---------|------------------------------|
| 上板 `msopprof` / `msprof op` | 硬件 PMU **聚合计数器**（PipeUtilization、带宽、Cache 命中率等 CSV） | ❌ 只有聚合图表，无指令级时序 |
| **仿真 `msopprof simulator`** | 逐指令模拟执行，完整流水线时序（trace.json + 指令级 CSV） | ✅ MTE2 等流水线时序图、指令热度图、代码行映射 |

**结论**：只要目标是"看流水线时序图"，就必须走仿真模式。上板采集用于性能验收，仿真用于流水 overlap 分析（如 CV 并行、DoubleBuffer 是否生效、流水线气泡定位）。

---

## 前提条件

1. **msopprof 存在**：`$ASCEND_HOME_PATH/tools/msopt/bin/msopprof`（CANN 8.5+；老版本在 `tools/msopprof/bin/`）
2. **仿真器支持目标芯片**：`ls $ASCEND_HOME_PATH/x86_64-linux/devlib/simulator/`，确认有所需 SoC（如 `Ascend910B2`）
3. **kernel 带调试信息**（生成代码行映射的前提）：
   - **Agent 生成的算子**（torch extension）：检查 kernel 工程 `CMakeLists.txt` 是否 `CMAKE_BUILD_TYPE=Debug`（tilelang2ascend 模板默认强制 Debug，已满足）。验证方式：
     ```bash
     readelf -S <device_obj>.o | grep debug_info
     ```
   - **ops 仓库算子**（asc_opc 构建）：需在 `build_binary_opc_gen_task.sh` 中追加 `--op_debug_level=1`（参考 grouped_bias_add_grad 性能测试案例），否则仿真无法生成流水线映射数据
4. **kernel 的编译 SoC 与仿真 SoC 一致**（如都必须是 `Ascend910B2`，注意大小写，首字母大写）

---

## Step 1：编写单 case 运行脚本

在算子目录（含 `model_new_ascendc.py` 与 `<op>.json` 用例文件）下创建 `run_one_case.py`：

```python
#!/usr/bin/env python3
"""Run a single case of the AscendC impl once (msopprof simulator target)."""
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
CASE_IDX = 0  # ← 修改为要采集的 case 序号

sys.path.insert(0, str(HERE))
from model_new_ascendc import ModelNew  # noqa: E402

def main():
    with open(HERE / "<op_name>.json") as f:      # ← 替换为实际用例文件名
        case = json.loads(f.readlines()[CASE_IDX])
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    args = []
    for spec in case["inputs"]:
        if spec["type"] == "tensor":
            t = torch.randn(spec["shape"], dtype=torch.float32)
            args.append(t.to(dtype_map[spec["dtype"]]).npu())
        else:                                      # attr（如 scale）
            args.append(spec["value"])

    model = ModelNew()
    model(*args)                    # 第一次调用（含 padding buffer 初始化）
    torch.npu.synchronize()
    model(*args)                    # 稳态调用
    torch.npu.synchronize()
    print("case", CASE_IDX, "ok")

if __name__ == "__main__":
    main()
```

**case 选择建议**：
- 仿真逐指令执行，**远慢于上板**。优先选中小规模 case（单 kernel 仿真耗时控制在秒级~分钟级）
- 想看多核负载/多 block 流水，选能切出多个 block 的 case；只想看单核内 CV overlap，选最小 case 即可
- 多核 case 会生成巨大的 `visualize_data.bin`（实测 18 个 AIC block → 415MB），注意磁盘

**先在真机上冒烟验证**（不通过则仿真必然失败）：

```bash
source $ASCEND_HOME_PATH/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<空闲NPU>
python3 run_one_case.py
```

---

## Step 2：确认 kernel 名

`--kernel-name` 用于过滤 wrapper 中 padding/拷贝等额外 kernel，只采集目标 kernel。kernel 名通常带 dtype 后缀：

```bash
# 从已构建的静态库中查
nm -C kernel/build/lib/libkernels.a | grep " T " | grep -v launch
# 典型输出：scaled_dot_product_attention_bf16 / _fp16 / _fp32
```

**注意**：`--kernel-name` 的 dtype 后缀必须与所选 case 的 dtype 一致，否则匹配不到任何 kernel。

---

## Step 3：仿真采集

```bash
cd <算子目录>
source $ASCEND_HOME_PATH/set_env.sh
export PATH=$ASCEND_HOME_PATH/tools/msopt/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES=<空闲NPU>

msopprof simulator --soc-version=Ascend910B2 \
  --kernel-name=<op>_<dtype> \
  --launch-count=1 \
  --timeout=120 \
  --output=./msopprof_sim \
  --application="python3 run_one_case.py"
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--soc-version=Ascend910B2` | 必须与仿真器目录名、kernel 编译 SoC 完全一致，**大小写敏感** |
| `--kernel-name=<op>_<dtype>` | 只采集目标 kernel，过滤 padding/拷贝等辅助 kernel |
| `--launch-count=1` | 只采集 1 次 launch，控制输出体积 |
| `--timeout=120` | 仿真慢，给足超时（分钟） |
| `--application="python3 ..."` | python 脚本需加引号作为整体传入 |

---

## Step 4：查看结果

### 输出结构

```
msopprof_sim/OPPROF_<timestamp>/
├── dump/
│   ├── aicore_binary.o          # kernel 二进制
│   └── object_dump.txt
└── simulator/
    ├── visualize_data.bin       # ← MindStudio Insight 打开这个文件！
    ├── trace.json               # 顶层流水线时序
    ├── core0.cubecore0/         # 每个 AIC 的 cube 核追踪
    ├── core0.veccore0/          # 每个 AIC 的 vector 核追踪
    │   ├── trace.json
    │   ├── core0.veccore0_instr_exe.csv   # 指令级执行追踪
    │   └── core0.veccore0_code_exe.csv    # 代码执行统计
    └── ...
```

### MindStudio Insight 可视化

打开 `simulator/visualize_data.bin`，可查看：
- **流水线时序图**：各 pipe（MTE2/MTE3/Cube/Vec/Scalar）逐指令时间线，用于分析 CV overlap、DoubleBuffer 效果、流水线气泡
- **指令热度图**：定位热点指令
- **代码行映射**：指令 ↔ kernel 源码行（依赖 Debug 构建的 DWARF 信息）

### 命令行快速查看

```bash
# 指令级执行追踪
cat msopprof_sim/OPPROF_*/simulator/core0.veccore0/core0.veccore0_instr_exe.csv
```

---

## 常见问题

| 问题 | 原因与解决 |
|------|-----------|
| `--soc-version is invalid` | SoC 名大小写错误。必须与 `$ASCEND_HOME_PATH/x86_64-linux/devlib/simulator/` 下目录名一致（如 `Ascend910B2` 不是 `ascend910b`） |
| 采集结果为空 / 没有目标 kernel | `--kernel-name` 的 dtype 后缀与 case dtype 不匹配；或不加 `--kernel-name` 先全量采集确认实际 kernel 名 |
| Insight 中无代码行映射 | kernel 未带调试信息构建。torch extension 工程检查 CMAKE_BUILD_TYPE=Debug；ops 仓库需 `--op_debug_level=1` |
| 仿真运行超时 | case 太大。换小 case，或加大 `--timeout` |
| visualize_data.bin 过大（数百 MB） | 多核多 block 导致。选更小 case、`--launch-count=1` |
| 上板采集后 Insight 看不到流水线 | 上板只有 PMU 聚合计数器，无指令级时序。必须用仿真模式 |

---

## 完整示例（13_ScaledDotProductAttention，CANN 8.5.1 + 910B2）

```bash
cd workspace/13_ScaledDotProductAttention
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export PATH=/usr/local/Ascend/cann-8.5.1/tools/msopt/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES=2

# case 12: bf16 Q[3,6,59,64] K/V[3,6,65,64]
msopprof simulator --soc-version=Ascend910B2 \
  --kernel-name=scaled_dot_product_attention_bf16 \
  --launch-count=1 --timeout=120 \
  --output=./msopprof_sim \
  --application="python3 run_one_case.py"

# 结果：msopprof_sim/OPPROF_*/simulator/visualize_data.bin → MindStudio Insight
```

---

## 相关资源

| 文件 | 内容 |
|------|------|
| [`msopprof-simulator-ops-transformer-guide.md`](msopprof-simulator-ops-transformer-guide.md) | **ops-transformer 仓（CANN 仓）算子仿真采集**：build.sh --pkg → aclnn 测试程序 → msopprof simulator，含 `--op_debug_level=1` 注入细节 |
| [`pipeline-bubble-analysis.md`](pipeline-bubble-analysis.md) | 指令流水空泡分类、因果归因、周期性模式检测（仿真 trace 的下游分析） |
| [`../../ops-profiling/references/msprof-op-guide.md`](../../ops-profiling/references/msprof-op-guide.md) | 上板 msprof op 采集（PMU 聚合指标、瓶颈判定） |
| [`../../ops-profiling/references/msprof-guide.md`](../../ops-profiling/references/msprof-guide.md) | msprof 深度瓶颈分析 |
| [`../../ops-profiling/references/optimization_quickref.md`](../../ops-profiling/references/optimization_quickref.md) | 从流水线时序发现瓶颈后的优化方法 |
