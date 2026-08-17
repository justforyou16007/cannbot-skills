# Ops-Transformer 仓仿真流水线采集（CANN 仓库算子）

> 目标：对 **ops-transformer**（CANN 仓 `attention/`、`ffn/`、`moe/` 等）中的算子，通过 `msopprof simulator` 采集指令级流水线时序，在 MindStudio Insight 中查看 **MTE2/Cube/Vec 流水线时序图**。

## 与 torch extension 路径的区别

| 维度 | torch extension（Agent 生成） | ops-transformer 仓（CANN 仓） |
|------|------------------------------|------------------------------|
| kernel 形态 | Python wheel（`kernel/dist/*.whl`） | CANN opp vendored `.run` 包 |
| 测试程序 | `python3 run_one_case.py`（torch_npu） | C++ aclnn 测试程序（aclnn + ascendcl） |
| 编译方式 | `kernel/CMakeLists.txt`（CMAKE_BUILD_TYPE=Debug） | `build.sh --pkg --soc=... --ops=...`（opc 构建） |
| op_debug_level 注入 | CMakeLists.txt 的 Debug 模式自动带 | 需要修改 cmake 模板注入 `--op_debug_level=1` |
| 部署 | pip install whl | bash `.run` 包部署到 CANN opp/vendors |

---

## Step 1：确保使用支持目标 SoC 的 CANN

ops-transformer 的 master 分支已注册 `ascend950` / `mc62` 等新芯片，需要 CANN 9.0.0+ 的 opbuild 才能识别这些 SoC（否则 `Invalid socVersion` 导致编译失败）。

**推荐**：

| CANN 版本 | 适用场景 |
|-----------|---------|
| 9.1.0 | 最佳，原生支持 ascend950 + mc62，无需 hack |
| 9.0.0 | 可用，支持 ascend950/mc62 |
| 8.5.1 | 仅 910B 系列。若必须用 8.5.1：需将 `tools/opbuild/op_build` 替换为 exit-0 wrapper（见常见问题） |

本指南使用 **CANN 9.1.0**（`/home/l00951729/env/cann-9.1.0/cann-9.1.0/`）。

---

## Step 2：修改构建脚本注入 `--op_debug_level=1`

ops-transformer 的 kernel 编译通过 `opc` / `asc_opc` 生成，默认不输出流水线映射数据。需在两个文件中向 opc 命令追加 `--op_debug_level=1`。

### 修改位置 1：`cmake/scripts/util/ascendc_bin_param_build.py`

找到两处 `--op_mode=dynamic`，追加 `--op_debug_level=1`：

```python
# 修改前（第 491-496 行）：
        if const_var.CHECK_ASC_DEVKIT_VERSION:
            bin_cmd_str = 'res=$(asc_opc $1 --main_func={fun} ... --op_mode=dynamic '
        else:
            bin_cmd_str = 'res=$(opc $1 --main_func={fun} ... --op_mode=dynamic '

# 修改后：
        if const_var.CHECK_ASC_DEVKIT_VERSION:
            bin_cmd_str = 'res=$(asc_opc $1 --main_func={fun} ... --op_mode=dynamic --op_debug_level=1 '
        else:
            bin_cmd_str = 'res=$(opc $1 --main_func={fun} ... --op_mode=dynamic --op_debug_level=1 '
```

### 修改位置 2：`cmake/scripts/util/const_var.py`

```python
# 修改前（第 62-67 行）：
    BIN_CMD = 'asc_opc $1 --main_func={fun} ... --op_mode=dynamic\n'
    BIN_CMD = 'opc $1 --main_func={fun} ... --op_mode=dynamic\n'

# 修改后：
    BIN_CMD = 'asc_opc $1 --main_func={fun} ... --op_mode=dynamic --op_debug_level=1\n'
    BIN_CMD = 'opc $1 --main_func={fun} ... --op_mode=dynamic --op_debug_level=1\n'
```

> `--op_debug_level=1` 的作用：启用 TBE pipe_all，生成 CCE-Python 映射文件，是仿真流水线追踪的前提。**不影响上板性能和正确性**。

### 验证修改

```bash
grep -n "op_debug_level=1" \
  cmake/scripts/util/ascendc_bin_param_build.py \
  cmake/scripts/util/const_var.py
# 应各输出 2 行匹配
```

---

## Step 3：编译算子包

```bash
cd /home/l00951729/ops-transformer
source /home/l00951729/env/cann-9.1.0/cann-9.1.0/set_env.sh

# 编译单个算子（使用 snake_name，如 fused_infer_attention_score）
bash build.sh --pkg --soc=ascend910b --ops=fused_infer_attention_score -j16
```

编译成功后在 `build_out/` 生成 `cann-ops-transformer-custom_linux-x86_64.run`。

**kernel 编译耗时参考**：FA 类大算子约 30-60 分钟（含 opc 逐个编译变体）。

---

## Step 4：部署

```bash
source /home/l00951729/env/cann-9.1.0/cann-9.1.0/set_env.sh
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --nox11
```

部署路径：`$ASCEND_HOME_PATH/opp/vendors/custom_transformer/`

验证部署：

```bash
ls $ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/include/aclnnop/
ls $ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so
```

---

## Step 5：编译 aclnn 测试程序

以 `fused_infer_attention_score` 为例：

```bash
source /home/l00951729/env/cann-9.1.0/cann-9.1.0/set_env.sh
cd attention/fused_infer_attention_score/examples

ASCEND=$ASCEND_HOME_PATH
VENDOR_INC=$ASCEND/opp/vendors/custom_transformer/op_api/include
VENDOR_LIB=$ASCEND/opp/vendors/custom_transformer/op_api/lib

g++ -std=c++17 \
  -I$ASCEND/include \
  -I$ASCEND/include/aclnnop \
  -I$ASCEND/x86_64-linux/include \
  -I$VENDOR_INC \
  -L$ASCEND/lib64 \
  -L$VENDOR_LIB \
  test_aclnn_fused_infer_attention_score_v4_gqa_noquant.cpp \
  -lascendcl -lnnopbase -lcust_opapi \
  -Wl,-rpath=$VENDOR_LIB \
  -o test_fias_npu
```

**编译参数说明**：

| 参数 | 说明 |
|------|------|
| `-I$ASCEND/include` | `acl/acl.h`、`aclnn_base.h` |
| `-I$ASCEND/include/aclnnop` | `aclnn_util.h`（算子头文件内部依赖） |
| `-I$ASCEND/x86_64-linux/include` | `aclnn/opdev/fp16_t.h` 等设备类型 |
| `-I$VENDOR_INC` | 算子头文件（如 `aclnn_fused_infer_attention_score_v4.h`） |
| `-L$VENDOR_LIB -lcust_opapi` | 自定义算子库 |
| `-Wl,-rpath=$VENDOR_LIB` | 将库路径写入 ELF RUNPATH |

---

## Step 6：功能冒烟验证

```bash
source /home/l00951729/env/cann-9.1.0/cann-9.1.0/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}

./test_fias_npu
# 预期：所有 result[0..255] = 1.0，exit=0
```

> 不通过则不要进入仿真步骤——仿真器中的错误定位远比上板困难。

---

## Step 7：msopprof simulator 仿真采集

```bash
source /home/l00951729/env/cann-9.1.0/cann-9.1.0/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}

$ASCEND_HOME_PATH/tools/msopprof/bin/msopprof simulator \
  --soc-version=Ascend910B2C \
  --output=./msopprof_output_sim \
  ./test_fias_npu
```

**关键参数**：
| 参数 | 说明 |
|------|------|
| `--soc-version=Ascend910B2C` | 必须与当前 NPU 芯片型号一致。大小写敏感。`npu-smi info` 可查看 |
| `--output=./msopprof_output_sim` | 输出目录 |
| `./test_fias_npu` | 直接传可执行文件（C++ 程序），无需 `--application=` |

**仿真耗时参考**：小 case（B=1,N=2,Sq=1,Skv=16,D=128,fp16）约 5 秒。

---

## Step 8：查看结果

### 输出结构

```
msopprof_output_sim/OPPROF_<timestamp>/
├── dump/
│   ├── aicore_binary.o          # kernel 二进制
│   ├── object_dump.txt
│   └── pc_start_addr.txt
└── simulator/
    ├── visualize_data.bin       # ← MindStudio Insight 打开这个文件
    ├── trace.json               # 顶层流水线时序
    ├── core0.cubecore0/
    │   ├── trace.json           # Cube 核流水线
    │   └── core0.cubecore0_instr_exe.csv
    ├── core0.veccore0/
    │   ├── trace.json           # Vector 核 0 流水线
    │   └── core0.veccore0_instr_exe.csv
    └── core0.veccore1/
        ├── trace.json           # Vector 核 1 流水线
        └── core0.veccore1_instr_exe.csv
```

### MindStudio Insight 中查看

1. 打开 MindStudio → **File → Open**
2. 选择 `simulator/visualize_data.bin`
3. 可查看：
   - **流水线时序图**：各 pipe（MTE2/MTE3/Cube/Vec/Scalar）逐指令时间线
   - **CV overlap 分析**：Cube 和 Vector 指令的并行度
   - **指令热度图**：热点指令定位
   - **代码行映射**（需 `--op_debug_level=1` 生成）

### 命令行快速查看

```bash
# 仿真核心耗时汇总（运行时输出 stderr）：
# core0.cubecore0   7.54 us
# core0.veccore0    9.40 us
# core0.veccore1    8.13 us

# 顶层时序
cat msopprof_output_sim/OPPROF_*/simulator/trace.json | python3 -m json.tool | head -50

# 指令级执行追踪
cat msopprof_output_sim/OPPROF_*/simulator/core0.veccore0/core0.veccore0_instr_exe.csv | head -20
```

---

## 一键命令汇总

```bash
# === 1. 环境 ===
cd /home/l00951729/ops-transformer
source /home/l00951729/env/cann-9.1.0/cann-9.1.0/set_env.sh
export ASCEND_HOME_PATH=$(pwd)/../env/cann-9.1.0/cann-9.1.0  # 实际路径
OP_SNAKE=<算子 snake_name>   # 如 fused_infer_attention_score
OP_CPP=<测试程序文件名>.cpp    # 如 test_aclnn_fused_infer_attention_score_v4_gqa_noquant.cpp
OP_BIN=test_<op>_npu

# === 2. 修改构建脚本（仅首次） ===
sed -i 's/--op_mode=dynamic /--op_mode=dynamic --op_debug_level=1 /g' \
  cmake/scripts/util/ascendc_bin_param_build.py
sed -i 's/--op_mode=dynamic\\n/--op_mode=dynamic --op_debug_level=1\\n/g' \
  cmake/scripts/util/const_var.py

# === 3. 编译 kernel ===
bash build.sh --pkg --soc=ascend910b --ops=$OP_SNAKE -j16

# === 4. 部署 ===
bash build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --nox11

# === 5. 编译测试程序 ===
cd attention/$OP_SNAKE/examples
VENDOR_INC=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/include
VENDOR_LIB=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib
g++ -std=c++17 \
  -I$ASCEND_HOME_PATH/include -I$ASCEND_HOME_PATH/include/aclnnop \
  -I$ASCEND_HOME_PATH/x86_64-linux/include -I$VENDOR_INC \
  -L$ASCEND_HOME_PATH/lib64 -L$VENDOR_LIB \
  $OP_CPP -lascendcl -lnnopbase -lcust_opapi \
  -Wl,-rpath=$VENDOR_LIB -o $OP_BIN

# === 6. 功能验证 ===
LD_LIBRARY_PATH=$VENDOR_LIB:${LD_LIBRARY_PATH} ./$OP_BIN

# === 7. 仿真采集 ===
$ASCEND_HOME_PATH/tools/msopprof/bin/msopprof simulator \
  --soc-version=Ascend910B2C \
  --output=./msopprof_output_sim \
  ./$OP_BIN

# === 8. MindStudio Insight 打开 ===
# msopprof_output_sim/OPPROF_*/simulator/visualize_data.bin
```

---

## 常见问题

| 问题 | 原因与解决 |
|------|-----------|
| `Invalid socVersion ascend950 of op <X>` | opbuild 版本不支持新芯片。解决：(1) 升级到 **CANN 9.0.0+**（推荐）；(2) 或用 8.5.1 时给 opbuild 包 exit-0 wrapper 绕过 |
| 编译报错 `expected primary-expression before 'inputDesc0_0'` | opbuild 生成了含语法错误的 aclnnInner 代码。通常与 CANN 版本不匹配有关——CANN 8.5 的 opbuild 不理解 CANN 9.x 风格的 OpDef。换 CANN 9.0+ 即可 |
| 测试程序 `undefined symbol: aclnn<Op>GetWorkspaceSize` | LD_LIBRARY_PATH 中 `customize` 优先于 `custom_transformer`。将 `custom_transformer/op_api/lib` 前置 |
| `all NPU devices are busy` | 仿真器不需要独占 NPU，但 `Init(deviceId=0)` 会检查设备状态。换空闲 NPU 的 deviceId 或用 `export ASCEND_RT_VISIBLE_DEVICES=<空闲NPU>` |
| Insight 中无代码行映射 / 流水线为空 | 未启用 `--op_debug_level=1`。检查 Step 2 修改是否生效（`grep op_debug_level=1` 构建日志） |
| 仿真成功但 MindStudio Insight 打开的 visual_data.bin 没有内容 / 报错 | 通常是 kernel 二进制不带 debug_line section。确认 opc 命令带了 `--op_debug_level=1`；可以用 `readelf -S dump/aicore_binary.o \| grep debug` 验证 |
| 多核 case visualize_data.bin 巨大 | 逐 AIC block 会各生成追踪数据。选小 case（单 block、小 shape）控制体积 |
