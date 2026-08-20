# P117 Playbook: 双 BMM 链 MatmulApiTiling Host 配方

> **强制流程**。本 Playbook 指导子 agent 为 attention 类算子的 QK^T/PV 双 matmul 链在 host 端接入 `MatmulApiTiling`，并落地 SetBufferSpace / SetFixSplit / depthA1 三个配方细节。必须按 Step 1-5 逐步执行。

## Step 1: 定位关键结构

把下面 grep 结果写入 `/tmp/p117_locations.txt`：

```bash
# 1.1 双 matmul 链现状（手写 cube 还是 vector 伪 matmul）
grep -nE "Nd2Nz|LoadData|Mmad|Fixpipe|Mul\\s*\\(|ReduceSum" \
    shared/original/op_kernel/*.{cpp,h} > /tmp/p117_locations.txt
# 1.2 host tiling 结构与 shape 参数
grep -nE "TilingData|s1Base|s2Base|dSize|headDim|SetShape|SetOrgShape" \
    shared/original/op_host/*.{cpp,h} >> /tmp/p117_locations.txt
# 1.3 平台参数查询（L1/L0C 容量来源）
grep -nE "PlatformAscendC|GetCoreMemSize|L1|L0_C|CoreMemType" \
    shared/original/op_host/*.cpp >> /tmp/p117_locations.txt
# 1.4 kernel 侧 matmul 消费方式
grep -nE "REGIST_MATMUL_OBJ|MatmulImpl|IterateAll|mmTiling" \
    shared/original/op_kernel/*.{cpp,h} >> /tmp/p117_locations.txt
```

**交付物**：写入 `implementation_note.txt` "Playbook Step 1"，列出 bmm1/bmm2 的 (M, N, K)、行 stride、L1/L0C 容量来源。

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| bmm1 (M,N,K) | ? | (min(s1Base,sq), min(s2Base*nRatio,skv), d) | op_host/*_tiling.cpp:L? |
| bmm1 B 转置 | ? | transpose=true（K^T） | op_host/*_tiling.cpp:L? |
| bmm2 (M,N,K) | ? | (min(s1Base,sq), d, bmm1N) | op_host/*_tiling.cpp:L? |
| OrgShape stride | ? | 真实 layout 行 stride | op_host/*_tiling.cpp:L? |
| CType | ? | DT_FLOAT | op_host/*_tiling.cpp:L? |
| C position | 缺失 | **与 kernel `MatmulType` C 模板参数严格一致**：kernel C 在 VECCALC（片内消费，X60 零 GM scratch 路线）→ host 必须 `SetCType(TPosition::VECCALC, ...)`；GM≠VECCALC 混配即 507015/静默错 | op_host/*_tiling.cpp:L? |
| SetBufferSpace | 缺失 | (l1Size, l0cSize) + 返回值检查 | op_host/*_tiling.cpp:L? |
| SetFixSplit | 缺失 | 仅 d∈(64,128] 时启用 | op_host/*_tiling.cpp:L? |
| depthA1 微调 | 缺失 | baseK>=singleCoreK && depthA1==2 → 3 | op_host/*_tiling.cpp:L? |
| TilingData 字段 | 缺失 | bmm1TilingData / bmm2TilingData 内嵌 | op_host/*_tiling.h:L? |

## Step 3: 代码改造

### 3A. 平台容量查询

```cpp
auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
uint64_t l1Size = 512 * 1024, l0cSize = 128 * 1024;   // 兜底默认值
if (platform != nullptr) {
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
}
```

### 3B. BuildMatmulTiling（bmm1/bmm2 各一份）

按卡片 `## 代码骨架` 落地，要点逐条核对：SetAType/BType/CType → SetShape → SetOrgShape（真实 stride）→ SetBias(false) → SetBufferSpace（**检查返回值，失败则缩 s1Base/s2Base 重试或 return 错误码**）→ d∈(64,128] 时 SetFixSplit → GetTiling → depthA1 条件微调 → shareMode/shareL1Size/shareL0CSize 回写。

### 3C. TilingData 内嵌与 kernel 侧消费

```cpp
// host TilingData 内嵌两个 TCubeTiling
struct MyTilingData {
    TCubeTiling bmm1TilingData;
    TCubeTiling bmm2TilingData;
    ...
};
// kernel 侧（配合 P46 高阶 API）
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), bmm1, &tilingData->bmm1TilingData);
```

### 3D. TilingData GM→local 对齐拷贝（direct-launch 场景）

```cpp
// 8 字节对齐字传输 + 尾部字节补齐，禁止逐字节 GM 事务
constexpr uint32_t words = sizeof(MyTilingData) / sizeof(uint64_t);
for (uint32_t i = 0; i < words; ++i) dstWords[i] = srcWords[i];
for (uint32_t i = words * 8; i < sizeof(MyTilingData); ++i) dstBytes[i] = srcBytes[i];
```

## Step 4: 约束复核

在 `implementation_note.txt` "Playbook Step 4" 报告：

```
P117 约束核查：
  1. bmm1 B 矩阵 transpose=true？ yes/no
  2. bmm1/bmm2 的 SetOrgShape 行 stride = ___（与 layout 一致？）
  3. CType = DT_FLOAT？ yes/no
  4. SetBufferSpace 返回值是否检查并有 fallback？ yes/no
  5. SetFixSplit 是否仅在 d∈(64,128] 启用？ yes/no，d = ___
  6. depthA1 微调是否满足 baseK>=singleCoreK && depthA1==2 前置？ yes/no
  7. bmm1/bmm2 是否各自独立 GetTiling（未共享 TCubeTiling）？ yes/no
```

## Step 5: 编码后自检（5 条 grep，全部必须过）

```bash
# 检查 1: MatmulApiTiling 已接入（bmm1 + bmm2）
(grep -cE "MatmulApiTiling" modified_files/op_host/*.cpp; true)
# 期望: >= 2

# 检查 2: SetBufferSpace 调用
(grep -cE "SetBufferSpace" modified_files/op_host/*.cpp; true)
# 期望: >= 2

# 检查 3: SetFixSplit 条件启用
(grep -cE "SetFixSplit" modified_files/op_host/*.cpp; true)
# 期望: >= 1（d≤64 或 d>128 的固定 shape 可豁免，需 note 说明）

# 检查 4: depthA1 微调
(grep -cE "depthA1" modified_files/op_host/*.cpp; true)
# 期望: >= 1

# 检查 5: kernel 侧消费 bmm tiling
(grep -cE "bmm1TilingData|bmm2TilingData" modified_files/op_kernel/*.{cpp,h} modified_files/op_host/*.{cpp,h}; true)
# 期望: >= 2
```

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P117 Playbook Completion]
Step 1: done
Step 2: plan table filled
Step 3: MatmulApiTiling applied for bmm1+bmm2 (fixsplit=___, depthA1=___)
Step 4: constraints checked (stride=___, buffer_fallback=___)
Step 5: all 5 grep checks passed
```

## 常见陷阱

- bmm1 的 B（K 矩阵）忘加 transpose → shape 错位。
- **转置三处一致性**（3_FusionAttention 实测）：host tiling `SetBType(..., transB)`、kernel `mmQK_.SetTensorB(kGM, transB)`、`MatmulType<TPosition, CubeFormat, T, transB>` 模板参数，三处必须同值；host=true/device=false 时 Cube 按错误布局读 K，不报错、结果全错。
- SetBufferSpace 不查返回值 → 超配 L1 时产出不可用 tiling。
- depthA1 无条件改 3 → K 多轮循环场景读越界 A1。
- 两个 bmm 共享一份 TCubeTiling → kernel 侧必挂。
- TilingData 逐字节从 GM 拷贝 → direct-launch 启动开销被放大。
- **custom-op（torch extension）路径下 Matmul 高阶 API 的 507015 脆弱性**（3_FusionAttention aic_v2 三个并行全部复现）：
  - 生成的 launch wrapper 已 `SetSysWorkspaceForce`，kernel 内 `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), ...)` 直接用即可，**禁止再把 user-workspace 参数设为 system workspace**；
  - **禁止**自定义 `SPLIT_CORE_CUBE/SPLIT_CORE_VEC` 宏去配合 `KERNEL_TYPE_AIC_ONLY`——宏会触发 Matmul server/client 分裂路径，与 AIC-only 声明冲突，首 case 即 507015；
  - workspace 用 `MatmulGetTmpBufSizeV2(tiling, bufSize)` 计算后取 max 并留余量（实测 ≥1MiB 兜底），不足时大 seq（≥4096）同步超时 507015；
  - **能跑的 .so ≠ 能 rebuild 的源码**：上述任一参数不一致时旧 build artifact 仍可运行、重编译即崩，达标后必须把 build 产物连同源码一起归档。
