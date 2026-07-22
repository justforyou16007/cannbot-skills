# MIX (Cube+Vector 融合) 模式 API 使用规范

适用：`KERNEL_TYPE_MIX_AIC_1_1` / `MIX_AIC_1_2` 的 Cube+Vector 融合 kernel 开发。
来源：gated_delta_rule CV 融合重写实证（`a5_ops_slim/workspace/gated_delta_rule_cv`）。

---

## 1. 核类型守卫（第一铁律）

MIX 启动后，**同一份 `__global__` 代码在 AIC 和 AIV 上都执行**。所有核类型专属代码必须用编译期宏守卫：

```cpp
// ✅ 正确：宏自带括号，不要再加 ()
if ASCEND_IS_AIC {
    // Cube 专属代码
}
if ASCEND_IS_AIV {
    // Vector 专属代码
}

// ❌ 错误：if (ASCEND_IS_AIC) — 宏展开后是 constexpr(...)，外加括号编译报错
```

## 2. 队列/Buffer 归属规则

| 资源 | 所属核 | 说明 |
|------|--------|------|
| `TQue<TPosition::A1, N>` | AIC | L1 操作数 A 队列 |
| `TQue<TPosition::B1, N>` | AIC | L1 操作数 B 队列 |
| `TQue<TPosition::A2, N>` | AIC | L0A 队列 |
| `TQue<TPosition::B2, N>` | AIC | L0B 队列 |
| `TQue<TPosition::CO1, N>` | AIC | L0C（累加器）队列 |
| `TBuf<TPosition::VECCALC>` | AIV | UB 工作 buffer |
| `TBuf<TPosition::VECIN/VECOUT>` | AIV | UB 输入/输出 buffer |

**越界即死锁**（100% AICore 空转，无报错）：
- AIV 趟执行 `InitBuffer(A1/B1/A2/B2/CO1)` 或 AllocTensor 这些队列 → 硬件级死锁
- AIC 趟触碰 UB buffer → 同样死锁
- `InitBuffer` 必须与使用该 buffer 的代码**同守卫**，且在循环外一次性调用

```cpp
__aicore__ inline void Init(...) {
    if ASCEND_IS_AIC {
        pipe_.InitBuffer(a1_, 1, SZ_A1);
        pipe_.InitBuffer(b1_, 1, SZ_B1);
        pipe_.InitBuffer(a2_, 1, SZ_A2);
        pipe_.InitBuffer(b2_, 1, SZ_B2);
        pipe_.InitBuffer(c1_, 1, SZ_C1);
    } else {
        pipe_.InitBuffer(tUb_, UB_SIZE);   // VECCALC
    }
}
```

## 3. KERNEL_TASK_TYPE 声明位置

```cpp
extern "C" __global__ __aicore__ void kernel_x(GM_ADDR ...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);   // 函数体内第一行
    KernelX k; k.Init(...); k.Process();
}
```

| 类型 | 含义 |
|------|------|
| `KERNEL_TYPE_AIC_ONLY` | 纯 Cube（Phase 0 derisk 用） |
| `KERNEL_TYPE_AIV_ONLY` | 纯 Vector |
| `KERNEL_TYPE_MIX_AIC_1_1` | 1 AIC : 1 AIV 配对（AIV idx == AIC idx） |
| `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC : 2 AIV（AIV idx = AIC idx × 2 + {0,1}） |

逻辑核号换算：
```cpp
int32_t coreIdx;
if ASCEND_IS_AIC { coreIdx = GetBlockIdx(); }
if ASCEND_IS_AIV { coreIdx = GetBlockIdx() / GetSubBlockNum(); }  // 1:2 时 /2，1:1 时 GetSubBlockNum()=1
```

## 4. SyncAll 语义（最常踩的坑）

```cpp
SyncAll<false>();   // ✅ MIX 全局 barrier：AIC+AIV 全部核握手（ffts 硬件同步）
SyncAll();          // ❌ 无参模板默认 isAIVOnly=true —— 只同步 AIV！MIX 下等于没有跨核同步
```

**两侧计数一致铁律**：AIC 路径和 AIV 路径的 `SyncAll<false>()` 调用**次数和顺序必须严格一致**（包括循环内——见 §6 padded loop）。每改一侧，数另一侧。

## 5. 最小 Mmad 链（GM → L1 → L0 → Mmad → Fixpipe → GM）

```cpp
// GM(ND) → L1(NZ)：srcDValue 是 GM 行步长，单位【元素】
Nd2NzParams p; p.ndNum=1; p.nValue=m; p.dValue=n; p.srcNdMatrixStride=0;
p.srcDValue=ld; p.dstNzC0Stride=m; p.dstNzNStride=1; p.dstNzMatrixStride=0;
DataCopy(l1Tensor, gmTensor, p);

// L1(NZ) → L0A/L0B：LoadData3DParamsV2
LoadData3DParamsV2<half> lp;
lp.l1H=1; lp.l1W=colC0Stride; lp.channelSize=k; lp.kExtension=k; lp.mExtension=m;
lp.strideH=1; lp.strideW=1; lp.filterH=1; lp.filterW=1;
lp.dilationFilterH=1; lp.dilationFilterW=1;
LoadData(l0Tensor, l1Tensor, lp);

// Mmad：cmatrixInitVal 1=覆盖 L0C；0=在 L0C 上累加（链式乘加关键）
MmadParams mp{}; mp.m=m; mp.n=n; mp.k=k; mp.cmatrixInitVal=1;
Mmad(c1Tensor, a2Tensor, b2Tensor, mp);

// Fixpipe L0C → GM：NZ2ND 默认 CFG_ROW_MAJOR，dstStride 单位【元素】
// （非 NZ 模式 dstStride 单位是 32B，注意区分）
FixpipeParamsV220 fp; fp.nSize=n; fp.mSize=m; fp.srcStride=m; fp.dstStride=n;
fp.ndNum=1; fp.srcNdStride=0; fp.dstNdStride=0;
// fp.quantPre=QuantMode_t::F322F16;   // fp32→fp16 输出时加
Fixpipe(gmOut, c1Tensor, fp);
```

**队列完整性**：每个 Mmad 操作数必须 `Alloc→Load→EnQue→DeQue→Mmad→Free` 完整走 a1_/b1_/a2_/b2_ 队列；Mmad→Fixpipe、Fixpipe→下次 GM 读之间 `PipeBarrier<PIPE_ALL>`。缺事件同步 = aicore exception（有报错，非 hang）。

## 6. Padded Loop + valid 守卫（循环内 SyncAll 的前提）

循环体内含 `SyncAll<false>()` 时，**所有核的循环次数必须一致**：

```cpp
int32_t slots = (totalWork + ncore - 1) / ncore;   // 全核相同
for (int32_t s = 0; s < slots; ++s) {
    int32_t work = s * ncore + core;
    bool valid = work < totalWork;
    if (valid) { /* 实际工作 */ }
    SyncAll<false>();   // 越界核也走到，只同步不做工
    if (valid) { /* 下一阶段 */ }
    SyncAll<false>();
}
```

## 7. GM Workspace 交换模式

AIC 与 AIV **不能共享 L0/L1/UB**，中间结果必须经 GM workspace 交接：

```
AIC: Mmad → Fixpipe 写 GM ws slot → SyncAll<false>()
AIV: DataCopy 读 GM ws slot → elementwise 处理 → DataCopy 写另一 slot → SyncAll<false>()
```

工程实践（gated_delta_rule_cv 实例）：
- wsF（float slot）/ wsH（half slot）两个 workspace，编译期常量定义 slot 偏移布局，host 与 kernel 两侧保持一致
- host 预写固定内容（如 Neumann 链的 identity 矩阵 `at::eye`）
- 一次性 elementwise/布局预处理（transpose、cumsum、decay mask、fp16 缩放）放 **host 侧 torch** 完成，device kernel 只做 matmul + 必要 elementwise

## 8. Host 启动：必须走 OpCommand::RunOpApi

```cpp
// ✅ 正确：RunOpApi 入队，与 torch 预处理/fill 算子同流顺序执行
EXEC_KERNEL_CMD(kernel_name, blockDim, args...);   // 内部 OpCommand::RunOpApi

// ❌ 危险：裸 aclrtlaunch_* 与 torch 算子可能流竞争
// 实例：torch::zeros 分配的输出被 fill kernel 在 kernel 写完后覆盖为全 0
aclrtlaunch_kernel(blockDim, stream, ...);
```

参考实现：`tilelang2ascend-operator-project-init/templates/ascend-kernel/csrc/utils/torch_kernel_helper.h` 的 `EXEC_KERNEL_CMD` 宏。

---

## 相关文档

- MIX hang 排查（R1-R6 根因）：`ascendc-crash-debug/references/mix-hang-debug.md`
- scalar→MIX 重写决策与四阶段工作流：evolution-strategies 卡 `P102_scalar_matmul_to_cv_fusion_rewrite.md`
- Cube Neumann 三角求逆：evolution-strategies 卡 `P103_cube_neumann_triangular_solve.md`
