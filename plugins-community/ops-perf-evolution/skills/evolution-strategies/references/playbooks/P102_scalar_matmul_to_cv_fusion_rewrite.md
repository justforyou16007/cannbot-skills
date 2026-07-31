# P102 Playbook: 标量/向量模拟矩阵运算 → MIX CV 融合重写实操流程

> **强制流程**。P102 把标量/Vector 模拟的矩阵运算架构级重写为 MIX CV 融合（Cube Mmad 承担 matmul，Vector 承担 elementwise/softmax/状态更新）。**这是已验证可行的路径**（gated_delta_rule，910B2C：scalar/VEC 3.3x → MIX_AIC_1_1 geomean **8.85x**）。失败几乎必然是实现问题，不是平台问题——本 Playbook 的 derisk 流程就是为了把"实现问题"和"平台问题"严格分开。

## Step 0: 必读调试手册（强制，先读后写）

**在写任何 MIX 代码之前，必须完整 Read 以下文件**：

```
plugins-community/ops-perf-evolution/skills/evolution-strategies/references/mix-cv-fusion-guide.md
```

该手册包含：MIX 核心机制、核类型守卫与队列归属、最小 Mmad 链参数规范、Hang 根因分类 R1-R6、双缓冲决策矩阵、调试工具箱。本 Playbook 的 Step 3-5 大量引用其内容。**不读该手册直接写 MIX 代码 = 盲写，hang 后无法定位。**

在 `implementation_note.txt` "Playbook Step 0" 段落确认已阅读，并摘录与本算子最相关的 2-3 条（如队列归属、dstStride 单位）。

## Step 1: 定位关键结构 + 适用性确认

执行下面的 grep（ops 仓路径用 `op_kernel/*.cpp`，lingxi/直出工程用 `kernel/*.cpp`），结果写入 `/tmp/p102_locations.txt`：

```bash
# 1.1 现有 matmul 实现方式
grep -n "Matmul|matmul|Cube|Mmad|mmad" kernel/*.cpp > /tmp/p102_locations.txt
# 1.2 L0 buffer 分配现状
grep -n "L0A|L0B|L0C|l0a|l0b|l0c" kernel/*.cpp >> /tmp/p102_locations.txt
# 1.3 同步方式
grep -n "HardEvent|unitFlag|SetFlag|WaitFlag" kernel/*.cpp >> /tmp/p102_locations.txt
# 1.4 数据搬运链路
grep -n "LoadData|Fixpipe|Nd2Nz" kernel/*.cpp >> /tmp/p102_locations.txt
# 1.5 标量/向量模拟矩阵运算的证据（P102 适用信号）
grep -nE "for\s*\(.*\+\+.*\)\s*\{?\s*for\s*\(" kernel/*.cpp >> /tmp/p102_locations.txt
grep -nE "Mul\(.*\).*ReduceSum|ReduceSum" kernel/*.cpp >> /tmp/p102_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- 当前 matmul 实现方式（库 / 手动 / 标量模拟 / Vector 伪 matmul）：文件 + 行号
- L0 buffer 分配现状、同步方式：文件 + 行号
- 矩阵维度确认：被模拟的矩阵运算维度 ≥ 64×64？（< 64 应改用 P94/P95，在 note 中声明换策略理由）

## Step 2: 改造计划表（不填完不得进入 Step 3）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Matmul | `?`（库API / 标量模拟 / Vector伪matmul） | 手动 Mmad（L0A/L0B → Mmad → L0C → Fixpipe） | `kernel/*.cpp:L?` |
| L0 buffer | `?`（库管理 / 无） | 手动分配 ping/pong 双缓冲 | `kernel/*.cpp:L?` |
| 同步 | `?`（库隐式 / PipeBarrier / 无） | HardEvent 显式配对 | `kernel/*.cpp:L?` |
| unitFlag | `?`（无） | 条件设置（末次 0b11，其余 0b10） | `kernel/*.cpp:L?` |
| AIC/AIV 分工 | 无 | Cube=全部 matmul；AIV=elementwise/softmax/状态更新；中间结果经 GM workspace 交接 | — |

**约束复核**（写入 note "Playbook Step 2"）：
- 极高代码复杂度：手动 L0 双缓冲需要精确配对 SetFlag/WaitFlag
- 同步错误风险：HardEvent 配对错误 → 死锁（AICore 100% 空转无输出）
- UB/L0 用量：L0A/L0B 各 64KB×2，L0C 128KB×2，总 L0 512KB 上限（910B2C）
- MmadParams 的 baseM/baseN/baseK、unitFlag 时机需实测调优

## Step 3: Phase 0 — 纯 AIC 单路径 derisk（Cube 链路先对，强制）

> **禁止跳过本步直接写 MIX。** 本轮 GDA 进化的三个变体全部因跳过 Phase 0 直接写全双工 MIX 而失败并误诊为"平台不支持"。正确顺序永远是：先把 Cube 数据通路在单核路径下做对，再加 AIV。

**3A. 形态识别**（在 note "Playbook Step 3A" 声明 `form: alpha | gamma`）：
- **形态 α — 完整手动流水线**：L0 双缓冲 + HardEvent + unitFlag + 条件 PipeBarrier。P102 强制要求此形态。
- **形态 γ — 退化路径**：仅当编译验证确认 CANN 不支持 HardEvent API 时降级（WaitFlag/SetFlag 全部移除，Mmad 后加 `PipeBarrier<PIPE_ALL>`）。**必须做编译验证，不能无证据声称不支持**，并在 note 记录降级声明与缺失 API 名。

**3B. 最小 matmul 探针（AIC_ONLY）**：`KERNEL_TYPE_AIC_ONLY` 写一个最小 matmul（一个 Mmad），验证 Nd2Nz/LoadData/Mmad/Fixpipe 完整参数链。关键参数单位陷阱（来自 mix-cv-fusion-guide §3）：

```cpp
// GM(ND) → L1(NZ)：srcDValue 是 GM 行步长，单位【元素】
// L1(NZ) → L0A/L0B：LoadData3DParamsV2（channelSize/kExtension/mExtension）
// Mmad：cmatrixInitVal 1=覆盖 L0C；0=在 L0C 上累加（链式乘加关键）
// Fixpipe L0C → GM：NZ2ND 默认 CFG_ROW_MAJOR，dstStride 单位【元素】
// （非 NZ 模式 dstStride 单位是 32B）；fp32→fp16 必须加 quantPre=F322F16
```

**3C. Canonical Template（形态 α，K-loop 双缓冲）**：

```cpp
TBuf<TPosition::A1> l0aBuf;  // 32K * 2 (ping/pong)
TBuf<TPosition::B1> l0bBuf;  // 32K * 2
TBuf<TPosition::C1> l0cBuf;  // 64K * 2
constexpr uint8_t L0AB_EVENT = 0;

for (int k = 0; k < kLoops; k++) {
    int pp = k % 2;
    WaitFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);      // 等上轮 Mmad 释放 L0
    LoadData(l0a[pp], l1Key[kvIdx], load3DParams);
    LoadData(l0b[pp], l1Query[qpIdx], load2DParams);
    SetFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);
    WaitFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);      // 等 DMA 完成
    MmadParams mp;
    mp.unitFlag = (k == kLoops - 1) ? 0b11 : 0b10;     // 末次触发 Fixpipe
    Mmad(l0c[pp], l0a[pp], l0b[pp], mp);
    SetFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);
    // Fixpipe 由 unitFlag bit0 自动触发，无需显式调用
}
```

**3D. 编码后自检**（任一失败 → 回 3C 重做，禁止在 note 中把失败 grep 标记为"不适用"）：

```bash
grep -cE "Mmad|mmad" kernel/*.cpp          # ≥1
grep -cE "HardEvent|unitFlag" kernel/*.cpp # ≥1（形态 γ 改为查 PipeBarrier<PIPE_ALL>）
grep -cE "l0a|l0b|l0c|L0A|L0B|L0C" kernel/*.cpp  # ≥1
grep -cE "WaitFlag|SetFlag.*MTE1" kernel/*.cpp   # ≥1（形态 γ 同上豁免）
grep -cE "Matmul<|mm\.SetTensor|mm\.IterateAll" kernel/*.cpp  # ==0
```

**3E. 已知陷阱**：

| 现象 | 诊断 | 修复 |
|---|---|---|
| 死锁 / AICore 100% 空转 | HardEvent 配对检查 | 每个 `WaitFlag(X, id)` 有对应 `SetFlag(X, id)` 且方向正确 |
| 数值错误（非确定 NaN） | unitFlag 条件错误 | 末次 `unitFlag=0b11`，其余 `0b10` |
| Fixpipe 输出全零 | LoadData shape/stride 错误 | load3D/load2D 参数与 Nd2Nz 配置一致 |
| 精度不通过 | L0C→GM 搬运错误 | dstStride 单位（NZ=32B / ND=元素）；fp32→fp16 缺 `quantPre=F322F16` |
| 编译错 "Mmad not declared" | CANN 版本问题 | 确认 ≥8.x 且含 `kernel_operator.h`，记录缺失 API |

**退出标准**：单 matmul 数值与 torch 参考一致 + 3D 全部 grep 自检通过。

## Step 4: Phase 1+2 — MIX 骨架与配对探针（只证明"MIX 不挂"）

**4A. MIX 骨架**：切 `KERNEL_TYPE_MIX_AIC_1_1`，加核类型守卫，**AIV 路径留空**，结尾一次 `SyncAll<false>()`。

```cpp
class KernelX {
    __aicore__ inline void Init(...) {
        if ASCEND_IS_AIC {
            pipe_.InitBuffer(a1_, 1, SZ_A1);   // A1/B1/A2/B2/CO1 — AIC 专属
            pipe_.InitBuffer(b1_, 1, SZ_B1);
            pipe_.InitBuffer(a2_, 1, SZ_A2);
            pipe_.InitBuffer(b2_, 1, SZ_B2);
            pipe_.InitBuffer(c1_, 1, SZ_C1);
        } else {
            pipe_.InitBuffer(ub_, UB_SIZE);    // VECCALC — AIV 专属
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { ProcessAIC(); } else { ProcessAIV(); }
        // 两侧 SyncAll<false>() 调用次数/顺序严格一致
    }
};
extern "C" __global__ __aicore__ void kernel_x(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);   // 函数体内第一行
    KernelX k; k.Init(...); k.Process();
}
```

**五条铁律**（违反即 hang，无报错只有 100% AICore 空转）：
1. **AIV 永不触碰 A1/B1/A2/B2/CO1；AIC 永不触碰 UB**——包括 InitBuffer。
2. **两侧 `SyncAll<false>()` 计数一致**——无参模板默认 `isAIVOnly=true` 只同步 AIV，MIX 全局 barrier 必须显式 `<false>`。
3. **所有核循环次数一致**——`slots = ceil(work/cores)` padded loop，越界核只走 barrier 不做工（`valid` 守卫工作体）。
4. **每个 Mmad 操作数完整走队列**：Alloc→Load→EnQue→DeQue→Mmad→Free；Mmad→Fixpipe、Fixpipe→下次 GM 读之间 `PipeBarrier<PIPE_ALL>`。
5. **中间结果走 GM workspace 交换**（AIC Fixpipe 写 → barrier → AIV DataCopy 读，或反向），不要试图共享 L0/L1/UB。

**退出标准（Phase 1）**：不 hang 且 AIC 结果与 Step 3 完全一致。挂了 → 按 Step 6 的 R1~R4 对照检查（几乎都是守卫缺失）。

**4B. 配对探针（Phase 2）**：AIV 路径写 `GetBlockIdx()/GetBlockNum()` 到 GM，host 读回打印。**退出标准**：确认 pairing（MIX_AIC_1_1 下 AIV idx == AIC idx）与实际核数。

## Step 5: Phase 3 — 逐 stage 增量 + 端到端验证

- 每个 stage：AIC 写中间矩阵到 GM workspace → `SyncAll<false>()` → AIV 处理 → `SyncAll<false>()` → ...
- **每个 stage 落地即验证**：host 临时 return workspace 张量，与 torch 参考逐 stage 对比。
- **必须测多 chunk / 多迭代（T≥128）**：单 chunk 会掩盖状态依赖路径的 bug（GDR 实例连踩两次：decay mask inclusive vs strict、漏乘 scale，单 chunk 均恰好正确）。
- **诊断前先 torch 算法仿真**：纯 torch 复刻 kernel 的分解公式再对参考，不花编译时间抓公式错误。
- **退出标准**：全 stage 对齐后，端到端精度 PASS。

**完成清单**（写入 `implementation_note.txt` "Playbook Step 5 Completion"）：

```
[P102 Playbook Completion]
Step 0: guide 已阅读（摘录 2-3 条）
Step 1: done（matmul 现状 + 维度判定）
Step 2: plan table filled
Step 3: form = alpha / gamma（声明降级理由如适用）; AIC_ONLY 探针数值一致; grep 自检 5 条通过
Step 4: MIX 骨架不 hang; 配对探针确认 pairing
Step 5: 逐 stage 对参考通过（多 chunk T≥128）; 端到端精度 PASS
```

## Step 6: 失败归因纪律（Hang/NaN 排查，强制执行）

**Hang 根因分类（R1-R6，来自 mix-cv-fusion-guide §7）**：

| # | 根因 | 现象 | 检查方法 |
|---|------|------|---------|
| R1 | AIV 执行 AIC 队列/指令（无核类型守卫） | 100% AICore 空转，无输出 | 所有 `TPosition::A1/B1/A2/B2/CO1` 操作是否在 `ASCEND_IS_AIC` 内 |
| R2 | `InitBuffer` 在错误的核趟执行 | 同上，启动即挂 | InitBuffer 必须与使用该 buffer 的代码同守卫 |
| R3 | AIC/AIV 两侧 `SyncAll` 次数/顺序不一致 | 死锁（部分核先到，永远等待） | 数两侧的 barrier 数；逐段打印 |
| R4 | 各核迭代次数不同 + 循环内全局 barrier | 死锁 | 循环上界必须全核一致（padded loop + `valid` 守卫） |
| R5 | 自定义 Mmad 链缺管道事件同步 | aicore exception（非 hang，有报错） | L0A/L0B 走完整 EnQue/DeQue；Mmad→Fixpipe、Fixpipe→GM 读之间 `PipeBarrier<PIPE_ALL>` |
| R6 | **设备被前一次 hang 的 kernel 搞 wedge** | 后续运行在初始化就 hang；或 NaN **位置逐 run 变化** | 换一块干净 NPU 重试，不要在挂死的卡上诊断 |

**"平台/环境不支持"结论的证据标准（四条缺一即禁止下此结论，只能记为 impl_error 继续修）**：
1. Step 3 的 AIC_ONLY 最小探针已失败（不是完整 MIX 失败）；
2. `rm -rf build` 清缓存后复现（build 缓存会让"改了没生效"造成假诊断）；
3. 换一块干净 NPU 复现（排除 R6 wedge；**NaN 位置逐 run 变化 = 设备污染，不是 kernel bug**）；
4. 引用具体编译/运行报错原文（API 缺失、版本不支持），而非"产生了 NaN/hang"这类症状描述。

**其他误诊教训（实例）**：
- 改了两样东西（launch type + 参数）时，hang 不一定归因于最后改的那个——AIC_ONLY 编译运行正常 ⇒ kernel 逻辑无问题，launch 模式是病根，不要回退参数空转。
- 带病 NPU：被前一次 hang kernel 污染的设备会让 cross-core barrier 失效，同 binary 换一块干净 NPU PASS ⇒ 卡问题，与代码无关。
- 单 chunk 不算验证；stage 中间张量逐个对参考；改码必 `rm -rf build`。

**一页纸速查**：

```
MIX hang?  → ① 守卫（AIC/AIV 各碰各的） ② barrier 两侧同数 ③ 循环次数全核一致
           → ④ Mmad 链完整队列事件 ⑤ 换块干净 NPU 再下结论
开发顺序   → AIC_ONLY 单 matmul 验证 → MIX 骨架 → 配对探针 → 逐 stage 增量（多 chunk!）
NaN 判别   → 位置逐 run 变化 = 设备 sync 污染（换卡）；固定位置系统性错误 = kernel bug
性能杠杆   → 串行链上的标量循环先上 Cube，再砍 barrier，再排流水
```
