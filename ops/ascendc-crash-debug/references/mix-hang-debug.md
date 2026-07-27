# MIX (Cube/Vector 融合) Kernel Hang 排查

适用：Ascend 910B 上 `KERNEL_TYPE_MIX_AIC_1_1` / `MIX_AIC_1_2` 的 Cube+Vector 融合 kernel 卡死/异常。
以 gated_delta_rule CV 融合重写为实例总结（源：`a5_ops_slim/workspace/gated_delta_rule_cv/docs/MIX_DEBUG_WORKFLOW.md`）。

---

## 1. MIX 模式核心机制（先理解，再动手）

| 机制 | 要点 |
|------|------|
| 单一二进制双核执行 | MIX 启动后，**同一份 `__global__` 代码在 AIC 和 AIV 上都执行**。不做核类型分支 = AIV 执行 AIC 专属指令 = **死锁** |
| `ASCEND_IS_AIC` / `ASCEND_IS_AIV` | 编译期宏，每个编译趟（AIC 趟/AIV 趟）静态求值，`if ASCEND_IS_AIC {...}` 会被干净裁剪（**宏自带括号，不要再加 `()`**） |
| `GetBlockIdx()` 配对 | MIX_AIC_1_1 下 **AIV 核 i 与 AIC 核 i 一一配对**，`GetBlockIdx()/GetBlockNum()` 两边一致 |
| `SyncAll<false>()` | AIC+AIV **全局 barrier**（ffts 硬件握手）。无参模板默认 `isAIVOnly=true` 只同步 AIV——**必须显式 `<false>`** |
| Buffer 初始化 | `pipe_.InitBuffer(A1/B1/A2/B2/CO1 队列)` 只能在 AIC 趟执行；`TBuf<VECCALC>` 只能在 AIV 趟 |

**MIX hang 的第一性原理**：MIX 模式本身可行且是必需架构（纯 AIC/AIV 的 matmul 密集算子不合格）。
hang 几乎全部来自"错误类型的核执行了不该执行的指令"或"两侧 barrier/迭代次数不对齐"。

---

## 2. 根因分类（R1-R6）

| # | 根因 | 现象 | 检查方法 |
|---|------|------|---------|
| R1 | AIV 执行 AIC 队列/指令（无核类型守卫） | 100% AICore 空转，无输出 | 代码审查：所有 `TPosition::A1/B1/A2/B2/CO1` 操作是否在 `ASCEND_IS_AIC` 内 |
| R2 | `InitBuffer` 在错误的核趟执行 | 同上，启动即挂 | InitBuffer 必须与使用该 buffer 的代码同守卫 |
| R3 | AIC/AIV 两侧 `SyncAll` 次数/顺序不一致 | 死锁（部分核先到，永远等待） | 数两侧的 barrier 数；逐段打印 |
| R4 | 各核迭代次数不同 + 循环内全局 barrier | 死锁 | 循环上界必须全核一致（padded loop + `valid` 守卫工作体） |
| R5 | 自定义 Mmad 链缺管道事件同步 | aicore exception（非 hang，有报错） | L0A/L0B 加载必须走 a2_/b2_ 完整 EnQue/DeQue；Mmad→Fixpipe、Fixpipe→下次 GM 读之间 `PipeBarrier<PIPE_ALL>` |
| R6 | **设备被前一次 hang 的 kernel 搞 wedge** | 后续运行在 `.npu()` 张量初始化阶段就 hang（根本到不了 kernel）；或输出 NaN 且**位置逐 run 变化** | 换一块干净 NPU 重试，不要在被挂死的卡上诊断 |

---

## 3. 修复模式（可直接套用的代码结构）

```cpp
class KernelX {
    __aicore__ inline void Init(...) {
        if ASCEND_IS_AIC {
            pipe_.InitBuffer(a1_, 1, SZ_A1);   // A1/B1/A2/B2/CO1 — AIC 专属
            // ...
        } else {
            pipe_.InitBuffer(ub_, UB_SIZE);    // UB/VECCALC — AIV 专属
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC {
            ProcessAIC();          // 全部 Cube 工作
        } else {
            ProcessAIV();          // 全部 Vector 工作
        }
        SyncAll<false>();          // AIC+AIV 全局 barrier；两侧调用次数/顺序严格一致
    }
};
extern "C" __global__ __aicore__ void kernel_x(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    ...
}
```

铁律：
1. **AIV 永不触碰 A1/B1/A2/B2/CO1；AIC 永不触碰 UB**——包括 InitBuffer。
2. **两侧 barrier 计数一致**——每改一侧，数另一侧。
3. **所有核循环次数一致**——`slots = ceil(work/cores)` padded loop，越界核只走 barrier 不做工。
4. **每个 Mmad 操作数完整走队列**：Alloc→Load→EnQue→DeQue→Mmad→Free；Mmad→Fixpipe、Fixpipe→下次 GM 读之间 `PipeBarrier<PIPE_ALL>`。
5. **中间结果走 GM 工作区交换**（AIC Fixpipe 写 → barrier → AIV DataCopy 读，或反向），不要试图共享 L0/L1/UB。

---

## 4. 预防式开发流程（四阶段，每阶段有明确退出标准）

### Phase 0 — 纯 AIC 单路径 derisk（Cube 链路先对）
- 用 `KERNEL_TYPE_AIC_ONLY` 写最小 matmul（一个 Mmad），验证 Nd2Nz/LoadData/Mmad/Fixpipe 的 shape/stride 配置。**退出标准**：单 matmul 数值与 torch 参考一致。
- 常见坑：Nd2Nz `srcDValue` 是 GM 行步长（元素），Fixpipe NZ2ND（默认 CFG_ROW_MAJOR）`dstStride` 单位是**元素**，非 NZ 模式是 32B。

### Phase 1 — MIX 骨架（只证明"MIX 不挂"）
- 切 `KERNEL_TYPE_MIX_AIC_1_1`，加核类型守卫，AIV 路径留空，结尾一次 `SyncAll<false>()`。
- **退出标准**：不 hang 且 AIC 结果与 Phase 0 完全一致。
- 挂了 → 按 §2 R1~R4 对照检查（几乎都是守卫缺失）。

### Phase 2 — 配对与同步探针（证明"核对应关系"）
- AIV 路径写 `GetBlockIdx()/GetBlockNum()` 到 GM，host 读回打印。
- **退出标准**：确认 pairing（1:1 下 AIV idx == AIC idx）与实际核数。

### Phase 3 — 逐 stage 增量（一次一个 stage，各自对参考）
- 每个 stage：AIC 写一个中间矩阵到 GM 工作区 → barrier → AIV 处理 → barrier → ...
- **每个 stage 落地即验证**：host 临时 return 工作区张量，与 torch 参考逐 stage 对比。
- **必须测多 chunk（T≥128）**：单 chunk 会掩盖状态依赖路径的 bug。
- **退出标准**：全 stage 对齐后，端到端精度 PASS。

---

## 5. 误诊教训（实例，引以为戒）

1. **改了两样东西时，hang 不一定归因于最后改的那个**：某次切 MIX launch type + 改 Fixpipe dstStride 后 hang，回退 dstStride 仍 hang——根因是 launch 模式（AIV 执行 AIC 指令），不是参数。AIC_ONLY 编译运行正常 ⇒ kernel 逻辑无问题。
2. **单 chunk 掩盖状态依赖 bug**（连踩两次）：decay mask 应为含对角 inclusive（step2 是严格下三角）；`o_inter` 漏乘 q 的 scale（chunk0 的 QH=q@0=0）——单 chunk 测试恰好都正确。
3. **UB 溢出**：32KB 数据写进 16KB TBuf——按 §1 UB 预算表逐 buffer 核对。
4. **build 缓存陷阱**：构建脚本检测 .so 存在会跳过 cmake——**改代码必须 `rm -rf build`**，否则"改了没生效"造成假诊断。

---

## 6. 调试工具箱

| 工具 | 用法 | 解决什么 |
|------|------|---------|
| `timeout --signal=KILL N` | 包裹每次上板运行 | 防止 hang 住会话 |
| `npu-smi info` AICore% | hang 时查看：100% = kernel 空转（R1~R5）；0% = host 侧挂（查 R6 wedge） | 区分 device/host hang |
| 换 NPU 重试 | hang 后设备可能 wedge；同 binary 换卡 PASS ⇒ 卡问题 | 避免在坏卡上误诊 |
| GM 探针 | 核内写 blockIdx/中间值到 GM，host 读回 | 配对验证、数据流追踪 |
| host 临时 return 工作区 | op 暂时返回 wsF/wsH 而非正式输出 | 逐 stage 数值二分 |
| torch 算法仿真 | 纯 torch 复刻 kernel 的分解公式再对参考 | 不花编译时间就能抓公式错误 |
| build 缓存清理 | `rm -rf build` 后重编 | 防止"改了没生效"的假诊断 |

---

## 7. 一页纸速查

```
MIX hang?  → ① 守卫（AIC/AIV 各碰各的） ② barrier 两侧同数 ③ 循环次数全核一致
           → ④ Mmad 链完整队列事件 ⑤ 换块干净 NPU 再下结论
开发顺序   → AIC_ONLY 单 matmul 验证 → MIX 骨架 → 配对探针 → 逐 stage 增量（多 chunk!）
验证纪律   → 单 chunk 不算验证；stage 中间张量逐个对参考；改码必 rm -rf build
NaN 判别   → 位置逐 run 变化 = 设备 sync 污染（换卡）；固定位置系统性错误 = kernel bug
```

---

## 8. 相关文档

- **TileLang DSL 层的 C/V 同步问题**：`tilelang-perf-optimization` skill 的
  `references/cv_sync_debug.md`——DSL 原语（`T.Pipelined` / cross flag /
  自动 CV 同步失效）、组屏障 AND 归约的子块参与、GM 可见性 fence、
  竞态放大、"正确但慢"的锁步调度诊断。分工约定：kernel 是 TileLang 写的先查
  该文档；生成的 AscendC 代码结构本身有错（守卫/barrier/队列事件）回到本文档 R1-R6。
