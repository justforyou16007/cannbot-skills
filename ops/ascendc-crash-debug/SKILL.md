---
name: ascendc-crash-debug
description: Ascend C 算子卡死/崩溃/内存错误调试路由技能。用于处理程序无法运行完或执行异常崩溃的场景：(1) 程序卡死/挂起/超时，Kernel 无响应，(2) 程序崩溃（Segmentation Fault、Abort），(3) Buffer 冲突/死锁导致的核心挂起，(4) 需要解析 plog 日志定位卡死/崩溃位置，(5) 偶发崩溃/结果异常（怀疑内存越界踩踏），需要主动检测内存错误。触发关键词：卡死、挂起、超时、崩溃、hang、crash、deadlock、Segmentation Fault、Abort、Kernel hang、内存越界、plog、memcheck、内存检查、内存错误、偶发崩溃、aic error、非法读写。
---

# Ascend C 算子卡死/崩溃调试

系统化调试 Ascend C 算子无法运行完的问题，包括程序卡死/挂起、Kernel超时、程序崩溃。

## 快速诊断

```
程序无法运行完或偶发崩溃
    │
    ├─ 程序崩溃（Segmentation Fault / Abort）？
    │   ├─ 启用 coredump → ulimit -c unlimited
    │   ├─ 生成 core 文件 → 运行程序
    │   └─ GDB 分析 → gdb <exe> <core> → bt / bt full / info locals
    │
    ├─ 程序卡死/超时？
    │   ├─ 查看 plog 日志 → 定位卡死位置
    │   ├─ 检查 Buffer 配对 → AllocTensor/FreeTensor
    │   ├─ 检查同步 → EnQue/DeQue 配对
    │   ├─ Kernel 调试 → AscendC::PRINTF / DumpTensor / msDebug
    │   └─ MIX (CV融合) kernel 卡死（100% AICore 空转）？ → [MIX hang 排查](references/mix-hang-debug.md)
    │       （核类型守卫 / SyncAll 两侧对齐 / 循环次数一致 / 带病 NPU）
    │
    └─ 偶发崩溃/怀疑内存错误导致异常？
        └─ 内存错误主动检测 → mssanitizer
            ├─ 越界读写（Illegal Read/Write）
            ├─ 多核踩踏（Multi-core Overwrite）
            ├─ 非对齐访问（Misaligned Access）
            └─ 内存泄漏（Memory Leak）
```

## 症状-原因速查表

| 症状 | 可能原因 | 诊断方向 |
|------|----------|----------|
| **程序卡死/超时** | Buffer 未释放 / 死锁 | 检查 AllocTensor/FreeTensor 配对、EnQue/DeQue 配对 |
| **核心超时/挂起** | Buffer 冲突/死锁 | 检查 Alloc / Free 配对 |
| **Segmentation Fault** | 空指针解引用 / 内存越界 | GDB 分析 coredump → bt 查看调用栈；如堆栈不清晰 → [mssanitizer 内存检测](references/memcheck/) |
| **Abort** | 断言失败 / 异常终止 | GDB 分析 coredump → 检查断言条件 |
| **栈溢出** | 递归过深 / 大数组 | 检查 Kernel 内数组大小 |
| **aic error** | 内存访问越界 / 非对齐访问 | 检查 DataCopy 长度、32B 对齐；如代码逻辑复杂 → [mssanitizer 内存检测](references/memcheck/) |
| **AIV 卡死** | 跨核同步缺失 | 检查 CrossCoreWaitFlag 对应 SetFlag |
| **drain 卡死** | 未发送 AIV→AIC flag | 检查 flag 发送链路 |
| **MIX kernel 启动即挂、100% AICore 空转** | AIV 执行 AIC 专属指令（队列/InitBuffer 未按核类型守卫） | [MIX hang 排查](references/mix-hang-debug.md) R1/R2：所有 A1/B1/A2/B2/CO1 操作须在 `ASCEND_IS_AIC` 内 |
| **MIX 两侧 barrier 不对齐** | AIC/AIV 的 SyncAll 次数/顺序不一致，或各核循环次数不同 | [MIX hang 排查](references/mix-hang-debug.md) R3/R4：数两侧 barrier；padded loop + valid 守卫 |
| **MIX 输出 NaN 且位置逐 run 变化** | 带病 NPU（前次 hang 残留导致 cross-core barrier 失效） | 换干净 NPU 重跑：同 binary PASS ⇒ 卡问题（[R6](references/mix-hang-debug.md)） |
| **偶发崩溃无法复现** | 内存踩踏 / 竞争条件 | [mssanitizer 内存检测](references/memcheck/)（加 -g 编译选项定位） |

## 按场景进入详细流程

| 场景 | 说明 | 详细步骤 |
|------|------|----------|
| 程序崩溃 | Segmentation Fault / Abort | [Coredump 调试](references/crash_workflow.md) |
| 程序卡死/超时 | Kernel 无响应、挂起 | [Kernel 挂起调试](references/crash_workflow.md) |
| MIX (CV融合) kernel 卡死 | 100% AICore 空转、barrier 死锁、带病 NPU | [MIX hang 排查](references/mix-hang-debug.md) |
| Buffer 冲突/死锁 | Alloc/Free 不配对、同步缺失 | [Buffer 问题排查](references/crash_workflow.md) |
| 内存错误主动检测 | 越界读写/多核踩踏/非对齐访问/内存泄漏/偶发崩溃 | [mssanitizer 内存检测](references/memcheck/) |

## 详细资源

- **[crash_workflow.md](references/crash_workflow.md)**：完整调试流程 + 调试工具速查
- **[mix-hang-debug.md](references/mix-hang-debug.md)**：MIX (Cube+Vector 融合) kernel hang 专项排查（R1-R6 根因分类、守卫代码模式、四阶段预防式开发流程）
- **[parse_plog.py](scripts/parse_plog.py)**：plog 日志解析脚本（自动识别卡死/崩溃/硬件异常信号）
- **[mssanitizer 内存检测](references/memcheck/)**：内存错误检测完整工作流
  - [自动化工作流指南](references/memcheck/automated_workflow.md) — 3步完成编译-安装-检测
  - [用户使用指南](references/memcheck/README.md) — 配置文件模板 + 常见问题
  - [msSanitizer 工具指南](references/memcheck/mssanitizer_guide.md) — 6类内存异常详解
- **[自动化检测脚本](scripts/run_memcheck_pre.sh)**：一键执行内存检测
- **[配置文件模板](scripts/memcheck_input.json.template)**：memcheck_input.json 模板
- **[ascendc-env-check skill](skill:ascendc-env-check)**：环境变量配置指南
