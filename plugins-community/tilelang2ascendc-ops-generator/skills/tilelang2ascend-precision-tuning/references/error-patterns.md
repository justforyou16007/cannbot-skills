# 错误模式参考

AscendC kernel 调试中的常见错误模式及其根因。

## 类型 A - 简单 Vector 算子错误模式

| 模式 | 根因 | 定位方法 |
|------|------|----------|
| 所有值偏差固定常数 | 缺少 bias/scale | 检查 Adds/Muls 操作 |
| 每隔 N 个值出错 | stride/alignment 问题 | 验证 DataCopy stride，检查向量对齐 |
| NaN 或 Inf | 除零/溢出 | 检查分母，验证输入范围 |
| 首尾值出错 | 边界/padding 问题 | 检查 tile alignment，边界处理 |
| 误差累积 | 未初始化变量/队列同步/流水同步 | 检查初始化，验证 EnQue/DeQue 和流水同步 |
| 随机零散错误 | 竞态条件/队列深度 | 增加 BUFFER_NUM，检查同步 |
| 输出全零 | 缺少计算/队列错误 | 验证 Compute 调用，检查队列 |
| 输出等于输入 | 计算未执行 | 验证操作是否真正执行 |

## 类型 B - 复杂 Cube+Vector 算子错误模式

| 模式 | 根因 | 定位方法 |
|------|------|----------|
| C输出正确但 V输出错误 | Softmax/累加逻辑问题 | 对比 `C输出`、`V输入`、`V输出`，检查 Vector 侧计算参数 |
| 每次执行数据不一致 | 缺少同步或同步时机错误、slot 错位 | 检查跨核 flag、MTE2/V/MTE3 流水同步，数据是否缺少必要同步 |
| 部分 slot 数据错误 | Ring buffer slot 管理问题 | 追踪特定 slot 序号，检查 slot % RING_SLOTS 计算 |
| 尾块（tail）错误 | tailValid/kvRows 处理问题 | 只用触发尾块的 case，检查 isTailKV 分支和 mask 处理 |
| 首次迭代正确后续错误 | 状态累加/softmax flash 问题 | 对比 isFirst=true 和 isFirst=false 的中间状态，检查 prevStateBase |
| 最终输出部分行错误 | qSeqLen 边界处理问题 | 检查 FinalizeOutputChunk 的 globalRowStart 计算，验证 dealRows clamp |
| dim 对齐导致的输出错误 | dim vs actualDim 处理问题 | 检查 dim != actualDim 时的逐行 Cast/DataCopy |
| MM1 结果错误 | K 加载或 Mmad 参数问题 | 验证 kvRowsAlign 计算，检查 LoadNdGmToNzL1 的 kvRows 参数 |
| MM2 结果错误 | V 加载或 P 格式问题 | 验证 LoadNzL1ToZnL0B 参数，检查 P 的 NZ 格式处理 |
| 生产端 dump 正确、消费端 dump 错误 | 同步/地址/slot 读写错位 | 对比 `C输出 -> V输入` 或上一段 `V输出 -> 下一段 C输入`，先查同步和 offset |

## 类型 C - 验证基准/环境侧误差模式（候选无缺陷，勿烧 D 类轮次）

| 模式 | 根因 | 定位方法 |
|------|------|----------|
| fp32 matched_ratio 单项超阈（allclose/MERE 全过），候选 vs CPU-fp64 精确值=1.0000 | **参考模型 fp32 conv 走 HF32 模式**（cube 截断乘法输入尾数 24→11bit，MERE ~1.4e-4），`torch_npu.npu.conv.allow_hf32` 默认开 | ① grep model.py 是否设 `torch_npu.npu.conv.allow_hf32=False`；② 测 floor=参考 NPU vs CPU 精确值的 matched_ratio，floor<0.9 即结构性不可达；③ 注意标志**在首个 conv op 执行时锁定**，消融实验必须每轮起新进程 |
| fp16/bf16 matched_ratio 0.85-0.92 卡边界，误差从 softmax 起系统性偏离 | 参考 conv epilogue 把含 bias 的 logits 舍入到存储 dtype 后才 softmax，候选用精确 fp32 logits | 三方对照（候选 NPU / 参考 NPU / CPU fp64 精确链）逐级比对，首个 cand/ref 分叉 stage 即根因；修复=候选对 logits+bias RoundTrip 到存储 dtype 再做 softmax |
| 误差集中在少数 channel，最坏 channel 大量失配 | 多为参考自身 per-channel 舍入噪声，**非分块/尾块 bug** | 查最坏 channel 的 mod-64（或分块宽度）余数分布：均匀分布即与分块无关；再查候选归约是否纯 fp32（对精确值 ~1e-6 级即排除自身） |
| 同一代码多次官方评测，边界 case 时过时不过 | 参考模型惰性权重种子含 `torch.dtype` hash，受 **PYTHONHASHSEED 随机化** → 每次评测权重重抽 | 修复目标 matched_ratio 必须 ≥0.93（0.90-0.92 的边界修复会被权重重抽翻转）；floor/稳定性结论须跨多次官方运行确认 |
| evaluate_tilelang.sh 本地模式必然失败（GE init `cce.product_init` / libtilelang_module.so undefined symbol） | CANN 8.5.1 与 tilelang-ascend 双 libtvm.so 共存冲突（SONAME+全局函数注册表），同进程不可共存，set_env.sh 不能解 | TileLang 验证拆进程（参考与候选分开跑）或代理验证（参考跑 CPU）；勿投入修复框架 |

## 诊断流程

1. 从 DumpTensor 屏幕输出或重定向后的 debug 文件识别错误模式
2. 先回溯上游输入是否正确，再匹配上表中的根因
3. 应用针对性修复
4. 重新运行验证
