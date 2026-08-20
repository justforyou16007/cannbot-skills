---
id: X54
origin: discovered
discovered_round: 3
discovered_from: "76_BatchMLAPagedAttention evolution round_3/parallel_2 (x0_x1_x1, open_exploration 延续)"
base_speedup: "全量 50 case msprof quick geomean 2.70x vs PyTorch（基线 0.113x -> 23.9x），39/50 case>1x，case43 0.043->0.807（19x）、case28 0.281->6.11、case39 0.093->1.161"
---

# Strategy X54: (q, head-block) 任务切分 + 全宽 per-head 归约（消除归约指令爆炸）+ 双缓冲 tile gather + PV 行树归约

## 核心思路
Batch MLA PagedAttention decode（nope/kpe split-KV）父实现工作项 = (q, 全部 H head)，对 H=32/Dc=512 等长 KV 大 H case 有三个结构缺陷：
1. **并行度不足**：任务数 = totalQ；case43 totalQ=10 → 40 个 AIV 核只用 10 核（25% 利用率）。
2. **归约指令爆炸**：QK^T 按 64 宽 chunk 归约，per tile 的 ReduceSum 调用数 = H×nDc（case43: 32×8=256/tile），且每 chunk 一次 PipeBarrier<PIPE_ALL>（~15 次/tile），同步空泡巨大。
3. **UB 被 accO/qNopeF 占满**：H=32/Dc=512 时各 64KB → T_ 仅 4。

两个正交杠杆消除上述瓶颈：
1. **(q, head-block) 任务切分**：head 维切 nHb 块，任务数 = totalQ×nHb。host 按 AIV 核数选 hb（**先用尽核，再少 KV 重读**），case43 → hb=8/nHb=4 → 40 任务铺满全部核。
2. **全宽 per-head 归约**：连续 DMA 拉 [Tn,Dc]/[Tn,Dr] 整块，每 head 对 nope/kpe 各 1 次 ReduceSum → [Tn]。ReduceSum 调用数 = 2H/tile（case43: 64/tile vs 原 256/tile）。

配套：**PV 全宽行树归约**（e 广播→[Tn,Dc] 积→对半行树归约，log2 级）+ **双缓冲 tile gather**（连续 DMA 预取下一 tile MTE2 与当前 tile 计算 V 重叠，消除内存延迟暴露，H=1 大 KV case 关键）+ **同步最小化**（每 tile 仅 1 次 PipeBarrier<PIPE_ALL>）。

## 适用场景
- decode 场景 Batch MLA / PagedAttention（nope/kpe split-KV、大 H 长 KV）向量内核
- 并行度不足 case（totalQ×H < AIV 核数，任务数=totalQ 导致欠核）→ head-block 切分填核
- QK^T 归约指令爆炸（按 64 宽 chunk 归约，ReduceSum 调用数 = H×nDc/tile）
- KV tile gather 内存延迟暴露（H=1 大 KV case）→ 双缓冲 tile gather

## 实现要点
1. host 搜索 (hb, T_)：T_∈{128..4}，hb∈{H..1}（nHb≤8），先铺满核再少 KV 重读，UB≤172KB。
2. 全宽归约：连续 DMA 整块拉取 + 每 head 单次 ReduceSum({Tn,Dc})/ReduceSum({Tn,Dr}) → [Tn]。
3. 双缓冲 tile gather：MTE2 预取下一 tile 与 V 计算重叠，消除 GM 延迟暴露。
4. PV 行树归约：e 广播 [Tn,Dc] 积 → 对半行树 ReduceSum（log2 级），累加 accO。
5. 同步最小化：每 tile 仅 1 次 PipeBarrier<PIPE_ALL>，向量同管 RAW 由硬件处理。

## 来源
76_BatchMLAPagedAttention 进化 round_3/parallel_2（x0_x1_x1，open_exploration 延续）：全量 50 case msprof quick（device 15）geomean 2.70x vs PyTorch（基线 0.113x → 23.9x），39/50 case>1x，case43 0.043→0.807（19x）、case28 0.281→6.11、case39 0.093→1.161；残余慢 case（case49 0.385/case43 0.807 等长 KV 高 H）仍由 ReduceSum 调用开销主导，需 Cube 化（P120）进一步突破。
