---
id: X59
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_1
base_speedup: 3.25x (geomean 5 shapes; 310P W4A8 int4 权重分形转换)
---

# Strategy X59: Nibble 分形批转置（可分离偏移表 + 向量链整批 + DMA 合并）

## 核心思路
对"低精度打包数据（int4 nibble）→ Cube 分形布局（ZN fractal）"的 kernel 内转换预处理，
当逐小窗口（如 64k×32B）处理时向量指令序列化开销与碎 DMA 主导实际耗时（ Roofline 带宽
效率仅 ~2%），将转换粒度整批放大：同一 K 条带上连续多个窗口合并为一轮，nibble 提取的
half 域技巧链（And/Cast/Muls/Adds floor 技巧）一次跑满数千元素，把"每窗口 ~141 个小向量
op + ~170 个屏障"压成"每批 ~20 个大 op"；转置-打包用**单次大 gather + 可分离偏移表**
T[n][k] = P[n] + Q[k]（行常量 + 列常量的外积，用 Duplicate(P)+一次 Add(src1RepStride=0)
向量化构建，避免 N×K 次标量 SetValue 的 ~25us 级开销）；DMA 侧 MTE2 背靠背发射共享一组
屏障、MTE3 合并为一次连续大块拷贝（连续窗口的分形在目标区天然连续），且 fracBuf 复用
屏障推迟到下一批 Cast 之前，使 MTE3 与下一批向量链重叠。

## 适用场景
- 权重/激活量化打包格式的 kernel 内分形转换（int4/fp4 nibble → ZN fractal / ND→NZ 类），
  尤其在**无 vector ShiftLeft/ShiftRight 指令的老架构**（如 310P/DAV2002）上只能用 half
  域乘加技巧提取 nibble、每元素指令数多的场景
- 症状：转换段向量 op 密集且每个 op 后跟 PipeBarrier（同 pipe 按序发射，多数屏障可删/
  合并）；MTE2 以 32B 跨步小事务为主；整算子带宽效率 <10% 而计算强度低
- 转置类（Transpose/ND2NZ/格式重排）算子的向量化同样适用

## 实现要点
1. **批大小选择**：按 UB 预算取"同一 K 条带连续窗口数"最大化（本例 4 窗口 ≈ 176KB UB，
   含其他阶段驻留缓冲后 < 256KB）；批内 nibble 链一次 8192 元素
2. **可分离偏移表**：偏移 = 行基址 P[n]（Duplicate 广播）+ 列增量 Q[k]（Add 外积，
   src1RepStride=0 定源复读）——避免逐元素标量 SetValue；表项单位必须是字节（m200
   vgather 以字节解释偏移）
3. **数值红线**：half 域 floor 技巧常量（如 -0.46875 = floor(x/16)、-0.4375 = floor(x/8)）
   与符号修正（t - 16*floor(t/8)）必须与原实现逐位一致；批处理只是粒度放大不改公式
4. **同步守卫**：MTE2→vector 可见性屏障每批一次（背靠背 DMA 共享）；MTE3 写出与 fracBuf
   复用之间保留屏障但推迟到下一批 Cast 前，实现 MTE3/向量流水重叠；正确性加固同步
   （PIPE_ALL + MTE 守卫）不可省——省掉后实测出现偶发错值，加回后 evolved 绝对耗时
   +39% 但确定性正确（该 trade-off 值得）
5. **评估伪影提示（原地改写输入的算子）**：若算子预处理会原地改写输入（如 MsdPre 原地
   展开 x1），baseline 与 evolved 两侧性能/精度测量必须等调用次数（equal-call-count），
   否则出现伪精度失败与 ~1.5x 的虚高加速比

## 来源
自动发现于第 1 轮进化，算子 quant_batch_matmul_v4（ops-nn，Ascend310P3，W4A8 ND int4
→ I4ZN kernel 内转换），speedup 3.25x（T1）/3.53x（T2/T3）/3.50x（G1）/2.66x（G2），
target geomean 3.43x，5/5 shape 精度 bit-exact。
