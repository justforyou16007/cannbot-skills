---
id: X61
origin: discovered
discovered_round: 3
discovered_from: round_3/parallel_1
base_speedup: 3.99x (T2, 相对 HEAD; 单步 +56% vs 父变体 n3)
---

# Strategy X61: 硬件转置指令替代 vgather 通路（v4dtrans + strided UB copy 交织）

## 核心思路
当数据重排的 vgather 吞吐被"每 repeat 上限元素数"钉死在发射率墙（如 m200 Gather
128 half/repeat，32 条微指令/窗口不可再压）时，检查芯片 platform_config 的
**Intrinsic 清单**寻找专用转置指令：310P3 无 v4trans 但**有 v4dtrans/vtranspose**。
`Transpose(dst, src, tmp, TransposeParamsExt{NCHW2NHWC})` 每次完成一个
[C,HW] 行主序 tile 的 [HW,C] 转置（实测几何 dst[hw*C+c]=src[c*HW+hw]）。把 gather
源布局重排为 [C,HW] tile 形态后，每窗（或每 tile）用 **2 次 Transpose（两路分量）+
2 次 strided UB->UB DataCopy（行交织到目标布局）** 替代全部 vgather 微指令——每窗
4 个发射单元替换 32 条微指令。UB->UB DataCopy 走 V-pipe copy_ubuf_to_ubuf
（dav_m200 支持，blockCount≤4095），同样不是 MTE 事务。

## 适用场景
- gather/转置主导的布局重排（ND→NZ/fractal、量化 nibble 分离转置、解交织）在
  "Gatherb NOT_SUPPORT + Gather 元素率上限"双卡死的老架构（m200/310P 等）
- 症状：Vector busy 高、每窗口固定 N 条 gather 微指令、发射率 ~4-5ns/指令
- 前置动作：grep platform_config ini 的 Intrinsic_v*trans 清单——v4dtrans/
  vtranspose 支持则此路通；v4trans 不支持不代表没救

## 实现要点
1. **几何验证**：v4dtrans 在 310P 官方文档无几何记录（归档"未实测"）。用仓内已有
   Transpose 参考实现（如 layer_norm_grad_v3 的 DoTranspose）+ dav_m200 Brcb 内
   8x8 v4dtrans 调用交叉推导，上板首测 bit-exact 验证
2. **双路交织**：lo/hi（或奇/偶行）两路分量各做一次 Transpose 到临时 tile（E/H），
   再用 2 次 strided DataCopy{32 块, 4 单位=128B, srcStride=0, dstStride=4} 交织到
   目标行布局——纯 bit 搬运，无数值变换
3. **暂存复用**：E/H 覆盖流水线上"此刻已死"的 buffer（如抽取完成后、下次 LoadBatch
   前的 winBuf 槽），不新增 UB；offset 表随 gather 一并删除（再省 16KB）
4. **同步**：Transpose 与 copy 都在 V-pipe，用 PIPE_V 屏障序；无新跨引擎边

## 来源
自动发现于 quant_batch_matmul_v4 第二轮进化 round_3/parallel_1（ops-nn，Ascend310P3，
W4Trans 段 vgather 消除）。相对 HEAD（已含 X59/X60 链）：T1 2.51x / T2 3.99x /
T3 3.97x / G1 3.34x / G2 1.73x，精度 5/5 bit-exact。流水线：Vector 567→163μs
（-404μs），MTE2 141μs 跃升为第二瓶颈。
