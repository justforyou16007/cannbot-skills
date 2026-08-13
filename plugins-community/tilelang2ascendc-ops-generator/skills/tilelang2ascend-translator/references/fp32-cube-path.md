# FP32 Cube 路径要点

A2/A3 上 fp16 能跑通的 cube 代码换 fp32 挂掉，几乎都源于分形几何差异：
**NZ/ZN 分形恒为 [16 行, C0 列]（512B），fp32 时 C0=8 而非 16**。
凡是把行数从 C0 推出来的代码（`m/C0`、`C0*i*k` 之类），fp16 下恰好等于 /16 全对，fp32 下全错。

## 规则清单

1. **分形行数恒为 16**。分形计数、grid stride、dst 行块偏移里的"16"禁止写成 C0。
   反例：L0B 分形数 `(k/C0)*(n/C0)` → fp32 过拷贝 2 倍溢出 L0B slot；正确 `(k/C0)*(n/16)`。

2. **Mmad fp32 的 M/N/K 必须 16 对齐**（`AlignUp(·,16)`，不是 C0=8），且
   `MmadParams.kDirectionAlign = true`。kvRowsAlign=24/40 这类 8 对齐值会让 NZ 列块
   stride 出现 1.5 个分形，数据全错。

3. **L0A（NZ→ZZ）**：`SetFmatrix(1, m, PAD, FMATRIX_LEFT)` + 原 LoadData3D 即可。
   SetFmatrix 是**粘性状态**，每次 fp32 LoadData3D 前必须重设（否则被上一次 RIGHT 污染）。

4. **L0B 需要转置时（B 存 [k,n]，如 P@V 的 V）**：普通 LoadData3D / LoadData2D 搬块都
   无法完成 [16,8]→[8,16] 分形内转置 + 网格重排。两条可用路径：
   - ZZ L1（Nd2Nz 变体：16 行一组，组 stride `AlignUp(w,16)*16`）+ `LoadDataWithTranspose`
     （`LoadData2dTransposeParams`：`repeatTimes=n/16`，`dstFracGap=n/16-1`，循环 k/16 次）；推荐
   - 或 NZ L1 保持 + `SetFmatrix(RIGHT)` + LoadData3D `fMatrixCtrl=true`
     （`kExtension=nAlign, mExtension=kAlign, channelSize=nAlign`）

5. **L0B 不需要转置时（B=K^T，K 按 [n,k] 存储）**：连续 LoadData2D 搬块即可，
   L1 NZ grid 与 L0B grid 天然一致，各 dtype 通用；只需按规则 1 算对分形数。

6. **L0C 与输入 dtype 无关**（累加器恒 fp32，分形 [16,16]），Fixpipe 路径不需要
   dtype 分支；但注意 `mSize=行 / nSize=列 / srcStride=行对齐 / dstStride=GM 行宽`，
   m/n 写反时方阵尺寸（64×64）恰好全对、非方阵（如 d_k=80）才暴露。

7. **Vec 侧**：fp32 直通跳过 `Cast`（float→float 的 CAST_ROUND 会把值舍入成 one-hot）；
   绕过 queue 的 EnQue/DeQue 后必须手动补 `V_MTE3`（算完再搬）和 `MTE3_MTE2`
   （搬完再复用 buffer）两个同步。

8. **Buffer 容量翻倍**：fp32 下 L0A/L0B 双 buffer 可能恰好顶满 64KB 硬件上限
   （如 2×[64,128]×4B），放不下就先降 queue 深度。
