# flash_attention FP32 支持与精度修复报告（2026-08-12）

## 最终结果

| 指标 | 结果 |
|---|---|
| **全量 13 个用例** | ✅ 全部通过（fp16 / bf16 / fp32） |
| **fp32 精度** | MERE 6.7e-8 ~ 1.4e-7（阈值 1.221e-4），matched_ratio = 1.0 |
| **d_k=80 fp16** | ✅ 通过（修复前 matched_ratio=0.016） |
| **matmul_tile.h** | 未改动（按约束保持现有 copy 函数） |

分支：`flash_attention_test_cases`（基于 `br_830_fa`）
提交：

- `9820c4c` test(flash_attention): add 5 test cases and fix build for 910B3
- `50729a6` wip(flash_attention): add FP32 entry, partial fixes and extended cases
- `3224100` fix(flash_attention): fix FP32 L0 load paths and d_k=80 remainder fixpipe

验证环境：Ascend 910B3 / CANN 9.0.0 / Python 3.12 / torch_npu 2.10.0。

---

## 1. 测试用例（general_case.csv，共 13 个）

| # | batch | heads | q/kv seq | d_k | dtype | 覆盖点 |
|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 64 | 128 | fp16 | 最小对齐 |
| 1 | 2 | 4 | 128 | 128 | fp16 | 多 batch/head |
| 2 | 1 | 2 | 64/128 | 128 | bf16 | kv > q |
| 3 | 1 | 1 | 256 | 128 | fp16 | 多 kv loop |
| 4 | 1 | 1 | 64 | 128 | bf16 | bf16 基线 |
| 5 | 4 | 10 | 101 | 64 | bf16 | 非对齐 seq（tail） |
| 6 | 3 | 12 | 131 | 64 | fp32 | fp32 + tail |
| 7 | 4 | 10 | 137 | 80 | fp16 | **d_k=80 remainder（原 bug）** |
| 8 | 2 | 9 | 89 | 64 | bf16 | 非对齐 seq |
| 9 | 2 | 24 | 419 | 64 | fp16 | 大 seq 多 loop |
| 10 | 4 | 4 | 17 | 32 | fp32 | fp32 极小 seq + 小 d_k |
| 11 | 4 | 8 | 79 | 64 | fp32 | fp32 + tail |
| 12 | 2 | 16 | 509 | 64 | fp32 | fp32 大 seq |

---

## 2. FP32 支持链路（前一阶段已完成）

FP32 入口为机械新增，不涉及本次核心 bug：

| 文件 | 内容 |
|---|---|
| `kernel/op_kernel/flash_attention_fp32.cpp` | `FlashAttentionKernel<float>` 实例化，`KERNEL_TYPE_MIX_AIC_1_2` |
| `kernel/CMakeLists.txt` | 加入 fp32.cpp |
| `kernel/ops.h` / `register.cpp` | `flash_attention_fp32` 声明与注册 |
| `kernel/op_host/flash_attention.cpp` | fp32 dispatch；workspace 按 `q.element_size()` 分配（原硬编码 `sizeof(uint16_t)`） |
| `model.py` / `model_new_ascendc.py` | fp32 dtype 映射与 dispatch；.so 后缀用 `sysconfig.EXT_SUFFIX` 动态获取 |

**vec.h 的 fp32 修复（参考 18_SelfAttention 的 Bug 1/2）：**

- `ComputeVec1`：fp32 跳过 `Cast`（float→float 的 `CAST_ROUND` 会把 P 舍入成 one-hot），直通 `DataCopy(pSlot, sUb)`；直通前补 `SetWaitFlag<V_MTE3>()`（否则 SoftmaxFlashV2 还没算完就被 MTE3 搬走），写完后补 `SetWaitFlag<MTE3_MTE2>()`（否则 inputQue1_ buffer 复用时被下个 chunk 的 MTE2 覆写）
- `FinalizeOutputChunk`：fp32 直接 `DataCopy(outGm, oUb)`，不经过 half Cast
- cube 侧对齐：`MMAD_ALIGN = fp32 ? 16 : C0`，所有 `AlignUp(·, C0)` 改 `AlignForMmad`（fp32 NZ 分形 [16 行, C0=8]，M/N/K 必须 16 对齐，否则列块 stride 出现 1.5 分形错位）；两处 Mmad 加 `kDirectionAlign = true`

这一阶段把 fp32 误差从 MERE ~1e6（完全错误）降到 0.002~0.12，但仍远超 1.221e-4 阈值 —— 剩余问题在 cube 的 L0 加载路径。

---

## 3. 本次修复的 4 个 bug（全部在 cube.h）

### Bug A — MM1 K→L0B 分形计数 2 倍过拷贝（fp32）

**位置**：`ComputeMM1` 两处 `DoMM1Tile` 调用的 `kRepeatTimes` 实参

**机制**：
- NZ/ZN 分形恒为 **16 行 × C0 列**（512B，dtype 无关）
- L0B 需要的分形数 = `(k/C0) × (n/16)`（k 按 C0 分列、n 按 16 分行）
- 原代码两个维度都除以 `C0`：fp16 时 C0=16 恰好正确；fp32 C0=8 时 n 维多算一倍 → **拷贝量 2 倍于有效数据**，溢出 L0B slot 写进相邻 ping-pong buffer

**修复**：

```cpp
BASE_K, (BASE_K / C0) * (kvRowsAlign / 16U),   // 原: (kvRowsAlign / C0)
kAligned, (kAligned / C0) * (kvRowsAlign / 16U),
```

fp16/bf16 下 `kvRowsAlign/16 == kvRowsAlign/C0`，行为不变。

### Bug B — MM1 Q→L0A 的 LoadData2D 循环 C0/16 混淆（fp32）

**位置**：`DoMM1Tile` 的 L0A 加载循环

**机制**：原自写循环三处把"16"写成了"C0"：
- 循环次数 `mActAlign / C0`（行分形数应为 `mActAlign/16`）：fp32 下 8 次而非 4 次
- `srcStride = mActAlign / C0`（L1 每个 k 列块的分形数应为 `/16`）：fp32 下 stride=8 而非 4
- dst 偏移 `aL0[C0 * i * kSize]`（每个 16 行块应为 `16*i*kSize` 元素）：fp32 下相邻行块**重叠写一半**

fp16 时 C0=16 三处全部恰好正确，所以 fp16 一直能过。

**修复**：fp32 改走参考实现验证过的配方 —— `SetFmatrix(LEFT) + LoadData3D`（即 matmul_tile.h 现有的 `LoadNzL1ToZzL0A`，调用前设置粘性 fMatrix 状态）；fp16/bf16 保持原 LoadData2D 循环不动：

```cpp
if constexpr (IS_FP32) {
    constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
    SetFmatrix(1, static_cast<uint16_t>(mActAlign), PAD_LIST, FmatrixMode::FMATRIX_LEFT);
    LoadNzL1ToZzL0A(aL0, qL1[qL1Off], mActAlign, kSize, mActAlign);
} else {
    // 原 LoadData2D 循环，fp16/bf16 专用（C0=16 时语义正确）
}
```

### Bug C — MM2 V→L0B 需要分形内转置，普通 LoadData3D/LoadData2D 做不到（fp32 核心 bug）

**位置**：`ComputeMM2Tile` 的 V 加载

**机制**：
- MM2 的 B 矩阵是 V 本身 `[k=kv, n=dim]`，L1 NZ 分形为 `[16 kv][C0 dim]`
- fp32 下 L0B 目标布局（nZ）分形为 `[C0 kv][16 dim]`，需要做 **[16,8]→[8,16] 分形内转置 + 网格重排**
- fp16 分形是 [16,16] 方阵，转置对称性让 `LoadNzL1ToZnL0B`（LoadData3D）恰好能处理；fp32 非对称分形下它和普通 LoadData2D 搬块都产生错误布局
- 对比：MM1 的 B = K^T，K 按 `[kv, dim]` 存储天然等于 B 的转置存储，所以 MM1 连续拷贝即可（Bug A 修好计数就行）；MM2 没这个运气

**修复**（参考实现的官方 ZZ 路径，easyasc tensorutils 复刻）：

1. V 的 GM→L1 改用 **ZZ 布局**（`GmToL1Nd2ZzFp32`：按 16 行分组，组内 NZ 分形 [16, C0=8]，组 stride = `AlignUp(w,16)*16`，尾行单独一个 Nd2Nz）
2. L1→L0B 用 **`LoadDataWithTranspose`**（`L1ZzToL0BZnFp32`：`repeatTimes=n/16`，`dstFracGap=n/16-1`，循环 `k/16` 次）—— 这是唯一能同时完成分形内转置和网格重排的 MTE1 原语

```cpp
if constexpr (IS_FP32) {
    GmToL1Nd2ZzFp32(kvL1, vGm_[vSliceOffset], kvRows, nSize, tiling_.dim);
    ...
    L1ZzToL0BZnFp32(bL0, kvL1, kvRowsAlign, nSize);
} else {
    LoadNdGmToNzL1(kvL1, vGm_[vSliceOffset], kvRows, nSize, tiling_.dim, kvRowsAlign);
    ...
    LoadNzL1ToZnL0B(bL0, kvL1, kvRowsAlign, nSize, kvRowsAlign);
}
```

两个 helper 写在 `flash_attention_cube.h` 内（遵守"不改 matmul_tile.h"约束），仅 fp32 分支实例化。

### Bug D — MM2 remainder 的 Fixpipe m/n 写反（d_k=80 根因，影响全 dtype）

**位置**：`FixpipeMM2Remainder`

**机制**：原实现 `FixpipeNzL0cToNdGm(oSlot[wsBase], cL0, nAligned, BLOCK_M)` 展开后：

| 参数 | 错误值 | 正确值 |
|---|---|---|
| mSize（行数） | 80（nAligned） | 64（BLOCK_M） |
| nSize（列数） | 64（BLOCK_M） | 80（nAligned） |
| srcStride（L0C 行分形 stride） | 80 | 64 |
| dstStride（GM 行 stride） | 64 | dimAlign_=80 |

d_k=64 时 nAligned=BLOCK_M=dimAlign_=64，**四个参数恰好全对**，所以 d_k=64 的 case 一直能过；d_k=80（非 BASE_K 整数倍）第一次走到这条路径就全错（matched_ratio=0.016）。

**修复**：改为与 `FixpipeMM2Main` 一致的显式参数：

```cpp
fixParams.mSize     = BLOCK_M;
fixParams.nSize     = nAligned;
fixParams.srcStride = AlignForMmad(BLOCK_M);
fixParams.dstStride = dimAlign_;
```

---

## 4. 验证结果（2026-08-12）

```
case[0]  fp16  1x1x64x128     matched_ratio=1.0  MERE=6.36e-05 (thr 9.77e-04)
case[1]  fp16  2x4x128x128    matched_ratio=1.0  MERE=4.45e-05
case[2]  bf16  1x2x64/128x128 matched_ratio=1.0  MERE=3.85e-04 (thr 7.81e-03)
case[3]  fp16  1x1x256x128    matched_ratio=1.0  MERE=3.25e-05
case[4]  bf16  1x1x64x128     matched_ratio=1.0  MERE=4.67e-04
case[5]  bf16  4x10x101x64    matched_ratio=1.0  MERE=3.16e-04
case[6]  fp32  3x12x131x64    matched_ratio=1.0  MERE=9.51e-08 (thr 1.22e-04)
case[7]  fp16  4x10x137x80    matched_ratio=1.0  MERE=3.84e-05   ← d_k=80 修复
case[8]  bf16  2x9x89x64      matched_ratio=1.0  MERE=3.21e-04
case[9]  fp16  2x24x419x64    matched_ratio=1.0  MERE=1.88e-05
case[10] fp32  4x4x17x32      matched_ratio=1.0  MERE=6.69e-08
case[11] fp32  4x8x79x64      matched_ratio=1.0  MERE=8.03e-08
case[12] fp32  2x16x509x64    matched_ratio=1.0  MERE=1.41e-07
Result: pass
```

fp32 精度比参考实现报告的值（max_abs_diff ≤ 1.2e-6）还好约一个量级。

---

## 5. 修改文件清单

| 文件 | 修改 |
|---|---|
| `kernel/op_kernel/flash_attention_cube.h` | Bug A/B/C/D + 两个 fp32 ZZ helper（GmToL1Nd2ZzFp32 / L1ZzToL0BZnFp32） |
| `kernel/op_kernel/flash_attention_vec.h` | fp32 跳过 Cast + V_MTE3/MTE3_MTE2 同步；输出直通 |
| `kernel/op_kernel/flash_attention_fp32.cpp` | 新增：fp32 kernel 实例化 |
| `kernel/op_kernel/kernel_common.h` | 内联 CopyTiling（替代缺失的 common/tiling_copy.h） |
| `kernel/op_host/flash_attention.cpp` | fp32 dispatch；workspace element_size 修正 |
| `kernel/ops.h` / `register.cpp` / `CMakeLists.txt` | fp32 注册与构建 |
| `kernel/setup.py` | 新增（复用 _kernel_setup_common） |
| `model.py` / `model_new_ascendc.py` | fp32 dtype 支持；动态 .so 后缀 |
| `general_case.csv` | 新增 13 个用例 |
| `kernel/op_kernel/matmul_tile.h` | **未改动** |

## 6. 已知边界（当前用例未覆盖，暂不影响）

- fp32 下 dim 非 16 对齐（如 dim=40）：`AlignForMmad` 会把 remainder 对到 48，GM 读取越出本行、Fixpipe 写出超出 dimAlign —— 如需支持要在 host 侧拒绝或单独处理
- LoadQ 对 q 尾块固定读 BLOCK_M 行，最后一个 (batch, head) 会越界读 GM 尾部（fp16 同样存在，靠 GM 余量没炸；输出侧有 clamp 兜底）
- fp32 L0A/L0B 双 buffer 恰好占满 64KB 硬件上限（2×32KB），暂未观察到分配失败
