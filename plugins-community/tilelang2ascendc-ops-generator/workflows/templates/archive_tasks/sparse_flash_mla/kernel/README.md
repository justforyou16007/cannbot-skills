# sparse_flash_mla kernel

This directory ships the **canonical AscendC implementation** of
`SparseFlashMla`（照搬自 ops-transformer `attention/sparse_flash_mla` 源码，
含 op_host + op_kernel 全部文件，未做修改），并配套了本机 CANN 框架的
**构建/部署/直调**路径，与 `flash_attention_gqa` 的 kernel 目录同构：

```
kernel/
├── op_host/                  # 打包算子 host 侧（照搬）
│   ├── sparse_flash_mla_def.cpp     # OpDef（输入/输出/属性定义）
│   ├── sparse_flash_mla_proto.cpp   # RegisterOperator 注册
│   ├── sparse_flash_mla_tiling.cpp  # tiling（metadata 任务划分 + tiling 数据）
│   └── sparse_flash_mla_tiling.h    # SparseFlashMlaTilingData 结构
├── op_kernel/                # 打包算子 kernel 侧（照搬，TTK 模板算子）
│   ├── sparse_flash_mla.cpp                  # 模板 kernel 入口（22 个 GM 指针）
│   ├── sparse_flash_mla_metadata.h           # FA/FD metadata 格式
│   ├── sparse_flash_mla_template_tiling_key.h # ASCENDC_TPL_ARGS_DECL 模板键
│   ├── arch22/               # A2（910B）实现：swa/csa kernel + block cube/vector
│   └── arch35/               # A3/950 实现：swa/csa kernel + kvcache/flash_decode
├── sparse_flash_mla_kernel_swa_decode.cpp   # TileLang 设计对 swa_decode 配置的发射快照
└── README.md                 # 本文档
```

## 为什么不能用 gqa 那种 ascendc cmake 直启构建

`sparse_flash_mla.cpp` 是 **TTK 模板算子**（`ASCENDC_TPL_ARGS_DECL` 声明
7 个模板参数的组合空间，`ASCENDC_TPL_PRE/ASCENDC_TPL_KERNEL` 预处理机制由
CANN 算子框架构建工具链驱动），CANN 9.0 的 `ascendc_kernel_cmake`
（legacy_modules）只支持普通 `__global__ __aicore__` 函数的
`ascendc_library` 构建，**不支持 TTK 模板实例化**。因此必须走 CANN 算子
框架构建（即 ops-transformer 的构建系统），产物由框架运行时加载。

## 本机构建与部署（已验证，910B2C + CANN 9.0.0）

前置：`ops-transformer` 工作区（`/home/y00889327/ops-transformer`）已配置
好 CANN 构建环境（`source $ASCEND_HOME_PATH/set_env.sh`）。

```bash
cd /home/y00889327/ops-transformer/build
# 1) 只配置 sparse_flash_mla（+ metadata 算子）两个算子
cmake .. -DASCEND_OP_NAME=sparse_flash_mla,sparse_flash_mla_metadata
# 2) 构建 kernel（TTK 实例化，asc_opc 编译）
cmake --build . --target sparse_flash_mla_ascend910b -j $(nproc)
# 3) 构建 host 侧（aclnn 接口 + tiling + proto + opmaster）
cmake --build . --target op_host_aclnn cust_opapi optiling cust_opmaster \
                       cust_proto cust_aicpu_kernels ops_config_ascend910b -j $(nproc)
# 4) 组装并安装 custom_transformer vendor 到 CANN OPP
cmake --build . --target install -j $(nproc)
cd /home/y00889327/ops-transformer/build_out && bash install.sh
# 5) 运行期环境
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

产物：

- kernel 二进制：`build/binary/ascend910b/bin/sparse_flash_mla/SparseFlashMla_*.o`
  （2 个 TTK 实例，`gen/` 下有编译日志）
- aclnn 接口：`build/libcust_opapi.so` 导出 `aclnnSparseFlashMla` /
  `aclnnSparseFlashMlaGetWorkspaceSize` / `aclnnSparseFlashMlaMetadata` /
  `aclnnSparseFlashMlaMetadataGetWorkspaceSize`
- 安装位置：`$ASCEND_OPP_PATH/vendors/custom_transformer/`（config.ini 里
  `load_priority=custom_transformer`）

## 调用契约（model_new_ascendc.py 使用）

算子为 metadata-first：先 `sparse_flash_mla_metadata`（AICPU 算子，输出
`[1024]` int32 的 FA/FD 任务表），再 `sparse_flash_mla`（TTK 主算子）。
BSND 布局张量（batch, seq, num_heads, head_dim），KV_N=1、D=512 为算子约束，
K 与 V 共享存储（ori_kv/cmp_kv）。

```python
metadata = torch.ops.npu.sparse_flash_mla_metadata(
    num_heads_q, num_heads_kv, head_dim,
    seqused_q, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv,
    batch_size, max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv,
    ori_topk, cmp_topk, cmp_ratio,
    ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
    layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=...,
)
out, _lse = torch.ops.npu.sparse_flash_mla(
    q, ori_kv=..., cmp_kv=..., cmp_sparse_indices=..., sinks=..., metadata=...,
    softmax_scale=..., cmp_ratio=..., ori_mask_mode=..., cmp_mask_mode=...,
    ori_win_left=..., ori_win_right=...,
    layout_q="BSND", layout_kv="BSND", topk_value_mode=1, return_softmax_lse=False,
)
```

`torch.ops.npu.*` 由 kernel/build 下构建的 torch 扩展注册（直调适配层，见
`kernel/op_host/sparse_flash_mla.cpp` + `kernel/register.cpp`，aclnn 调用照搬
ops-transformer `torch_extension/cann_ops_transformer/ops/csrc/sparse_flash_mla.cpp`
模式；`model_new_ascendc.py` 优先 import 该扩展，也可回退到 CANN 打包的
`cann_ops_transformer` 同名接口）。

### 与打包实现核对过的契约细节

- **`sinks` 必填**：tiling 的 `GetSinks` 拒绝 nullptr（产品 README 虽标为
  可选）；传 `[H]` 零向量等价 golden 的无 sinks 语义。
- **`cmp_residual_kv` 在 `cmp_kv` 存在且 `cmp_mask_mode == 3`、
  `cmp_ratio != 1` 时必填**（`CheckSingleParaCmpResidualKv`）：恢复压缩前
  长度供 cmp 侧 causal 阈值计算 `ori_len = cmp_len * ratio + residual`；
  全量 BSND 时按 `seqused_ori_kv=[ori_seq]`、`seqused_cmp_kv=[cmp_seq]`、
  `cmp_residual_kv=[ori_seq % cmp_ratio]` 传参（与参考 pytest 一致）。
- **CSA topk 语义**（对照 arch35 CSA kernel `ComputeS2LoopInfo`/
  `GetRealCmpS2Idx` 验证）：每查询行仅考虑 `cmp_sparse_indices` 行的前
  `min(topk, threshold)` 项；`-1` 终止扫描；`>= threshold` 的项被跳过但
  扫描继续。`topk_value_mode=1`。
- **metadata 格式**：`[1024]` int32 = FA 槽（36 AIC 核 × 9 字段）+ FD 槽
  （72 AIV 核 × 8 字段），见 `op_kernel/sparse_flash_mla_metadata.h`。
- **平台约束**：A5/950PR：`Q_N` ∈ [2, 128]，`ori_mask_mode` ∈ {0, 3, 4}，
  `cmp_mask_mode` ∈ {0, 3}，`cmp_ratio` ∈ [1, 128]；A2/A3：`ori_mask_mode`
  仅 4、`cmp_mask_mode` 仅 3、win 127/0，HCA `cmp_ratio` = 128，CSA
  `cmp_ratio` = 4，`topk` ∈ {512, 1024}。

### 场景

| scenario | ori_kv | cmp_kv | cmp_sparse_indices |
|----------|--------|--------|--------------------|
| SWA (sliding-window attention) | 全量 KV，band/causal/no-mask 窗口 | 不用 | 不用 |
| HCA (heavily compressed attention) | KV 窗口 | 压缩前缀，前 `floor(causal / ratio)` 行 | 不用 |
| CSA (compressed sparse attention) | KV 窗口 | 压缩前缀 | 每查询 top-k 行索引（前 min(K, threshold) 项、-1 停止） |

## 部署注意事项（本机验证过程中确认）

- TTK kernel 只能用框架构建（见上），`ascendc_kernel_cmake` 无法直启；
- vendor 布局需与框架期望一致：`op_tiling/lib/linux/x86_64/liboptiling.so`、
  `libcust_opmaster_rt2.0.so` 在 `op_impl/ai_core/tbe/` 下，AICPU kernel
  （metadata）在 `op_impl/cpu/aicpu_kernel/impl/`，配置在
  `op_impl/cpu/config/`；
- 运行期 `BinaryGetFunction` 失败排查：确认 `binary_info_config.json`
  （kernel 注册表，`op_impl/ai_core/tbe/kernel/config/ascend910b/`）由
  `ops_config_ascend910b` 目标重新生成（旧构建配置生成的 json 只含当时
  配置的算子）、AICPU kernel json 已随 install 安装、LD_LIBRARY_PATH 含
  vendored `op_api/lib`。

## TileLang 设计发射快照

`design/tile_level/sparse_flash_mla.py` 是 SWA/HCA 的 TileLang 设计（单实例
1D 网格、ori 窗口 + cmp 前缀单循环、flash-softmax 携带 sinks 虚拟 key，
已在 910B 上对 general_case.csv 全部用例验证）。其发射的 AscendC 对头号
`swa_decode` 配置的快照见：

- `sparse_flash_mla_kernel_swa_decode.cpp`

重新生成（任一配置）：

```python
import sys
sys.path.insert(0, "design")
from tile_level.sparse_flash_mla import sparse_flash_mla_fwd, SparseFlashMlaConfig
cfg = SparseFlashMlaConfig(batch=1, q_seq=1, heads=128, dim=512, ori_seq=2048,
                           cmp_seq=0, cmp_ratio=1, ori_mask_mode=4,
                           cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
                           dtype="bfloat16")
print(sparse_flash_mla_fwd(cfg).get_kernel_source())
```

CSA（cmp_kv 行的 topk gather）尚不能以分支无关 TileLang kernel 表达；
`model_new_tilelang.py` 对其抛 `NotImplementedError`，golden（`model.py`）
与 AscendC 路径保持参考——扩展点讨论见
`design/block_level/sparse_flash_mla.py`。
