#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# flash_attention 手写同步变体：AUTO_SYNC 关闭，全部旗语显式。
#
# 结构：C1/V1 生产者 + C2/V2 消费者，prelaunch=2 / ring=3 耦合环；
# K 双槽预取；V2 压缩（免零初始化 + 整幅 broadcast-mul / 行循环分档）；
# V1 brcb-sub（dim<=128 门控，brcb 物化 64x8 替代整幅 2D 广播物化）。
#
# 旗语清单（sync_only + flag_blocking 模拟审计导出）：
#   RAW（同迭代）：
#     V1: V->MTE2(1) acc_s_ub 下一载；MTE3->V(1) acc_s_half cast 前
#   跨核（设计原生）：SIG_S/P/O_READY 不变。
#   同管 V 链（mul/sub/exp/reduce/broadcast/div）零屏障——in-order FIFO
#   （CANN sample 裸链 + sync pass 同管免屏障规则）。
# ----------------------------------------------------------------------------------------------------------
from dataclasses import dataclass
import os

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout
from tvm.tir import BufferRegion
from tvm.ir import Range


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,   # 手写同步：编译器不插
}

# 模拟器回退开关（GQA/s 系列同款）：row_expand 切片形态超 sim bridge
# validator，模拟构建回退行循环标量 sub（同步/数据流同构）
_SIM_NO_BRCB = os.environ.get("FA_SIM_NO_BRCB") == "1"

SIG_S_READY = 0   # C1 -> V1: ws_s 槽就绪
SIG_P_READY = 1   # V1 -> C2: ws_p 槽就绪
SIG_O_READY = 2   # C2 -> V2: ws_o 槽就绪

MAX_CORES = 20
# dim<=512：dim_align 分档定 block_m 保 L0C/UB 容量
MAX_DIM = 512


@dataclass(frozen=True)
class FlashAttentionConfig:
    batch: int
    heads: int
    q_seq_len: int
    kv_seq_len: int
    dim: int
    dtype: str = "float16"
    causal: bool = False


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs, target='ascendc')
def flash_attention_handsync_fwd(cfg: FlashAttentionConfig):
    batch = cfg.batch
    heads = cfg.heads
    q_seq_len = cfg.q_seq_len
    kv_seq_len = cfg.kv_seq_len
    dim = cfg.dim
    dtype = cfg.dtype
    causal = cfg.causal

    if dim > MAX_DIM:
        raise ValueError(f"head dim {dim} exceeds MAX_DIM={MAX_DIM}")

    c0 = 32 // (4 if dtype == "float32" else 2)
    dim_align = ((dim + c0 - 1) // c0) * c0
    if dim_align > 128:
        # chunk 宽度对齐：dim>128 时分配宽度按 128 整 chunk 取齐——
        # 否则 dim_align=208 这类宽度会让 chunk 的 L0 装载/ws_o 落盘切片
        # 越界 48 列（case56 全行垃圾的根因）；填充区零值不参与计算。
        dim_align = ((dim + 127) // 128) * 128

    # dim_align 越大 block_m 越小，L0C(acc_o=bm*128*4B + acc_s=bm*128*4B)
    # 与 UB 守恒
    if dtype == "float32":
        # fp32 元素 4B：L0A 放不下 128 行 lhs+p，block_m 减半避免别名
        block_m = 64 if dim_align <= 128 else 32
    elif dim_align <= 128:
        block_m = 128
    else:
        block_m = 64   # dim>128 也 64
    block_n = 128
    cube_k = dim_align if dim_align < 256 else 256
    prelaunch = 2
    ring_slots = prelaunch + 1
    accum_dtype = "float"

    q_blocks = (q_seq_len + block_m - 1) // block_m
    block_num = q_blocks * heads * batch
    used_core_num = MAX_CORES
    tasks_per_core = (block_num + used_core_num - 1) // used_core_num
    kv_loops = (kv_seq_len + block_n - 1) // block_n
    qk_k_loops = (dim + cube_k - 1) // cube_k
    d_chunks = (dim_align + 255) // 256   # K 维 chunk 数（cube_k=256）

    dim_pad = dim_align
    # causal 右下对齐（GQA golden 同语义）：行 i 可见列 j <= i + diff_s
    diff_s = kv_seq_len - q_seq_len
    if causal and q_seq_len > kv_seq_len:
        raise ValueError("causal requires q_seq_len <= kv_seq_len")

    q_shape = [batch, heads, q_seq_len, dim]
    kv_shape = [batch, heads, kv_seq_len, dim]
    out_shape = [batch, heads, q_seq_len, dim]

    tail_valid = kv_seq_len % block_n
    # brcb 门控：dim>128 档 acc_o/acc_o_ub 各 64KB，UB 无余量（brcb_tmp
    # 虽小但无处可放）；dim<=128 档非 causal 余 ~30KB，2KB 轻松
    use_brcb = dim_align <= 128

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, dtype),
        k: T.Tensor(kv_shape, dtype),
        v: T.Tensor(kv_shape, dtype),
        output: T.Tensor(out_shape, dtype),
        workspace_s: T.Tensor([block_num, ring_slots, block_m, block_n], accum_dtype),
        workspace_p: T.Tensor([block_num, ring_slots, block_m, block_n], dtype),
        workspace_o: T.Tensor([block_num, ring_slots, block_m, dim_align], accum_dtype),
        sm_scale: T.float32,
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_m, dim_pad], dtype)
            k_l1 = T.alloc_L1([2, block_n, dim_pad], dtype)   # K 双槽（kv_slots=2）
            v_l1 = T.alloc_L1([block_n, dim_pad], dtype)
            acc_s_l1 = T.alloc_L1([block_m, block_n], dtype)
            T.annotate_layout({
                q_l1: make_zn_layout(q_l1),
                k_l1: make_zn_layout(k_l1),
                v_l1: make_zn_layout(v_l1),
                acc_s_l1: make_zn_layout(acc_s_l1),
            })

            lhs_l0 = T.alloc_L0A([block_m, cube_k], dtype)
            rhs_l0 = T.alloc_L0B([cube_k, block_n], dtype)
            p_l0 = T.alloc_L0A([block_m, block_n], dtype)
            v_l0 = T.alloc_L0B([block_n, cube_k], dtype)
            acc_s_l0c = T.alloc_L0C([block_m, block_n], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_m, cube_k], accum_dtype)

            acc_o = T.alloc_ub([block_m // 2, dim_align], accum_dtype)
            acc_o_ub = T.alloc_ub([block_m // 2, dim_align], accum_dtype)
            acc_o_half = T.alloc_ub([block_m // 2, dim_align], dtype)
            sumexp = T.alloc_ub([block_m // 2, 1], accum_dtype)
            m_i = T.alloc_ub([block_m // 2, 1], accum_dtype)
            alpha_ring = T.alloc_ub([ring_slots, block_m // 2, 1], accum_dtype)
            sumexp_i_ring = T.alloc_ub([ring_slots, block_m // 2, 1], accum_dtype)
            acc_s_ub = T.alloc_ub([block_m // 2, block_n], accum_dtype)
            acc_s_half = T.alloc_ub([block_m // 2, block_n], dtype)
            m_i_2d = T.alloc_ub([block_m // 2, block_n], accum_dtype)
            # brcb 物化 64x8（不用时缩成 8 元素哑元，分配无条件——
            # "if 内 alloc 不可见"陷阱，尺寸走 builder 常量三元）
            brcb_tmp = T.alloc_ub(
                [block_m // 2 * 8 if use_brcb else 8], accum_dtype)
            mask_col = T.alloc_ub([block_n], accum_dtype)
            # causal（仅 causal=True 时使用；col_2d 复用 m_i_2d——mask 在
            # reduce_max 前消费完，m_i_2d 的真实用途在其后）
            col_idx = T.alloc_ub([block_n], accum_dtype)
            bound_vec = T.alloc_ub([block_m // 2], accum_dtype)
            bound_2d = T.alloc_ub([block_m // 2, block_n], accum_dtype)
            causal_cmp = T.alloc_ub([block_m // 2, block_n], "uint8")

            for local_idx in T.serial(tasks_per_core):
                task_idx = cid * tasks_per_core + local_idx
                if task_idx < block_num:
                    bx = task_idx % q_blocks
                    by = task_idx // q_blocks % heads
                    bz = task_idx // q_blocks // heads % batch

                    with T.Scope("C"):
                        T.copy(q[bz, by, bx * block_m: (bx + 1) * block_m, 0:dim], q_l1)
                        # K 槽信用预置 + 预加载 K(0) 到槽 0（K 双缓冲预取）
                        T.set_flag("mte1", "mte2", 0)   # K 槽 0 空闲
                        T.set_flag("mte1", "mte2", 1)   # K 槽 1 空闲
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(k[bz, by, 0:block_n, 0:dim], k_l1[0, :, :])
                        T.set_flag("mte2", "mte1", 0)   # K 槽 0 就绪

                    with T.Scope("V"):
                        # 免零初始化：acc_o/sumexp 由 V2(0) 直接建立；
                        # 单元素填充只是活性锚点——免填后 planner 会把 acc_o
                        # 活区间收缩到首个 V2 之后并与 acc_s_ub 别名
                        # （memplan 别名 WAR 缺口），锚点拉回 task 首
                        T.tile.fill(acc_o[0, :], 0.0)
                        T.tile.fill(sumexp[0, :], 0.0)
                        T.tile.fill(m_i, -2 ** 30)

                    for t in T.serial(kv_loops + prelaunch):
                        # ---------- 生产者：C1(t) / V1(t) ----------
                        if t < kv_loops:
                            slot_prod = t % ring_slots

                            with T.Scope("C"):
                                # 预取 K(t+1) 到槽 (t+1)%2（MTE2，与下方 QK 的 Cube/MTE1 重叠）
                                if t + 1 < kv_loops:
                                    T.wait_flag("mte1", "mte2", (t + 1) % 2)
                                    T.copy(k[bz, by, (t + 1) * block_n: (t + 2) * block_n,
                                             0:dim], k_l1[(t + 1) % 2, :, :])
                                    T.set_flag("mte2", "mte1", (t + 1) % 2)
                                # QK 用槽 t%2 的 K(t)
                                T.wait_flag("mte2", "mte1", t % 2)
                                for cc in T.serial(d_chunks):
                                    T.set_flag("m", "mte1", 1)
                                    T.wait_flag("m", "mte1", 1)
                                    T.copy(q_l1[0, cc * cube_k], lhs_l0)
                                    T.copy(k_l1[t % 2, 0, cc * cube_k], rhs_l0, transpose=True)
                                    T.set_flag("mte1", "m", 0)
                                    T.wait_flag("mte1", "m", 0)
                                    T.set_flag("fix", "m", 1)
                                    T.wait_flag("fix", "m", 1)
                                    T.mma(lhs_l0, rhs_l0, acc_s_l0c, init=(cc == 0))
                                T.set_flag("m", "fix", 0)
                                T.wait_flag("m", "fix", 0)
                                T.copy(acc_s_l0c, workspace_s[task_idx, slot_prod, :, :])
                                # 释放槽 t%2（供 C1(t+1) 预取复用）
                                T.set_flag("mte1", "mte2", t % 2)
                                # C1 跨迭代消费者完成标记（供 ring 尾 drain）
                                T.set_flag("m", "mte1", 3)
                                T.set_flag("fix", "m", 3)
                                T.set_cross_flag("FIX", SIG_S_READY)

                            with T.Scope("V"):
                                T.wait_cross_flag(SIG_S_READY)
                                h_start = vid * (block_m // 2)
                                T.set_flag("v", "mte2", 1)
                                T.wait_flag("v", "mte2", 1)
                                T.copy(m_i, alpha_ring[slot_prod, :, :])
                                T.copy(
                                    workspace_s[task_idx, slot_prod,
                                                h_start: h_start + block_m // 2, :],
                                    acc_s_ub,
                                )
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                if tail_valid != 0:
                                    if t == kv_loops - 1:
                                        T.tile.fill(mask_col, T.float32(-2 ** 30))
                                        T.tile.fill(
                                            BufferRegion(mask_col, [Range(0, tail_valid)]),
                                            T.float32(0.0),
                                        )
                                        for h_i in range(block_m // 2):
                                            T.tile.add(acc_s_ub[h_i, :], acc_s_ub[h_i, :],
                                                       mask_col)
                                # causal 对角块 mask：行 r 可见列 j <=
                                # (s0 + h_start + r) + diff_s；仅块内末列超出
                                # 首行可见界才需要（向量化）
                                if causal:
                                    if (t + 1) * block_n - 1 > bx * block_m + diff_s:
                                        # col_2d[r][c] = t*bn + c（m_i_2d 复用）；
                                        # bound_2d[r][c] = s0 + h_start + diff_s + r
                                        # （mask_col 作 bound ramp 复用）
                                        T.tile.createvecindex(
                                            col_idx, T.cast(t * block_n, "float32"))
                                        T.tile.createvecindex(
                                            bound_vec, T.cast(bx * block_m + h_start + diff_s, "float32"))
                                        T.tile.broadcast(m_i_2d, col_idx, axis=0)
                                        T.tile.broadcast(bound_2d, bound_vec, axis=1)
                                        T.tile.compare(
                                            causal_cmp, m_i_2d, bound_2d, "LE")
                                        T.tile.select(
                                            acc_s_ub, causal_cmp, acc_s_ub,
                                            T.float32(-2 ** 30),
                                            "VSEL_TENSOR_SCALAR_MODE")
                                # online softmax（同管链，零屏障，in-order FIFO）
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, alpha_ring[slot_prod, :, :])
                                T.tile.sub(alpha_ring[slot_prod, :, :], alpha_ring[slot_prod, :, :], m_i)
                                T.tile.exp(alpha_ring[slot_prod, :, :], alpha_ring[slot_prod, :, :])
                                # brcb 化（catlass CalcExp 同构）：brcb 物化
                                # 64x8 替代整幅 2D 广播；dim>128/sim 构建回退
                                # （行循环免 scratch）
                                if use_brcb and not _SIM_NO_BRCB:
                                    T.tile.row_expand_sub_experiment(
                                        BufferRegion(acc_s_ub, [Range(0, block_m // 2),
                                                                Range(0, 64)]),
                                        BufferRegion(acc_s_ub, [Range(0, block_m // 2),
                                                                Range(0, 64)]),
                                        m_i, brcb_tmp)
                                    T.tile.row_expand_sub_experiment(
                                        BufferRegion(acc_s_ub, [Range(0, block_m // 2),
                                                                Range(64, 128)]),
                                        BufferRegion(acc_s_ub, [Range(0, block_m // 2),
                                                                Range(64, 128)]),
                                        m_i, brcb_tmp)
                                elif use_brcb:
                                    for h_i in range(block_m // 2):
                                        T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :],
                                                   m_i[h_i, 0])
                                else:
                                    T.tile.broadcast(m_i_2d, m_i)
                                    T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i_ring[slot_prod, :, :], dim=-1)
                                T.set_flag("mte3", "v", 1)
                                T.wait_flag("mte3", "v", 1)
                                T.copy(acc_s_ub, acc_s_half)
                                T.set_flag("v", "mte3", 0)
                                T.wait_flag("v", "mte3", 0)
                                T.copy(
                                    acc_s_half,
                                    workspace_p[task_idx, slot_prod,
                                                h_start: h_start + block_m // 2, :],
                                )
                                T.set_cross_flag("MTE3", SIG_P_READY)

                        # ---------- 消费者：C2(t-2) / V2(t-2) ----------
                        if t >= prelaunch:
                            now_k = t - prelaunch
                            slot_cons = now_k % ring_slots

                            with T.Scope("C"):
                                T.wait_cross_flag(SIG_P_READY)
                                # C2 的 acc_s_l1/v_l1 WAR/RAW 让位给 K 槽信用（MTE1_MTE2
                                # 0/1、MTE2_MTE1 0/1），改用 3/2（flag id 让位）。
                                T.set_flag("mte1", "mte2", 3)
                                T.wait_flag("mte1", "mte2", 3)
                                T.copy(workspace_p[task_idx, slot_cons, :, :], acc_s_l1)
                                T.copy(v[bz, by, now_k * block_n: (now_k + 1) * block_n,
                                         0:dim], v_l1)
                                T.set_flag("mte2", "mte1", 2)
                                T.wait_flag("mte2", "mte1", 2)
                                T.set_flag("m", "mte1", 1)
                                T.wait_flag("m", "mte1", 1)
                                T.copy(acc_s_l1, p_l0)
                                for cc in T.serial(d_chunks):
                                    T.set_flag("m", "mte1", 1)
                                    T.wait_flag("m", "mte1", 1)
                                    T.copy(v_l1[0, cc * cube_k], v_l0)
                                    T.set_flag("mte1", "m", 0)
                                    T.wait_flag("mte1", "m", 0)
                                    T.set_flag("fix", "m", 1)
                                    T.wait_flag("fix", "m", 1)
                                    T.mma(p_l0, v_l0, acc_o_l0c, init=True)
                                    T.set_flag("m", "fix", 0)
                                    T.wait_flag("m", "fix", 0)
                                    T.copy(
                                        acc_o_l0c,
                                        workspace_o[task_idx, slot_cons, :,
                                                    cc * cube_k: (cc + 1) * cube_k],
                                    )
                                # C2 跨迭代消费者完成标记（供 ring 尾 drain）
                                T.set_flag("m", "mte1", 5)
                                T.set_flag("fix", "m", 4)
                                T.set_flag("mte1", "mte2", 4)
                                T.set_cross_flag("FIX", SIG_O_READY)

                            with T.Scope("V"):
                                T.wait_cross_flag(SIG_O_READY)
                                h_start = vid * (block_m // 2)
                                T.set_flag("v", "mte2", 1)
                                T.wait_flag("v", "mte2", 1)
                                T.copy(
                                    workspace_o[task_idx, slot_cons,
                                                h_start: h_start + block_m // 2, :],
                                    acc_o_ub,
                                )
                                T.set_flag("mte2", "v", 0)
                                T.wait_flag("mte2", "v", 0)
                                # 压缩 Vector（s10 同构）：首 item 免零初始化
                                # 直接建立；dim_align==128 档整幅 broadcast-mul
                                # 替代 64 路行循环（da!=128 回退行循环，见下）
                                if now_k == 0:
                                    T.copy(acc_o_ub, acc_o)
                                    T.copy(sumexp_i_ring[slot_cons, :, :], sumexp)
                                elif dim_align == 128:
                                    T.tile.broadcast(m_i_2d, alpha_ring[slot_cons, :, :])
                                    T.tile.mul(acc_o, acc_o, m_i_2d)
                                    T.tile.add(acc_o, acc_o, acc_o_ub)
                                    T.tile.mul(sumexp, sumexp,
                                               alpha_ring[slot_cons, :, :])
                                    T.tile.add(sumexp, sumexp,
                                               sumexp_i_ring[slot_cons, :, :])
                                else:
                                    # dim_align != 128 回退行循环标量乘：
                                    # - da>128：acc_o 比 m_i_2d 宽且 UB 放不下加宽版；
                                    # - da<128：tile.mul 平坦化要求 pitch(acc_o)==
                                    #   pitch(m_i_2d)==128，da<128 时逐元素错位
                                    #   （dim=64/80/96/112 kv_loops>=2 实测全错，
                                    #   见 ascendc/GENERALIZE_DEBUG_LOG.md ISSUE-A1）
                                    for h_i in range(block_m // 2):
                                        T.tile.mul(acc_o[h_i, :], acc_o[h_i, :],
                                                   alpha_ring[slot_cons, h_i, 0])
                                    T.tile.add(acc_o, acc_o, acc_o_ub)
                                    T.tile.mul(sumexp, sumexp,
                                               alpha_ring[slot_cons, :, :])
                                    T.tile.add(sumexp, sumexp,
                                               sumexp_i_ring[slot_cons, :, :])

                        # ring 尾定向 drain（去 PIPE_ALL）：5 个定向跨迭代消费者
                        with T.Scope("C"):
                            if t < kv_loops:  # 本迭代跑了 C1(t)
                                T.wait_flag("m", "mte1", 3)   # mma 读 rhs_l0 完成
                                T.wait_flag("fix", "m", 3)    # FIX 读 acc_s_l0c 完成
                            if prelaunch <= t:
                                now_k = t - prelaunch
                                if now_k < kv_loops:  # 本迭代跑了 C2(now_k)
                                    T.wait_flag("m", "mte1", 5)   # mma 读 v_l0 完成
                                    T.wait_flag("fix", "m", 4)    # FIX 读 acc_o_l0c 完成
                                    T.wait_flag("mte1", "mte2", 4)  # MTE1 读 acc_s_l1 完成

                    # C 侧槽信用 drain：task 内预置的 K 槽空闲信用（MTE1_MTE2 0/1）
                    # 经预取/释放后每槽恒残留 1 个未消费 set，task 尾 wait 配对清零
                    # （防跨 task 电平残留累积死锁）。
                    with T.Scope("C"):
                        T.wait_flag("mte1", "mte2", 0)   # K 槽 0 空闲信用清零
                        T.wait_flag("mte1", "mte2", 1)   # K 槽 1 空闲信用清零

                    # ---- 收尾：归一化并写出 ----
                    with T.Scope("V"):
                        for h_i in range(block_m // 2):
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i, 0])
                        T.copy(acc_o, acc_o_half)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        row_start = bx * block_m + vid * (block_m // 2)
                        T.copy(
                            acc_o_half[:, 0:dim],
                            output[bz, by, row_start: row_start + block_m // 2, 0:dim],
                        )

    return main
