#!/usr/bin/env python3
"""Tile-level direct 1x1 Conv2D as [Co,Ci] @ [Ci,H*W]."""

from dataclasses import dataclass

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

from ..block_level.conv2d_base import (
    BASE_K, BASE_M, BASE_N, C0, ceildiv, round_up,
)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass(frozen=True)
class Direct1x1Plan:
    """Static storage, grid, and layout contract for the direct path."""

    hw: int
    ci_storage: int
    co_storage: int
    hw_storage: int
    m_blocks: int
    n_blocks: int
    k_panels: int
    total_blocks: int
    used_cores: int
    tasks_per_core: int
    a_gm_layout: str = "raw OIHW viewed as [Co,Ci]"
    b_gm_layout: str = "raw NCHW viewed as [N,Ci,HW]"
    c_gm_layout: str = "[N,Co_storage,HW_storage] FP32"
    a_l0_layout: str = "L0A Zz"
    b_l0_layout: str = "L0B Zn"


def make_direct_1x1_plan(cfg):
    """Build and validate the flipped-GEMM plan used by the kernel."""
    if not (
        cfg.data_format == "NCHW"
        and cfg.kernel_h == cfg.kernel_w == 1
        and cfg.stride_h == cfg.stride_w == 1
        and cfg.pad_h == cfg.pad_w == 0
        and cfg.dilation_h == cfg.dilation_w == 1
        and cfg.ci % C0 == 0
    ):
        raise ValueError("configuration does not satisfy direct 1x1 routing")
    hw = cfg.height * cfg.width
    ci_storage = round_up(cfg.ci, BASE_K)
    co_storage = round_up(cfg.co, BASE_M)
    hw_storage = round_up(hw, BASE_N)
    m_blocks = co_storage // BASE_M
    n_blocks = hw_storage // BASE_N
    total_blocks = cfg.batch * m_blocks * n_blocks
    used_cores = min(20, total_blocks)
    return Direct1x1Plan(
        hw=hw,
        ci_storage=ci_storage,
        co_storage=co_storage,
        hw_storage=hw_storage,
        m_blocks=m_blocks,
        n_blocks=n_blocks,
        k_panels=ci_storage // BASE_K,
        total_blocks=total_blocks,
        used_cores=used_cores,
        tasks_per_core=ceildiv(total_blocks, used_cores),
    )


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_direct(cfg):
    """Mirror the direct branch of ``Conv2dBaseKernel::Process``."""
    plan = make_direct_1x1_plan(cfg)

    @T.prim_func
    def process_direct(
        weight_oihw: T.Tensor(
            (cfg.co, cfg.ci, 1, 1), cfg.dtype
        ),
        x_nchw: T.Tensor(
            (cfg.batch, cfg.ci, cfg.height, cfg.width), cfg.dtype
        ),
        acc: T.Tensor(
            (cfg.batch, plan.co_storage, plan.hw_storage), "float32"
        ),
    ):
        with T.Kernel(plan.used_cores, is_npu=True) as (cid, _):
            # Each L1 tile is one complete Cube K panel. T.Pipelined gives
            # these buffers a two-stage lifetime: GM prefetch then L0 load.
            a_l1 = T.alloc_L1((BASE_M, BASE_K), cfg.dtype)
            b_l1 = T.alloc_L1((BASE_K, BASE_N), cfg.dtype)
            T.annotate_layout({
                a_l1: make_zn_layout(a_l1),
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)
            b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
            c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
            for local_task in T.serial(plan.tasks_per_core):
                block_id = cid + local_task * plan.used_cores
                if block_id < plan.total_blocks:
                    image = block_id // (plan.m_blocks * plan.n_blocks)
                    remainder = block_id % (
                        plan.m_blocks * plan.n_blocks
                    )
                    co_start = (remainder // plan.n_blocks) * BASE_M
                    hw_start = (remainder % plan.n_blocks) * BASE_N
                    with T.Scope("C"):
                        for k_panel in T.Pipelined(
                            plan.k_panels, num_stages=2
                        ):
                            ci_start = k_panel * BASE_K
                            # Prefetch + LoadA/LoadB: GM ND -> L1 NZ.
                            T.copy(
                                weight_oihw[co_start, ci_start, 0, 0], a_l1
                            )
                            T.copy(
                                x_nchw[
                                    image, ci_start,
                                    hw_start // cfg.width,
                                    hw_start % cfg.width,
                                ],
                                b_l1,
                            )
                            # SplitA/SplitB: L1 NZ -> L0A Zz/L0B Zn.
                            T.copy(a_l1, a_l0)
                            T.copy(b_l1, b_l0)
                            # Compute: cmatrixInitVal is true on panel zero.
                            T.mma(a_l0, b_l0, c_l0, init=(k_panel == 0))
                        # CopyOut: L0C -> direct FP32 accumulator workspace.
                        T.copy(c_l0, acc[image, co_start, hw_start])

    return process_direct


# Compatibility alias used by model_new_tilelang.py.
conv2d_base_direct_1x1 = conv2d_base_process_direct


__all__ = [
    "Direct1x1Plan",
    "make_direct_1x1_plan",
    "conv2d_base_process_direct",
    "conv2d_base_direct_1x1",
]
