#!/usr/bin/env python3
"""Block schedule for direct 1x1 Conv2D."""

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout

from .conv2d_base import BASE_K, BASE_M, BASE_N, ceildiv, round_up

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def direct_1x1_block_contract(cfg):
    """Return the complete block domain and producer/consumer contract."""
    hw = cfg.height * cfg.width
    ci_storage = round_up(cfg.ci, BASE_K)
    co_storage = round_up(cfg.co, BASE_M)
    hw_storage = round_up(hw, BASE_N)
    m_blocks = co_storage // BASE_M
    n_blocks = hw_storage // BASE_N
    total_blocks = cfg.batch * m_blocks * n_blocks
    cores = min(20, total_blocks)
    return {
        "logical_gemm": "[Co,Ci] @ [Ci,H*W]",
        "ci_storage": ci_storage,
        "co_storage": co_storage,
        "hw_storage": hw_storage,
        "m_blocks": m_blocks,
        "n_blocks": n_blocks,
        "total_blocks": total_blocks,
        "cores": cores,
        "tasks_per_core": ceildiv(total_blocks, cores),
        "k_panels": ci_storage // BASE_K,
        "pipeline_stages": 2,
        "gm_layouts": (
            "raw OIHW, viewed as [Co,Ci] for KH=KW=1",
            "raw NCHW, viewed as [N,Ci,HW]",
            "[N,Co_storage,HW_storage] FP32",
        ),
        "dataflow": "GM->L1->L0->MMA->L0C->GM",
        "producer_contracts": ("raw OIHW view", "raw NCHW view"),
        "consumer_contract": "common epilogue/direct",
    }


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def conv2d_base_process_direct_block(cfg):
    contract = direct_1x1_block_contract(cfg)
    ci_storage = contract["ci_storage"]
    co_storage = contract["co_storage"]
    hw_storage = contract["hw_storage"]
    m_blocks = contract["m_blocks"]
    n_blocks = contract["n_blocks"]

    @T.prim_func
    def process_direct(
        weight: T.Tensor((cfg.co, cfg.ci, 1, 1), cfg.dtype),
        x: T.Tensor(
            (cfg.batch, cfg.ci, cfg.height, cfg.width), cfg.dtype
        ),
        acc: T.Tensor((cfg.batch, co_storage, hw_storage), "float32"),
    ):
        with T.Kernel(contract["cores"], is_npu=True) as (cid, _):
            # Expert buffers represent one K panel at every memory level.
            a_l1 = T.alloc_L1((BASE_M, BASE_K), cfg.dtype)
            b_l1 = T.alloc_L1((BASE_K, BASE_N), cfg.dtype)
            T.annotate_layout({
                a_l1: make_zn_layout(a_l1),
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((BASE_M, BASE_K), cfg.dtype)
            b_l0 = T.alloc_L0B((BASE_K, BASE_N), cfg.dtype)
            c_l0 = T.alloc_L0C((BASE_M, BASE_N), "float32")
            for task in T.serial(contract["tasks_per_core"]):
                block = cid + task * contract["cores"]
                if block < contract["total_blocks"]:
                    image = block // (m_blocks * n_blocks)
                    rem = block % (m_blocks * n_blocks)
                    co_start = (rem // n_blocks) * BASE_M
                    hw_start = (rem % n_blocks) * BASE_N
                    with T.Scope("C"):
                        for panel in T.Pipelined(
                            contract["k_panels"],
                            num_stages=contract["pipeline_stages"],
                        ):
                            ci_start = panel * BASE_K
                            # Prefetch + LoadA/LoadB.
                            T.copy(
                                weight[co_start, ci_start, 0, 0], a_l1
                            )
                            T.copy(
                                x[
                                    image, ci_start,
                                    hw_start // cfg.width,
                                    hw_start % cfg.width,
                                ],
                                b_l1,
                            )
                            # SplitA/SplitB then Compute.
                            T.copy(a_l1, a_l0)
                            T.copy(b_l1, b_l0)
                            T.mma(a_l0, b_l0, c_l0, init=(panel == 0))
                        # CopyOut.
                        T.copy(c_l0, acc[image, co_start, hw_start])

    return process_direct


conv2d_base_direct_1x1_block = conv2d_base_process_direct_block


__all__ = [
    "direct_1x1_block_contract",
    "conv2d_base_process_direct_block",
    "conv2d_base_direct_1x1_block",
]
