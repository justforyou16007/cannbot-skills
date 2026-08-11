---
schema_version: okf.v1
kind: discovered_strategy
id: X28
origin: discovered
title: 小 S Prefill FA 自适应 KV Chunk 粒度
description: FA/GQA prefill 的 kvPad=align_up(S, nChunk) 在 nChunk=256/128 时对小 S 产生 4-85x 序列 padding 浪费（attention 在 mask 覆盖列白算）。按 S 自适应 nChunk（小 S 用 64，kvPad=align_up(S,64)）并让 kernel 支持尾块 partial chunk，消除 padding 冗余计算。GroupedQueryAttention 上小/中 case attention device 时间降 2-4x，geomean 0.500->0.588。
confidence: single_op
status: active
op_families: ['attention', 'flash_attention']
bottlenecks: ['undersize_transfer', 'tiling_imbalance']
tags: ['attention', 'flash_attention', 'gqa', 'prefill', 'padding', 'kv-chunk', 'tiling']
---
# X28: 小 S Prefill FA 自适应 KV Chunk 粒度

详见策略库 disc_X28.md。核心：小 S 用 nChunk=64（而非 256/128），kvPad=align_up(S,64)，kernel 支持 16 对齐尾块 partial chunk。验证于 GroupedQueryAttention（910B2C, device 13），geomean 0.500->0.588。
