# DAV_2201 经验卡片索引

本目录存放经 **DAV_2201 架构上的探针实测**认证的可复用经验。

## 为什么按架构分目录

硬件机制与架构绑定。DAV_2201 用 set/wait 强配对同步、没有 CV 直通通路、
不支持 FP8/MXFP Cube MMAD —— 这些差异使得本目录的经验不能直接套到
DAV_3510 上。准入闸会拦截跨架构认证的卡片。

## 怎么加卡片

```bash
python3 ../../scripts/hwkb.py validate --card draft.md   # 先自检
python3 ../../scripts/hwkb.py add --card draft.md        # 通过才入库
```

准入要求 `certified_by` 带 job 号、架构、实测变化量。没有实测背书的
经验不得入库 —— 那会让知识库退化成主观笔记合集。

## 怎么检索

```bash
python3 ../../scripts/hwkb.py query --symptom "mte2 占比高" --arch DAV_2201
```

检索以**症状**为主键，因为实际场景是「我观测到了这个，有人遇到过吗」。

## 现有卡片

| id | 机制 | 触发症状 |
|---|---|---|
| `R-2201-001` | UB 常驻复用消除跨 tile 的重复 GM 读 | MTE2 占比显著高于 VEC，且数据被反复搬入 |
