# DAV_3510 经验卡片索引

本目录存放经 **DAV_3510 架构上的探针实测**认证的可复用经验。

## 目前为空

这是正常状态。知识库靠 campaign 实测积累，不预填 —— 预填的条目没有
实测背书，恰恰是准入闸要拦的东西。

## 本架构的特别之处

相对 DAV_2201，以下差异会让经验不可互通：

| 差异 | 影响 |
|---|---|
| BufferID 同步（替代 set/wait 强配对） | 同步相关经验不可互换 |
| CV 直通通路（L0C→UB、UB→L1） | 部分数据流动不出现在 MTE2/MTE3，搬运量推断方式不同 |
| FP8 / MXFP8 / MXFP4 / HiF8 Cube MMAD | 算力利用率的基准不同 |
| UB 248KB / L0C 256KB（vs 192KB / 128KB） | 容量权衡要重新量化 |

详见 [`../../../forge-reference-analysis/references/arch-3510.md`](../../../forge-reference-analysis/references/arch-3510.md)。

## 怎么加卡片

```bash
python3 ../../scripts/hwkb.py validate --card draft.md
python3 ../../scripts/hwkb.py add --card draft.md
```
