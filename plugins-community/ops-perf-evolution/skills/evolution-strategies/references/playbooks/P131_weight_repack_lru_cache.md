# P131 Playbook: 权重 Repack LRU 缓存落地 SOP

> **强制流程**。纯 host 侧改动。完整实战记录见 `references/discovered/disc_X20.md`。

## Step 1: 取证

```bash
grep -nE "repack|wrepack|Repack" op_host/*.cpp op_kernel/*.{cpp,h} > /tmp/p131_locations.txt
```
确认：repack kernel 每次调用都发射？权重在评测/推理循环中是否不变？

## Step 2: 缓存实现（host C++/ATen）

1. 缓存键：`data_ptr + sizes + dtype + version counter`——**version 必需**（in-place 修改 bump version 强制重打包，否则脏数据）
2. 条目持有 weight **强引用**（防 allocator 回收地址被复用造成 stale 命中）；workspace 张量由条目持活
3. LRU 容量个位数条目；miss 时整体重写
4. 多目标布局（normal k 序 vs fused k 序）→ **布局标志进缓存键**
5. 留环境变量开关（`*_WREPACK_CACHE=0`）便于 A/B 与排障

## Step 3: 验证

- 权重不变反复调用：第二次起 repack kernel 零发射
- in-place 改权重（`copy_`/`fill_`）后必须重打包（version 机制生效）
- 多布局混用时结果正确（布局标志在键中）
