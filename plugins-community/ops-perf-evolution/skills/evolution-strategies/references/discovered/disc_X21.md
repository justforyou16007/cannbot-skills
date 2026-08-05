---
id: X21
origin: discovered
discovered_round: 2
discovered_from: conv3dbf_cb round_2/parallel_1 (x0_fix1) + round_2/parallel_0 (n2_s1 交叉实证)
base_speedup: 1/20 failed → 20/20, avg_speedup 0.1085x（cann-bench Conv3DBackpropFilter）
---

# Strategy X21: 910B2C Cube 化平台纪律 + 数据驱动 Kahan 精度路由

## 核心思路
在 910B2C（DAV_2201）上把 AIV 算子 Cube 化时，正确性杀手往往不是算法而是**平台纪律**：
四条独立实证的平台陷阱会静默产出垃圾/hang，且 CANN 不报错。逐一避坑后，Cube 路线
才能真正落地；对 fp32 L0C 累加必然超差的相消重元素 case，用**数据驱动预扫描**（非 shape
硬编码）按一般性能量规则分流到 Kahan 兜底，保住全量精度。

四条平台纪律（均为实测实证，文档无载）：
1. **AIC_ONLY 内核必须显式声明**：`__global__ __aicore__`（无 `__vector__`）的内核被编译为
   MIX 双入口（mix_aic+mix_aiv），AIV 入口执行 cube 指令产垃圾甚至 hang。修复：内核首行
   `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`。
2. **跨 pipe 同步只用 SetFlag/WaitFlag HardEvent**：`PipeBarrier<PIPE_MTE2>` 不建立
   MTE2→V 顺序（DMA 未落地即被读取，输出恒 0）；`MTE2_S` 事件在本平台不存在，GetValue
   标量读前用 `MTE2_MTE3 + V_S` 链替代。
3. **TransDataTo5HD（scatter_vnchwconv_b16）在本平台静默杀死内核**：保留该指令时整个
   内核 no-op（rc=0 无日志，魔数写入都不落地）。替代：标量 UB gather
   （`dst[hw*16+lane]=src[lane*HW+hw]`）→ S_V → 精确 cast round-trip → V_MTE3 → DataCopyPad。
4. **vendor fractal N 序必须探针实测**：3D conv backprop-filter 的 dw 分形序为
   **(kd, ci1, kh, kw, ci0)**（kd 在 ci1 之外），与 X18 的 2D forward k 序 (ci1,kh,kw,ci0)
   不同；ci1G=1 或 kd=1 时两序重合会"蒙对"。判别探针：固定 ci1G=2，分别跑 kd=1（通过）
   与 kd=3（失败）即可定序。

## 适用场景
- **算子类型**: 任何在 910B2C 上新增的 Cube（AIC）内核；conv3d/conv2d backward-filter
  及一切 vendor fractal 布局读写的算子
- **瓶颈类型**: Cube 化移植阶段的正确性失败（垃圾值/hang/静默 no-op）；fp32 累加精度
  上限导致的边界 case 超差
- **前提条件**: 无（纪律类）；精度路由需要一次 AIV 预扫描（每 block max|v|/sum|v|，
  ~20-78us/call + 一次 D2H 同步 10-20us）
- **预期收益**: Cube 路线从"全错"到可用（本例 1/20 → 20/20，0.1085x vs PyTorch，
  为 AIV 向量化天花板的 46 倍）

## 实现要点
1. **崩溃 salvage 的数据判读陷阱**：cann-bench compare.py 在 NaN 掩码不匹配时提前返回，
   MERE/MARE 停留默认 0.0——"MERE=0 但 FAIL"**不代表**有限值精确。先用全 1 输入探针
   （任何布局排列下结果不变量）确认基本正确性，再信 compare 数值
2. **探针方法**：全 1（布局不变量）→ ramp（k 序）→ one-hot（落点）；单 tile → 多 unit
   → 多 ci1 逐级放大；host 侧 aclrtMemcpy 分级 dump 工作区定位故障 stage
3. **精度路由规则**（数据驱动，禁止 shape 硬编码）：输入含 inf/NaN → Kahan；
   max|x|·max|g|·S > 5e10（相消误差超 fp32 能力）→ Kahan；fp16 且任一 stride>1
   （边界 tap 累加拓扑 + 紧容差）→ Kahan。fp32 L0C 树序/原子加随机序在相消到 O(100)
   的元素上必然超 bf16 7.8% 容差，这是精度上限不是 bug
4. **Kahan 兜底必须移植带补偿的完整实现**（TwoSum/逐行 Kahan/inf-guard），plain-fp32
   重写在紧容差 case 上 maxrel 可达 0.24
5. 排查纪律：对每个嫌疑留排除证据（毒药测试预填魔数、单核 CONV_DW_AICNUM=1、相位间
   host 全同步等），避免误修

## 来源
自动发现于 conv3dbf_cb 第 2 轮进化（x0_fix1 修复 open_exploration 节点，六根因全链
排查），n2_s1 兄弟变体交叉实证 R1/R2。算子 Conv3DBackpropFilter，speedup 0.1085x（20/20）。
残余杠杆：TransDataTo5HD 的标量 gather 替代占 cube case 60-85% 时间，16x16 块转置或
ping-pong 流水是下一步方向。
