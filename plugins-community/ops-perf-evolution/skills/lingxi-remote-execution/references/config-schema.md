# remote_hosts.json 配置模式

本地 CLI（`remote_exec.py`）通过 `--config <path>`（或环境变量
`LINGXI_REMOTE_CONFIG`，默认 `./remote_hosts.json`）读取主机配置。
配置在加载时做**严格校验**，危险值直接拒绝（而不是事后转义）。

## 字段说明

| 字段 | 必填 | 类型/约束 | 说明 |
|---|---|---|---|
| `hosts` | 是 | object | host alias -> 配置 |
| `hosts.<alias>.transport` | 否 | `"ssh"`（默认）/`"local"` | `local` 不经 ssh 直接本机执行 worker（调试/测试用） |
| `hosts.<alias>.ssh_host` | ssh 模式必填 | `[A-Za-z0-9@._-]+` 且不能以 `-` 开头 | ssh 目标（~/.ssh/config alias 或 `user@host`）。**禁止出现空白、引号、`;|&<>\`$` 等元字符** |
| `hosts.<alias>.ssh_args` | 否 | 字符串数组 | 额外 ssh 参数，逐项校验（同上字符集约束），如 `-o BatchMode=yes` |
| `hosts.<alias>.ssh_bin` | 否 | 字符串 | ssh 可执行文件路径（默认 `ssh`） |
| `hosts.<alias>.remote_root` | 是 | 绝对 POSIX 路径 | 远端工作根目录。所有远端文件操作都限制在该目录内 |
| `hosts.<alias>.remote_python` | 否 | 字符串 | 远端 python 解释器（默认 `python3`，要求 >= 3.6，仅标准库） |
| `hosts.<alias>.env_script` | 实际 NPU 执行通常必填 | 绝对路径 | **可信**的 CANN 环境脚本（如 `set_env.sh`）。仅当 sshd 启动的非交互环境本身已完整配置 CANN 时才可省略 |
| `hosts.<alias>.devices` | 否 | `"auto"` 或 int 数组 | 设备池。`auto` = preflight 时优先用稳定的 `npu-smi info -m` 映射表发现，旧版本才回退主表 |
| `hosts.<alias>.max_build_slots` | 否 | int | 构建并发槽位数（默认 4）。构建不占设备锁，只受此信号量约束 |
| `hosts.<alias>.required_paths` | 否（推荐） | 绝对路径数组 | 远端验证和 Profiling 脚本，例如 `verification_ascendc.py`、`msprof_perf_summary.py`；preflight 检查文件存在，避免任务跑到中途才发现工具缺失 |

## 安全规则

1. **不落敏感凭据**：配置只描述"连哪台机器"，口令/私钥内容一律不写。
   认证依赖用户现有 SSH 配置（`~/.ssh/config`、agent、交互输入）。
2. **配置注入防护**：`ssh_host` / `ssh_args` / `remote_root` 等字段在加载时
   校验；所有远端 argv 再逐参数引用。危险值直接报 `exit 2`。
3. **remote_root 是唯一根**：远端 worker 的所有路径（job 目录、lease、日志、
   产物）都在 `remote_root` 之下；spec 中的相对路径经包含性校验 + realpath
   双重检查，拒绝 `..`、绝对路径与符号链接逃逸。
4. **设备号不可由 job 指定**：spec 中禁止 `ASCEND_RT_VISIBLE_DEVICES`；
   设备唯一来源是 session lease。
5. **活跃租约期间配置不可漂移**：多个 session 可并发执行相同配置的 preflight；
   只要仍有有效租约，worker 就拒绝更改 devices、max_build_slots 或 env_script，
   防止某个客户端在其他算子运行中途移除卡或切换 CANN 环境。

## lease TTL 与心跳

- `lease acquire --ttl <秒>` 设置租约有效期（默认 3600）。
- 单次刷新：`lease heartbeat --lease-id <id>`；整段进化持续刷新：追加
  `--watch --interval 30`，并让该进程覆盖 baseline 到最终复测。
- 远端 runner 在整个 job（含构建、排队和设备阶段）也持续续租，因此 SSH 断开
  不会让正在执行的 job 中途丢卡；watch 负责两个 job 之间的设计时间。
- TTL 过期的租约在下次分配时被回收（session 崩溃且无人续租的兜底）。
- session 结束时务必 `lease release`，不要等 TTL。
