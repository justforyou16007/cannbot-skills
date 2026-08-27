---
name: lingxi-remote-execution
description: 通过 SSH 在一台远程 NPU 主机上拉起算子 build、精度验证与性能采集（build/verify/profile 分阶段、job 私有 pip 目录、主机级设备池跨 session 原子分配、sticky 绑卡）。当本地无 NPU、需要多算子绑不同卡并行实验、或本地与远端环境不一致时使用。
---

# lingxi-remote-execution：远程 NPU 执行后端（可选）

本地仍负责进化编排、世界模型、代码修改和报告；本 skill 只把**确定性的
build / verify / profile** 三个阶段提交到一台远程 NPU 主机执行。local 执行
方式完全保留，remote 是可选后端，两者产出相同的产物契约。

## 何时使用

- 本地机器没有 NPU（或有卡但环境不可用），NPU 在一台可通过 SSH 登录的远端主机上
- 多个独立算子 session 需要各自绑定不同卡并行进化（共享远端"主机级"设备池）
- 本地与远端 Python/CANN 环境不一致，希望评估始终落在同一台确定性环境上

不适用：本地已有 NPU 且无跨机需求（继续用原 local 流程，不要引入 SSH 开销）。

## 快速开始

```bash
RE=plugins-community/ops-perf-evolution/skills/lingxi-remote-execution

# 0) 准备配置（不写口令/私钥，认证依赖你现有的 ~/.ssh 配置）
cp $RE/scripts/examples/remote_hosts.example.json remote_hosts.json  # 按需修改

# 1) 部署 worker + 环境预检（SSH / python / remote_root / CANN env / npu-smi / 构建工具 / 设备池）
python3 $RE/scripts/remote_exec.py --config remote_hosts.json --host npu-a preflight

# 2) session 申请设备租约（记录文件必须放进本次 EVO_DIR）
STATE=output/FastGELU_evo_20260827/remote_session.json
python3 $RE/scripts/remote_exec.py --config remote_hosts.json --host npu-a \
    --state-file "$STATE" lease acquire \
    --session FastGELU_evo_20260827 --owner FastGELU --ttl 7200

# 3) 在整个进化期间持续续租；job 内部也会续租，这个进程覆盖两次 job 之间的设计时间
python3 $RE/scripts/remote_exec.py --config remote_hosts.json --host npu-a \
    --state-file "$STATE" lease heartbeat --lease-id <lease_id> \
    --watch --interval 30 > output/FastGELU_evo_20260827/remote_heartbeat.log 2>&1 &
HEARTBEAT_PID=$!

# 4) 提交 job。验证/Profiling 脚本是远端可信仓库中的绝对路径，不是示例占位脚本
python3 $RE/scripts/remote_exec.py --config remote_hosts.json --host npu-a \
    --state-file "$STATE" job submit \
    --job-id FastGELU_evo_20260827_baseline --session FastGELU_evo_20260827 \
    --src /path/to/kernel_proj \
    --preset ascendc-eval --kernel-dir kernel --soc Ascend910B3 \
    --verify-cmd '["python3","/srv/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/scripts/verification_ascendc.py","{bundle_dir}"]' \
    --profile-cmd '["python3","/srv/cannbot-skills/ops/ops-profiling/scripts/msprof_perf_summary.py","--compare","--output-dir","{bundle_dir}"]' \
    --artifacts evaluation_results.json performance.json performance.log perf_report.md profiling \
    --wait --fetch --fetch-dest output/.../baseline

# 5) 进化结束后停止本地续租并显式释放
kill "$HEARTBEAT_PID"
python3 $RE/scripts/remote_exec.py --config remote_hosts.json --host npu-a \
    --state-file "$STATE" lease release --lease-id <lease_id>
```

`batch` 是“一组评估 job”的薄入口，不会替你执行完整的世界模型进化。完整的
多算子并行方式是：为每个算子启动一个独立的顶层 `lingxi-evo` 会话；它们共享
同一个远端设备池，各拿一张卡。若只需并行跑一组已定义的评估 job，可用：

```bash
python3 $RE/scripts/remote_exec.py --host npu-a batch --plan plan.json
```

plan.json 格式见 `references/job-protocol.md`。

## 核心行为（为什么这样设计）

| 行为 | 说明 |
|---|---|
| 绑卡一致性 | lease 是 sticky 的：session 第一次拿到哪张卡，之后所有 job（baseline、每轮候选、最终复测）都在这张卡上执行，避免跨卡噪声 |
| 跨进程原子分配 | 设备分配在远端用 `flock + lease.json` 完成，多个本地进程/多个算子 session 同时抢卡不会出现"各自认为卡空闲" |
| 同卡串行 / 异卡并行 | 每张卡有独立的执行锁：verify/profile 在同一张卡上排队串行；不同卡上的算子天然并行 |
| 构建不占卡 | build 阶段走独立的 build slot（可并发），不会拿着设备锁等整个构建 |
| SSH 断开不杀任务 | job runner 以独立会话（setsid）运行，重连后 `job status` / `recover` 即可续上 |
| 幂等提交 | 源码内容摘要是 spec 的一部分；同一 job_id、同一源码和命令返回已有状态，不同源码或命令报冲突（exit 3）。上传先进入独立暂存目录，不会覆盖已存在 job |
| 私有 Python 环境 | wheel 一律 `pip install --target <job>/pylibs`，verify 阶段通过该 job 的 `PYTHONPATH` 使用；**绝不** `pip install --force-reinstall` 污染远端全局环境（这是原 evaluate_ascendc.sh 的并发污染点，远程模式已解耦为 build/wheel/install/verify/profile，并在需要时追加不占卡的 profile_collect） |
| Profiling 证据归档 | 当 profile 命令是 `msprof_perf_summary.py --compare` 时，preset 自动追加 `--keep-prof`；设备阶段结束后再把 `/tmp` 中、且由 performance.json 精确引用的 PROF_GROUP 复制到 job 的 `bundle/profiling/`，复制过程不占卡，回传后路径为本地 `profiling/` |
| 命令安全 | 所有远端命令以 argv 数组执行，spec 中的 shell 元字符直接拒绝；SSH 传输逐参数引用 |
| 路径安全 | 所有路径限制在 remote_root/job 目录内；上传、产物清单和回传均拒绝 `..`、链接与特殊文件；删除只针对一个已终止 job |
| 取消安全 | cancel 只杀属于该 job 的进程组，杀前同时核对精确 job_id 与 `/proc` 启动时间，防 PID 复用误杀 |

## 失败分类

`transport` / `environment` / `build` / `correctness` / `profile` / `cancelled` /
`lease_lost`。分类写在 job 的 `state.json` 的 `failure_class` 字段；本地 CLI
错误也带 `failure_class`。远端 runner 在上传后的任务执行期间持续续租；上面的
`lease heartbeat --watch` 覆盖任务之间的设计时间。两者都停止后才由 TTL 回收。

## 脚本工具

- `scripts/remote_exec.py` - 本地 CLI（deploy/preflight/lease/job/recover/batch，均有 --help）
- `scripts/remote_worker.py` - 远端 worker（由 deploy 推送到 remote_root/lib/，仅标准库）

## 参考资料

- `references/config-schema.md` - remote_hosts.json 配置模式与校验规则
- `references/job-protocol.md` - job spec / state / lease / batch plan 协议
- `scripts/examples/remote_hosts.example.json` - 配置示例

## 已知限制

- 需要 `python3`（>=3.6）与 `bash` 在远端可用；文件传输用 tar 流，不依赖 rsync。
- 远端必须有与本地流程匹配的验证脚本和 ops-profiling 仓库。把它们的绝对路径
  写入配置的 `required_paths`，preflight 会检查；命令路径仍由 `--verify-cmd` /
  `--profile-cmd` 明确传入，工具不会猜测远端仓库位置。
- 自动测试覆盖真实 preset 的目录、私有 wheel 安装、并发、租约和传输协议，但
  不伪造真实 NPU 结论；首次接入一台机器仍需用一个真实算子做联调验收。
- 单个 `--host` 对应一台 NPU 主机；跨多台主机的池化分配不在本 skill 范围内。
