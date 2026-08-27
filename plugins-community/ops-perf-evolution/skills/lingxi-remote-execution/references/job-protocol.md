# job / lease / batch 协议

本文定义 remote_exec.py（本地）与 remote_worker.py（远端）之间的可校验、
可审计的协议。所有跨机数据都是 JSON（job spec 走 base64 通道）。

## 目录布局（远端 remote_root 下）

```
remote_root/
├── lib/remote_worker.py          # deploy 推送的 worker（只读使用）
├── pool/
│   ├── pool_config.json          # {devices, max_build_slots, env_script}
│   ├── allocation.lock           # 整池分配的全局锁（发现/抢占互斥）
│   ├── device_<id>.lock          # 该卡 lease 变更原子锁（flock）
│   ├── device_<id>.lease.json    # 当前租约
│   ├── device_<id>.exec.lock     # 该卡 verify/profile 串行锁
│   └── build_slots/slot_<i>.lock # 构建并发信号量
├── uploads/<job_id>/<digest>-<token>/bundle   # 源码上传暂存区（submit 前落位）
└── jobs/<job_id>/
    ├── spec.json                 # 提交时的 spec（不可变，幂等比对依据）
    ├── state.json                # 运行态（见下）
    ├── bundle/                   # 源码（从暂存区整体搬入，digest 已验证）
    ├── pylibs/                   # pip --target 私有安装目录
    ├── logs/<stage>.log          # 每阶段日志（含完整命令行，可审计）
    ├── artifacts/                # 产物目录
    └── state.lock                # state.json 写锁
```

## 源码上传（staged upload）

本地 `job submit` 带 `--src` 时的提交流程：

1. 本地计算源码树的 sha256（`source_digest`）并生成随机 `upload_token`；
2. `upload-prepare` 在远端创建隔离暂存目录
   `uploads/<job_id>/<digest>-<token>/`（路径各分量均校验）；
3. 源码以 tar 流推入暂存区 `bundle/`（多个 --src 依次落 `bundle/`、
   `bundle/extra_N/`）；推送失败自动 `upload-discard` 清理暂存区；
4. `job-submit` 携带 `source_digest` + `source_upload`，远端**重算暂存树
   digest 并比对**，不一致拒绝（exit 2）并清理；
5. 通过后把 `bundle/` 原子搬入 `jobs/<job_id>/`（`.creating-*` 临时目录 +
   `os.replace`），spec 落盘前剥离 `source_upload`（digest 保留用于审计）。

`--preset ascendc-eval` 必须至少给一个 `--src`（且首个 `--src` 下存在
`--kernel-dir` 指定的内核目录）；`--stage-json` 可以不带源码。

## job spec

```json
{
  "job_id": "FastGELU_evo_20260827_r1_p0",   // ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$
  "session_id": "FastGELU_evo_20260827",     // 同上；绑定设备租约
  "source_digest": "<sha256>",               // 由 CLI 生成（有 --src 时）
  "stages": [
    {
      "name": "build",              // ^[a-z][a-z0-9_]{0,31}$
      "kind": "cmd",                // cmd | pip_install
      "commands": [["cmake", "..."], ["make", "-j8"]],   // argv 数组，非 shell 字符串
      "needs_device": false,        // true 则走设备执行锁 + lease 设备
      "failure_class": "build",     // build|correctness|profile|environment
      "cwd": "bundle/kernel/build", // job 目录内的相对路径（可空）
      "env": {"PYTHONPATH": "{pylibs_dir}:{bundle_dir}"},
      "timeout_s": 3600
    },
    {
      "name": "install", "kind": "pip_install",
      "wheel": "bundle/kernel/dist/*.whl", "target": "pylibs",
      "failure_class": "build"
    }
  ],
  "artifacts": ["bundle/performance.json", "bundle/perf_report.md", "logs"]
}
```

校验规则（违反即 exit 2，job 不落盘）：

- `commands` 只能是 argv 数组；程序名含空白/引号/`;|&<>\`$` 等元字符直接拒绝；
  绝对或相对程序路径都不能含 `.` / `..` / 空分量。
- `env` 键名合法、值不含 NUL；**`ASCEND_RT_VISIBLE_DEVICES` 保留**（设备来自
  lease，运行时由 worker 注入）。
- `cwd` / `wheel` / `target` / `artifacts` 均为 job 目录内相对路径，拒绝
  `..`、绝对路径、符号链接逃逸。
- `stage` 名称唯一；`failure_class` 取值受限。

占位符（运行时由 worker 替换）：`{job_dir}` `{bundle_dir}` `{pylibs_dir}`
`{root}` `{python}` `{ascend_home_path}`（来自 env_script 导出）`{device_id}`。

## stage 执行顺序与并发

1. `needs_device=false` 的阶段（build/wheel/install）先拿 build slot（信号量，
   默认 4 并发），**不占设备锁**——多个候选的构建可以并行，不会拿着卡等编译。
2. `needs_device=true` 的阶段（verify/profile）先校验 session lease 仍有效，
   然后拿该卡的 `exec.lock` 排队，并在真正获得锁后再次核对精确 lease token：
   同卡串行、异卡并行。
   `ASCEND_RT_VISIBLE_DEVICES` 由 worker 设为 lease 的设备号。
3. runner 持有租约期间由后台心跳线程按 `ttl/3`（100ms~30s 区间）自动续租；
   租约丢失（被回收/重分配）时立刻 SIGTERM 当前命令组，job 归为 `lease_lost`。
   本地 CLI 侧在 `--wait` 期间同样维护心跳（远端心跳是兜底，双保险）。
4. 任一阶段失败即终止，`failure_class` 取该阶段声明（命令找不到归
   `environment`，lease 失效归 `lease_lost`，取消归 `cancelled`）。
5. preset 识别到 `msprof_perf_summary.py --compare` 时自动追加 `--keep-prof`，
   随后的 `profile_collect` 阶段在释放设备锁后运行。它只接受
   `performance.json` 引用的 `/tmp/msprof_{ref|asc}_bundle_cN*/PROF_GROUP_*`
   路径，拒绝链接和特殊文件，并复制到 `bundle/profiling/` 供 manifest 回传。

## state.json（job 状态）

```json
{
  "job_id": "...", "session_id": "...",
  "status": "submitted|running|succeeded|failed|cancelled",
  "failure_class": null | "build|correctness|profile|environment|lease_lost|cancelled",
  "error": "...",
  "device_id": 3, "lease_id": "...", "pgid": 12345,
  "runner_pid": 12345, "runner_start_token": "250577545",
  "cancel_requested": false,
  "created_ts": 0, "started_ts": 0, "ended_ts": 0,
  "stages": [{"name": "build", "status": "ok", "exit_code": 0, "error": null,
              "started_ts": 0, "ended_ts": 0}],
  "artifacts_manifest": {"files": ["bundle/performance.json", "logs"], "generated_ts": 0}
}
```

- runner 以独立会话（setsid）运行：SSH 断开不影响；`job status` 会恢复卡在
  `submitted` 或 `running`、但 runner 已死亡的状态。
- `runner_start_token` 是 runner 进程的启动时间戳（来自 /proc/<pid>/stat）。
  `job cancel` 杀进程组前做**双重身份核对**：`/proc/<pgid>/cmdline` 同时包含
  `remote_worker.py`、`_job-runner --job-id <精确 job_id>`，且当前 start token
  与 state 记录一致——两者
  任一不符都拒绝杀（防 PID 复用误杀）。先 SIGTERM 后 SIGKILL。

## 幂等与退出码

- 相同 job_id + 相同 spec 与 `source_digest` 重复提交：返回现有状态，
  `idempotent: true`，不重跑，且清理遗留的暂存上传目录。
- 相同 job_id + 不同命令、不同源码内容或残留不完整目录：exit 3（冲突），
  已存在的 bundle 不会被新上传覆盖。
- submit 全程持有 `jobs/.submit_<job_id>.lock`：并发的重复提交只有一个会
  派生 runner。
- worker 退出码：0 成功；2 校验错误；3 job 冲突；4 未找到；5 锁/租约超时；
  6 远端环境错误。本地 CLI 另有 7 = transport 失败（`failure_class: transport`）。

## lease（设备租约）

```
{"lease_id": "<hex>", "session_id": "...", "owner": "<op_name>",
 "device_id": 3, "created_ts": 0, "heartbeat_ts": 0, "ttl_s": 3600}
```

- 分配：sticky 查找和空闲卡选择都在 `pool/allocation.lock` 下完成，再用
  `device_<id>.lock` 写 lease，保证并发请求同一 session 只产生一个租约。
- 配置发布也使用 `allocation.lock`；存在有效租约时只允许重复写入相同语义配置，
  不允许改变设备列表、构建槽位或环境脚本。
- sticky：`lease acquire` 先扫描本 session 仍有效的租约，命中即复用同一张卡
  （进化 session 绑卡一致的实现基础）。
- 回收：`now - heartbeat_ts >= ttl` 视为失效，可被其他 session 抢占。
- 释放：`lease release --lease-id` 显式删除；本地 CLI 会在本地状态中标记
  已释放，`recover` 不会再误认为持有。

## batch plan（多算子批量入口）

```json
{
  "sessions": [
    {"session_id": "opA_evo_x", "owner": "opA", "job_id": "opA_base",
     "src": "/abs/local/opA_proj",
     "stage_json": "/abs/local/opA_spec.json",
     "artifacts": ["performance.json", "logs"]}
  ]
}
```

每个 session 一条线程：先 `lease acquire`（拿不到卡就阻塞等待，因此**活跃算子
数天然受设备池约束**），再上传 src、提交 job、等待终态、fetch 结果、finally
释放租约。session 顶层字段即 job 参数（`session_id`/`owner` 除外），可覆盖
preset 参数（`kernel_dir`/`soc`/`verify_cmd`/`profile_cmd` 等）或直接给
`stage_json`。

这个 batch 只执行 plan 中定义的一组评估 job，不包含世界模型初始化、候选生成和
报告。完整多算子进化应启动多个独立 `lingxi-evo` 顶层会话，让它们共享设备池。

## 本地 session 状态文件

`--state-file`（默认 `./remote_session_state.json`，带 `.lock` 伴生锁文件）
记录 `host / session_id / lease_id / device_id / jobs`，供 `recover` 断线重连：

```bash
python3 remote_exec.py --host npu-a recover --session <sid> [--reattach]
```

`--reattach` 只接受原 device_id；若原卡不可用而分配器返回另一张卡，CLI 会立即
释放新租约并要求新建 session、重跑 baseline，避免静默破坏性能可比性。
