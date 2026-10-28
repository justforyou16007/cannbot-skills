#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""封存产物库：跨 campaign 保存参考与各代产出。

论文 §3.5 的自我提升建立在一个前提上：**上一代的产出，下一代还拿得到**。
「参考只被当作可测量的二进制使用，所以 ForgeKernel 自己产出的 kernel
可以原样充当这个角色」—— 但前提是它还在。

此前 promote 只把产物路径记进 campaign state。那个目录被清掉、被覆盖、
被后续迭代改写，第 n 代就对不上第 n-1 代了，而且**没有任何地方能看出
参考变过**。跑到第三代时第一代产物多半已经不在原处。

本库解决四件事：

  留存   产物复制进库，不是记个路径。campaign 的工作目录可以随便清
  可核   每代带校验和。参考变了就能查出来 —— 参考一变，之前所有相对它
         的加速比全部失效，这件事必须是可发现的
  身份   记录环境（CANN 版本、soc、芯片）。不同环境下的参考不可比，
         把两者的数字放在一起看会得出错误结论
  血统   每代记录从哪一代来、增益多少、机制有没有核实过

关于 CANN 内置实现
------------------
`builtin_aclnn` 的产物是厂商的，我们封存不了，也不该去拷。这一类入库时
只记**可观测身份**：架构签名快照、实测时间、环境。CANN 升级后签名会漂，
`verify` 能据此发现「参考变了」，这正是需要的那个信号。

关于 cann-bench
---------------
bench 的 `tasks/level*/<算子>/` 提供任务定义（proto.yaml、cases.yaml、
golden.py、desc.md），其中 `baseline_perf_us` 是**评分基准线**。
它可以作为 campaign 的目标标注挂进库里，但**不是参考产物** ——
bench 本地不提供「最优版本」的 kernel，论文里的 public best 是从托管
排行榜读的，而且明说它从不作为循环的输入。

用法:
    reference_library.py register --library ~/.forge-kernel/lib \\
        --operator add_layer_norm --arch DAV_2201 \\
        --input-shape "B=8,S=4096,fp16" \\
        --kind builtin_aclnn --id aclnnAddLayerNorm \\
        --signature reference_signature.json --time-us 456.5

    reference_library.py promote --library ~/.forge-kernel/lib \\
        --operator add_layer_norm --arch DAV_2201 \\
        --input-shape "B=8,S=4096,fp16" \\
        --artifact ./best_kernel --signature sig.json \\
        --time-us 350.0 --gain 1.30 --mechanism-verified

    reference_library.py resolve --library ~/.forge-kernel/lib \\
        --operator add_layer_norm --arch DAV_2201 --input-shape "B=8,S=4096,fp16"

    reference_library.py verify --library ~/.forge-kernel/lib
    reference_library.py lineage --library ~/.forge-kernel/lib --operator add_layer_norm ...

退出码: 0 = 正常; 1 = 校验失败 / 查无此项; 2 = 用法或 IO 错误
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"
INDEX_FILENAME = "index.json"
MANIFEST_FILENAME = "manifest.json"
SIGNATURE_FILENAME = "signature.json"
ARTIFACT_DIRNAME = "artifact"

#: 没有产物可拷的来源 —— 厂商实现不归我们保管。
ARTIFACTLESS_KINDS = {"builtin_aclnn"}

ALL_KINDS = ("builtin_aclnn", "opp_install", "sealed_kernel")

#: 签名漂移超过这个比例就认为「参考变了」。
#: 工程判断，无文档出处；与机制行的移动判据同量级。
SIGNATURE_DRIFT_PCT = 5.0

_DRIFT_EPS = 1e-6


class LibraryError(Exception):
    """库操作失败。"""


# ---------------------------------------------------------------------------
# 跨任务互斥
# ---------------------------------------------------------------------------

#: 取锁超时。入库要复制产物，大目录可能慢，所以给得比普通 IO 宽。
DEFAULT_LOCK_TIMEOUT = 120.0


@contextlib.contextmanager
def exclusive(library: Path, timeout: float = DEFAULT_LOCK_TIMEOUT):
    """对整个库加排他锁，覆盖「读索引 → 改 → 写回」的全过程。

    为什么非锁不可：多个演进任务并行跑（各占一张卡）时，它们**共用同一个
    产物库**。add_generation 的流程是

        load_index  →  算出 gen 号  →  复制产物到 genNNN/  →  save_index

    不加锁会出两种事，而且都不报错：

      索引丢更新   A 读到索引、B 也读到，A 写回、B 再写回 —— A 这次入库
                   在索引里消失了。产物还在磁盘上，但查不到。

      产物被覆盖   同一算子+形状的两个任务同时提升时，双方算出同一个 gen
                   号，后者 rmtree 掉前者刚复制进去的产物。前者的 manifest
                   记着自己算的校验和，于是 verify 之后会报「内容已变更」——
                   那时已经查不出到底发生过什么了。

    锁文件放在库根目录，粒度是整库而不是单条目：入库不频繁（一次 campaign
    才几次），而按条目分锁要先读索引才知道锁哪个，那又回到了竞争上。
    """
    library.mkdir(parents=True, exist_ok=True)
    lock_path = library / ".lock"
    deadline = time.time() + timeout
    handle = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= deadline:
                    raise LibraryError(
                        f"取库锁超时（{timeout:.0f}s）：{lock_path}\n"
                        "  另一个任务正在写库。入库要复制产物，大目录会慢一些。\n"
                        "  若确认没有别的任务在跑，删掉这个锁文件即可。"
                    )
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(handle)


# ---------------------------------------------------------------------------
# 身份
# ---------------------------------------------------------------------------

def shape_key(input_shape: str) -> str:
    """把输入形状压成可做目录名的键。

    形状里有逗号、等号、空格，不能直接当目录名；但也不能只取哈希 ——
    那样 `list` 出来人看不懂。所以保留可读部分，再缀一段哈希防撞。
    """
    normalised = re.sub(r"\s+", "", input_shape.strip())
    readable = re.sub(r"[^A-Za-z0-9=_.-]+", "-", normalised).strip("-")
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:48]}-{digest}" if readable else digest


def entry_key(arch: str, operator: str, input_shape: str) -> str:
    return f"{arch}/{operator}/{shape_key(input_shape)}"


def detect_environment() -> Dict[str, Any]:
    """尽可能记录环境身份。

    不同 CANN 版本下的参考不是同一个东西，把两者的数字放在一起比会得出
    错误结论。所以这里尽量多记，**记不到的如实标 unknown，不猜**。
    """
    env: Dict[str, Any] = {}
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME")
    env["ascend_home"] = home or "unknown"

    version = "unknown"
    if home:
        # CANN 的版本信息文件名随版本而异，这里按几个常见名字找，
        # 找到就原样记录内容的第一行；找不到就如实说 unknown。
        for name in ("version.info", "ascend_toolkit_install.info",
                     "version.cfg"):
            candidate = Path(home) / name
            if candidate.is_file():
                try:
                    text = candidate.read_text(
                        encoding="utf-8", errors="ignore").strip()
                    if text:
                        version = text.splitlines()[0][:200]
                        env["version_source"] = str(candidate)
                        break
                except OSError:
                    continue
    env["cann_version"] = version
    env["soc_version"] = os.environ.get("SOC_VERSION", "unknown")
    return env


#: 算子族推断规则，按顺序匹配，先命中先算。
#:
#: 族的划分沿用 ops/ascendc-performance-best-practices 的分类体系 ——
#: 那边按族组织优化知识，这边按族找参考，两边对齐才查得通。
#:
#: 推断只是省事，**推错比没有更糟**，所以结果会标明是推断还是显式指定，
#: 显式给的 --family 永远优先。
FAMILY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("attention", ("attention", "flash", "_fa", "fa_", "mla", "gqa", "mha",
                   "sdpa")),
    ("matmul", ("matmul", "batch_matmul", "bmm", "gemm", "linear", "_mm_",
                "grouped_mm")),
    ("convolution", ("conv1d", "conv2d", "conv3d", "conv_", "depthwise")),
    ("sort", ("topk", "sort", "argsort", "kthvalue", "argmax", "argmin")),
    ("quantization", ("quant", "dequant", "fakequant")),
    ("reduction", ("reduce", "softmax", "layernorm", "layer_norm", "rmsnorm",
                   "rms_norm", "groupnorm", "group_norm", "batchnorm",
                   "mean", "_sum", "logsumexp", "norm")),
    ("conversion", ("transpose", "concat", "split", "cast", "permute",
                    "reshape", "gather", "scatter")),
    ("broadcast", ("broadcast",)),
    ("random", ("random", "dropout", "uniform", "normal_")),
    ("elementwise", ("add", "mul", "sub", "div", "sin", "cos", "abs", "exp",
                     "relu", "gelu", "silu", "swish", "sigmoid", "tanh",
                     "mish", "sqrt", "rsqrt", "pow", "clip")),
)


def infer_family(operator: str) -> Optional[str]:
    """从算子名推断算子族。推不出来返回 None，不硬塞。"""
    name = operator.lower()
    for family, keywords in FAMILY_RULES:
        if any(keyword in name for keyword in keywords):
            return family
    return None


def cumulative_gain(generations: List[Dict[str, Any]], upto: int) -> float:
    """算到第 upto 代为止、相对 gen0 的累计增益。

    用累计增益而不是绝对耗时来评最优，因为**绝对耗时跨环境不可比**：
    多卡机器上卡与卡之间有系统性差异，不同时段的并发水平也不同。
    而 gain_over_parent 是同一次配对测量里得到的比值 —— 那个比值在哪张卡上
    测都成立，这正是论文坚持同 job 配对计时的理由。
    """
    total = 1.0
    for record in generations:
        if record["generation"] == 0 or record["generation"] > upto:
            continue
        gain = record.get("gain_over_parent")
        if isinstance(gain, (int, float)) and gain > 0:
            total *= gain
    return total


def pick_best(entry: Dict[str, Any]) -> Tuple[Optional[int], str]:
    """选出当前最优的一代，并说明依据。

    优先按累计增益（跨环境可比）；整条链一个增益都没记时才退回绝对耗时，
    并在依据里标明它只在同一环境内可比。
    """
    generations = entry.get("generations", [])
    if not generations:
        return None, "无代次"

    has_gain = any(isinstance(g.get("gain_over_parent"), (int, float))
                   for g in generations)
    if has_gain:
        best = max(generations,
                   key=lambda g: cumulative_gain(generations, g["generation"]))
        return best["generation"], "累计增益（跨环境可比）"

    timed = [g for g in generations
             if isinstance(g.get("time_us"), (int, float)) and g["time_us"] > 0]
    if timed:
        best = min(timed, key=lambda g: g["time_us"])
        return best["generation"], "绝对耗时（仅同一环境内可比）"

    # 一个可比数字都没有时仍要给出一代，否则刚 ingest 完的参考会查不到 ——
    # 「还没测过」不等于「不存在」。取最新一代并讲明依据不足。
    return generations[-1]["generation"], "无可比数据，取最新一代"


# ---------------------------------------------------------------------------
# 索引读写
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def index_path(library: Path) -> Path:
    return library / INDEX_FILENAME


def load_index(library: Path) -> Dict[str, Any]:
    path = index_path(library)
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "entries": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LibraryError(f"无法读取索引 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LibraryError(f"索引 {path} 不是合法 JSON: {exc}") from exc
    if not isinstance(doc, dict) or "entries" not in doc:
        raise LibraryError(f"索引 {path} 结构不对")
    return doc


def save_index(library: Path, index: Dict[str, Any]) -> None:
    library.mkdir(parents=True, exist_ok=True)
    path = index_path(library)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise LibraryError(f"无法写入索引 {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# 产物封存
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_artifact(source: Path, target: Path) -> List[Dict[str, Any]]:
    """把产物复制进库，并逐文件记校验和。

    复制而不是软链：campaign 的工作目录随时可能被清，而库要能在那之后
    仍然拿得出这一代的产物。
    """
    if not source.exists():
        raise LibraryError(f"待封存产物不存在: {source}")
    if target.exists():
        shutil.rmtree(target)

    if source.is_file():
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / source.name)
    else:
        shutil.copytree(source, target)

    files: List[Dict[str, Any]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        files.append({
            "relative": str(path.relative_to(target)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    if not files:
        raise LibraryError(f"产物目录为空: {source}")
    return files


def load_signature_snapshot(path: Optional[str]) -> Optional[Dict[str, float]]:
    """读签名并压成 {通道: 数值} 快照。

    与 campaign.snapshot_signature 保持同一形状 —— 两边都用它做漂移比对。
    """
    if not path:
        return None
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise LibraryError(f"无法读取签名 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LibraryError(f"签名 {path} 不是合法 JSON: {exc}") from exc

    out: Dict[str, float] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            channel = node.get("channel")
            value = node.get("value")
            if (isinstance(channel, str) and node.get("present") is True
                    and isinstance(value, (int, float))):
                out.setdefault(channel, float(value))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(doc)
    exposure = doc.get("exposure") if isinstance(doc, dict) else None
    if isinstance(exposure, dict) and exposure.get("available"):
        for key in ("exposed_pct", "exposed_us", "covered_us"):
            value = exposure.get(key)
            if isinstance(value, (int, float)):
                out.setdefault(f"derived:exposure.{key}", float(value))
    return out or None


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------

def _entry_dir(library: Path, key: str) -> Path:
    return library / key


def _generation_dir(library: Path, key: str, gen: int) -> Path:
    return _entry_dir(library, key) / f"gen{gen:03d}"


def add_generation(library: Path, *, operator: str, arch: str,
                   input_shape: str, kind: str, identifier: str,
                   artifact: Optional[str] = None,
                   signature: Optional[str] = None,
                   time_us: Optional[float] = None,
                   gain: Optional[float] = None,
                   mechanism_verified: Optional[bool] = None,
                   bench_task: Optional[str] = None,
                   bench_baseline_us: Optional[float] = None,
                   family: Optional[str] = None,
                   device_id: Optional[int] = None,
                   note: str = "") -> Dict[str, Any]:
    """登记一代。gen 0 是外来参考，之后是本系统的产出。"""
    if kind not in ALL_KINDS:
        raise LibraryError(f"未知来源 '{kind}'；可选: {list(ALL_KINDS)}")

    # 整个读-改-写在锁内完成，包括 gen 号的计算与产物复制 ——
    # 三者任何一步被另一个任务插进来都会出事，见 exclusive 的说明。
    with exclusive(library):
        return _add_generation_locked(
            library, operator=operator, arch=arch, input_shape=input_shape,
            kind=kind, identifier=identifier, artifact=artifact,
            signature=signature, time_us=time_us, gain=gain,
            mechanism_verified=mechanism_verified, bench_task=bench_task,
            bench_baseline_us=bench_baseline_us, family=family,
            device_id=device_id, note=note)


def _add_generation_locked(library: Path, *, operator: str, arch: str,
                           input_shape: str, kind: str, identifier: str,
                           artifact: Optional[str] = None,
                           signature: Optional[str] = None,
                           time_us: Optional[float] = None,
                           gain: Optional[float] = None,
                           mechanism_verified: Optional[bool] = None,
                           bench_task: Optional[str] = None,
                           bench_baseline_us: Optional[float] = None,
                           family: Optional[str] = None,
                           device_id: Optional[int] = None,
                           note: str = "") -> Dict[str, Any]:
    """add_generation 的实体。调用方必须已持有库锁。"""
    if kind not in ALL_KINDS:
        raise LibraryError(f"未知来源 '{kind}'；可选: {list(ALL_KINDS)}")

    index = load_index(library)
    key = entry_key(arch, operator, input_shape)
    entry = index["entries"].setdefault(key, {
        "key": key, "arch": arch, "operator": operator,
        "input_shape": input_shape, "generations": [], "best_gen": None,
        "created_at": _now(),
    })

    # 算子族用于按类型检索。显式给的优先；推断的标明来源，
    # 因为推错比没有更糟 —— 查的人会以为这一族里没有参考。
    if family:
        entry["family"] = family
        entry["family_source"] = "explicit"
    elif "family" not in entry:
        inferred = infer_family(operator)
        if inferred:
            entry["family"] = inferred
            entry["family_source"] = "inferred"

    if bench_task:
        entry["bench_task"] = bench_task
    if bench_baseline_us is not None:
        # bench 的 baseline_perf_us 是评分基准线，不是参考产物 ——
        # 它作为目标标注存在，不参与 best 的评选。
        entry["bench_baseline_us"] = bench_baseline_us

    gen = len(entry["generations"])
    parent = gen - 1 if gen else None

    if gen > 0 and mechanism_verified is False:
        raise LibraryError(
            "机制未核实的产出不得入库作为参考。\n"
            "  封存一个解释不清的结果，会让后续所有代都建立在它之上，\n"
            "  而没人知道它为什么快。先把机制核实了再 promote。\n"
            "  （确需强行入库请用 --force，并在 note 里写明理由）")

    gen_dir = _generation_dir(library, key, gen)
    gen_dir.mkdir(parents=True, exist_ok=True)

    files: List[Dict[str, Any]] = []
    if kind in ARTIFACTLESS_KINDS:
        if artifact:
            raise LibraryError(
                f"{kind} 的产物属于厂商，不应拷入库。"
                "这一类只记可观测身份（签名 + 实测 + 环境）")
    else:
        if not artifact:
            raise LibraryError(f"{kind} 必须提供 --artifact")
        files = copy_artifact(Path(artifact), gen_dir / ARTIFACT_DIRNAME)

    snapshot = load_signature_snapshot(signature)
    if snapshot:
        (gen_dir / SIGNATURE_FILENAME).write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "key": key, "generation": gen, "kind": kind, "id": identifier,
        "sealed_at": _now(),
        "environment": detect_environment(),
        "device_id": device_id,
        "time_us": time_us,
        "gain_over_parent": gain,
        "promoted_from": parent,
        "mechanism_verified": mechanism_verified,
        "artifact_files": files,
        "artifact_present": bool(files),
        "signature_fields": len(snapshot) if snapshot else 0,
        "source_path": artifact or None,
        "note": note,
    }
    (gen_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    entry["generations"].append({
        "generation": gen, "kind": kind, "id": identifier,
        "device_id": device_id,
        "sealed_at": manifest["sealed_at"], "time_us": time_us,
        "gain_over_parent": gain, "promoted_from": parent,
        "mechanism_verified": mechanism_verified,
        "artifact_present": bool(files),
        "dir": str(gen_dir.relative_to(library)),
    })

    best_gen, basis = pick_best(entry)
    entry["best_gen"] = best_gen
    entry["best_basis"] = basis
    entry["updated_at"] = _now()

    save_index(library, index)
    return manifest


# ---------------------------------------------------------------------------
# 查询与校验
# ---------------------------------------------------------------------------

def resolve(library: Path, operator: str, arch: str,
            input_shape: str, generation: Optional[int] = None
            ) -> Dict[str, Any]:
    """查出当前该用哪一代作参考。"""
    index = load_index(library)
    key = entry_key(arch, operator, input_shape)
    entry = index["entries"].get(key)
    if not entry:
        known = sorted(index["entries"])
        raise LibraryError(
            f"库里没有 {key}。\n"
            + (f"  已有: {known}" if known else "  库是空的，先 register 一个参考"))

    gen = generation if generation is not None else entry.get("best_gen")
    if gen is None:
        raise LibraryError(
            f"{key} 下没有带实测时间的代次，无法确定最优。"
            "register / promote 时带上 --time-us")

    for record in entry["generations"]:
        if record["generation"] == gen:
            out = dict(record)
            out["key"] = key
            out["abs_dir"] = str(library / record["dir"])
            out["artifact_dir"] = (
                str(library / record["dir"] / ARTIFACT_DIRNAME)
                if record.get("artifact_present") else None)
            out["bench_task"] = entry.get("bench_task")
            out["bench_baseline_us"] = entry.get("bench_baseline_us")
            return out
    raise LibraryError(f"{key} 下没有第 {gen} 代")


def _read_manifest(library: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    path = library / record["dir"] / MANIFEST_FILENAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LibraryError(f"无法读取 {path}: {exc}") from exc


def verify(library: Path, operator: Optional[str] = None
           ) -> Tuple[List[str], int]:
    """校验库里每一代的产物是否与封存时一致。

    参考一变，之前所有相对它的加速比全部失效。这件事必须是**可发现的**，
    否则会带着一堆不可比的数字继续往下跑。
    """
    index = load_index(library)
    problems: List[str] = []
    checked = 0

    for key, entry in sorted(index["entries"].items()):
        if operator and entry.get("operator") != operator:
            continue
        for record in entry["generations"]:
            checked += 1
            gen_dir = library / record["dir"]
            if not gen_dir.is_dir():
                problems.append(f"{key} gen{record['generation']}: 目录丢失")
                continue
            try:
                manifest = _read_manifest(library, record)
            except LibraryError as exc:
                problems.append(f"{key} gen{record['generation']}: {exc}")
                continue

            for file_record in manifest.get("artifact_files", []):
                path = gen_dir / ARTIFACT_DIRNAME / file_record["relative"]
                if not path.is_file():
                    problems.append(
                        f"{key} gen{record['generation']}: "
                        f"{file_record['relative']} 已删除")
                    continue
                if _sha256(path) != file_record["sha256"]:
                    problems.append(
                        f"{key} gen{record['generation']}: "
                        f"{file_record['relative']} 内容已变更 —— "
                        "该代产物不再是封存时的那个，"
                        "所有相对它的加速比都已失效")
    return problems, checked


def signature_drift(library: Path, operator: str, arch: str,
                    input_shape: str, generation: int,
                    current_signature: str) -> Dict[str, Any]:
    """比对当前签名与某一代封存时的签名。

    这是 `builtin_aclnn` 这类「拷不走产物」的参考唯一的变更检测手段：
    CANN 升级后内置实现变了，签名会漂。漂了就说明参考不是原来那个，
    之前的加速比不可比。
    """
    key = entry_key(arch, operator, input_shape)
    sealed_path = _generation_dir(library, key, generation) / SIGNATURE_FILENAME
    if not sealed_path.is_file():
        raise LibraryError(
            f"{key} gen{generation} 没有封存签名，无法比对漂移。"
            "register 时带 --signature 才能做这件事")
    try:
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LibraryError(f"无法读取 {sealed_path}: {exc}") from exc

    current = load_signature_snapshot(current_signature)
    if not current:
        raise LibraryError("当前签名里没有有效读数，无法比对")

    drifted: List[Dict[str, Any]] = []
    compared = 0
    for channel, old in sealed.items():
        if channel not in current:
            continue
        compared += 1
        new = current[channel]
        denominator = max(abs(old), _DRIFT_EPS)
        pct = (new - old) / denominator * 100.0
        if abs(pct) > SIGNATURE_DRIFT_PCT:
            drifted.append({"channel": channel, "sealed": old,
                            "current": new, "drift_pct": pct})

    return {
        "key": key, "generation": generation,
        "compared_fields": compared,
        "drifted": drifted,
        "stable": not drifted,
        "threshold_pct": SIGNATURE_DRIFT_PCT,
    }


def find(library: Path, *, family: Optional[str] = None,
         operator: Optional[str] = None, arch: Optional[str] = None,
         shape: Optional[str] = None,
         require_artifact: bool = False) -> List[Dict[str, Any]]:
    """按类型检索参考。

    forge-kernel 要迭代某一类算子时，第一件事是「这一类有没有现成的参考
    可以当起点」。精确键（架构/算子/形状）查不了这个 —— 它要求你已经知道
    确切的形状。所以这里按族、按算子名子串、按架构筛，返回每个条目当前
    最优的那一代。
    """
    index = load_index(library)
    out: List[Dict[str, Any]] = []

    for key, entry in sorted(index["entries"].items()):
        if arch and entry.get("arch") != arch:
            continue
        if family and entry.get("family") != family:
            continue
        if operator and operator.lower() not in str(entry.get("operator", "")).lower():
            continue
        if shape and shape.lower() not in str(entry.get("input_shape", "")).lower():
            continue

        best_gen = entry.get("best_gen")
        record = None
        for candidate in entry.get("generations", []):
            if candidate["generation"] == best_gen:
                record = candidate
                break
        if record is None:
            continue
        if require_artifact and not record.get("artifact_present"):
            continue

        out.append({
            "key": key,
            "operator": entry.get("operator"),
            "arch": entry.get("arch"),
            "input_shape": entry.get("input_shape"),
            "family": entry.get("family"),
            "family_source": entry.get("family_source"),
            "generations": len(entry.get("generations", [])),
            "best_gen": best_gen,
            "best_basis": entry.get("best_basis"),
            "cumulative_gain": cumulative_gain(
                entry.get("generations", []), best_gen or 0),
            "kind": record.get("kind"),
            "id": record.get("id"),
            "time_us": record.get("time_us"),
            "artifact_present": record.get("artifact_present"),
            "artifact_dir": (str(library / record["dir"] / ARTIFACT_DIRNAME)
                             if record.get("artifact_present") else None),
            "bench_task": entry.get("bench_task"),
        })
    return out


def ingest(library: Path, *, source: Path, operator: str, arch: str,
           input_shape: str, identifier: Optional[str] = None,
           family: Optional[str] = None,
           signature: Optional[str] = None,
           time_us: Optional[float] = None,
           device_id: Optional[int] = None,
           work_dir: Optional[Path] = None,
           bench_task: Optional[str] = None,
           bench_baseline_us: Optional[float] = None,
           note: str = "") -> Dict[str, Any]:
    """把本地一个算子目录纳入库，作为自定义参考（gen0）。

    这条路径做三件事，少任何一件这个参考都不算「封好了」：

      1. 校验目录确实像一个算子实现，不是随手指的路径
      2. **封存**：产出禁读清单与专有名词清单。有源码的参考必须封，
         否则 no-code 边界形同虚设 —— agent 可以随便读它，而读了之后
         架构结论就是抄的
      3. 登记进库作为 gen0

    手工串这三步容易漏第 2 步，而漏了之后一切照常运转，只是方法失效了。
    """
    if not source.exists():
        raise LibraryError(f"参考目录不存在: {source}")
    if not source.is_dir():
        raise LibraryError(f"参考必须是目录: {source}")

    source_files = [p for p in source.rglob("*") if p.is_file()]
    if not source_files:
        raise LibraryError(f"参考目录下没有任何文件: {source}")

    seal_report: Dict[str, Any] = {}
    work = work_dir or (library / "_seals" / entry_key(
        arch, operator, input_shape).replace("/", "_"))
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import seal_reference as sealer  # noqa: PLC0415

        manifest = sealer.seal(source, operator, work)
        seal_report = {
            "work_dir": str(work),
            "files": manifest["file_count"],
            "source_files": manifest["source_file_count"],
            "identifiers": manifest["identifier_count"],
            "sealed_paths": str(work / sealer.SEALED_PATHS_FILENAME),
            "denylist": str(work / sealer.DENYLIST_FILENAME),
        }
    except Exception as exc:  # noqa: BLE001 - 封存失败不能静默放行
        raise LibraryError(
            f"封存失败: {exc}\n"
            "  有源码的参考必须先封存，否则 no-code 边界形同虚设 —— \n"
            "  agent 可以随便读它，而读了之后架构结论就是抄来的。"
        ) from exc

    record = add_generation(
        library, operator=operator, arch=arch, input_shape=input_shape,
        kind="opp_install", identifier=identifier or operator,
        artifact=str(source), signature=signature, time_us=time_us,
        family=family, device_id=device_id, bench_task=bench_task,
        bench_baseline_us=bench_baseline_us,
        note=note or f"自定义参考，源自 {source}",
    )
    record["seal"] = seal_report
    return record


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def cmd_register(args: argparse.Namespace) -> int:
    try:
        manifest = add_generation(
            Path(args.library), operator=args.operator, arch=args.arch,
            input_shape=args.input_shape, kind=args.kind,
            identifier=args.id, artifact=args.artifact,
            signature=args.signature, time_us=args.time_us,
            family=args.family, device_id=args.device_id,
            bench_task=args.bench_task,
            bench_baseline_us=args.bench_baseline_us,
            note=args.note,
        )
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(f"已登记 gen{manifest['generation']}: {manifest['kind']}:{manifest['id']}")
    print(f"  键      {manifest['key']}")
    print(f"  环境    CANN {manifest['environment']['cann_version']}")
    if manifest["artifact_present"]:
        print(f"  产物    {len(manifest['artifact_files'])} 个文件已复制进库")
    else:
        print("  产物    无（厂商实现不归我们保管，只记可观测身份）")
    print(f"  签名    {manifest['signature_fields']} 个字段")
    if manifest["environment"]["cann_version"] == "unknown":
        print("\n提示: 没探到 CANN 版本。不同版本下的参考不是同一个东西，"
              "建议 source set_env.sh 后重做，或在 --note 里写明版本。",
              file=sys.stderr)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    if not args.mechanism_verified and not args.force:
        print("错误: 机制未核实的产出不得入库作为参考。\n"
              "  封存一个解释不清的结果，会让后续所有代都建立在它之上，\n"
              "  而没人知道它为什么快 —— 下一代的「增益」也就无从归因。\n"
              "  campaign.py record 判为 accept（而非 accept_unverified）时\n"
              "  才算核实过。确需强行入库请加 --force 并在 --note 写明理由。",
              file=sys.stderr)
        return 1
    try:
        manifest = add_generation(
            Path(args.library), operator=args.operator, arch=args.arch,
            input_shape=args.input_shape, kind="sealed_kernel",
            identifier=args.id or f"{args.operator}_promoted",
            artifact=args.artifact, signature=args.signature,
            time_us=args.time_us, gain=args.gain,
            device_id=args.device_id,
            mechanism_verified=bool(args.mechanism_verified),
            note=args.note,
        )
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(f"已封存 gen{manifest['generation']}")
    print(f"  来自    gen{manifest['promoted_from']}")
    if manifest["gain_over_parent"]:
        print(f"  增益    {(manifest['gain_over_parent'] - 1) * 100:+.2f}%")
    print(f"  产物    {len(manifest['artifact_files'])} 个文件")
    print(f"  机制    {'已核实' if manifest['mechanism_verified'] else '未核实（强行入库）'}")
    print("\n下一代可以用它作参考：")
    print(f"  reference_library.py resolve --library {args.library} \\")
    print(f"      --operator {args.operator} --arch {args.arch} \\")
    print(f"      --input-shape {json.dumps(args.input_shape, ensure_ascii=False)}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    try:
        record = resolve(Path(args.library), args.operator, args.arch,
                         args.input_shape, args.generation)
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    print(f"gen{record['generation']}  {record['kind']}:{record['id']}")
    print(f"  实测      {record.get('time_us')} us")
    print(f"  机制核实  {record.get('mechanism_verified')}")
    if record["artifact_dir"]:
        print(f"  产物      {record['artifact_dir']}")
        print("\n接到 campaign 上：")
        print(f"  reference_adapter.py prepare --kind {record['kind']} \\")
        print(f"      --op-name {record['id']} --path {record['artifact_dir']} \\")
        print("      --work-dir ./forge_work")
    else:
        print("  产物      无（厂商实现，调用即可，不需要路径）")
    if record.get("bench_baseline_us"):
        print(f"\n  bench 基准线 {record['bench_baseline_us']} us"
              "（评分用，不是参考产物）")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """把本地某个目录下的算子纳入库作为自定义参考。"""
    try:
        record = ingest(
            Path(args.library), source=Path(args.source),
            operator=args.operator, arch=args.arch,
            input_shape=args.input_shape, identifier=args.id,
            family=args.family, signature=args.signature,
            time_us=args.time_us, device_id=args.device_id,
            work_dir=Path(args.work_dir) if args.work_dir else None,
            bench_task=args.bench_task,
            bench_baseline_us=args.bench_baseline_us, note=args.note,
        )
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    seal = record.get("seal", {})
    print(f"已纳入自定义参考 gen{record['generation']}: {record['id']}")
    print(f"  键        {record['key']}")
    print(f"  产物      {len(record['artifact_files'])} 个文件已复制进库")
    print(f"  封存      源码 {seal.get('source_files')} 个，"
          f"抽出专有名词 {seal.get('identifiers')} 个")
    print(f"  禁读清单  {seal.get('sealed_paths')}")
    print(f"  名词清单  {seal.get('denylist')}")
    print()
    print("开 campaign 时把这两份清单接上，no-code 边界才真正生效：")
    print(f"  export FORGE_SEALED_PATHS={seal.get('sealed_paths')}")
    print(f"  fence_check.py --denylist {seal.get('denylist')} ...")
    if record["environment"]["cann_version"] == "unknown":
        print("\n提示: 没探到 CANN 版本。不同版本下的参考不是同一个东西，"
              "建议 source set_env.sh 后重做，或在 --note 写明。",
              file=sys.stderr)
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    """按类型检索：这一类算子有没有现成的参考可以当起点。"""
    try:
        hits = find(Path(args.library), family=args.family,
                    operator=args.operator, arch=args.arch,
                    shape=args.shape, require_artifact=args.with_artifact)
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return 0 if hits else 1

    if not hits:
        print("没有命中。可用的族：")
        index = load_index(Path(args.library))
        families = sorted({e.get("family") for e in index["entries"].values()
                           if e.get("family")})
        print("  " + (", ".join(families) if families else "（库是空的）"))
        return 1

    print(f"命中 {len(hits)} 项\n")
    for hit in hits:
        mark = "" if hit["family_source"] == "explicit" else "（推断）"
        print(f"{hit['operator']}  [{hit['arch']}]  族 {hit['family']}{mark}")
        print(f"  形状     {hit['input_shape']}")
        print(f"  最优     gen{hit['best_gen']} / 共 {hit['generations']} 代"
              f"  累计 {(hit['cumulative_gain'] - 1) * 100:+.2f}%")
        print(f"  依据     {hit['best_basis']}")
        if hit["artifact_dir"]:
            print(f"  产物     {hit['artifact_dir']}")
        else:
            print(f"  产物     无（{hit['kind']}，调用即可）")
        print()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    try:
        index = load_index(Path(args.library))
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    entries = index["entries"]
    if args.as_json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0
    if not entries:
        print("库是空的")
        return 0

    for key, entry in sorted(entries.items()):
        best = entry.get("best_gen")
        print(f"{key}")
        print(f"  形状 {entry['input_shape']}  共 {len(entry['generations'])} 代"
              + (f"，最优 gen{best}" if best is not None else "，无实测"))
        if entry.get("bench_task"):
            print(f"  bench 任务 {entry['bench_task']}"
                  + (f"  基准线 {entry['bench_baseline_us']} us"
                     if entry.get("bench_baseline_us") else ""))
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    try:
        index = load_index(Path(args.library))
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    key = entry_key(args.arch, args.operator, args.input_shape)
    entry = index["entries"].get(key)
    if not entry:
        print(f"错误: 库里没有 {key}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        return 0

    print(f"{key}")
    print(f"形状 {entry['input_shape']}")
    if entry.get("bench_baseline_us"):
        print(f"bench 基准线 {entry['bench_baseline_us']} us（评分用）")
    print()
    cumulative = 1.0
    for record in entry["generations"]:
        gain = record.get("gain_over_parent")
        if isinstance(gain, (int, float)):
            cumulative *= gain
        mark = "★" if record["generation"] == entry.get("best_gen") else " "
        gain_text = (f"{(gain - 1) * 100:+6.2f}%"
                     if isinstance(gain, (int, float)) else "     —")
        verified = {True: "已核实", False: "未核实", None: "—"}[
            record.get("mechanism_verified")]
        print(f"{mark} gen{record['generation']:<3} {record['kind']:<14} "
              f"{str(record.get('time_us')):>9} us  {gain_text}  {verified}")
    if cumulative != 1.0:
        print(f"\n累计相对 gen0: {(cumulative - 1) * 100:+.2f}%")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        problems, checked = verify(Path(args.library), args.operator)
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if not problems:
        print(f"校验通过：{checked} 代产物与封存时一致")
        return 0
    print(f"校验失败：{checked} 代中有 {len(problems)} 处问题\n", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print("\n参考一变，之前所有相对它的加速比都不再可比。"
          "继续跑之前先确认这些变动是不是预期的。", file=sys.stderr)
    return 1


def cmd_drift(args: argparse.Namespace) -> int:
    try:
        report = signature_drift(
            Path(args.library), args.operator, args.arch, args.input_shape,
            args.generation, args.signature)
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["stable"] else 1

    print(f"{report['key']} gen{report['generation']}：比对 "
          f"{report['compared_fields']} 个字段")
    if report["stable"]:
        print("签名稳定，参考没变。")
        return 0
    print(f"\n{len(report['drifted'])} 个字段漂移超过 "
          f"{report['threshold_pct']}%：", file=sys.stderr)
    for item in report["drifted"]:
        print(f"  {item['channel']}: {item['sealed']} → {item['current']}"
              f"（{item['drift_pct']:+.1f}%）", file=sys.stderr)
    print("\n参考的行为变了（多半是 CANN 升级或环境变化）。"
          "之前相对它的加速比不可比，应重新 register 一代并重测噪声底。",
          file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--library", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", dest="as_json",
                        default=argparse.SUPPRESS)

    identity = argparse.ArgumentParser(add_help=False)
    identity.add_argument("--operator", required=True)
    identity.add_argument("--arch", required=True)
    identity.add_argument("--input-shape", required=True)

    parser = argparse.ArgumentParser(
        description="封存产物库：跨 campaign 保存参考与各代产出")
    parser.add_argument("--library", default=str(
        Path.home() / ".forge-kernel" / "library"),
        help="库根目录（也可写在子命令后面）")
    parser.add_argument("--json", action="store_true", dest="as_json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="登记一个外来参考（gen0）",
                           parents=[common, identity])
    p_reg.add_argument("--kind", required=True, choices=list(ALL_KINDS))
    p_reg.add_argument("--id", required=True)
    p_reg.add_argument("--artifact", default=None,
                       help="产物路径；builtin_aclnn 不需要也不允许")
    p_reg.add_argument("--signature", default=None,
                       help="signature_probe 输出，用于日后检测参考漂移")
    p_reg.add_argument("--time-us", type=float, default=None)
    p_reg.add_argument("--bench-task", default=None,
                       help="挂到 cann-bench 任务，如 tasks/level2/add_layer_norm")
    p_reg.add_argument("--bench-baseline-us", type=float, default=None,
                       help="cases.yaml 的 baseline_perf_us；评分基准线，非参考产物")
    p_reg.add_argument("--family", default=None,
                       help="算子族；不给则从算子名推断")
    p_reg.add_argument("--device-id", type=int, default=None)
    p_reg.add_argument("--note", default="")
    p_reg.set_defaults(func=cmd_register)

    p_pro = sub.add_parser("promote", help="把本系统产出封存为下一代参考",
                           parents=[common, identity])
    p_pro.add_argument("--artifact", required=True)
    p_pro.add_argument("--id", default=None)
    p_pro.add_argument("--signature", default=None)
    p_pro.add_argument("--time-us", type=float, default=None)
    p_pro.add_argument("--gain", type=float, default=None,
                       help="相对上一代的增益")
    p_pro.add_argument("--mechanism-verified", action="store_true",
                       help="campaign 判为 accept（而非 accept_unverified）时才给")
    p_pro.add_argument("--force", action="store_true",
                       help="机制未核实也强行入库")
    p_pro.add_argument("--device-id", type=int, default=None)
    p_pro.add_argument("--note", default="")
    p_pro.set_defaults(func=cmd_promote)

    p_res = sub.add_parser("resolve", help="查当前该用哪一代作参考",
                           parents=[common, identity])
    p_res.add_argument("--generation", type=int, default=None,
                       help="指定代次；不给则取实测最快的一代")
    p_res.set_defaults(func=cmd_resolve)

    p_lin = sub.add_parser("lineage", help="展示代际链条",
                           parents=[common, identity])
    p_lin.set_defaults(func=cmd_lineage)

    p_ing = sub.add_parser(
        "ingest", help="把本地算子目录纳入库作为自定义参考（含封存）",
        parents=[common, identity])
    p_ing.add_argument("--source", required=True, help="本地算子实现目录")
    p_ing.add_argument("--id", default=None)
    p_ing.add_argument("--family", default=None,
                       help="算子族；不给则从算子名推断")
    p_ing.add_argument("--signature", default=None)
    p_ing.add_argument("--time-us", type=float, default=None)
    p_ing.add_argument("--device-id", type=int, default=None,
                       help="测这个时间用的是哪张卡。多卡机器上卡间有系统性"
                            "差异，记下来才知道哪些数字可比")
    p_ing.add_argument("--work-dir", default=None,
                       help="封存清单的落地目录；默认放库内 _seals/")
    p_ing.add_argument("--bench-task", default=None)
    p_ing.add_argument("--bench-baseline-us", type=float, default=None)
    p_ing.add_argument("--note", default="")
    p_ing.set_defaults(func=cmd_ingest)

    p_find = sub.add_parser(
        "find", help="按算子族/名字/架构检索参考", parents=[common])
    p_find.add_argument("--family", default=None,
                        help="算子族，如 matmul / reduction / attention")
    p_find.add_argument("--operator", default=None, help="算子名子串")
    p_find.add_argument("--arch", default=None)
    p_find.add_argument("--shape", default=None, help="形状子串")
    p_find.add_argument("--with-artifact", action="store_true",
                        help="只要有产物可取的（排除厂商实现）")
    p_find.set_defaults(func=cmd_find)

    p_lst = sub.add_parser("list", help="列出库里有什么", parents=[common])
    p_lst.set_defaults(func=cmd_list)

    p_ver = sub.add_parser("verify", help="校验产物是否与封存时一致",
                           parents=[common])
    p_ver.add_argument("--operator", default=None)
    p_ver.set_defaults(func=cmd_verify)

    p_dft = sub.add_parser("drift", help="比对当前签名与封存签名，检测参考变更",
                           parents=[common, identity])
    p_dft.add_argument("--generation", type=int, default=0)
    p_dft.add_argument("--signature", required=True)
    p_dft.set_defaults(func=cmd_drift)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except LibraryError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
