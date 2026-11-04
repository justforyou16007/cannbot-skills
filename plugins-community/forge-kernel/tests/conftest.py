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
"""测试不得改动随包发布的知识库。

这条守卫的由来是一次真实污染。`seal_reference.seal()` 的 `kb_root` 默认值
指向**随包发布**的 `skills/forge-hwkb/references/` —— 那是对的，L-DENY 这道
闸就是靠这根接线喂数据的（它曾经因为两段脱节而完全失效）。

但测试里有十来处 `seal(...)` 不传 `kb_root`，于是每跑一次测试，
`demo` / `demo_attn` / `KernelSecret` 这些**测试符号**就被写进随包目录。
后果不是脏一点：

  - 用户装上插件之后，这些词会让他自己的卡片误报 L-DENY
  - 而误报的代价是整条规则被豁免掉 —— 真越界也跟着放过去

这类污染很难自己暴露：文件被 .gitignore 挡住，所以 `git status` 干净；
测试全绿；只有真去 `cat` 那个文件才看得见。所以要有一道会响的守卫。

守卫**不**替测试改行为：它在整轮测试前后各拍一次快照，变了就让整轮失败，
并指名哪个文件变了。要写就传 `kb_root=tmp_path`。
"""

import sys
from pathlib import Path

import pytest

#: 随包发布的知识库根。和 seal_reference.DEFAULT_KB_ROOT、
#: hwkb.DEFAULT_KB_ROOT 指向同一处 —— 三者一致由
#: test_seal_and_boundary.py 的用例盯着。
SHIPPED_KB = (Path(__file__).resolve().parent.parent
              / "skills" / "forge-hwkb" / "references")


def _snapshot() -> dict:
    """随包知识库的内容快照：相对路径 → 字节数与内容摘要。

    连内容一起记，不只记文件列表 —— 一次 seal 往已有的 denylist 里**追加**
    几个词时，文件数不变。
    """
    import hashlib
    out = {}
    if not SHIPPED_KB.is_dir():
        return out
    for path in sorted(SHIPPED_KB.rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            data = b"<unreadable>"
        out[str(path.relative_to(SHIPPED_KB))] = hashlib.sha256(data).hexdigest()[:12]
    return out


@pytest.fixture(scope="session", autouse=True)
def _redirect_default_kb_root(tmp_path_factory):
    """整轮测试期间，把「随包知识库」这个默认值指到临时目录。

    为什么改默认值而不是给十六个调用点各传一次 `kb_root=tmp_path`：

      一处写清理由，胜过十六处重复      —— 而且新增的调用点自动被覆盖。
                                           逐点传参的办法下一次有人加一个
                                           seal() 调用就又漏了，而漏掉的
                                           表现是「测试全绿但随包目录被改」。

    这**不**削弱覆盖：没有任何用例断言「写进随包目录这件事本身」。真正要
    钉住的是「写方和读方拼的是同一个路径」（L-DENY 当初失效就是因为两段
    脱节），那由常量相等的用例盯着，与这些常量具体指向哪里无关。

    夹具顺序要紧：它和 shipped_kb_is_not_mutated 都是 session 级 autouse，
    这个先生效，那个才拍得到一份没被动过的快照。pytest 按定义顺序建立
    同级夹具，所以这个定义在前面。
    """
    stand_in = tmp_path_factory.mktemp("shipped_kb_stand_in")
    patched = []
    for module_name, attr in (("seal_reference", "DEFAULT_KB_ROOT"),
                              ("hwkb", "DEFAULT_KB_ROOT")):
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, attr):
            continue
        patched.append((module, attr, getattr(module, attr)))
        setattr(module, attr, stand_in)
    yield stand_in
    for module, attr, original in patched:
        setattr(module, attr, original)


@pytest.fixture(scope="session", autouse=True)
def shipped_kb_is_not_mutated():
    before = _snapshot()
    yield
    after = _snapshot()
    if before == after:
        return

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    lines = ["测试改动了随包发布的知识库 "
             f"{SHIPPED_KB}："]
    if added:
        lines.append(f"  新增: {added}")
    if removed:
        lines.append(f"  删除: {removed}")
    if changed:
        lines.append(f"  内容变了: {changed}")
    lines += [
        "",
        "多半是某处 seal() / hwkb 命令没传 kb_root，走了指向随包目录的默认值。",
        "传 kb_root=tmp_path。",
        "",
        "为什么这算失败而不是无害的副作用：写进去的是**测试符号**"
        "（demo / KernelSecret 之类），",
        "用户装上插件后它们会让他自己的卡片误报 L-DENY —— 而误报的代价是"
        "整条规则被豁免掉。",
        "这些文件被 .gitignore 挡着，所以 git status 是干净的、测试是全绿的，",
        "只有真去 cat 那个文件才看得见。所以这里必须响。",
    ]
    pytest.fail("\n".join(lines), pytrace=False)
