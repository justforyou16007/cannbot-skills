#!/usr/bin/env python3
"""Operator description file and test case validator.

Modes:
  static   – AST-level structural checks (no torch needed)
  runtime  – import + execute every case (needs torch)
  coverage – analyze dtype/shape/attr/boundary/annotation completeness
  all      – static + runtime + coverage

Usage:
  python3 validate_op.py --py /path/to/10_LayerNorm.py --mode static
  python3 validate_op.py --py /path/to/10_LayerNorm.py --mode runtime
  python3 validate_op.py --py /path/to/10_LayerNorm.py --mode coverage --op-class normalization
  python3 validate_op.py --py /path/to/10_LayerNorm.py --mode all --op-class normalization
"""

import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_json(py_path: str) -> str:
    """Find the matching .json file for a given .py path."""
    base = os.path.splitext(py_path)[0]
    for ext in (".json", ".jsonl"):
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    # try without id prefix (CANN format: py has id_, json doesn't)
    fname = os.path.basename(py_path)
    parts = fname.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        candidate = os.path.join(os.path.dirname(py_path), parts[1])
        if os.path.exists(candidate):
            return candidate
    return ""


def _load_json_lines(json_path: str) -> list[dict]:
    """Load JSONL file, returning list of parsed objects."""
    cases = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _extract_id_and_name(py_path: str) -> tuple[str, str]:
    """Extract (id_prefix, op_name) from filename like '10_LayerNorm.py'."""
    fname = os.path.splitext(os.path.basename(py_path))[0]
    parts = fname.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[0], parts[1]
    return "", fname


# ---------------------------------------------------------------------------
# static checks (AST-based, no torch needed)
# ---------------------------------------------------------------------------

def _check_static(py_path: str, json_path: str) -> dict:
    """Run structural checks without importing the module."""
    issues = []

    # file existence
    if not os.path.exists(py_path):
        return {"passed": False, "issues": [f"PY file not found: {py_path}"]}
    if json_path and not os.path.exists(json_path):
        issues.append(f"JSON file not found (expected: {json_path})")

    # parse AST
    with open(py_path, "r") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"passed": False, "issues": [f"Syntax error: {e}"]}

    # find definitions
    classes = {node.name: node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef)}
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}

    # check Model class
    if "Model" not in classes:
        issues.append("Missing class: Model(nn.Module)")
    else:
        model = classes["Model"]
        # check Model inherits nn.Module
        bases = [ast.unparse(b) for b in model.bases if hasattr(ast, 'unparse')]
        if not bases:
            bases_str = []
            for b in model.bases:
                try:
                    bases_str.append(ast.dump(b))
                except Exception:
                    bases_str.append(str(b))
            issues.append("Model class should inherit from nn.Module")

        # check forward method exists
        forward_methods = [n for n in ast.walk(model)
                           if isinstance(n, ast.FunctionDef) and n.name == "forward"]
        if not forward_methods:
            issues.append("Model class missing forward() method")

    # check required top-level functions
    for fn_name in ("get_input_groups", "get_init_inputs"):
        if fn_name not in functions:
            issues.append(f"Missing function: {fn_name}()")

    # check for NPU imports (forbidden)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "torch_npu" in alias.name or "framework" in alias.name:
                    issues.append(f"Forbidden NPU import: {alias.name}")

    # check JSON line count matches len(get_init_inputs) heuristically
    if json_path and os.path.exists(json_path):
        json_lines = sum(1 for line in open(json_path) if line.strip())

    passed = len(issues) == 0
    return {"passed": passed, "issues": issues}


# ---------------------------------------------------------------------------
# runtime checks (needs torch)
# ---------------------------------------------------------------------------

def _check_runtime(py_path: str) -> dict:
    """Import the module and run every case."""
    result = {"passed": True, "total": 0, "passed_cases": 0,
              "failed_cases": 0, "failures": []}

    spec = importlib.util.spec_from_file_location("_op_module", py_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        result["passed"] = False
        result["failures"].append({
            "case_idx": -1, "error_type": "ImportError",
            "error_msg": str(e)[:2000]
        })
        return result

    try:
        groups = module.get_input_groups()
        inits = module.get_init_inputs()
    except Exception as e:
        result["passed"] = False
        result["failures"].append({
            "case_idx": -1, "error_type": "CallError",
            "error_msg": f"get_input_groups() or get_init_inputs() failed: {e}"[:2000]
        })
        return result

    # Normalize: if get_init_inputs() returns [] (no init args needed),
    # pad to match group count. If it returns a non-empty list, length must match.
    if not inits:
        # Empty list means no init args — pad for all cases
        inits = [[] for _ in groups]
    elif len(groups) != len(inits):
        result["passed"] = False
        result["failures"].append({
            "case_idx": -1, "error_type": "LengthMismatch",
            "error_msg": f"len(groups)={len(groups)} != len(inits)={len(inits)}"
        })
        return result

    result["total"] = len(groups)

    for i, (init_args, group) in enumerate(zip(inits, groups)):
        try:
            if init_args:
                model = module.Model(*init_args)
            else:
                model = module.Model()
            output = model(*group)

            # check output type
            if not isinstance(output, (__import__("torch").Tensor, tuple, list)):
                result["failed_cases"] += 1
                result["failures"].append({
                    "case_idx": i, "error_type": "OutputType",
                    "error_msg": f"Expected Tensor, got {type(output)}"
                })
                continue

            # check for NaN/Inf (skip for isnan/isinf operators)
            op_name = os.path.basename(py_path).lower()
            skip_nan_check = any(kw in op_name for kw in ("isnan", "isinf", "isfinite", "isinf"))
            if not skip_nan_check:
                tensors = [output] if isinstance(output, __import__("torch").Tensor) else list(output)
                for t in tensors:
                    if isinstance(t, __import__("torch").Tensor):
                        if __import__("torch").isnan(t).any():
                            result["failed_cases"] += 1
                            result["failures"].append({
                                "case_idx": i, "error_type": "NaNOutput",
                                "error_msg": "Model output contains NaN"
                            })
                            break
                        if __import__("torch").isinf(t).any():
                            result["failed_cases"] += 1
                            result["failures"].append({
                                "case_idx": i, "error_type": "InfOutput",
                                "error_msg": "Model output contains Inf"
                            })
                            break

            result["passed_cases"] += 1

        except Exception as e:
            result["failed_cases"] += 1
            result["failures"].append({
                "case_idx": i, "error_type": type(e).__name__,
                "error_msg": str(e)[:2000]
            })

    result["passed"] = (result["failed_cases"] == 0)
    return result


# ---------------------------------------------------------------------------
# coverage analysis
# ---------------------------------------------------------------------------

# Default dtype sets per operator class
DEFAULT_DTYPES = {
    "elementwise": {"float16", "float32", "bfloat16"},
    "reduction": {"float16", "float32", "bfloat16"},
    "normalization": {"float16", "float32", "bfloat16"},
    "index": {"float16", "float32"},
    "comparison": {"float16", "float32", "int32", "bool"},
    "matmul": {"float16", "float32", "bfloat16"},
    "creation": {"float16", "float32", "int32"},
    "manipulation": {"float16", "float32"},
    "composite": {"float16", "float32", "bfloat16"},
    "backward": {"float16", "float32"},
}

# Boundary requirements per operator class
BOUNDARY_REQUIREMENTS = {
    "elementwise": ["nonaligned_shape", "scalar_shape", "large_shape"],
    "reduction": ["nonaligned_shape", "scalar_shape", "large_shape", "neg_dim", "keepdim_both"],
    "normalization": ["nonaligned_shape", "large_shape", "missing_optional"],
    "index": ["neg_dim", "boundary_index", "duplicate_index", "scalar_shape"],
    "comparison": ["broadcast", "scalar_shape"],
    "matmul": ["nonaligned_k", "large_matrix", "scalar_shape"],
    "creation": ["zero_size", "scalar_shape", "large_shape"],
    "manipulation": ["neg_dim", "scalar_shape", "nonaligned_shape"],
    "composite": ["nonaligned_shape", "scalar_shape", "large_shape"],
    "backward": ["nonaligned_shape", "scalar_shape", "large_shape"],
}


def _infer_op_class(py_path: str) -> str:
    """Try to infer operator class from the forward implementation."""
    with open(py_path, "r") as f:
        source = f.read().lower()

    # normalization patterns
    if any(kw in source for kw in ("layer_norm", "group_norm", "batch_norm", "rms_norm")):
        return "normalization"
    # reduction patterns
    if any(kw in source for kw in (".sum(", ".mean(", ".max(", ".min(",
                                     "softmax", "log_softmax", "argmax", "argmin")):
        return "reduction"
    # index patterns
    if any(kw in source for kw in ("index_select", "gather", "scatter", "index_put", "nonzero")):
        return "index"
    # comparison patterns
    if any(kw in source for kw in (".eq(", ".ne(", ".gt(", ".lt(", ".ge(", ".le(",
                                     "logical_and", "logical_or", "logical_not",
                                     "torch.eq", "torch.ne", "torch.gt", "torch.lt")):
        return "comparison"
    # matmul patterns
    if any(kw in source for kw in ("matmul", "bmm", "linear", "@")):
        return "matmul"
    # creation patterns
    if any(kw in source for kw in ("torch.eye", "torch.zeros", "torch.ones",
                                     "torch.full", "torch.arange")):
        return "creation"
    # manipulation patterns
    if any(kw in source for kw in ("permute", "cat(", "split(", "pad(", "repeat(", "reshape")):
        return "manipulation"
    # composite
    if "chunk" in source and "silu" in source:
        return "composite"
    # backward (gradient computation)
    if any(kw in source for kw in ("grad_output", "backward", "_backward",
                                     "self_or_result", "is_result")):
        return "backward"

    return "elementwise"  # default


def _analyze_dtype_coverage(cases: list[dict], op_class: str) -> dict:
    """Score dtype coverage."""
    expected = DEFAULT_DTYPES.get(op_class, {"float16", "float32"})
    max_score = 25
    details = []
    score = 0

    # collect observed dtypes
    observed = []
    has_inputs_format = False
    for case in cases:
        inputs = case.get("inputs", [])
        if inputs:
            has_inputs_format = True
            for inp in inputs:
                if inp.get("dtype") and inp.get("type") == "tensor":
                    observed.append(inp["dtype"])
        else:
            # CANN flat format
            if "dtype" in case:
                observed.append(case["dtype"])

    from collections import Counter
    counts = Counter(observed)
    total = len(observed)

    if not observed:
        details.append("No tensor dtype found in JSON")
        return {"score": 0, "max": max_score, "details": details}

    # float16 ratio (10 pts)
    fp16_pct = counts.get("float16", 0) / total * 100
    if fp16_pct >= 30:
        score += 10
        details.append(f"float16 占比 {fp16_pct:.0f}% ≥ 30%: OK (10/10)")
    else:
        pts = max(0, int(fp16_pct / 30 * 10))
        score += pts
        details.append(f"float16 占比 {fp16_pct:.0f}% < 30%: {pts}/10")

    # float32 (5 pts)
    if "float32" in expected and counts.get("float32", 0) >= 2:
        score += 5
        details.append(f"float32: {counts['float32']} cases ≥ 2: OK (5/5)")
    elif "float32" in expected:
        details.append(f"float32: {counts.get('float32', 0)} cases < 2: 0/5")
    else:
        score += 5

    # bfloat16 (5 pts)
    if "bfloat16" in expected and counts.get("bfloat16", 0) >= 2:
        score += 5
        details.append(f"bfloat16: {counts['bfloat16']} cases ≥ 2: OK (5/5)")
    elif "bfloat16" in expected:
        details.append(f"bfloat16: {counts.get('bfloat16', 0)} cases < 2: 0/5 (建议 ≥ 2)")
    else:
        score += 5

    # integer/bool (5 pts) — only for classes that need them
    int_bool = sum(counts.get(d, 0) for d in ("int32", "int64", "int8", "uint8", "bool"))
    if "int32" in expected or "bool" in expected:
        if int_bool >= 2:
            score += 5
            details.append(f"整数/bool: {int_bool} cases ≥ 2: OK (5/5)")
        else:
            details.append(f"整数/bool: {int_bool} cases < 2: 0/5")
    else:
        score += 5

    return {"score": score, "max": max_score, "details": details}


def _analyze_shape_coverage(cases: list[dict]) -> dict:
    """Score shape coverage."""
    max_score = 25
    score = 0
    details = []

    shapes = []
    for case in cases:
        inputs = case.get("inputs", [])
        if inputs:
            for inp in inputs:
                if inp.get("shape") is not None and inp.get("type") == "tensor":
                    shapes.append(tuple(inp["shape"]))
                elif inp.get("type") == "tensor_list":
                    # tensor_list: collect shapes from "shapes" field (format 1)
                    # or "tensors" array (format 2)
                    for s in inp.get("shapes", []):
                        shapes.append(tuple(s))
                    for t in inp.get("tensors", []):
                        if t.get("shape"):
                            shapes.append(tuple(t["shape"]))
        elif "input_shape" in case:
            try:
                shapes.append(tuple(eval(case["input_shape"])))
            except Exception:
                pass

    if not shapes:
        details.append("No tensor shapes found in JSON")
        return {"score": 0, "max": max_score, "details": details}

    unique_shapes = set(shapes)
    numels = [abs(int(math.prod(s))) if s else 0 for s in unique_shapes]

    # ≥ 3 rank types (8 pts)
    ranks = set(len(s) for s in unique_shapes)
    if len(ranks) >= 3:
        score += 8
        details.append(f"维度数种类: {len(ranks)} ({ranks}) ≥ 3: OK (8/8)")
    else:
        pts = max(0, len(ranks) * 2)
        score += pts
        details.append(f"维度数种类: {len(ranks)} ({ranks}) < 3: {pts}/8")

    # nonaligned shape (5 pts)
    nonaligned = [s for s in unique_shapes
                  if any(d not in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
                         for d in s)]
    if nonaligned:
        score += 5
        details.append(f"非对齐 shape: {nonaligned[:3]}... OK (5/5)")
    else:
        details.append("缺少非对齐 shape: 0/5")

    # large shape (5 pts)
    large = [s for s, n in zip(unique_shapes, numels) if n > 2**18]
    if large:
        score += 5
        details.append(f"超大 shape (numel>2^18): {large[:3]}... OK (5/5)")
    else:
        details.append("缺少超大 shape (numel>2^18): 0/5")

    # scalar/tiny shape (7 pts)
    tiny = [s for s, n in zip(unique_shapes, numels) if n <= 2 and len(s) > 0] + \
           [s for s in unique_shapes if s == () or s == tuple()]
    if tiny:
        score += 7
        details.append(f"标量/极小 shape: {tiny[:3]}... OK (7/7)")
    else:
        all_small = any(n <= 4 for n in numels)
        if all_small:
            score += 4
            details.append(f"有小 shape 但不含标量: 4/7")
        else:
            details.append("缺少标量/极小 shape ([]/[1]/[2]): 0/7")

    return {"score": score, "max": max_score, "details": details}


def _analyze_attr_coverage(cases: list[dict]) -> dict:
    """Score attribute coverage."""
    max_score = 15
    score = 0
    details = []

    # collect attrs by name
    attrs = {}
    for case in cases:
        inputs = case.get("inputs", [])
        for inp in inputs:
            if inp.get("type") == "attr":
                name = inp["name"]
                if name not in attrs:
                    attrs[name] = []
                attrs[name].append(inp.get("value"))

    if not attrs:
        details.append("无 attr 参数")
        return {"score": max_score, "max": max_score, "details": details}

    bool_score = 5
    enum_score = 5
    dim_score = 5

    for name, values in attrs.items():
        unique = set(str(v) for v in values)
        # bool attr
        if all(str(v).lower() in ("true", "false") for v in unique):
            if len(unique) >= 2:
                bool_score = 5
                details.append(f"bool attr '{name}': True+False 都覆盖 OK")
            else:
                bool_score = 0
                details.append(f"bool attr '{name}': 仅覆盖 {unique}，缺少另一个取值")
        # dim-like attr
        elif all(str(v).lstrip("-").isdigit() for v in unique if v is not None):
            has_pos = any(int(v) > 0 for v in unique)
            has_neg = any(int(v) < 0 for v in unique)
            has_zero = any(int(v) == 0 for v in unique)
            if has_pos and has_neg:
                dim_score = 5
                details.append(f"dim attr '{name}': 正/负/0 覆盖 OK")
            elif has_pos and has_zero:
                dim_score = 4
                details.append(f"dim attr '{name}': 缺少负数 dim")
            else:
                dim_score = 2
                details.append(f"dim attr '{name}': dim 取值不够 (缺正/负/0 中的维度)")
            # also count as enum coverage
            if len(unique) >= 3:
                enum_score = 5
                details.append(f"数值 attr '{name}': {len(unique)} 种取值 ≥ 3 OK")
            else:
                enum_score = max(0, len(unique) * 1)
                details.append(f"数值 attr '{name}': {len(unique)} 种取值 < 3")
        # list attr
        elif all(v is not None and (isinstance(v, list) or str(v).startswith("[")) for v in values):
            lens = set(len(v) if isinstance(v, list) else len(eval(str(v))) for v in values)
            if len(lens) >= 2:
                enum_score = 5
                details.append(f"list attr '{name}': 不同长度 {lens} OK")
            else:
                details.append(f"list attr '{name}': 仅一种长度")
        # str/enum attr
        else:
            if len(unique) >= 2:
                enum_score = 5
                details.append(f"enum attr '{name}': {len(unique)} 种取值 ({unique}) OK")
            else:
                details.append(f"enum attr '{name}': 仅 1 种取值 ({unique})，建议覆盖全部合法值")

    score = bool_score + enum_score + dim_score
    return {"score": min(score, max_score), "max": max_score, "details": details}


def _analyze_boundary_coverage(cases: list[dict], op_class: str) -> dict:
    """Score boundary case coverage."""
    max_score = 20
    score = 0
    details = []
    required = BOUNDARY_REQUIREMENTS.get(op_class, ["nonaligned_shape", "scalar_shape", "large_shape"])

    # check for optional params omitted (required=false)
    has_optional = False
    optional_omitted = False
    for case in cases:
        inputs = case.get("inputs", [])
        for inp in inputs:
            if inp.get("required") is False or inp.get("required") == "false":
                has_optional = True
        # check if any case has fewer inputs than the max
        lens = [len(case.get("inputs", [])) for case in cases]
        if len(set(lens)) > 1:
            optional_omitted = True

    if has_optional and optional_omitted and "missing_optional" in required:
        score += 8
        details.append("required=false 参数被省略: OK (8/8)")
    elif "missing_optional" in required:
        details.append("required=false 参数未被省略: 0/8")
    else:
        score += 8

    # broadcast shapes (for multi-input ops)
    has_multiple_tensors = False
    for case in cases:
        tensors = [inp for inp in case.get("inputs", []) if inp.get("type") == "tensor"]
        if len(tensors) >= 2:
            has_multiple_tensors = True
            shapes = [tuple(t["shape"]) for t in tensors]
            if len(set(shapes)) > 1:
                if "broadcast" in required:
                    score += 6
                    details.append("广播 shape 组合: OK (6/6)")
                    break
    else:
        if not has_multiple_tensors:
            score += 6  # single-input op, N/A
        elif "broadcast" in required:
            details.append("缺少广播 shape 组合: 0/6")

    # index boundary (for index ops)
    if "boundary_index" in required or "duplicate_index" in required:
        score += 6
        details.append("索引边界/重复: 由 runtime 检查保障 (6/6)")
    else:
        score += 6

    return {"score": min(score, max_score), "max": max_score,
            "details": details}


def _analyze_annotation_quality(py_path: str) -> dict:
    """Score annotation/comment quality via AST inspection."""
    max_score = 15
    score = 0
    details = []

    with open(py_path, "r") as f:
        source = f.read()
    tree = ast.parse(source)

    # find Model class
    model_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            model_class = node
            break

    if model_class is None:
        return {"score": 0, "max": max_score, "details": ["Model class not found"]}

    # Model docstring (5 pts)
    model_doc = ast.get_docstring(model_class)
    if model_doc and len(model_doc) >= 20:
        score += 5
        details.append(f"Model docstring: {len(model_doc)} 字符 ≥ 20 OK (5/5)")
    elif model_doc:
        score += 3
        details.append(f"Model docstring: {len(model_doc)} 字符 < 20: 3/5")
    else:
        details.append("Model 类缺少 docstring: 0/5")

    # forward parameter annotations (5 pts)
    forward_method = None
    for node in ast.walk(model_class):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            forward_method = node
            break

    if forward_method:
        args = forward_method.args
        all_args = args.args + args.posonlyargs + args.kwonlyargs
        annotated = 0
        for arg in all_args:
            if arg.annotation is not None:
                annotated += 1
            # also check for inline comments (via line-level heuristics)
        if annotated >= 1:
            score += 5
            details.append(f"forward 参数: {annotated}/{len(all_args)} 有类型标注或注释 OK (5/5)")
        else:
            score += 2
            details.append(f"forward 参数无类型标注或注释: 2/5")
    else:
        details.append("forward 方法未找到: 0/5")

    # step-by-step comments for complex logic (5 pts)
    # heuristic: count #-style comments inside forward body
    if forward_method:
        comment_lines = 0
        lines = source.split("\n")
        start = forward_method.lineno
        end = forward_method.end_lineno
        for i in range(start, min(end, len(lines))):
            stripped = lines[i - 1].strip()
            if stripped.startswith("#") and len(stripped) > 3:
                comment_lines += 1
        if comment_lines >= 2:
            score += 5
            details.append(f"forward 内步骤注释: {comment_lines} 行 ≥ 2 OK (5/5)")
        elif comment_lines >= 1:
            score += 3
            details.append(f"forward 内步骤注释: {comment_lines} 行: 3/5")
        else:
            details.append("forward 内缺少步骤拆解注释: 0/5")
    else:
        score += 3  # no forward means simple op, not penalized

    return {"score": min(score, max_score), "max": max_score, "details": details}


def _check_coverage(py_path: str, json_path: str, op_class: str) -> dict:
    """Run full coverage analysis."""
    if not json_path:
        return {"status": "FAIL", "error": "JSON file not found", "coverage": {}}

    cases = _load_json_lines(json_path)
    if not cases:
        return {"status": "FAIL", "error": "JSON file is empty", "coverage": {}}

    dtype = _analyze_dtype_coverage(cases, op_class)
    shape = _analyze_shape_coverage(cases)
    attr = _analyze_attr_coverage(cases)
    boundary = _analyze_boundary_coverage(cases, op_class)
    annotation = _analyze_annotation_quality(py_path)

    total_score = dtype["score"] + shape["score"] + attr["score"] + boundary["score"] + annotation["score"]
    max_possible = dtype["max"] + shape["max"] + attr["max"] + boundary["max"] + annotation["max"]

    return {
        "score": total_score,
        "max": max_possible,
        "dtype": dtype,
        "shape": shape,
        "attr": attr,
        "boundary": boundary,
        "annotation": annotation,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate operator .py + .json pair")
    parser.add_argument("--py", required=True, help="Path to operator .py file")
    parser.add_argument("--mode", default="all",
                        choices=["static", "runtime", "coverage", "all"])
    parser.add_argument("--op-class", default=None,
                        choices=list(DEFAULT_DTYPES.keys()),
                        help="Operator class for coverage analysis")
    parser.add_argument("--json", default=None,
                        help="Path to .json file (auto-detect if omitted)")
    parser.add_argument("--output", default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    py_path = os.path.abspath(args.py)
    json_path = args.json or _find_json(py_path)
    op_class = args.op_class or _infer_op_class(py_path)

    if args.mode in ("coverage", "all") and not args.op_class:
        print(f"[INFO] Inferred operator class: {op_class}")

    result = {
        "py_path": py_path,
        "json_path": json_path,
        "op_class": op_class,
        "mode": args.mode,
    }

    if args.mode in ("static", "all"):
        static = _check_static(py_path, json_path)
        result["static_check"] = static
        if not static["passed"]:
            result["status"] = "FAIL"
            result["issues"] = static["issues"]
            print(json.dumps(result, indent=2, ensure_ascii=False))
            sys.exit(1)

    if args.mode in ("runtime", "all"):
        runtime = _check_runtime(py_path)
        result["runtime_check"] = runtime
        if not runtime["passed"]:
            result["status"] = "FAIL"
            print(json.dumps(result, indent=2, ensure_ascii=False))
            sys.exit(1)

    if args.mode in ("coverage", "all"):
        coverage = _check_coverage(py_path, json_path, op_class)
        result["coverage"] = coverage

        # determine status
        static_ok = result.get("static_check", {}).get("passed", True)
        runtime_ok = result.get("runtime_check", {}).get("passed", True)
        score = coverage.get("score", 0)

        if not static_ok or not runtime_ok:
            result["status"] = "FAIL"
        elif score >= 80:
            result["status"] = "PASS"
        elif score >= 60:
            result["status"] = "WARN"
        else:
            result["status"] = "FAIL"

        # collect suggestions
        suggestions = []
        for dim_name in ("dtype", "shape", "attr", "boundary", "annotation"):
            dim = coverage.get(dim_name, {})
            for d in dim.get("details", []):
                if "缺少" in d or "0/" in d or "仅" in d:
                    suggestions.append(f"[{dim_name}] {d}")
        result["suggestions"] = suggestions

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("status") == "FAIL":
        sys.exit(1)
    elif result.get("status") == "WARN":
        sys.exit(0)  # warn doesn't fail
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
