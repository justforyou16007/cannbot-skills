#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Pure helpers shared by single-task multi-NPU verification and its tests."""

import re


def parse_index_list(raw, label):
    """Parse an ordered, duplicate-free comma-separated integer list."""
    if raw is None or not str(raw).strip():
        return None
    result = []
    seen = set()
    for token in str(raw).split(','):
        token = token.strip()
        if not token:
            raise ValueError(f"{label} contains an empty value")
        value = int(token)
        if value < 0:
            raise ValueError(f"{label} must contain non-negative integers")
        if value in seen:
            raise ValueError(f"{label} contains duplicate value: {value}")
        seen.add(value)
        result.append(value)
    return result


def distribute_case_indices_round_robin(case_indices, npu_list):
    """Return one serial case shard per NPU, omitting idle devices."""
    if not npu_list:
        raise ValueError("npu_list must not be empty")
    shards = {npu_id: [] for npu_id in npu_list}
    for offset, case_index in enumerate(case_indices):
        shards[npu_list[offset % len(npu_list)]].append(case_index)
    return {npu_id: indices for npu_id, indices in shards.items() if indices}


def _ordered_input_summaries(reports):
    """Restore flattened input summary lines to global case order."""
    rows = []
    fallback_order = 0
    for report in reports:
        for line in report.get('inputs', []):
            match = re.match(r"inputs\[(\d+)\]", line)
            case_index = int(match.group(1)) if match else float('inf')
            rows.append((case_index, fallback_order, line))
            fallback_order += 1
    rows.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in rows]


def merge_case_shard_reports(op, reports, npu_list, expected_case_indices,
                             default_report):
    """Merge device-shard reports and validate exact case coverage."""
    base = next((report for report in reports if report.get('task_dir')), None)
    merged = dict(base or default_report)
    case_rows = []
    errors = []

    for report in reports:
        indices = report.get('case_indices') or report.get(
            '_assigned_case_indices', [])
        comparisons = report.get('comparisons', [])
        case_oks = report.get('case_oks', [])
        if report.get('error'):
            errors.append(f"NPU {report.get('_npu')}: {report['error']}")
        for offset, case_index in enumerate(indices):
            comparison = (
                comparisons[offset] if offset < len(comparisons)
                else f"case[{case_index}]: missing shard result"
            )
            ok = case_oks[offset] if offset < len(case_oks) else False
            case_rows.append((case_index, comparison, ok, report.get('_npu')))

    case_rows.sort(key=lambda row: row[0])
    actual_indices = [row[0] for row in case_rows]
    expected_indices = sorted(expected_case_indices)
    merged['op'] = str(op)
    merged['device'] = f"npu:{npu_list}"
    merged['device_ids'] = list(npu_list)
    merged['case_indices'] = actual_indices
    merged['case_devices'] = {str(row[0]): row[3] for row in case_rows}
    merged['n_cases_total'] = len(expected_case_indices)
    merged['inputs'] = _ordered_input_summaries(reports)
    merged['comparisons'] = [row[1] for row in case_rows]
    merged['comparison'] = '\n'.join(merged['comparisons'])
    merged['case_oks'] = [row[2] for row in case_rows]
    merged['error'] = '; '.join(errors)
    merged['ok'] = (
        actual_indices == expected_indices
        and not errors
        and all(merged['case_oks'])
    )
    return merged
