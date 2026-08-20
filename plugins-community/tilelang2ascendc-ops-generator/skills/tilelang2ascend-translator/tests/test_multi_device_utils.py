#!/usr/bin/env python3
"""CPU-only tests for single-task multi-NPU verification helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "multi_device_utils.py"
SPEC = importlib.util.spec_from_file_location("test_multi_device_utils_impl", SCRIPT)
UTILS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UTILS
SPEC.loader.exec_module(UTILS)


class MultiDeviceUtilsTest(unittest.TestCase):
    def test_parse_index_list(self):
        self.assertEqual(UTILS.parse_index_list("3, 0,5", "--npu-list"), [3, 0, 5])
        for raw in ("1,1", "1,,2", "-1"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                UTILS.parse_index_list(raw, "--npu-list")

    def test_round_robin_and_idle_device_omission(self):
        self.assertEqual(
            UTILS.distribute_case_indices_round_robin([0, 1, 2, 3, 4], [1, 3]),
            {1: [0, 2, 4], 3: [1, 3]},
        )
        self.assertEqual(
            UTILS.distribute_case_indices_round_robin([7], [1, 3, 5]),
            {1: [7]},
        )

    def test_empty_device_list_is_rejected(self):
        with self.assertRaises(ValueError):
            UTILS.distribute_case_indices_round_robin([0], [])

    def test_reports_merge_in_case_order_and_validate_coverage(self):
        reports = [
            {
                "task_dir": "/task", "_npu": 0,
                "case_indices": [0, 2], "case_oks": [True, True],
                "comparisons": ["case[0]: ok", "case[2]: ok"],
                "inputs": ["inputs[0]: a", "inputs[2]: c"],
            },
            {
                "task_dir": "/task", "_npu": 1,
                "case_indices": [1, 3], "case_oks": [True, True],
                "comparisons": ["case[1]: ok", "case[3]: ok"],
                "inputs": ["inputs[1]: b", "inputs[3]: d"],
            },
        ]
        merged = UTILS.merge_case_shard_reports(
            "/task", list(reversed(reports)), [0, 1], [0, 1, 2, 3],
            {"task_dir": "", "error": ""},
        )
        self.assertTrue(merged["ok"])
        self.assertEqual(merged["case_indices"], [0, 1, 2, 3])
        self.assertEqual(
            merged["inputs"],
            ["inputs[0]: a", "inputs[1]: b", "inputs[2]: c", "inputs[3]: d"],
        )
        self.assertEqual(merged["case_devices"], {"0": 0, "1": 1, "2": 0, "3": 1})

    def test_missing_case_fails_merged_report(self):
        merged = UTILS.merge_case_shard_reports(
            "/task",
            [{
                "_npu": 0, "case_indices": [0], "case_oks": [True],
                "comparisons": ["case[0]: ok"],
            }],
            [0, 1], [0, 1], {"task_dir": "", "error": ""},
        )
        self.assertFalse(merged["ok"])


if __name__ == "__main__":
    unittest.main()
