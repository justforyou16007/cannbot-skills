#!/usr/bin/env python3
"""CPU-only tests for msprof case sharding."""

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "msprof_perf_summary.py"
SPEC = importlib.util.spec_from_file_location("test_msprof_perf_summary", SCRIPT)
PROFILER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILER
SPEC.loader.exec_module(PROFILER)


class DeviceListTest(unittest.TestCase):
    def test_parse_device_list(self):
        self.assertEqual(PROFILER._parse_device_ids("0, 2,5"), [0, 2, 5])
        for raw in ("1,1", "1,,2", "-1", "bad"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                PROFILER._parse_device_ids(raw)


class ParallelCaseLoopTest(unittest.TestCase):
    def test_parallel_shards_merge_in_global_case_order(self):
        barrier = threading.Barrier(2)
        args = SimpleNamespace(cases=None)

        def fake_loop(_out, _cases, _count, _args, device_id,
                      case_indices=None):
            barrier.wait(timeout=2)
            rows = [
                {
                    "case": index,
                    "device_id": device_id,
                    "ref_us": float(index + 2),
                    "asc_us": float(index + 1),
                    "speedup": float(index + 2) / float(index + 1),
                }
                for index in reversed(case_indices)
            ]
            return rows, [], [], []

        rows, speedups, refs, asc, assignment, selected = (
            PROFILER._run_loop_multi_device(
                fake_loop, Path("unused"), [{}] * 5, 5, args, [2, 6]
            )
        )

        self.assertEqual([row["case"] for row in rows], [0, 1, 2, 3, 4])
        self.assertEqual(assignment, {"0": 2, "1": 6, "2": 2, "3": 6, "4": 2})
        self.assertEqual(selected, 5)
        self.assertEqual(len(speedups), 5)
        self.assertEqual(len(refs), 5)
        self.assertEqual(len(asc), 5)

    def test_explicit_case_subset_is_preserved(self):
        args = SimpleNamespace(cases="4,1")
        self.assertEqual(PROFILER._selected_case_indices(6, args), [4, 1])


if __name__ == "__main__":
    unittest.main()
