#!/usr/bin/env python3
"""CPU-only tests for ops multi-device sharding and session leases."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load_module("test_evaluate_ops_direct", "evaluate_ops_direct.py")
lease_mod = _load_module("test_device_lease", "device_lease.py")


class DeviceListTest(unittest.TestCase):
    def test_parse_and_reject_invalid_lists(self):
        self.assertEqual(evaluate.parse_device_ids("3, 1,7"), (3, 1, 7))
        self.assertIsNone(evaluate.parse_device_ids(None))
        for raw in ("0,0", "0,,1", "-1", "x"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                evaluate.parse_device_ids(raw)

    def test_round_robin_omits_idle_devices(self):
        items = [{"name": name} for name in "abc"]
        shards = evaluate.build_device_shards(items, (2, 4, 6, 8))
        self.assertEqual(
            [(device, [item["name"] for item in shard]) for device, shard in shards],
            [(2, ["a"]), (4, ["b"]), (6, ["c"])],
        )


class ParallelEvaluationTest(unittest.TestCase):
    def test_results_are_mapped_and_restored_to_input_order(self):
        shape_plan = [
            {"name": name, "group": "target"} for name in "abcde"
        ]
        cfg = evaluate.DirectEvalConfig(
            op_name="demo",
            call_spec_path="unused.json",
            baseline_path="baseline",
            evolved_path="evolved",
            device_ids=(1, 3),
        )

        def fake_shard(_cfg, _spec, device_id, shard, *_paths):
            baseline = [
                {
                    "name": item["name"], "group": item["group"],
                    "device_id": device_id, "time_us": 10.0,
                }
                for item in reversed(shard)
            ]
            evolved = [
                {
                    "name": item["name"], "group": item["group"],
                    "device_id": device_id, "time_us": 5.0,
                }
                for item in reversed(shard)
            ]
            return baseline, evolved

        with mock.patch.object(evaluate, "_run_device_shard", side_effect=fake_shard):
            baseline, evolved = evaluate._run_parallel_device_shards(
                cfg, {}, shape_plan, "baseline_profile", "evolved_profile"
            )

        self.assertEqual([row["name"] for row in baseline], list("abcde"))
        self.assertEqual([row["name"] for row in evolved], list("abcde"))
        self.assertEqual(
            {row["name"]: row["device_id"] for row in baseline},
            {"a": 1, "b": 3, "c": 1, "d": 3, "e": 1},
        )

    def test_cache_is_rejected_when_device_mapping_changes(self):
        shape_plan = [
            {"name": "a", "group": "target"},
            {"name": "b", "group": "target"},
        ]
        cache = {
            "shape_results": {
                "target": [
                    {"name": "a", "baseline_time_us": 10, "device_id": 0},
                    {"name": "b", "baseline_time_us": 20, "device_id": 1},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps(cache), encoding="utf-8")
            self.assertIsNone(
                evaluate._load_baseline_shape_cache(
                    str(path), shape_plan, {"a": 1, "b": 0}
                )
            )


class MultiDeviceLeaseTest(unittest.TestCase):
    def test_device_groups_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            lease_mod, "_get_device_memory_usage", return_value=0.0
        ):
            first = lease_mod.SessionDeviceLease(tmp)
            second = lease_mod.SessionDeviceLease(tmp)
            third = lease_mod.SessionDeviceLease(tmp)
            try:
                self.assertEqual(
                    first.acquire_session_devices(
                        "op-a", "session-a", 2, [0, 1, 2], timeout=0
                    ),
                    [0, 1],
                )
                self.assertEqual(
                    second.acquire_session_devices(
                        "op-b", "session-b", 1, [0, 1, 2], timeout=0
                    ),
                    [2],
                )
                with self.assertRaises(TimeoutError):
                    third.acquire_session_devices(
                        "op-c", "session-c", 1, [0, 1, 2], timeout=0
                    )
            finally:
                first.release_session()
                second.release_session()


if __name__ == "__main__":
    unittest.main()
