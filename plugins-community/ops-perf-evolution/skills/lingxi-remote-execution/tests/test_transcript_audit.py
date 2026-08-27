#!/usr/bin/env python3
"""Remote execution must satisfy the existing anti-fake evaluation audit."""

import argparse
import importlib.util
import json
from pathlib import Path


AUDIT = (Path(__file__).resolve().parents[2] / "evolution-world-model" /
         "scripts" / "transcript_audit.py")
spec = importlib.util.spec_from_file_location("transcript_audit_remote_test", AUDIT)
transcript_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcript_audit)


def _write_transcript(path, command):
    row = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": command}},
    ]}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_remote_job_submit_satisfies_lingxi_s1(tmp_path):
    transcript = tmp_path / "remote.jsonl"
    _write_transcript(
        transcript,
        "python3 plugins-community/ops-perf-evolution/skills/lingxi-remote-execution/"
        "scripts/remote_exec.py --host npu-a job submit --job-id j --session s --wait")
    args = argparse.Namespace(partial_type="lingxi-partial", transcript=str(transcript))
    assert transcript_audit._check_s1_invocation(args) == []


def test_passive_remote_exec_reference_does_not_satisfy_s1(tmp_path):
    transcript = tmp_path / "passive.jsonl"
    _write_transcript(transcript, "cat scripts/remote_exec.py; echo job submit")
    args = argparse.Namespace(partial_type="lingxi-partial", transcript=str(transcript))
    issues = transcript_audit._check_s1_invocation(args)
    assert issues and issues[0].startswith("S1:")


def test_remote_help_then_echo_is_not_mistaken_for_submit(tmp_path):
    transcript = tmp_path / "split-command.jsonl"
    _write_transcript(transcript, "python3 scripts/remote_exec.py --help; echo job submit")
    args = argparse.Namespace(partial_type="lingxi-partial", transcript=str(transcript))
    assert transcript_audit._check_s1_invocation(args)
