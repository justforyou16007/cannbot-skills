#!/usr/bin/env python3
"""Build matmul_gen AscendC kernel（deliverer Phase 5 工程化产物）。"""
import sys
from pathlib import Path

_COMMON_DIR = Path(__file__).resolve().parent.parent.parent  # archive_tasks/
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from _kernel_setup_common import kernel_setup

kernel_setup("matmul_gen", kernel_dir=__file__)
