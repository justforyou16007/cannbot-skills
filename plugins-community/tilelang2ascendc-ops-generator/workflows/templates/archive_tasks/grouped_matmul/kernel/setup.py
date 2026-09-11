#!/usr/bin/env python3
"""Build grouped_matmul AscendC kernel。"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _kernel_setup_common import kernel_setup

kernel_setup("grouped_matmul", kernel_dir=__file__)
