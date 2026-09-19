# -*- coding: utf-8 -*-
"""pytest 根路径注入：保证 `python -m pytest` 从仓库任意目录运行都能 import src.*。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
