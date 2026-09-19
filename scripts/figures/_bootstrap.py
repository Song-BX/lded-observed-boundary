"""Import-path bootstrap for direct execution of figure scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_pipeline_path() -> None:
    scripts_dir = Path(__file__).resolve().parents[1]
    scripts_dir_text = str(scripts_dir)
    if scripts_dir_text not in sys.path:
        sys.path.insert(0, scripts_dir_text)
