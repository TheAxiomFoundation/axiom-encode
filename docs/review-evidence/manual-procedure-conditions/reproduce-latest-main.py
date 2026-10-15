"""Compatibility entry point for the pinned-main corpus comparison.

Use the same --corpus-root and --output arguments as reproduce-corpus.py.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("reproduce-corpus.py")), run_name="__main__"
    )
