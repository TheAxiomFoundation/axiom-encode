#!/usr/bin/env python3
"""Check a pull request body's Encoding-Target and Fixes-Run lines.

A local entry point for ``axiom_encode.pr_linkage``; CI runs that module with
``python3 -I`` so it needs no installed environment. Examples::

    uv run python scripts/check_pr_linkage.py --body-file body.md
    GITHUB_TOKEN=$(gh auth token) uv run python scripts/check_pr_linkage.py \
        --body-file body.md --verify-runs
"""

from __future__ import annotations

from axiom_encode.pr_linkage import main

if __name__ == "__main__":
    raise SystemExit(main())
