#!/usr/bin/env python3
"""Print the digest and line offsets for a reviewed alternative-list entry.

Usage: uv run python scripts/reviewed_alternative_list_digest.py SOURCE.txt

The digest covers the text exactly as the gates-collapsed check scans it. See
docs/reviewed-alternative-lists.md.
"""

from axiom_encode.harness.reviewed_alternative_lists import main

if __name__ == "__main__":
    raise SystemExit(main())
