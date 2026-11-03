"""Measure what one cached corpus bucket keeps resident, against its charge.

Run from the repository root:

    uv run python scripts/measure_corpus_cache_memory.py

For synthetic provisions buckets of short, medium and long row bodies, this
binds a test-signed release, resolves one citation (which loads the bucket and
builds its citation index), and reports with ``tracemalloc``:

- the provisions file's bytes;
- the memory the cache retains after the call, and its ratio to file bytes;
- the peak during the load, as a ratio to file bytes;
- the resident charge the cache's row budget books for the bucket
  (``_RESIDENT_BYTES_PER_ARTIFACT_BYTE`` times file bytes).

The signing key is the fixed test key from ``tests/release_object_fixtures.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import tracemalloc
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from axiom_encode import corpus_resolver  # noqa: E402
from tests.release_object_fixtures import bind_test_corpus_release  # noqa: E402

VERSION = "2026-01-01-statute"
RELEASE = "memory-measurement"
CASES = (("short", 40), ("medium", 200), ("long", 1_200))


def _write_bucket(root: Path, rows: int, body_chars: int) -> Path:
    path = root / "data/corpus/provisions/us/statute" / f"{VERSION}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    filler = "x" * body_chars
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "id": f"row-{index}",
                        "citation_path": f"us/statute/1/{index}",
                        "body": f"({index}) {filler}"[:body_chars],
                        "heading": f"Section {index}",
                        "jurisdiction": "us",
                        "document_class": "statute",
                        "version": VERSION,
                        "source_path": f"sources/us/statute/{VERSION}/x.xml",
                        "source_as_of": "2026-01-02",
                        "expression_date": "2026-01-01",
                        "metadata": {"source_history": [f"Amended {index}."]},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def measure(rows: int, body_chars: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provisions = _write_bucket(root, rows, body_chars)
        release = bind_test_corpus_release(root, RELEASE, [("us", "statute", VERSION)])
        file_bytes = provisions.stat().st_size
        gc.collect()
        tracemalloc.start()
        try:
            corpus_resolver.resolve_local_corpus_source("us/statute/1/0", release)
            gc.collect()
            retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        bucket = release._resolution_cache.buckets[("us", "statute")]
        return {
            "file_mib": file_bytes / 2**20,
            "retained_mib": retained / 2**20,
            "retained_ratio": retained / file_bytes,
            "peak_ratio": peak / file_bytes,
            "charged_mib": bucket.row_bytes / 2**20,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=20_000)
    args = parser.parse_args()
    print(
        f"| Rows | Body chars | File MiB | Retained MiB | Retained / file | "
        f"Peak / file | Charged MiB ({corpus_resolver._RESIDENT_BYTES_PER_ARTIFACT_BYTE}x) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for _name, body_chars in CASES:
        result = measure(args.rows, body_chars)
        print(
            f"| {args.rows:,} | {body_chars:,} | {result['file_mib']:.1f} | "
            f"{result['retained_mib']:.1f} | {result['retained_ratio']:.2f}x | "
            f"{result['peak_ratio']:.2f}x | {result['charged_mib']:.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
