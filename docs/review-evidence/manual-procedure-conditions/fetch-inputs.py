"""Fetch a bounded batch of immutable, authenticated corpus inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = Path(__file__).resolve().parent / "corpus-source-binding.json"
DESTINATION = ROOT / ".hub-scratch/round6-corpus/provisions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    options = parser.parse_args()
    inventory = json.loads(INVENTORY.read_text())
    entries = list(inventory["files"].items())
    started = time.monotonic()
    for index in range(options.start, min(options.stop, len(entries))):
        if time.monotonic() - started > 500:
            raise SystemExit("Batch time budget reached; resume at next printed index")
        name, entry = entries[index]
        target = options.output_dir / name
        if target.exists():
            content = target.read_bytes()
            if (
                len(content) == entry["bytes"]
                and hashlib.sha256(content).hexdigest() == entry["sha256"]
            ):
                print(f"VERIFIED {index} {name}", flush=True)
                continue
            raise SystemExit(f"Existing input differs: {name}")
        url = (
            f"https://raw.githubusercontent.com/{inventory['corpus_repository']}/"
            f"{inventory['git_commit']}/data/corpus/provisions/{name}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "axiom-corpus-evidence"}
            )
            with (
                urllib.request.urlopen(request, timeout=45) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    if time.monotonic() - started > 525:
                        raise TimeoutError("Batch time budget reached")
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Downloaded input differs: {name}")
            temporary.rename(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        print(f"DOWNLOADED {index} bytes={size} {name}", flush=True)
    print(f"DONE files={min(options.stop, len(entries)) - options.start}", flush=True)


if __name__ == "__main__":
    main()
