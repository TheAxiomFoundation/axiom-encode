#!/usr/bin/env python3
"""Partition every short Hebrew prefix-letter run using pinned primary Hspell.

Run under the repository's capped runner:
    python scripts/generate_hebrew_prefix_lexicon.py HSpell_ARCHIVE [--check]

The input is the unmodified ivrix/hspell archive at the pinned commit below.
Upstream's Perl inflection and prefix-mask generators supply lexical readings;
no Hspell C executable, system installation, or production grammar is used.
Hspell licenses its dictionary and generated word lists under AGPLv3; the
upstream notices accompany the derived lexical membership fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f4c5f2e760e28bfb9df06e625dae84c98d655e8f"
ARCHIVE_SHA256 = "3d3961fe2f7f12ccd7bf127e89b0467530cfbf38de61841d94ac90ec2f34a5c3"
ARCHIVE_URL = f"https://codeload.github.com/ivrix/hspell/tar.gz/{COMMIT}"
PREFIX_LETTERS = "והבכלמשד"
FIXTURE = ROOT / "tests/fixtures/hebrew_prefix_lexicon.json"
MODULE = ROOT / "src/axiom_encode/harness/_hebrew_prefix_words.py"
LICENSE_ROOT = ROOT / "tests/fixtures/licenses/hspell"
HIFS = (
    "milot.hif",
    "extrawords.hif",
    "biza-verbs.hif",
    "biza-nouns.hif",
    "nouns.hif",
    "verbs.hif",
    "shemp.hif",
)


def _run(source: Path, arguments: list[str], output: str) -> None:
    with (source / output).open("wb") as stream:
        subprocess.run(arguments, cwd=source, stdout=stream, check=True, timeout=120)


def build(source: Path) -> None:
    """Run the upstream Makefile's dictionary operations, with fat verbs enabled."""

    _run(source, ["perl", "-I", ".", "-w", "wolig.pl", "-d", "wolig.dat"], "nouns.hif")
    _run(source, ["perl", "-I", ".", "-w", "woo", "-d", "woo.dat"], "verbs.hif")
    _run(source, ["perl", "-I", ".", "-w", "wolig.pl", "-d", "shemp.dat"], "shemp.hif")
    _run(source, ["perl", "-I", ".", "genprefixes.pl"], "finite-prefixes.c")
    _run(
        source,
        ["perl", "-I", ".", "pmerge", "-p", "finite-word-masks.bin", *HIFS],
        "finite-words.txt",
    )


def derive(source: Path) -> dict:
    """Read upstream-generated words/masks and retain the finite candidate slice."""

    words = (source / "finite-words.txt").read_bytes().decode("iso8859-8").splitlines()
    masks = (source / "finite-word-masks.bin").read_bytes()
    if len(words) != len(masks):
        raise ValueError("Hspell word/mask counts differ")
    word_masks = dict(zip(words, masks, strict=True))
    prefixes_source = (source / "finite-prefixes.c").read_bytes().decode("iso8859-8")
    prefix_section = re.search(
        r"prefixes_noH\[\]=\{(.*?)0\};", prefixes_source, re.DOTALL
    )
    mask_section = re.search(r"masks_noH\[\]=\{(.*?)-1\};", prefixes_source, re.DOTALL)
    assert prefix_section and mask_section
    prefixes = re.findall(r'"([^\"]*)"', prefix_section.group(1))
    prefix_masks = [int(value) for value in re.findall(r"\d+", mask_section.group(1))]
    if len(prefixes) != len(prefix_masks):
        raise ValueError("Hspell prefix/mask counts differ")
    candidates = sorted(
        "".join(letters)
        for length in range(1, 4)
        for letters in itertools.product(PREFIX_LETTERS, repeat=length)
    )
    candidate_set = set(candidates)
    analyses: dict[str, set[tuple[str, str, bool]]] = defaultdict(set)
    acronym_analyses: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for name in HIFS:
        for line in (source / name).read_bytes().decode("iso8859-8").splitlines():
            if not line or line.startswith("#"):
                continue
            word = line.split()[0]
            if word[0] in "LB":
                word = word[1:]
            word = word.removesuffix("-")
            unquoted = word.translate(str.maketrans("", "", "\"'״׳"))
            if unquoted != word and unquoted in candidate_set and len(unquoted) > 1:
                acronym_analyses[unquoted].add((name, line))
            if word in candidate_set:
                analyses[word].add((name, line, "פרטי" in line))
    rows = []
    for word in candidates:
        readings = []
        for prefix, prefix_mask in zip(prefixes, prefix_masks, strict=True):
            if not word.startswith(prefix):
                continue
            stem = word[len(prefix) :]
            common_mask = prefix_mask & word_masks.get(stem, 0)
            if not common_mask:
                continue
            descriptions = sorted(analyses[stem])
            readings.append(
                {
                    "prefix": prefix,
                    "stem": stem,
                    "compatible_mask": common_mask,
                    "analyses": [
                        {"hif": name, "description": line, "proper_name": proper}
                        for name, line, proper in descriptions
                    ],
                }
            )
        ordinary = any(
            not analysis["proper_name"]
            for reading in readings
            for analysis in reading["analyses"]
        )
        category = (
            "single_letter_prefix"
            if len(word) == 1
            else "lexical_word"
            if ordinary
            else "proper_name"
            if readings
            else "acronym"
            if acronym_analyses[word]
            else "nonlexical_prefix_chain"
        )
        rows.append(
            {
                "token": word,
                "category": category,
                "readings": readings,
                "quoted_acronym_analyses": [
                    {"hif": name, "description": description}
                    for name, description in sorted(acronym_analyses[word])
                ],
            }
        )
    return {
        "source": {
            "project": "Hspell 1.4",
            "repository": "https://github.com/ivrix/hspell",
            "official_downloads": "https://hspell.sourceforge.net/download.html",
            "commit": COMMIT,
            "archive_url": ARCHIVE_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_bytes": 570733,
            "copyright": "Copyright (C) 2000-2017 Nadav Har'El and Dan Kenigsberg",
            "license": "AGPL-3.0-only",
            "license_sha256": hashlib.sha256(
                (source / "LICENSE").read_bytes()
            ).hexdigest(),
            "copying_sha256": hashlib.sha256(
                (source / "COPYING").read_bytes()
            ).hexdigest(),
        },
        "generator": "scripts/generate_hebrew_prefix_lexicon.py",
        "build": "Upstream Perl inflection generators, fat-verb forms; pmerge; genprefixes without interrogative He.",
        "dictionary_words": len(words),
        "prefix_mask_pairs": len(prefixes),
        "prefix_masks": dict(zip(prefixes, prefix_masks, strict=True)),
        "candidate_stem_masks": {word: word_masks.get(word, 0) for word in candidates},
        "alphabet": PREFIX_LETTERS,
        "candidate_count": len(candidates),
        "candidates": rows,
    }


def render_module(fixture: dict, source: Path) -> str:
    words = [
        row["token"]
        for row in fixture["candidates"]
        if row["readings"] and len(row["token"]) > 1
    ]
    chunks = [" ".join(words[index : index + 12]) for index in range(0, len(words), 12)]
    return (
        '"""Generated Hspell lexical membership; do not edit by hand.\n\n'
        "Source: ivrix/hspell, copyright 2000-2017 Nadav Har'El and Dan Kenigsberg.\n"
        "Upstream dictionary license: AGPL-3.0-only; the original notice and full\n"
        "license text are reproduced as comments below and ship with this module.\n"
        "See the 584-row primary-lexicon fixture\n"
        "tests/fixtures/hebrew_prefix_lexicon.json and its generator.\n"
        '"""\n\n'
        f'HSPELL_COMMIT = "{COMMIT}"\n'
        "HSPELL_ARCHIVE_SHA256 = (\n"
        f'    "{ARCHIVE_SHA256}"\n'
        ")\n"
        "HSPELL_PREFIX_WORDS = frozenset(\n"
        + "".join(f'    "{chunk} "\n' for chunk in chunks[:-1])
        + f'    "{chunks[-1]}".split()\n)\n'
        + "".join(
            f"\n# HSPELL-UPSTREAM-{name}-BEGIN\n"
            + "".join(
                f"# {line}\n" if line else "#\n"
                for line in (source / name).read_text().split("\n")[:-1]
            )
            + f"# HSPELL-UPSTREAM-{name}-END\n"
            for name in ("LICENSE", "COPYING")
        )
    )


def _write_or_check(path: Path, data: bytes, check: bool) -> None:
    if check:
        if path.read_bytes() != data:
            raise SystemExit(f"Generated Hspell artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("Unexpected Hspell archive SHA256")
    scratch = ROOT / ".review-scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hspell-prefix-", dir=scratch) as directory:
        with tarfile.open(args.archive) as archive:
            archive.extractall(directory, filter="data")
        source = Path(directory) / f"hspell-{COMMIT}"
        build(source)
        fixture = derive(source)
        _write_or_check(
            FIXTURE,
            (json.dumps(fixture, ensure_ascii=False, indent=2) + "\n").encode(),
            args.check,
        )
        _write_or_check(MODULE, render_module(fixture, source).encode(), args.check)
        for name in ("LICENSE", "COPYING"):
            _write_or_check(
                LICENSE_ROOT / name, (source / name).read_bytes(), args.check
            )


if __name__ == "__main__":
    main()
