"""The fixture differential reports every changed verdict and nothing else."""

from __future__ import annotations

import copy
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "source_structure_fixture_differential.py"
)
SPEC = spec_from_file_location("_source_structure_fixture_differential", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIFFERENTIAL = module_from_spec(SPEC)
sys.modules[SPEC.name] = DIFFERENTIAL
SPEC.loader.exec_module(DIFFERENTIAL)

SOURCE = (
    "(1) A refund is allowed if the claimant is a resident and the claimant "
    "files a return.\n(2) The refund is $300 for an income less than $25,000.\n"
)


def _collect(tmp_path: Path, name: str) -> Path:
    root = tmp_path / "root"
    (root / "cases" / "one").mkdir(parents=True, exist_ok=True)
    (root / "cases" / "one" / "source.txt").write_text(SOURCE)
    output = tmp_path / name
    assert (
        DIFFERENTIAL.main(
            [
                "collect",
                "--root",
                str(root),
                "--glob",
                "cases/**/*.txt",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    return output


def test_identical_collections_report_no_change(tmp_path, capsys):
    before = _collect(tmp_path, "before.json")
    after = _collect(tmp_path, "after.json")
    assert (
        DIFFERENTIAL.main(["compare", "--before", str(before), "--after", str(after)])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["changes"] == []


def test_every_changed_verdict_is_listed(tmp_path, capsys):
    after = _collect(tmp_path, "after.json")
    collected = json.loads(after.read_text())
    (key,) = collected["texts"]
    assert collected["texts"][key]["branches"]
    before_value = copy.deepcopy(collected)
    removed = before_value["texts"][key]["branches"].pop()
    before_value["texts"][key]["boundaries"].append([["9"], 0, 1.0, "1", 0, 1])
    before = tmp_path / "before.json"
    before.write_text(json.dumps(before_value))
    assert (
        DIFFERENTIAL.main(["compare", "--before", str(before), "--after", str(after)])
        == 1
    )
    changes = json.loads(capsys.readouterr().out)["changes"]
    assert {(change["input"], change["verdict"]) for change in changes} == {
        (key, "branches"),
        (key, "boundaries"),
    }
    branch_change = next(c for c in changes if c["verdict"] == "branches")
    assert branch_change["added"] == [removed]
    assert branch_change["removed"] == []


def test_counts_bound_the_work_and_every_skip_is_recorded(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "long.txt").write_text("x" * 50)
    output = tmp_path / "out.json"
    DIFFERENTIAL.main(
        [
            "collect",
            "--root",
            str(root),
            "--glob",
            "*.txt",
            "--output",
            str(output),
            "--max-chars",
            "10",
        ]
    )
    assert json.loads(output.read_text())["skipped"] == {"long.txt": "50 characters"}


def test_corpus_rows_are_inputs_with_their_own_citations(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    rows = [
        {"citation_path": "us-co/statute/39/39-22-2003", "body": SOURCE},
        {"citation_path": "us-co/statute/39/39-22-2004", "body": " "},
    ]
    (root / "title-39.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    output = tmp_path / "rows.json"
    DIFFERENTIAL.main(
        [
            "collect",
            "--root",
            str(root),
            "--jsonl-glob",
            "*.jsonl",
            "--output",
            str(output),
        ]
    )
    collected = json.loads(output.read_text())
    assert set(collected["texts"]) == {"title-39.jsonl:1"}
    assert (
        collected["texts"]["title-39.jsonl:1"]["citation"]
        == "us-co/statute/39/39-22-2003"
    )
    assert collected["skipped"] == {"title-39.jsonl:2": "no body"}
