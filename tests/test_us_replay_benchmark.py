"""Offline US replay benchmark: committed fixtures and the replay-benchmark gate.

``tests/fixtures/us_replay_benchmark`` holds real candidates from
``targeted-reencode-failure`` artifacts (visible, public-source cases only).
``scripts/replay_us_failure_bundles.py`` replays ``analyze_complete_source_unit``
over them, and ``snapshot.json`` pins every fixture's exact findings.

The full replay runs in the ``replay benchmark`` CI job (``ci`` command), not
here: these tests check the snapshot's shape and the gate's logic on small
synthetic reports. Rewrite the snapshot with::

    python scripts/replay_us_failure_bundles.py snapshot --base-ref origin/main

See ``docs/us-replay-benchmark.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.replay_us_failure_bundles import (
    DEFAULT_FIXTURE_ROOT,
    FAILURE_CLASSES,
    FIXTURE_FILE,
    HELD_OUT,
    REPLAY_TRIGGER_PATHS,
    SNAPSHOT_FILE,
    SNAPSHOT_REPO_PATH,
    SNAPSHOT_SCHEMA,
    VISIBLE,
    ReplayCase,
    UnsafeBundlePath,
    _context_key,
    _read_inside,
    analyzer_module_paths,
    build_report,
    compare_reports,
    discover_bundle_modules,
    extract_fixture,
    fixture_directories,
    format_comparison,
    format_summary,
    gate_markdown,
    guard_violations,
    import_symbol_contents,
    load_bundle_cases,
    load_fixture,
    load_split_map,
    main,
    read_snapshot_at,
    recorded_issue_rule,
    replay_is_needed,
    run_gate,
    snapshot_drift,
    snapshot_from_report,
    unneeded_allowances,
)

FIXTURES = DEFAULT_FIXTURE_ROOT
SNAPSHOT = FIXTURES / SNAPSHOT_FILE
COMPLETENESS_RULES = {
    "complete-source-unit:deferral",
    "complete-source-unit:formula-output",
    "complete-source-unit:numeric-recall",
    "complete-source-unit:source-explicit-conditions",
    "complete-source-unit:structure",
    "complete-source-unit:tests",
}
RECORDED_FAILURE_CLASSES = COMPLETENESS_RULES | {
    "compile",
    "oracle",
    "proof:excerpt-not-found",
    "tests:wrong-expected-value",
}
MAX_FIXTURE_BYTES = 5 * 1024 * 1024
LABEL_KEYS = ("expected_class", "expected_subclass", "class_evidence", "class_source")
# Private corpus: never committed (see docs/us-replay-benchmark.md).
PRIVATE_CITATION_PREFIXES = ("us/regulation/12/1005", "us/regulation/12/1006")


def test_snapshot_pins_exactly_the_committed_fixtures():
    snapshot = json.loads(SNAPSHOT.read_text())

    assert snapshot["schema"] == SNAPSHOT_SCHEMA
    assert snapshot["context"] == "workspace"
    assert sorted(snapshot["cases"]) == [
        directory.name for directory in fixture_directories(FIXTURES)
    ]
    for case_id, row in snapshot["cases"].items():
        metadata = json.loads((FIXTURES / case_id / FIXTURE_FILE).read_text())
        assert row["citation"] == metadata["citation"], case_id
        assert row["control"] == (
            metadata.get("recorded_verdict", "rejected") != "rejected"
        ), case_id
        assert row["error"] is None, case_id


def test_snapshot_totals_match_cases():
    snapshot = json.loads(SNAPSHOT.read_text())
    cases = snapshot["cases"].values()

    assert snapshot["totals"]["cases"] == len(snapshot["cases"])
    assert snapshot["totals"]["issues"] == sum(case["issues"] for case in cases)
    assert snapshot["totals"]["clean"] == sum(
        1 for case in cases if not case["issues"] and not case["error"]
    )
    rule_totals: dict[str, int] = {}
    for case in cases:
        assert case["issues"] == sum(case["rules"].values()) == len(case["findings"])
        assert case["findings"] == sorted(case["findings"])
        assert {
            rule: sum(1 for finding in case["findings"] if f"[{rule}]" in finding)
            for rule in case["rules"]
        } == case["rules"]
        for rule, count in case["rules"].items():
            rule_totals[rule] = rule_totals.get(rule, 0) + count
    assert snapshot["totals"]["rules"] == rule_totals
    assert all(
        isinstance(reason, str) and reason.strip()
        for reason in snapshot["allowed_regressions"].values()
    )


def test_fixture_set_covers_failing_rule_classes_and_controls():
    snapshot = json.loads(SNAPSHOT.read_text())
    replayed = {rule for case in snapshot["cases"].values() for rule in case["rules"]}
    recorded = {
        rule
        for directory in fixture_directories(FIXTURES)
        for rule in json.loads((directory / FIXTURE_FILE).read_text())["recorded_rules"]
    }
    citations = {case["citation"] for case in snapshot["cases"].values()}

    assert COMPLETENESS_RULES <= replayed
    assert RECORDED_FAILURE_CLASSES <= recorded
    assert len(citations) >= 20
    assert any(not case["issues"] for case in snapshot["cases"].values())
    assert any(case["control"] for case in snapshot["cases"].values())


def test_committed_fixtures_are_visible_public_and_unlabeled():
    # Held-out cases, hand labels and private-corpus sources stay outside git.
    for directory in fixture_directories(FIXTURES):
        metadata = json.loads((directory / FIXTURE_FILE).read_text())
        assert metadata.get("split", VISIBLE) == VISIBLE, directory.name
        assert not set(LABEL_KEYS) & set(metadata), directory.name
        assert not metadata["citation"].startswith(PRIVATE_CITATION_PREFIXES), (
            directory.name
        )
        private_paths = ("12-cfr/1005", "12-cfr/1006", "regulations/12/1005")
        for key, relative in metadata["context_imports"].items():
            assert not any(part in key for part in private_paths), directory.name
            assert not any(part in relative for part in private_paths), directory.name


def test_fixtures_carry_provenance_and_stay_small():
    total = 0
    for directory in fixture_directories(FIXTURES):
        case = load_fixture(directory)
        provenance = case.provenance
        control = provenance.get("recorded_verdict", "rejected") != "rejected"

        assert re.fullmatch(r"[0-9]+", provenance["run_id"])
        assert directory.name.split("-")[0] == provenance["run_id"]
        assert provenance["run_url"].endswith(f"/actions/runs/{provenance['run_id']}")
        assert re.fullmatch(r"[0-9a-f]{40}", provenance["encoder_commit"])
        if not control:
            # A control's recorded issues are guard failures, not its candidate's.
            assert provenance["candidate_matches_bundle_inventory"] is True
            assert provenance["candidate_matches_recorded_issues"] is True
        assert provenance["source_matches_attestation"] is True
        assert set(provenance["context_origins"]) == set(provenance["context_imports"])
        assert set(provenance["context_origins"].values()) <= {
            "workspace",
            "rulespec_ref",
        }
        providers = {
            relative
            for key, relative in provenance["context_imports"].items()
            if provenance["context_origins"][key] == "rulespec_ref"
        }
        assert set(provenance["provider_git_blob_sha1"]) == providers
        if providers:
            assert re.fullmatch(r"[0-9a-f]{40}", provenance["rulespec_ref"])
        assert {"candidate.yaml", "source.txt", "issues.json"} <= set(
            provenance["files"]
        )
        assert case.citation in case.candidate
        total += sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file()
        )

    assert total <= MAX_FIXTURE_BYTES


def _copy_fixture(source: Path, tmp_path: Path) -> Path:
    copy = tmp_path / source.name
    for path in source.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return copy


def _small_fixture() -> Path:
    return min(
        fixture_directories(FIXTURES),
        key=lambda d: sum(p.stat().st_size for p in d.rglob("*") if p.is_file()),
    )


def _rewrite_metadata(copy: Path, **changes) -> None:
    metadata = json.loads((copy / FIXTURE_FILE).read_text())
    for key, value in changes.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    (copy / FIXTURE_FILE).write_text(json.dumps(metadata, indent=2) + "\n")


def test_tampered_fixture_bytes_are_rejected(tmp_path):
    copy = _copy_fixture(_small_fixture(), tmp_path)
    (copy / "candidate.yaml").write_text(
        (copy / "candidate.yaml").read_text() + "\n# edited\n"
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_fixture(copy)


def test_unlisted_fixture_files_are_rejected(tmp_path):
    copy = _copy_fixture(_small_fixture(), tmp_path)
    (copy / "context").mkdir(exist_ok=True)
    (copy / "context" / "extra.yaml").write_text("format: rulespec/v1\n")

    with pytest.raises(ValueError, match="files on disk differ"):
        load_fixture(copy)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"split": "test"}, "split must be one of"),
        ({"recorded_verdict": ""}, "recorded_verdict must be a non-empty string"),
        ({"recorded_verdict": 3}, "recorded_verdict must be a non-empty string"),
    ],
)
def test_invalid_split_and_verdict_are_rejected(tmp_path, changes, message):
    copy = _copy_fixture(_small_fixture(), tmp_path)
    _rewrite_metadata(copy, **changes)

    with pytest.raises(ValueError, match=re.escape(message)):
        load_fixture(copy)


def test_split_and_control_come_from_fixture_metadata(tmp_path):
    copy = _copy_fixture(_small_fixture(), tmp_path)
    _rewrite_metadata(copy, split=HELD_OUT, recorded_verdict="validated")

    case = load_fixture(copy)
    report = build_report([case])

    assert report["cases"][0]["split"] == HELD_OUT
    assert report["cases"][0]["control"] is True
    assert report["summary"]["splits"][HELD_OUT]["cases"] == 1
    assert report["summary"]["controls"]["cases"] == 1
    with pytest.raises(ValueError, match="visible fixtures only"):
        snapshot_from_report(report)


def test_failure_class_vocabulary():
    assert {code: set(subs) for code, subs in FAILURE_CLASSES.items()} == {
        "encoder": {"omission", "formula", "test", "harness"},
        "validator_fp": set(),
        "corpus": {"content", "metadata"},
        "dependency": set(),
        "selection": set(),
        "engine": set(),
        "migration": {"schema", "elided_excerpt"},
        "process": set(),
        "infra": set(),
        "unknown": set(),
    }


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ({"expected_class": "bogus"}, "unknown expected_class 'bogus'"),
        (
            {
                "expected_class": "selection",
                "expected_subclass": "schema",
                "class_evidence": "x",
                "class_source": "hand",
            },
            "expected_subclass 'schema' is not a subclass of 'selection'",
        ),
        (
            {"expected_class": "selection", "class_source": "hand"},
            "needs a non-empty class_evidence",
        ),
        (
            {"expected_class": "selection", "class_evidence": "x"},
            "class_source must be one of",
        ),
        ({"class_source": "hand"}, "class_source without expected_class"),
    ],
)
def test_invalid_failure_class_labels_are_rejected(tmp_path, label, message):
    copy = _copy_fixture(_small_fixture(), tmp_path)
    _rewrite_metadata(copy, **label)

    with pytest.raises(ValueError, match=re.escape(message)):
        load_fixture(copy)


def test_private_labels_are_reported_but_never_pinned(tmp_path):
    # Hand labels live in the private benchmark; the replay still reads them.
    copy = _copy_fixture(_small_fixture(), tmp_path)
    _rewrite_metadata(
        copy,
        expected_class="encoder",
        expected_subclass="test",
        class_evidence="synthetic label",
        class_source="hand",
    )
    labeled = load_fixture(copy)
    unlabeled = dataclasses.replace(labeled, case_id="unlabeled-copy", provenance={})

    report = build_report([labeled, unlabeled])

    rows = {row["id"]: row for row in report["cases"]}
    assert rows[labeled.case_id]["expected_class"] == "encoder"
    assert rows[labeled.case_id]["expected_subclass"] == "test"
    assert rows["unlabeled-copy"]["expected_class"] is None
    assert report["summary"]["expected_classes"] == {"encoder": 1, "unlabeled": 1}
    assert re.search(r"^  encoder +1$", format_summary(report), re.MULTILINE)
    assert "expected_class" not in json.dumps(snapshot_from_report(report))


def _report(cases, load_errors=()):
    return {
        "schema": "axiom-encode/us-replay-report/v1",
        "encoder_version": "0.0.0",
        "context": "workspace",
        "seconds": 0.1,
        "summary": {
            "cases": len(cases),
            "clean": sum(1 for c in cases if not c["issues"] and not c["error"]),
            "errors": sum(1 for c in cases if c["error"]),
            "issues": sum(len(c["issues"]) for c in cases),
            "rules": {},
        },
        "cases": cases,
        "load_errors": list(load_errors),
    }


def _case(case_id, issues, error=None, *, split=None, control=False):
    rules = {}
    for issue in issues:
        rule = re.search(r"\[([a-z-]+:[a-z-]+)\]", issue).group(1)
        rules[rule] = rules.get(rule, 0) + 1
    row = {
        "id": case_id,
        "citation": "us/statute/26/32",
        "issues": issues,
        "rules": rules,
        "error": error,
        "control": control,
        "imports_unresolved": [],
        "deferral_providers_unresolved": [],
    }
    if split:
        row["split"] = split
    return row


NUMERIC = "[complete-source-unit:numeric-recall] value 7 has no named scalar"
TESTS = "[complete-source-unit:tests] missing paired case"
STRUCTURE = "[complete-source-unit:structure] paragraph (b) has no rule"


# --- the snapshot pins exact findings ------------------------------------------


def test_drift_lists_every_added_and_removed_finding():
    pinned = snapshot_from_report(_report([_case("a", [NUMERIC, TESTS])]))

    assert snapshot_drift(pinned, _report([_case("a", [TESTS, NUMERIC])])) == []
    assert snapshot_drift(pinned, _report([_case("a", [TESTS])])) == [f"a - {NUMERIC}"]
    # Same counts per rule, different text: a reworded finding is drift too.
    reworded = NUMERIC.replace("value 7", "value 8")
    assert snapshot_drift(pinned, _report([_case("a", [reworded, TESTS])])) == [
        f"a - {NUMERIC}",
        f"a + {reworded}",
    ]
    assert snapshot_drift(pinned, _report([_case("a", [NUMERIC, NUMERIC, TESTS])])) == [
        f"a + {NUMERIC}"
    ]


def test_drift_flags_missing_new_erroring_and_reclassified_fixtures():
    pinned = snapshot_from_report(_report([_case("a", [TESTS])]))

    assert snapshot_drift(pinned, _report([_case("b", [TESTS])])) == [
        "a: pinned fixture is missing from the replay",
        "b: fixture is not pinned in the snapshot",
    ]
    assert snapshot_drift(
        pinned, _report([_case("a", [], error="ValueError: boom")])
    ) == ["a: replay error 'ValueError: boom'"]
    assert snapshot_drift(pinned, _report([_case("a", [TESTS], control=True)])) == [
        "a: control flag changed"
    ]
    with pytest.raises(ValueError, match="not a US replay snapshot"):
        snapshot_drift({}, _report([_case("a", [TESTS])]))
    old = {"schema": "axiom-encode/us-replay-snapshot/v1", "cases": {}}
    assert snapshot_drift(old, _report([]))[0].endswith("rewrite it")


def test_drift_output_is_bounded_per_case():
    pinned = snapshot_from_report(_report([_case("a", [])]))
    many = [f"[complete-source-unit:tests] case {n:03}" for n in range(30)]

    lines = snapshot_drift(pinned, _report([_case("a", many)]), limit=5)

    assert len(lines) == 6
    assert lines[-1] == "a + ... 25 more"


def test_snapshot_never_pins_errors():
    with pytest.raises(ValueError, match="never pinned"):
        snapshot_from_report(_report([_case("a", [], error="E: x")]))
    with pytest.raises(ValueError, match="never pinned"):
        snapshot_from_report(_report([], load_errors=[{"bundle": "x"}]))


# --- the guard: clean and control fixtures stay clean ---------------------------


def test_guard_refuses_clean_fixtures_that_become_refused_or_removed():
    base = snapshot_from_report(
        _report([_case("clean", []), _case("gone", []), _case("dirty", [TESTS])])
    )
    after = _report(
        [
            _case("clean", [NUMERIC]),
            _case("dirty", [TESTS, NUMERIC]),
            _case("new", [TESTS]),
        ]
    )

    violations = guard_violations(base, after)

    assert [line.split(" ")[0].rstrip(":") for line in violations] == ["clean", "gone"]
    assert "clean fixture is now refused with 1 findings" in violations[0]
    assert violations[1] == "gone: clean fixture was removed"


def test_guard_covers_new_controls_and_errors():
    base = snapshot_from_report(_report([_case("clean", [])]))
    after = _report(
        [
            _case("clean", [], error="ValueError: boom"),
            _case("ctl", [STRUCTURE], control=True),
            _case("ok-ctl", [], control=True),
        ]
    )

    violations = guard_violations(base, after)

    assert violations[0] == "clean: clean fixture now errors: ValueError: boom"
    assert violations[1].startswith("ctl (us/statute/26/32): control is now refused")
    assert len(violations) == 2


def test_unneeded_allowances_are_listed():
    base = snapshot_from_report(_report([_case("clean", []), _case("dirty", [TESTS])]))
    after = _report([_case("clean", [NUMERIC]), _case("dirty", [TESTS])])

    assert unneeded_allowances(base, after, {"clean": "x"}) == []
    assert unneeded_allowances(base, after, {"clean": "x", "dirty": "y"}) == ["dirty"]


def test_guard_accepts_a_recorded_reason_and_needs_a_base():
    base = snapshot_from_report(_report([_case("clean", []), _case("gone", [])]))
    after = _report([_case("clean", [NUMERIC])])

    assert (
        guard_violations(
            base, after, {"clean": "true defect: 7 is operative", "gone": "dup"}
        )
        == []
    )
    # A blank reason is no reason.
    assert len(guard_violations(base, after, {"clean": " ", "gone": "dup"})) == 1
    # Without a base snapshot only controls are guarded.
    assert guard_violations(None, after) == []
    # A v1 base snapshot (counts only) still guards its clean fixtures.
    v1 = {
        "schema": "axiom-encode/us-replay-snapshot/v1",
        "cases": {"clean": {"issues": 0, "error": None}},
    }
    assert len(guard_violations(v1, after)) == 1


# --- split-aware compare ----------------------------------------------------------


def test_compare_counts_issue_level_changes_by_rule():
    before = _report([_case("a", [NUMERIC, TESTS]), _case("b", [TESTS])])
    after = _report([_case("a", []), _case("b", [TESTS, NUMERIC])])

    diff = compare_reports(before, after)

    assert diff["removed_by_rule"] == {
        "complete-source-unit:numeric-recall": 1,
        "complete-source-unit:tests": 1,
    }
    assert diff["added_by_rule"] == {"complete-source-unit:numeric-recall": 1}
    assert diff["newly_clean"] == ["a"]
    assert (diff["issues_before"], diff["issues_after"]) == (3, 2)
    assert diff["splits"][VISIBLE]["compared"] == 2
    assert diff["splits"][HELD_OUT]["compared"] == 0


def test_compare_never_counts_a_crash_as_removed_issues():
    before = _report([_case("a", [NUMERIC, TESTS]), _case("b", [], error="E: x")])
    after = _report([_case("a", [], error="E: boom"), _case("b", [TESTS])])

    diff = compare_reports(before, after)

    assert diff["compared"] == 0
    assert [item["id"] for item in diff["errored"]] == ["a", "b"]
    assert diff["removed_by_rule"] == {} and diff["added_by_rule"] == {}
    assert diff["newly_clean"] == [] and diff["newly_failing"] == []
    assert (diff["clean_before"], diff["clean_after"]) == (0, 0)
    assert "ERRORED (not compared): 2" in format_comparison(diff)


def test_compare_splits_totals_and_hides_held_out_cases():
    before = _report(
        [
            _case("v1", [NUMERIC]),
            _case("v2", []),
            _case("h1", [TESTS, NUMERIC], split=HELD_OUT),
            _case("h2", [TESTS]),
        ]
    )
    after = _report(
        [
            _case("v1", []),
            _case("v2", [STRUCTURE]),
            _case("h1", [TESTS], split=HELD_OUT),
            _case("h2", []),
        ]
    )

    # h2 is held out only through the split map, which wins over report rows.
    diff = compare_reports(before, after, split_of={"h2": HELD_OUT, "v1": VISIBLE})

    visible, held_out = diff["splits"][VISIBLE], diff["splits"][HELD_OUT]
    assert (visible["compared"], held_out["compared"]) == (2, 2)
    assert (visible["issues_before"], visible["issues_after"]) == (1, 1)
    assert (held_out["issues_before"], held_out["issues_after"]) == (3, 1)
    assert visible["newly_clean"] == ["v1"] and visible["newly_failing"] == ["v2"]
    assert held_out["newly_clean"] == ["h2"]
    assert held_out["removed_by_rule"] == {
        "complete-source-unit:numeric-recall": 1,
        "complete-source-unit:tests": 1,
    }
    assert {c["id"]: c["split"] for c in diff["changed"]} == {
        "v1": VISIBLE,
        "v2": VISIBLE,
        "h1": HELD_OUT,
        "h2": HELD_OUT,
    }
    text = format_comparison(diff)
    assert "held_out: 2 modules, issues 3 -> 1" in text
    assert "h1 (" not in text and "h2 (" not in text
    assert "(2 changed held-out modules not listed)" in text
    assert "h1 (" in format_comparison(diff, show_held_out=True)


def test_split_map_reads_json_objects_and_manifest_lines(tmp_path):
    as_object = tmp_path / "split.json"
    as_object.write_text(json.dumps({"cases": {"a": "visible", "b": "held_out"}}))
    as_lines = tmp_path / "manifest.jsonl"
    as_lines.write_text(
        '{"case_id": "a", "split": "visible"}\n'
        '{"kind": "run", "run_id": "x"}\n\n'
        '{"case_id": "b", "split": "held_out"}\n'
    )
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": "test"}))

    assert load_split_map(as_object) == {"a": "visible", "b": "held_out"}
    assert load_split_map(as_lines) == {"a": "visible", "b": "held_out"}
    with pytest.raises(ValueError, match="split must be one of"):
        load_split_map(bad)


def test_compare_cli_hides_cases_held_out_only_in_the_before_report(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_report([_case("h", [TESTS], split=HELD_OUT)])))
    after.write_text(json.dumps(_report([_case("h", [])])))

    assert main(["compare", str(before), str(after), "--json"]) == 0
    assert '"h"' not in capsys.readouterr().out


def test_ci_cli_skips_the_replay_when_no_trigger_path_changed(tmp_path, capsys):
    head = subprocess.run(
        ["git", "-C", str(FIXTURES.parents[2]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert main(["ci", "--changed-since", head, "--fixtures", str(tmp_path)]) == 0
    assert "passes without a replay" in capsys.readouterr().out


def test_compare_cli_hides_held_out_cases_in_json(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    split = tmp_path / "split.json"
    before.write_text(json.dumps(_report([_case("v", [TESTS]), _case("h", [TESTS])])))
    after.write_text(json.dumps(_report([_case("v", []), _case("h", [])])))
    split.write_text(json.dumps({"h": "held_out"}))

    assert (
        main(["compare", str(before), str(after), "--split", str(split), "--json"]) == 0
    )
    output = capsys.readouterr().out
    diff = json.loads(output)
    assert [c["id"] for c in diff["changed"]] == ["v"]
    assert diff["newly_clean"] == ["v"] and diff["newly_clean_held_out"] == 1
    assert diff["splits"]["held_out"]["newly_clean"] == 1
    assert '"h"' not in output

    assert (
        main(
            [
                "compare",
                str(before),
                str(after),
                "--split",
                str(split),
                "--json",
                "--show-held-out",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["splits"]["held_out"]["newly_clean"] == [
        "h"
    ]


# --- the CI gate ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("paths", "needed"),
    [
        (["src/axiom_encode/harness/source_completeness.py"], True),
        (["src/axiom_encode/bare_structure.py"], True),
        (["src/axiom_encode/corpus_resolver.py"], True),
        (["tests/fixtures/us_replay_benchmark/snapshot.json"], True),
        (["scripts/replay_us_failure_bundles.py"], True),
        (["uv.lock"], True),
        (["docs/us-replay-benchmark.md", "README.md"], False),
        (["src/axiom_encode/cli.py"], False),
        (["src/axiom_encode/harness_extra.py"], False),
        (["src/axiom_encode/bare_structure.pyc"], False),
        ([], False),
    ],
)
def test_replay_trigger_paths(paths, needed):
    assert replay_is_needed(paths) is needed


def test_modules_the_replay_loads_trigger_it():
    loaded = analyzer_module_paths()

    assert "src/axiom_encode/harness/source_completeness.py" in loaded
    assert "src/axiom_encode/numeric_equality.py" in loaded
    assert replay_is_needed(["src/axiom_encode/numeric_equality.py"], loaded)
    assert not replay_is_needed(["src/axiom_encode/numeric_equality.py"])
    assert not replay_is_needed(["docs/us-replay-benchmark.md"], loaded)


def test_trigger_paths_exist():
    root = FIXTURES.parents[2]
    for trigger in REPLAY_TRIGGER_PATHS:
        assert (root / trigger).exists(), trigger


def test_gate_markdown_reports_verdict_and_changed_fixtures():
    base = snapshot_from_report(_report([_case("a", [TESTS]), _case("b", [])]))
    report = _report([_case("a", []), _case("b", [])])

    passed = gate_markdown(report, base, {"replay errors": []})
    failed = gate_markdown(report, base, {"replay errors": ["a: boom"]})

    assert "**passed**" in passed
    assert "| `a` | `us/statute/26/32` | 1 | 0 | 1 | 0 |" in passed
    assert "**FAILED: replay errors**" in failed
    assert "a: boom" in failed


def test_read_snapshot_at_reads_git_refs_and_fails_closed(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "README").write_text("x\n")
    git("add", "README")
    git("commit", "-q", "-m", "no snapshot")
    target = tmp_path / SNAPSHOT_REPO_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"schema": SNAPSHOT_SCHEMA, "cases": {}}))
    git("add", SNAPSHOT_REPO_PATH)
    git("commit", "-q", "-m", "snapshot")

    assert read_snapshot_at("HEAD", repo=tmp_path) == {
        "schema": SNAPSHOT_SCHEMA,
        "cases": {},
    }
    assert read_snapshot_at("HEAD^1", repo=tmp_path) is None
    with pytest.raises(ValueError, match="cannot resolve"):
        read_snapshot_at("no-such-ref", repo=tmp_path)


def test_recorded_issue_classification():
    assert (
        recorded_issue_rule("x.yaml: ci: Test case `c` output `o` expected 1, got 2.")
        == "tests:wrong-expected-value"
    )
    assert recorded_issue_rule("x.yaml: ci: [complete-source-unit:deferral] y") == (
        "complete-source-unit:deferral"
    )


def test_bundle_reads_refuse_symlinks_and_traversal(tmp_path):
    (tmp_path / "outside.txt").write_text("secret")
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")

    with pytest.raises(UnsafeBundlePath, match="symlink"):
        _read_inside(root, "link.txt")
    with pytest.raises(UnsafeBundlePath, match="unsafe relative path"):
        _read_inside(root, "../outside.txt")
    (tmp_path / "linked-root").symlink_to(root)
    (root / "plain.txt").write_text("ok")
    with pytest.raises(UnsafeBundlePath, match="symlinked root"):
        _read_inside(tmp_path / "linked-root", "plain.txt")


IMPORTING_CANDIDATE = """\
format: rulespec/v1
imports:
- us:statutes/26/1#tax_rate
module:
  source_verification:
    corpus_citation_path: us/statute/26/2
rules:
- name: tax
  kind: derived
  entity: TaxUnit
  dtype: Money
  period: Year
  versions:
  - effective_from: '2025-01-01'
    formula: income * tax_rate
"""
CONTEXT_MODULE = """\
format: rulespec/v1
rules:
- name: tax_rate
  kind: parameter
  dtype: Rate
  versions:
  - effective_from: '2025-01-01'
    formula: '0.15'
"""
RECALL_15_PERCENT = (
    "[complete-source-unit:numeric-recall] Authoritative corpus numeric value "
    "0.15 has no named scalar representation."
)


def _synthetic_bundle(tmp_path: Path, context_module: str = CONTEXT_MODULE) -> Path:
    root = tmp_path / "12345" / "x"
    candidate_dir = root / "generated/target/final-rejected-candidate"
    workspace = (
        root / "generated/target/_eval_workspaces/runner-a/us-statute-26-2/workspace"
    )
    (candidate_dir / "statutes/26").mkdir(parents=True)
    (workspace / "context/statutes/26").mkdir(parents=True)
    (root / "generated/target/runner-a/statutes/26").mkdir(parents=True)
    (candidate_dir / "statutes/26/2.yaml").write_text(IMPORTING_CANDIDATE)
    (root / "generated/target/runner-a/statutes/26/2.yaml").write_text(
        IMPORTING_CANDIDATE
    )
    (candidate_dir / "statutes/26/2.test.yaml").write_text("[]\n")
    (candidate_dir / "issues.json").write_text(
        json.dumps(
            {
                "citation": "us/statute/26/2",
                "encoder_version": "0.0.1",
                "issues": ["statutes/26/2.yaml: ci: [complete-source-unit:tests] x"],
            }
        )
    )
    (workspace / "source.txt").write_text("The tax is 15 percent of income.\n")
    (workspace / "context/statutes/26/1.yaml").write_text(context_module)
    (workspace / "context/statutes/26/9.yaml").write_text(CONTEXT_MODULE)
    (workspace / "context-manifest.json").write_text(
        json.dumps(
            {
                "policy_prefix": "us",
                "context_files": [
                    {
                        "import_path": "us:statutes/26/1",
                        "workspace_path": "context/statutes/26/1.yaml",
                    },
                    {
                        "import_path": "us:statutes/26/9",
                        "workspace_path": "context/statutes/26/9.yaml",
                    },
                ],
            }
        )
    )
    candidate_sha = hashlib.sha256(IMPORTING_CANDIDATE.encode())
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "citation": "us/statute/26/2",
                "encoder_commit": "a" * 40,
                "workflow_run_id": "12345",
                "files": [
                    {
                        "path": "target/final-rejected-candidate/statutes/26/2.yaml",
                        "sha256": candidate_sha.hexdigest(),
                    }
                ],
            }
        )
    )
    return root


def test_extract_copies_only_imported_context_and_authenticates(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    (module,) = discover_bundle_modules(bundle.parent)

    fixture = extract_fixture(module, tmp_path / "fixtures")
    case = load_fixture(fixture)
    provenance = json.loads((fixture / FIXTURE_FILE).read_text())

    assert module.runner == "runner-a"
    assert case.context == {"us:statutes/26/1": CONTEXT_MODULE}
    assert not (fixture / "context/statutes/26/9.yaml").exists()
    assert provenance["candidate_matches_bundle_inventory"] is True
    assert provenance["recorded_rules"] == {"complete-source-unit:tests": 1}
    with pytest.raises(FileExistsError):
        extract_fixture(module, tmp_path / "fixtures")


def test_imports_resolve_from_context_and_can_be_disabled(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    (module,) = discover_bundle_modules(bundle.parent)
    case = load_fixture(extract_fixture(module, tmp_path / "fixtures"))

    with_context = build_report([case])
    without_context = build_report([case], use_context=False)

    assert with_context["cases"][0]["imports_resolved"] == 1
    assert without_context["cases"][0]["imports_resolved"] == 0
    assert without_context["context"] == "none"
    assert not any(
        RECALL_15_PERCENT in issue for issue in with_context["cases"][0]["issues"]
    )
    assert any(
        RECALL_15_PERCENT in issue for issue in without_context["cases"][0]["issues"]
    )


def test_imported_value_must_match_to_satisfy_recall(tmp_path):
    bundle = _synthetic_bundle(tmp_path, CONTEXT_MODULE.replace("'0.15'", "'0.2'"))
    (module,) = discover_bundle_modules(bundle.parent)
    case = load_fixture(extract_fixture(module, tmp_path / "fixtures"))

    report = build_report([case])

    assert report["cases"][0]["imports_resolved"] == 1
    assert any(RECALL_15_PERCENT in issue for issue in report["cases"][0]["issues"])


def _extracted_root(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "fixtures"
    for name in names:
        bundle = _synthetic_bundle(tmp_path / name)
        extract_fixture(discover_bundle_modules(bundle.parent)[0], root, name=name)
    return root


def test_snapshot_cli_writes_checks_and_records_new_fixtures(tmp_path, capsys):
    root = _extracted_root(tmp_path, "first")
    snapshot_path = root / SNAPSHOT_FILE

    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 1
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 0
    pinned = json.loads(snapshot_path.read_text())
    assert pinned["schema"] == SNAPSHOT_SCHEMA
    assert pinned["cases"]["first"]["findings"]

    # Any edit to a pinned finding is drift until the snapshot is rewritten.
    edited = json.loads(snapshot_path.read_text())
    edited["cases"]["first"]["findings"][0] += " (edited)"
    snapshot_path.write_text(json.dumps(edited))
    capsys.readouterr()
    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 1
    assert "first - " in capsys.readouterr().out
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    assert json.loads(snapshot_path.read_text()) == pinned

    _synthetic = _synthetic_bundle(tmp_path / "second")
    extract_fixture(discover_bundle_modules(_synthetic.parent)[0], root, name="second")
    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 1
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    assert set(json.loads(snapshot_path.read_text())["cases"]) == {"first", "second"}


def test_snapshot_cli_needs_a_reason_to_refuse_a_clean_fixture(tmp_path, capsys):
    root = _extracted_root(tmp_path, "first")
    snapshot_path = root / SNAPSHOT_FILE
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    # Pretend the base pinned the fixture clean.
    clean = json.loads(snapshot_path.read_text())
    clean["cases"]["first"].update(issues=0, rules={}, findings=[])
    snapshot_path.write_text(json.dumps(clean))

    assert main(["snapshot", "--fixtures", str(root)]) == 1
    assert "clean fixture is now refused" in capsys.readouterr().err
    assert json.loads(snapshot_path.read_text()) == clean
    assert (
        main(["snapshot", "--fixtures", str(root), "--allow-regression", "first"]) == 1
    )
    assert (
        main(["snapshot", "--fixtures", str(root), "--allow-regression", "nope=x"]) == 1
    )
    assert (
        main(
            [
                "snapshot",
                "--fixtures",
                str(root),
                "--allow-regression",
                "first=true defect",
            ]
        )
        == 0
    )
    written = json.loads(snapshot_path.read_text())
    assert written["allowed_regressions"] == {"first": "true defect"}
    # The next rewrite drops reasons it is not given again.
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    assert json.loads(snapshot_path.read_text())["allowed_regressions"] == {}


def test_run_gate_passes_and_fails_on_each_condition(tmp_path):
    root = _extracted_root(tmp_path, "first")
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    snapshot = json.loads((root / SNAPSHOT_FILE).read_text())

    report, failures = run_gate(root, base_snapshot=snapshot)
    assert not any(failures.values()), failures
    assert report["summary"]["cases"] == 1

    # A base that pinned the fixture clean: the guard fails.
    clean_base = json.loads(json.dumps(snapshot))
    clean_base["cases"]["first"].update(issues=0, rules={}, findings=[])
    _, failures = run_gate(root, base_snapshot=clean_base)
    assert failures["clean or control fixtures refused"]
    assert not failures["findings differ from the committed snapshot"]

    # A finding that differs from the committed snapshot: drift fails.
    edited = json.loads(json.dumps(snapshot))
    edited["cases"]["first"]["findings"].pop()
    (root / SNAPSHOT_FILE).write_text(json.dumps(edited))
    _, failures = run_gate(root, base_snapshot=snapshot)
    assert failures["findings differ from the committed snapshot"]

    # A hand edit that keeps the findings but changes the counts: refused.
    hand = json.loads(json.dumps(snapshot))
    hand["cases"]["first"]["issues"] += 1
    (root / SNAPSHOT_FILE).write_text(json.dumps(hand))
    _, failures = run_gate(root, base_snapshot=snapshot)
    assert "a hand edit?" in failures["findings differ from the committed snapshot"][0]

    # A reason for a fixture that does not regress against the base: refused.
    stale = json.loads(json.dumps(snapshot))
    stale["allowed_regressions"] = {"first": "old reason"}
    (root / SNAPSHOT_FILE).write_text(json.dumps(stale))
    _, failures = run_gate(root, base_snapshot=snapshot)
    assert failures["clean or control fixtures refused"] == [
        "allowed_regressions names first, which does not regress against the "
        "base; drop it (rewrite the snapshot without that reason)"
    ]
    # Against a base that pinned it clean, the same reason is needed and passes.
    _, failures = run_gate(root, base_snapshot=clean_base)
    assert not any(failures.values()), failures

    # After that change merges, the base carries the reason and pins the
    # fixture refused: the inherited reason is neither stale nor an excuse.
    merged = json.loads(json.dumps(snapshot))
    merged["allowed_regressions"] = {"first": "old reason"}
    _, failures = run_gate(root, base_snapshot=merged)
    assert not any(failures.values()), failures
    inherited_clean = json.loads(json.dumps(clean_base))
    inherited_clean["allowed_regressions"] = {"first": "old reason"}
    _, failures = run_gate(root, base_snapshot=inherited_clean)
    assert (
        "clean fixture is now refused"
        in (failures["clean or control fixtures refused"][0])
    )

    # A fixture that does not load: a replay error.
    (root / SNAPSHOT_FILE).write_text(json.dumps(snapshot))
    (root / "first" / "candidate.yaml").write_text("tampered\n")
    _, failures = run_gate(root, base_snapshot=snapshot)
    assert "sha256 mismatch" in failures["replay errors"][0]


def test_ci_cli_exit_status_and_summary(tmp_path, capsys):
    root = _extracted_root(tmp_path, "first")
    summary = tmp_path / "summary.md"
    assert main(["ci", "--fixtures", str(root)]) == 1
    assert "no snapshot at" in capsys.readouterr().out
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    assert main(["ci", "--fixtures", str(root), "--summary", str(summary)]) == 0
    assert "**passed**" in summary.read_text()
    assert main(["ci", "--fixtures", str(root), "--base-ref", "no-such-ref-x"]) == 1


def test_rulespec_root_fills_deferral_providers_missing_from_context(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    candidate = bundle / "generated/target/final-rejected-candidate/statutes/26/2.yaml"
    candidate.write_text(
        IMPORTING_CANDIDATE.replace(
            "rules:\n",
            "  deferred_outputs:\n"
            "  - output: us:statutes/26/2#surtax\n"
            "    reason: Needs the base defined in section 32.\n"
            "    blocked_by:\n"
            "    - us:statutes/26/32#surtax_base\n"
            "rules:\n",
            1,
        )
    )
    (module,) = discover_bundle_modules(bundle.parent)
    rulespec = tmp_path / "rulespec-us"
    (rulespec / "us/statutes/26").mkdir(parents=True)
    (rulespec / "us/statutes/26/32.yaml").write_text(
        "format: rulespec/v1\nrules:\n- name: surtax_base\n  kind: derived\n"
    )

    without_root = load_fixture(extract_fixture(module, tmp_path / "a"))
    with_root = load_fixture(
        extract_fixture(module, tmp_path / "b", rulespec_root=rulespec)
    )
    gap = build_report([without_root])["cases"][0]
    filled = build_report([with_root])["cases"][0]

    assert gap["deferral_providers_unresolved"] == ["us:statutes/26/32#surtax_base"]
    assert filled["deferral_providers_unresolved"] == []
    assert filled["deferral_providers_resolved"] == ["us:statutes/26/32#surtax_base"]
    assert with_root.provenance["context_origins"] == {
        "us:statutes/26/1": "workspace",
        "us:statutes/26/32": "rulespec_ref",
    }
    assert with_root.provenance["context_imports"]["us:statutes/26/32"] == (
        "providers/us/statutes/26/32.yaml"
    )
    provider_bytes = (rulespec / "us/statutes/26/32.yaml").read_bytes()
    assert with_root.provenance["provider_git_blob_sha1"] == {
        "providers/us/statutes/26/32.yaml": hashlib.sha1(
            b"blob %d\0" % len(provider_bytes) + provider_bytes
        ).hexdigest()
    }

    (rulespec / "us/statutes/26/32.yaml").unlink()
    (rulespec / "us/statutes/26/32.yaml").symlink_to(tmp_path / "elsewhere.yaml")
    (tmp_path / "elsewhere.yaml").write_bytes(provider_bytes)
    with pytest.raises(UnsafeBundlePath, match="symlink"):
        extract_fixture(module, tmp_path / "c", rulespec_root=rulespec)


@pytest.mark.parametrize(
    ("target", "key"),
    (
        ("us:statutes/26/32", "us:statutes/26/32"),
        ("us:statutes/26/32.yaml", "us:statutes/26/32"),
        ("us-az:policies/des/faa5/x", "us-az:policies/des/faa5/x"),
        ("statutes/26/32", "us-ok:statutes/26/32"),
        ("/statutes/26/32/", "us-ok:statutes/26/32"),
        ("us:/statutes/26/32", None),
        ("us:statutes/26/32/", None),
        ("us:statutes//26/32", None),
        ("us:statutes/../26/32", None),
        ("us:sources/26/32", None),
        ("US:statutes/26/32", None),
    ),
)
def test_context_keys_follow_the_ci_resolver(target, key):
    assert _context_key(target, "us-ok") == key


def test_unprefixed_imports_without_a_policy_prefix_count_as_unresolved():
    case = ReplayCase(
        case_id="x",
        run_id="1",
        citation="us/statute/26/2",
        module_path="statutes/26/2.yaml",
        candidate=IMPORTING_CANDIDATE.replace(
            "us:statutes/26/1#tax_rate", "statutes/26/1#tax_rate"
        ),
        tests=None,
        source="The tax is 15 percent of income.",
        source_metadata=None,
        policy_prefix=None,
    )

    assert import_symbol_contents(case) == ([], ["statutes/26/1#tax_rate"])


def test_symlinked_bundle_roots_are_load_errors(tmp_path):
    real = _synthetic_bundle(tmp_path / "real")
    linked = tmp_path / "bundles" / "777"
    linked.mkdir(parents=True)
    (linked / "x").symlink_to(real)

    cases, errors = load_bundle_cases(tmp_path / "bundles")

    assert cases == []
    assert len(errors) == 1 and errors[0]["bundle"] == "777"
    assert "symlinked bundle root" in errors[0]["error"]


def test_malformed_bundles_are_reported_not_fatal(tmp_path):
    _synthetic_bundle(tmp_path / "bundles")
    bad = tmp_path / "bundles" / "999" / "x"
    bad.mkdir(parents=True)
    (bad / "metadata.json").write_text("[]")

    cases, errors = load_bundle_cases(tmp_path / "bundles")

    assert [case.run_id for case in cases] == ["12345"]
    assert errors == [
        {"bundle": "999", "error": "ValueError: metadata.json is not a JSON object"}
    ]
