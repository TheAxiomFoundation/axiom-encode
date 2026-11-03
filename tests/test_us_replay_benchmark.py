"""Offline US replay benchmark: committed failure fixtures and their ratchet.

``tests/fixtures/us_replay_benchmark`` holds real rejected US candidates from
``targeted-reencode-failure`` artifacts. ``scripts/replay_us_failure_bundles.py``
replays ``analyze_complete_source_unit`` over them, resolving each candidate's
imports from the bundle's workspace context, and ``snapshot.json`` pins the
per-fixture, per-rule issue counts.

The pin is a ratchet. A count that rises fails here as a regression. A count
that falls also fails until the same PR records it with::

    python scripts/replay_us_failure_bundles.py snapshot

Raising a pinned count needs ``--allow-increase`` and a justification in the
PR. See ``docs/us-replay-benchmark.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts.replay_us_failure_bundles import (
    DEFAULT_FIXTURE_ROOT,
    FAILURE_CLASSES,
    FIXTURE_FILE,
    SNAPSHOT_FILE,
    ReplayCase,
    SnapshotDeviations,
    UnsafeBundlePath,
    _context_key,
    _read_inside,
    build_report,
    compare_reports,
    discover_bundle_modules,
    extract_fixture,
    fixture_directories,
    fixture_label_errors,
    format_comparison,
    format_summary,
    import_symbol_contents,
    load_bundle_cases,
    load_fixture,
    load_fixtures,
    main,
    recorded_issue_rule,
    snapshot_deviations,
    snapshot_from_report,
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
MAX_FIXTURE_BYTES = 3 * 1024 * 1024
UPDATE_HINT = (
    "record improvements with `python scripts/replay_us_failure_bundles.py "
    "snapshot` in this PR"
)


def test_replay_matches_snapshot_ratchet():
    # One test replays every fixture, so CI shards never repeat the replay.
    report = build_report(load_fixtures(FIXTURES))
    snapshot = json.loads(SNAPSHOT.read_text())
    deviations = snapshot_deviations(snapshot, report)

    assert report["summary"]["errors"] == 0
    assert report["summary"]["imports_resolved"] > 0
    assert report["summary"]["imports_unresolved"] == 0
    assert report["summary"]["deferral_providers_unresolved"] == 0

    assert not deviations.increases, (
        "US replay benchmark regressed (pinned counts rose):\n  "
        + "\n  ".join(deviations.increases)
    )
    assert not deviations.decreases, (
        "US replay benchmark improved; "
        + UPDATE_HINT
        + ":\n  "
        + "\n  ".join(deviations.decreases)
    )
    assert not deviations.unpinned, (
        "new US replay fixtures; "
        + UPDATE_HINT
        + ":\n  "
        + "\n  ".join(deviations.unpinned)
    )


def test_snapshot_totals_match_cases():
    snapshot = json.loads(SNAPSHOT.read_text())
    cases = snapshot["cases"].values()

    assert snapshot["context"] == "workspace"
    assert snapshot["totals"]["cases"] == len(snapshot["cases"])
    assert snapshot["totals"]["issues"] == sum(case["issues"] for case in cases)
    assert snapshot["totals"]["clean"] == sum(
        1 for case in cases if not case["issues"] and not case["error"]
    )
    rule_totals: dict[str, int] = {}
    for case in cases:
        assert case["issues"] == sum(case["rules"].values())
        for rule, count in case["rules"].items():
            rule_totals[rule] = rule_totals.get(rule, 0) + count
    assert snapshot["totals"]["rules"] == rule_totals


def test_fixture_set_covers_failing_rule_classes():
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


def test_fixtures_carry_provenance_and_stay_small():
    total = 0
    for directory in fixture_directories(FIXTURES):
        case = load_fixture(directory)
        provenance = case.provenance

        assert re.fullmatch(r"[0-9]+", provenance["run_id"])
        assert provenance["run_url"].endswith(f"/actions/runs/{provenance['run_id']}")
        assert re.fullmatch(r"[0-9a-f]{40}", provenance["encoder_commit"])
        assert provenance["citation"].startswith("us")
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


def test_tampered_fixture_bytes_are_rejected(tmp_path):
    source = fixture_directories(FIXTURES)[0]
    copy = tmp_path / source.name
    for path in source.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (copy / "candidate.yaml").write_text(
        (copy / "candidate.yaml").read_text() + "\n# edited\n"
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_fixture(copy)


def test_unlisted_fixture_files_are_rejected(tmp_path):
    source = fixture_directories(FIXTURES)[0]
    copy = tmp_path / source.name
    for path in source.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (copy / "context").mkdir(exist_ok=True)
    (copy / "context" / "extra.yaml").write_text("format: rulespec/v1\n")

    with pytest.raises(ValueError, match="files on disk differ"):
        load_fixture(copy)


def _copy_fixture(source: Path, tmp_path: Path) -> Path:
    copy = tmp_path / source.name
    for path in source.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return copy


def test_committed_fixtures_carry_valid_failure_class_labels():
    expected_codes = {
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
    assert {code: set(subs) for code, subs in FAILURE_CLASSES.items()} == (
        expected_codes
    )
    for directory in fixture_directories(FIXTURES):
        metadata = json.loads((directory / FIXTURE_FILE).read_text())
        # fixture.json holds the hashes, so it never lists itself; adding a
        # label therefore leaves every recorded file hash valid.
        assert FIXTURE_FILE not in metadata["files"]
        assert metadata["expected_class"] in FAILURE_CLASSES, directory.name
        assert metadata["class_source"] == "hand", directory.name
        assert fixture_label_errors(metadata) == [], directory.name
        assert len(metadata["class_evidence"]) <= 200, directory.name


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ({"expected_class": "bogus"}, "unknown expected_class 'bogus'"),
        (
            {"expected_class": "selection", "expected_subclass": "schema"},
            "expected_subclass 'schema' is not a subclass of 'selection'",
        ),
        ({"class_evidence": ""}, "needs a non-empty class_evidence"),
        ({"class_source": "guess"}, "class_source must be one of"),
    ],
)
def test_invalid_failure_class_labels_are_rejected(tmp_path, label, message):
    copy = _copy_fixture(fixture_directories(FIXTURES)[0], tmp_path)
    metadata = json.loads((copy / FIXTURE_FILE).read_text())
    metadata.update(label)
    (copy / FIXTURE_FILE).write_text(json.dumps(metadata, indent=2) + "\n")

    with pytest.raises(ValueError, match=re.escape(message)):
        load_fixture(copy)


def test_unlabeled_fixtures_still_load_but_orphan_label_keys_do_not(tmp_path):
    copy = _copy_fixture(fixture_directories(FIXTURES)[0], tmp_path)
    metadata = json.loads((copy / FIXTURE_FILE).read_text())
    for key in ("expected_class", "expected_subclass", "class_evidence"):
        metadata.pop(key, None)
    (copy / FIXTURE_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    with pytest.raises(ValueError, match="class_source without expected_class"):
        load_fixture(copy)

    metadata.pop("class_source")
    (copy / FIXTURE_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    assert "expected_class" not in load_fixture(copy).provenance


def test_replay_reports_expected_class_per_fixture_and_in_total():
    labeled = [load_fixture(directory) for directory in fixture_directories(FIXTURES)]
    by_class: dict[str, ReplayCase] = {}
    for case in labeled:
        by_class.setdefault(case.provenance["expected_class"], case)
    first, second = list(by_class.values())[:2]
    unlabeled = dataclasses.replace(first, case_id="unlabeled-copy", provenance={})

    report = build_report([first, second, unlabeled])

    rows = {row["id"]: row for row in report["cases"]}
    assert rows[first.case_id]["expected_class"] == first.provenance["expected_class"]
    assert rows[first.case_id]["expected_subclass"] == first.provenance.get(
        "expected_subclass"
    )
    assert rows["unlabeled-copy"]["expected_class"] is None
    expected_counts = {
        first.provenance["expected_class"]: 1,
        second.provenance["expected_class"]: 1,
        "unlabeled": 1,
    }
    assert report["summary"]["expected_classes"] == dict(
        sorted(expected_counts.items())
    )
    summary = format_summary(report)
    assert "expected class (hand label)" in summary
    assert re.search(
        rf"^  {re.escape(second.provenance['expected_class'])} +1$",
        summary,
        re.MULTILINE,
    )
    # Labels never enter the pinned snapshot: the ratchet stays per-rule.
    assert "expected_classes" not in snapshot_from_report(report)["totals"]


def _report(cases):
    return {
        "schema": "axiom-encode/us-replay-report/v1",
        "encoder_version": "0.0.0",
        "context": "workspace",
        "summary": {"cases": len(cases), "clean": 0, "issues": 0, "rules": {}},
        "cases": cases,
    }


def _case(case_id, issues, error=None):
    rules = {}
    for issue in issues:
        rule = re.search(r"\[([a-z-]+:[a-z-]+)\]", issue).group(1)
        rules[rule] = rules.get(rule, 0) + 1
    return {
        "id": case_id,
        "citation": "us/statute/26/32",
        "issues": issues,
        "rules": rules,
        "error": error,
    }


NUMERIC = "[complete-source-unit:numeric-recall] value 7 has no named scalar"
TESTS = "[complete-source-unit:tests] missing paired case"


def test_ratchet_separates_regressions_from_improvements():
    pinned = snapshot_from_report(_report([_case("a", [NUMERIC, TESTS])]))

    worse = _report([_case("a", [NUMERIC, NUMERIC, TESTS])])
    better = _report([_case("a", [TESTS])])
    same = _report([_case("a", [TESTS, NUMERIC])])

    assert snapshot_deviations(pinned, worse) == SnapshotDeviations(
        ["a (us/statute/26/32) complete-source-unit:numeric-recall: 1 -> 2"], [], []
    )
    assert snapshot_deviations(pinned, better) == SnapshotDeviations(
        [], ["a (us/statute/26/32) complete-source-unit:numeric-recall: 1 -> 0"], []
    )
    assert not snapshot_deviations(pinned, same)


def test_ratchet_rejects_missing_unpinned_and_erroring_fixtures():
    pinned = snapshot_from_report(_report([_case("a", [TESTS])]))

    deviations = snapshot_deviations(pinned, _report([_case("b", [TESTS])]))
    assert deviations.increases == ["a: pinned fixture is missing from the replay"]
    assert deviations.unpinned == ["b: fixture is not pinned in the snapshot"]
    deviations = snapshot_deviations(
        pinned, _report([_case("a", [], error="ValueError: boom")])
    )
    assert deviations.increases[0] == "a: replay error changed to 'ValueError: boom'"
    with pytest.raises(ValueError, match="not a US replay snapshot"):
        snapshot_deviations({}, _report([_case("a", [TESTS])]))


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


def test_snapshot_cli_writes_checks_and_refuses_unflagged_increases(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    (module,) = discover_bundle_modules(bundle.parent)
    root = tmp_path / "fixtures"
    extract_fixture(module, root)
    snapshot_path = root / SNAPSHOT_FILE

    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 1
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 0

    pinned = json.loads(snapshot_path.read_text())
    (case_id,) = pinned["cases"]
    current = pinned["cases"][case_id]["issues"]
    assert current > 0
    lowered = json.loads(snapshot_path.read_text())
    lowered["cases"][case_id]["rules"] = {}
    lowered["cases"][case_id]["issues"] = 0
    snapshot_path.write_text(json.dumps(lowered))

    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 1
    assert main(["snapshot", "--fixtures", str(root)]) == 1
    assert json.loads(snapshot_path.read_text()) == lowered
    assert main(["snapshot", "--fixtures", str(root), "--allow-increase"]) == 0
    assert json.loads(snapshot_path.read_text())["cases"][case_id]["issues"] == (
        current
    )


def test_snapshot_cli_adds_new_fixtures_without_the_increase_flag(tmp_path):
    root = tmp_path / "fixtures"
    first = _synthetic_bundle(tmp_path / "first")
    second = _synthetic_bundle(tmp_path / "second")
    extract_fixture(discover_bundle_modules(first.parent)[0], root, name="first")
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    extract_fixture(discover_bundle_modules(second.parent)[0], root, name="second")

    assert main(["snapshot", "--fixtures", str(root), "--check"]) == 1
    assert main(["snapshot", "--fixtures", str(root)]) == 0
    assert set(json.loads((root / SNAPSHOT_FILE).read_text())["cases"]) == {
        "first",
        "second",
    }


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
