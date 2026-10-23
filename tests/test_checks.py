"""Declared checks, typed validator issues, and the untyped-issue ratchet."""

from __future__ import annotations

import ast
import functools
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from axiom_encode import checks
from axiom_encode.checks import (
    ISSUE_SCHEMA_VERSION,
    IssueSubject,
    ValidationIssue,
    check_definition_problems,
    declare_check,
    issue_record,
    issues_from_document,
    load_checks,
    map_issue_text,
    retype,
    typed_issue_document,
)
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.encoding_db import (
    Iteration,
    IterationError,
    iteration_payload,
)
from axiom_encode.harness.evals import (
    _format_validation_retry_feedback,
    summarize_validation_failures,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)
from axiom_encode.retry_feedback import bounded_validation_retry_feedback_item

REPO = Path(__file__).parents[1]
SRC = REPO / "src" / "axiom_encode"
WORKFLOW = REPO / ".github" / "workflows" / "targeted-signed-reencode.yml"

#: Untyped ``*issues.append(...)`` sites in ``src/axiom_encode``. Lower this
#: number when you convert sites to declared checks; never raise it. A new
#: validator issue is a ``ValidationIssue`` from a ``declare_check`` check.
#: (677 = 648 at this check's start plus 29 sites #1835 and #1989 added to
#: main before it merged.)
UNTYPED_ISSUE_APPEND_BASELINE = 677

#: ``echo ... >&2`` + ``exit 1`` refusals in the targeted re-encode workflow
#: that record no infra issue (the hardened steps after the signing supervisor
#: is provisioned, the protected encode step and the publish steps). A new
#: refusal adds ``python axiom-encode/scripts/infra_issue.py record
#: infra:<step> "<message>" || true`` between its ``echo`` and ``exit 1``.
BARE_WORKFLOW_REFUSAL_BASELINE = 42

#: Functions that build and return a ``ValidationIssue``.
TYPED_ISSUE_FACTORIES = frozenset(
    {
        "ValidationIssue",
        "_rules_engine_compile_failed_issue",
        "_rules_engine_compile_crashed_issue",
        "_out_of_scope_issue",
    }
)


def _issue_append_sites() -> tuple[int, int]:
    total = typed = 0
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
            ):
                continue
            target = node.func.value
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else ""
            )
            if not name.endswith("issues"):
                continue
            total += 1
            argument = node.args[0] if node.args else None
            if (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Name)
                and argument.func.id in TYPED_ISSUE_FACTORIES
            ):
                typed += 1
    return total, typed


def test_untyped_issue_append_sites_do_not_grow():
    total, typed = _issue_append_sites()
    untyped = total - typed
    assert untyped <= UNTYPED_ISSUE_APPEND_BASELINE, (
        f"{untyped} untyped `issues.append` sites (baseline "
        f"{UNTYPED_ISSUE_APPEND_BASELINE}). Emit new validator issues as "
        "`ValidationIssue(..., check=<declared check>)` instead of a bare string."
    )
    assert untyped == UNTYPED_ISSUE_APPEND_BASELINE, (
        f"Only {untyped} untyped sites remain: lower "
        "UNTYPED_ISSUE_APPEND_BASELINE to that number so the ratchet holds."
    )


def test_every_declared_check_explains_itself():
    registry = load_checks()
    assert len(registry) >= 29
    for check in registry.values():
        assert check_definition_problems(check) == [], check.id
        # The fix is shown to the model and to people: it must say what to do.
        assert len(check.fix) >= 40, check.id
        assert len(check.why) >= 40, check.id


def test_declaring_a_check_without_fix_or_why_fails():
    with pytest.raises(ValueError, match="`fix` is empty"):
        declare_check(
            "test:missing/fix",
            title="t",
            requires="r",
            why="w",
            fix="",
            cause="encoder_output",
            gate="ci",
            site="tests/test_checks.py:test",
        )
    with pytest.raises(ValueError, match="`why` is empty"):
        declare_check(
            "test:missing/why",
            title="t",
            requires="r",
            why=" ",
            fix="f",
            cause="encoder_output",
            gate="ci",
            site="tests/test_checks.py:test",
        )
    with pytest.raises(ValueError, match="cause"):
        declare_check(
            "test:bad/cause",
            title="t",
            requires="r",
            why="w",
            fix="f",
            cause="model",  # type: ignore[arg-type]
            gate="ci",
            site="tests/test_checks.py:test",
        )
    assert checks.get_check("test:missing/fix") is None


def test_check_sites_name_existing_code():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for check in load_checks().values():
        path, _, function = check.site.partition(":")
        if path.startswith(".github/"):
            assert f"- name: {function}\n" in workflow, check.id
            continue
        source = (REPO / "src" / path).read_text(encoding="utf-8")
        assert re.search(rf"^\s*def {re.escape(function)}\(", source, re.M), check.id


def test_check_modules_list_every_declaring_module():
    declaring = set()
    for path in SRC.rglob("*.py"):
        if path.name == "checks.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "declare_check(" in text or "def register_checks(" in text:
            relative = path.relative_to(SRC.parent).with_suffix("")
            declaring.add(".".join(relative.parts))
    assert declaring <= set(checks.CHECK_MODULES)


def test_committed_registry_matches_the_code():
    committed = (SRC / checks.COMMITTED_REGISTRY_PATH).read_text(encoding="utf-8")
    assert committed == checks.committed_registry_text(), (
        "Regenerate src/axiom_encode/data/check_registry.json: "
        "uv run python -c 'from axiom_encode.checks import committed_registry_text;"
        ' print(committed_registry_text(), end="")\' '
        "> src/axiom_encode/data/check_registry.json"
    )


def test_registry_document_is_versioned_and_digested():
    document = checks.registry_document(encoder_version="9.9.9")
    assert document["schema"] == "axiom-encode/check-registry/v1"
    assert document["encoder_version"] == "9.9.9"
    assert document["registry_sha256"] == checks.registry_digest()
    ids = [check["id"] for check in document["checks"]]
    assert ids == sorted(ids)
    tests_checks = [i for i in ids if i.startswith("complete-source-unit:tests/")]
    assert len(tests_checks) == 10
    for required in (
        "complete-source-unit:structure/branch-unencoded",
        "complete-source-unit:deferral/reason-imprecise",
        "complete-source-unit:numeric-recall/value-unnamed",
        "complete-source-unit:formula-output/branch-without-output",
        "complete-source-unit:source-explicit-conditions/gates-collapsed",
        "proof:source-evidence/not-found",
        "compile:rules-engine/failed",
        "infra:validate-atomic-source-inputs",
    ):
        assert required in ids


def test_checks_cli_prints_the_registry(tmp_path):
    out = tmp_path / "checks.json"
    subprocess.run(
        [sys.executable, "-m", "axiom_encode.cli", "checks", "--json", "--out", out],
        check=True,
        cwd=REPO,
    )
    document = json.loads(out.read_text(encoding="utf-8"))
    from axiom_encode import __version__

    assert document["encoder_version"] == __version__
    assert {check["id"] for check in document["checks"]} == set(load_checks())


def _issue() -> ValidationIssue:
    load_checks()
    return ValidationIssue(
        "[complete-source-unit:tests] Principal output `x` is never asserted by a "
        "companion test.",
        check="complete-source-unit:tests/output-unasserted",
        subject=IssueSubject(citation="us/statutes/7/2014/e", rule="x"),
        expected="at least one asserting companion case",
        found="no case asserts it",
    )


def test_typed_issue_is_its_message():
    issue = _issue()
    plain = str(issue)
    assert issue == plain
    assert hash(issue) == hash(plain)
    assert type(str(issue)) is str
    assert list(dict.fromkeys([issue, plain])) == [issue]
    assert json.dumps([issue]) == json.dumps([plain])
    assert issue.startswith("[complete-source-unit:tests]")


def test_typed_issue_fingerprint_is_stable_and_subject_bound():
    issue = _issue()
    same = _issue()
    assert issue.fingerprint == same.fingerprint
    assert len(issue.fingerprint) == 16
    other = ValidationIssue(
        str(issue),
        check=issue.check,
        subject=IssueSubject(citation="us/statutes/7/2014/e", rule="y"),
        expected=issue.expected,
    )
    assert other.fingerprint != issue.fingerprint
    # The found value varies between tries; the fingerprint does not.
    assert (
        ValidationIssue(
            str(issue),
            check=issue.check,
            subject=issue.subject,
            expected=issue.expected,
            found="something else",
        ).fingerprint
        == issue.fingerprint
    )


def test_typed_fields_survive_rewrites_and_pickling():
    issue = _issue()
    bounded = bounded_validation_retry_feedback_item("  " + issue + "  ")
    assert isinstance(bounded, str) and not isinstance(bounded, ValidationIssue)
    bounded = bounded_validation_retry_feedback_item(issue)
    assert isinstance(bounded, ValidationIssue)
    assert bounded.check == issue.check
    prefixed = retype(issue, f"policies/x.yaml: ci: {issue}")
    assert isinstance(prefixed, ValidationIssue)
    assert prefixed.fingerprint == issue.fingerprint
    mapped = map_issue_text(issue, lambda text: text.replace("asserts", "ASSERTS"))
    assert mapped.found == "no case ASSERTS it"
    restored = pickle.loads(pickle.dumps(issue))
    assert isinstance(restored, ValidationIssue)
    assert restored.subject == issue.subject
    assert map_issue_text("plain", str.upper) == "PLAIN"


def test_issue_record_is_a_superset_of_v1():
    record = issue_record(_issue(), gate="ci")
    v1_keys = {
        "gate",
        "kind",
        "message",
        "category",
        "locator",
        "line",
        "value",
        "clause",
    }
    assert v1_keys <= set(record)
    assert (
        record["kind"]
        == record["check"]
        == ("complete-source-unit:tests/output-unasserted")
    )
    assert record["category"] == "complete-source-unit:tests"
    assert record["locator"] == "rule:x"
    assert record["cause"] == "encoder_output"
    assert record["subject"] == {"citation": "us/statutes/7/2014/e", "rule": "x"}
    untyped = issue_record(
        "Ungrounded generated numeric literal: 7 does not appear", gate="ci"
    )
    assert untyped["check"] is None and untyped["kind"]


def test_typed_issue_document_is_bounded_and_self_describing():
    issues = [_issue()] + [f"untyped {index}" for index in range(5)]
    document = typed_issue_document([("ci", issues)], max_issues=3)
    assert document["schema"] == ISSUE_SCHEMA_VERSION
    assert document["total"] == 6 and document["typed"] == 1
    assert document["truncated"] == 3
    assert set(document["checks"]) == {"complete-source-unit:tests/output-unasserted"}
    assert document["check_registry_sha256"] == checks.registry_digest()
    assert typed_issue_document([("ci", ["only untyped"])]) is None


def test_stored_documents_retype_their_issues():
    issue = _issue()
    document = json.loads(json.dumps(typed_issue_document([(None, [issue])])))
    retyped = issues_from_document([str(issue), "other"], document)
    assert isinstance(retyped[0], ValidationIssue)
    assert retyped[0].fingerprint == issue.fingerprint
    assert not isinstance(retyped[1], ValidationIssue)
    assert issues_from_document(["x"], None) == ["x"]


def test_compile_issues_name_the_module_by_its_repo_relative_path(tmp_path):
    from axiom_encode.harness.validator_pipeline import (
        _compile_issue_module,
        _rules_engine_compile_crashed_issue,
        _rules_engine_compile_failed_issue,
    )

    def failed(root: Path, relative: str) -> ValidationIssue:
        module = _compile_issue_module(root / relative, (root, None))
        return _rules_engine_compile_failed_issue("invalid field at line 12", module)

    first = failed(tmp_path / "stage-a", "us/statutes/26/1/a.yaml")
    other = failed(tmp_path / "stage-a", "us/statutes/26/2/a.yaml")
    restaged = failed(tmp_path / "stage-b", "us/statutes/26/1/a.yaml")
    assert first.subject.module == "us/statutes/26/1/a.yaml"
    # Same basename, different modules: distinct identities.
    assert other.fingerprint != first.fingerprint
    # Same module in another staging directory: the same identity.
    assert restaged.fingerprint == first.fingerprint
    crashed = _rules_engine_compile_crashed_issue(
        RuntimeError("boom"),
        _compile_issue_module(
            tmp_path / "stage-c" / "us/statutes/26/1/a.yaml",
            (tmp_path / "elsewhere", tmp_path / "stage-c"),
        ),
    )
    assert crashed.subject.module == "us/statutes/26/1/a.yaml"


def test_long_issue_keeps_its_check_through_a_bundle_round_trip(tmp_path):
    from types import SimpleNamespace

    from axiom_encode.cli import (
        _emit_final_rejected_candidate,
        _load_initial_validation_retry_feedback,
    )
    from axiom_encode.harness.validator_pipeline import (
        _rules_engine_compile_failed_issue,
    )

    issue = _rules_engine_compile_failed_issue(
        "compiler diagnosis " + "x" * 2100, "statutes/26/1.yaml"
    )
    assert len(issue) > checks.MAX_RECORDED_MESSAGE_CHARS
    output_root = tmp_path / "out"
    output_file = output_root / "codex-terra" / "statutes/26/1.yaml"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("format: rulespec/v1\nrules: []\n")
    output_file.with_suffix(".test.yaml").write_text("[]\n")
    destination = tmp_path / "failed-encode"
    _emit_final_rejected_candidate(
        SimpleNamespace(runner="codex-terra", output_file=str(output_file)),
        output_root=output_root,
        destination=destination,
        citation="26 USC 1",
        validation_issues=(issue,),
        attempt_count=1,
    )
    metadata = json.loads((destination / "issues.json").read_text())
    record = metadata["typed_issues"]["issues"][0]
    assert record["message_truncated"] == len(issue) - checks.MAX_RECORDED_MESSAGE_CHARS
    assert record["message_sha256"] == checks.message_digest(str(issue))

    replayed = _load_initial_validation_retry_feedback(
        SimpleNamespace(
            repair_candidate_root=destination,
            repair_candidate_path=Path("statutes/26/1.yaml"),
            repair_candidate_rulespec_sha256=metadata["rulespec_sha256"],
            repair_candidate_tests_sha256=metadata["tests_sha256"],
            citation="26 USC 1",
        )
    )
    typed = [item for item in replayed if isinstance(item, ValidationIssue)]
    assert [item.check for item in typed] == ["compile:rules-engine/failed"]
    assert typed[0].fingerprint == issue.fingerprint
    # A different message with the same stored prefix does not match it.
    stranger = str(issue)[: checks.MAX_RECORDED_MESSAGE_CHARS] + "y" * 200
    assert not isinstance(
        issues_from_document([stranger], metadata["typed_issues"])[0],
        ValidationIssue,
    )


def test_typed_issue_document_bounds_its_whole_serialized_size():
    ci_checks = [check for check in load_checks().values() if check.gate == "ci"]
    issues = [
        ValidationIssue(
            f"[{check.category}] issue {index} " + "z" * 600,
            check=check.id,
            subject=IssueSubject(citation="us/statutes/26/1", rule=f"r{index}"),
            expected="e" * 300,
            found="f" * 300,
        )
        for index in range(200)
        for check in [ci_checks[index % len(ci_checks)]]
    ]
    document = typed_issue_document([("ci", issues)])
    size = len(json.dumps(document, ensure_ascii=False).encode("utf-8"))
    assert size <= checks.MAX_RECORDED_ISSUES_JSON_BYTES
    assert document["truncated"] == 200 - len(document["issues"]) > 0
    assert set(document["checks"]) == {r["check"] for r in document["issues"]}


def test_failure_counts_are_keyed_by_check_id():
    summary = summarize_validation_failures(
        [("ci", [_issue(), "Ungrounded generated numeric literal: 7 does not appear"])]
    )
    counts = summary["validation_failure_counts"]
    assert counts["ci:complete-source-unit:tests/output-unasserted"] == 1
    assert sum(counts.values()) == 2
    typed_failure = summary["validation_failures"][0]
    assert typed_failure["check"] == "complete-source-unit:tests/output-unasserted"
    assert summary["validation_issues"]["typed"] == 1
    assert "validation_issues" not in summarize_validation_failures(
        [("ci", ["plain issue"])]
    )


def test_retry_feedback_adds_each_failed_checks_fix():
    plain = _format_validation_retry_feedback(["plain issue"])
    assert "CHECK FIXES" not in plain
    typed = _format_validation_retry_feedback([_issue(), "plain issue"])
    assert "=== BEGIN CHECK FIXES (declared by each failed check) ===" in typed
    fix = checks.get_check("complete-source-unit:tests/output-unasserted").fix
    assert f"  Fix: {fix}" in typed
    assert (
        "  - rule:x; expected at least one asserting companion case; "
        "found no case asserts it"
    ) in typed
    # The validator messages themselves render exactly as before.
    assert json.dumps(str(_issue()), ensure_ascii=False) in typed


def test_iteration_payload_keeps_old_shape_without_typed_issues():
    bare = iteration_payload(
        Iteration(attempt=1, duration_ms=1, errors=[IterationError("validation", "m")])
    )
    assert set(bare["errors"][0]) == {
        "error_type",
        "message",
        "variable",
        "fix_applied",
    }
    document = typed_issue_document([(None, [_issue()])])
    typed = iteration_payload(
        Iteration(
            attempt=1,
            duration_ms=1,
            errors=[IterationError("validation", "m", issues=document)],
        )
    )
    assert typed["errors"][0]["issues"] == document


CITATION = "de/guidance/bzst-dakg-2025/a-19-2/document-1"


def test_complete_source_unit_emits_typed_issues():
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [{"name": "flag", "dtype": "Boolean"}],
        "rules": [
            {
                "name": "eligible",
                "kind": "derived",
                "dtype": "Judgment",
                "entity": "Child",
                "period": "Day",
                "source": CITATION + "(1)",
                "versions": [{"effective_from": "2025-01-01", "formula": "flag"}],
            }
        ],
    }
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        "(1) Das Kind ist berechtigt.",
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="de-DE"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    typed = [issue for issue in result.issues if isinstance(issue, ValidationIssue)]
    assert typed, result.issues
    no_suite = [i for i in typed if i.check == "complete-source-unit:tests/no-suite"]
    assert no_suite
    assert str(no_suite[0]) == (
        "[complete-source-unit:tests] Complete source-unit controls require "
        "a companion test suite covering outputs, branches, boundaries, "
        "exceptions, and rounding rules."
    )
    for issue in typed:
        assert issue.definition() is not None
        assert issue.startswith(f"[{issue.definition().category}]")


def test_workflow_refusals_use_declared_infra_checks():
    text = WORKFLOW.read_text(encoding="utf-8")
    used = set(re.findall(r"infra_issue\.py record (infra:[a-z0-9-]+) ", text))
    assert used, "the workflow emits no infra issues"
    assert used <= set(load_checks())
    lines = text.splitlines()
    # A record call sits between the refusal's own echo and exit; the refusal
    # never depends on the helper.
    record_re = re.compile(
        r"^(\s*)python axiom-encode/scripts/infra_issue\.py record infra:[a-z0-9-]+ "
        r'(".*") \|\| true$'
    )
    for index, line in enumerate(lines):
        match = record_re.match(line)
        if match is None:
            continue
        indent, message = match.groups()
        assert lines[index - 1] == f"{indent}echo {message} >&2", index + 1
        assert lines[index + 1] == f"{indent}exit 1", index + 1
    # Steps after the signing supervisor is provisioned run a trusted
    # interpreter with ``-I``; the PATH ``python`` helper stays out of them.
    supervisor = text.index("      - name: Provision protected signing supervisor\n")
    assert "infra_issue.py" not in text[supervisor:]
    # Each record names the check of the step it is in.
    step = None
    for line in lines:
        if line.startswith("      - name: "):
            step = line.removeprefix("      - name: ")
        for check_id in re.findall(r"infra_issue\.py record (infra:[a-z0-9-]+) ", line):
            assert load_checks()[check_id].site.endswith(f":{step}"), line
    bare = sum(
        1
        for previous, line in zip(lines, lines[1:])
        if line.strip() == "exit 1" and re.search(r"echo .* >&2$", previous)
    ) + sum(1 for line in lines if re.search(r"echo .* >&2; exit 1", line))
    assert bare <= BARE_WORKFLOW_REFUSAL_BASELINE, (
        f"{bare} bare `echo ... >&2; exit 1` refusals (baseline "
        f"{BARE_WORKFLOW_REFUSAL_BASELINE}); record each refusal with "
        "scripts/infra_issue.py record."
    )
    assert bare == BARE_WORKFLOW_REFUSAL_BASELINE, (
        "Lower BARE_WORKFLOW_REFUSAL_BASELINE to the new count."
    )


def test_infra_issue_script_emits_the_envelope(tmp_path):
    issues_file = tmp_path / "issues.jsonl"
    summary = tmp_path / "summary.md"
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")
    }
    env.update(
        AXIOM_INFRA_ISSUES_FILE=str(issues_file),
        GITHUB_STEP_SUMMARY=str(summary),
        GITHUB_RUN_ID="123",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "infra_issue.py"),
            "record",
            "infra:validate-atomic-source-inputs",
            "source bundles require legacy replacements to merge first",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.startswith(
        "::error title=infra%3Avalidate-atomic-source-inputs::"
    )
    assert "%0AFix: " in completed.stdout
    record = json.loads(issues_file.read_text(encoding="utf-8"))
    assert record["schema"] == ISSUE_SCHEMA_VERSION
    assert record["check"] == "infra:validate-atomic-source-inputs"
    assert record["cause"] == "infra"
    assert record["run"] == {"run_id": "123"}
    assert record["fix"] == checks.get_check("infra:validate-atomic-source-inputs").fix
    assert "Fix:" in summary.read_text(encoding="utf-8")


def test_infra_issue_script_rejects_unknown_usage():
    completed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infra_issue.py"), "record", "nope"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2


def test_boundary_fingerprint_survives_repair_of_another_boundary(tmp_path):
    from axiom_encode.harness import validator_pipeline as vp
    from tests.test_source_completeness_bare_markers import CITATION, _source_rule

    first = "The payment is at most 50 dollars, calculated as income divided by 2."
    second = "The bonus is at most 70 dollars, calculated as wages divided by 4."
    source = "a\n" + first + "\nb\n" + second + "\n"
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": "income", "dtype": "Money"},
            {"name": "wages", "dtype": "Money"},
        ],
        "rules": [
            _source_rule(
                "payment",
                "if income <= income_limit: income / divisor else: 0",
                first,
                f"{CITATION}(a)",
                dtype="Money",
            ),
            _source_rule(
                "bonus",
                "if wages <= wage_limit: wages / wage_divisor else: 0",
                second,
                f"{CITATION}(b)",
                dtype="Money",
            ),
        ],
    }
    for name, number, proof, branch in (
        ("income_limit", 50, "at most 50 dollars", "a"),
        ("divisor", 2, "income divided by 2", "a"),
        ("wage_limit", 70, "at most 70 dollars", "b"),
        ("wage_divisor", 4, "wages divided by 4", "b"),
    ):
        rule = _source_rule(
            name, str(number), proof, f"{CITATION}({branch})", dtype="Money"
        )
        rule["kind"] = "parameter"
        payload["rules"].append(rule)
    pipeline = vp.ValidatorPipeline(
        policy_repo_path=tmp_path / "nonexistent-rulespec",
        axiom_rules_path=tmp_path / "nonexistent-engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )

    def boundary_issue(pairs):
        cases = [
            {
                "period": "2026-07-01",
                "input": {"income": income, "wages": wages},
                "output": {
                    "payment": income / 2 if income <= 50 else 0,
                    "bonus": wages / 4 if wages <= 70 else 0,
                },
            }
            for income, wages in pairs
        ]
        issues = pipeline._complete_source_unit_issues(
            yaml.safe_dump(payload),
            validation_source_texts={CITATION: source},
            proof_source_texts={CITATION: source},
            test_cases=cases,
        )
        (issue,) = [
            issue
            for issue in issues
            if isinstance(issue, ValidationIssue)
            and issue.check == "complete-source-unit:tests/boundary-unexercised"
        ]
        return issue

    both_missing = boundary_issue([(0, 0), (10, 0), (99, 0)])
    second_repaired = boundary_issue([(0, 69), (10, 70), (99, 71), (5, 0)])
    assert "=70" in both_missing.found and "=70" not in second_repaired.found
    # Branch (a)'s boundary is still unresolved: the same identity.
    assert second_repaired.subject == both_missing.subject
    assert second_repaired.fingerprint == both_missing.fingerprint
