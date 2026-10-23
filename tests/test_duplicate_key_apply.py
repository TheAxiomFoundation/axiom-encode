"""Duplicate mapping keys are rejected before any load-and-dump rewrite.

``yaml.safe_load`` keeps only the last of two equal keys. Apply stamping, the
apply-time and eval-time repairs, materialization and the retry overlay all
write the generated files back from a loaded payload, so without a raw check
the first ``rules`` block, rule field or test output would silently vanish
while the engine, which rejects duplicate keys, compiles what is left.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.cli import (
    _can_attempt_apply,
    _generated_rulespec_yaml_nonparser_issue,
    _stamp_generated_source_attestation_for_apply,
    _validate_generated_encoding_in_policy_overlay,
)
from axiom_encode.harness.evals import (
    ValidationRetryCandidate,
    _apply_generated_eval_repairs,
    _materialize_eval_artifact,
    _normalize_main_eval_content,
    _overlay_validation_retry_candidate,
)
from axiom_encode.harness.validator_pipeline import _safe_load_unique_keys
from axiom_encode.yaml_serialization import (
    find_duplicate_mapping_key_issues,
    safe_dump_yaml,
)
from tests.test_cli import (
    _complete_source_attestation,
    _write_test_context_manifest,
)

APPLY_CITATION = "us/statute/26/1"

BASE = """format: rulespec/v1
module:
  summary: Fictional lantern review rates.
rules:
  - name: first_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.1'
  - name: second_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
"""
SHADOW_RULES = """  - name: shadow_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.3'
"""


def _once(old, new):
    assert BASE.count(old) >= 1, old
    return BASE.replace(old, new, 1)


DUPLICATES = [
    pytest.param(
        BASE + "rules:\n" + SHADOW_RULES,
        "document root has duplicate key `rules` at line 17, column 1 "
        "(first at line 4, column 1)",
        id="root-rules",
    ),
    pytest.param(
        _once("format: rulespec/v1\n", "format: rulespec/v1\nformat: rulespec/v1\n"),
        "document root has duplicate key `format` at line 2, column 1",
        id="root-format",
    ),
    pytest.param(
        _once("module:\n", "imports: []\nimports: []\nmodule:\n"),
        "document root has duplicate key `imports` at line 3, column 1",
        id="root-imports",
    ),
    pytest.param(
        _once("    kind: parameter\n", "    kind: parameter\n    kind: derived\n"),
        "rules[0] has duplicate key `kind` at line 7, column 5",
        id="rule-mapping",
    ),
    pytest.param(
        _once(
            "        formula: '0.1'\n", "        formula: '0.1'\n        formula: '9'\n"
        ),
        "rules[0].versions[0] has duplicate key `formula` at line 11, column 9",
        id="rule-version",
    ),
    pytest.param(
        _once("  summary:", "  summary: Discarded summary.\n  summary:"),
        "module has duplicate key `summary` at line 4, column 3",
        id="module-field",
    ),
    pytest.param(
        BASE + '"rules":\n' + SHADOW_RULES,
        "document root has duplicate key `rules` at line 17, column 1",
        id="double-quoted-root-key",
    ),
    pytest.param(
        BASE + "'rules':\n" + SHADOW_RULES,
        "document root has duplicate key `rules` at line 17, column 1",
        id="single-quoted-root-key",
    ),
    pytest.param(
        BASE + "? rules\n:\n" + SHADOW_RULES,
        "document root has duplicate key `rules` at line 17, column 3",
        id="explicit-root-key",
    ),
    pytest.param(
        BASE + "!!str rules:\n" + SHADOW_RULES,
        "document root has duplicate key `rules` at line 17, column 1",
        id="string-tagged-root-key",
    ),
    pytest.param(
        _once("    kind: parameter\n", "    kind: parameter\n    'kind': derived\n"),
        "rules[0] has duplicate key `kind` at line 7, column 5",
        id="quoted-rule-key",
    ),
    pytest.param(
        _once("    dtype: Float\n", '    dtype: Float\n    "dtype": Integer\n'),
        "rules[0] has duplicate key `dtype` at line 8, column 5",
        id="double-quoted-rule-key",
    ),
    pytest.param(
        _once(
            "      - effective_from: '2024-01-01'\n        formula: '0.1'\n",
            "      - {effective_from: '2024-01-01', formula: '0.1', formula: '9'}\n",
        ),
        "rules[0].versions[0] has duplicate key `formula` at line 9, column 56 "
        "(first at line 9, column 40)",
        id="flow-mapping",
    ),
    pytest.param(
        _once("module:\n", "x-key: &key rules\nmodule:\n") + "*key :\n" + SHADOW_RULES,
        "document root has duplicate key `rules` at line 18, column 1 "
        "(first at line 5, column 1)",
        id="aliased-root-key",
    ),
]

# Mappings whose keys PyYAML loads as distinct, or which use merge semantics.
DISTINCT = [
    pytest.param("values:\n  1: a\n  '1': b\n", id="int-and-string"),
    pytest.param("a:\n  x: 1\nb:\n  x: 1\n", id="same-key-in-sibling-mappings"),
    pytest.param("d: &d {a: 1}\nm:\n  <<: *d\n  a: 2\n", id="merge-override"),
    pytest.param(
        "d: &d {a: 1}\ne: &e {b: 1}\nm:\n  <<: *d\n  <<: *e\n", id="two-merges"
    ),
    pytest.param("d: &d {a: 1}\nm:\n  <<: *d\n  '<<': 2\n", id="quoted-merge-name"),
    pytest.param("a: &x\n  b: *x\n  c: 1\n", id="recursive-alias"),
    pytest.param("rules: rules\nrule: rules\n", id="key-equal-to-a-value"),
    pytest.param("", id="empty-document"),
    pytest.param("- a: 1\n- a: 1\n", id="same-key-in-list-items"),
]


@pytest.mark.parametrize(("content", "diagnostic"), DUPLICATES)
def test_raw_guard_reports_duplicate_keys_anywhere(content, diagnostic):
    issues = find_duplicate_mapping_key_issues(content)

    assert any(diagnostic in issue for issue in issues), issues
    # The engine rejects these files, and safe_load would collapse them.
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        _safe_load_unique_keys(content)


def _engine_compile(real_engine, root: Path, content: str):
    root = root.resolve() / "rulespec-us"
    module = root / "us/statutes/26/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(content)
    return subprocess.run(
        [
            str(real_engine.binary),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(root.parent / "compiled.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_engine_compiles_the_clean_module(real_engine, tmp_path):
    compiled = _engine_compile(real_engine, tmp_path, BASE)

    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.parametrize(("content", "diagnostic"), DUPLICATES)
def test_engine_rejects_raw_duplicates_that_a_rewrite_would_hide(
    real_engine, tmp_path, content, diagnostic
):
    """The defect's premise: only the load-and-dump rewrite made these compile."""

    raw = _engine_compile(real_engine, tmp_path / "raw", content)
    rewritten = _engine_compile(
        real_engine,
        tmp_path / "rewritten",
        safe_dump_yaml(yaml.safe_load(content), sort_keys=False, allow_unicode=True),
    )

    assert raw.returncode != 0
    assert "duplicate entry with key" in raw.stderr
    assert rewritten.returncode == 0, rewritten.stderr
    assert find_duplicate_mapping_key_issues(content)


@pytest.mark.parametrize("content", DISTINCT)
def test_raw_guard_accepts_keys_that_load_distinctly(content):
    assert find_duplicate_mapping_key_issues(content) == []


@pytest.mark.parametrize(
    ("first", "second"),
    [("1", "0x1"), ("1", "true"), ("1", "1.0"), ("true", "yes"), ("~", "null")],
)
def test_raw_guard_reports_spellings_that_load_as_one_key(first, second):
    content = f"values:\n  {first}: kept by nothing\n  {second}: kept by safe_load\n"
    assert len(yaml.safe_load(content)["values"]) == 1

    assert find_duplicate_mapping_key_issues(content) == [
        f"values has key `{second}` at line 3, column 3, which loads as the same "
        f"key as `{first}` at line 2, column 3."
    ]


def test_raw_guard_scans_a_recursive_alias_target_once():
    content = "a: &x\n  self: *x\n  c: 1\n  c: 2\nb: *x\n"

    assert find_duplicate_mapping_key_issues(content) == [
        "a has duplicate key `c` at line 4, column 3 (first at line 3, column 3)."
    ]


def test_raw_guard_bounds_its_report():
    content = "m:\n" + "".join(f"  k{'x' * 200}: {i}\n" for i in range(14))

    issues = find_duplicate_mapping_key_issues(content)

    assert len(issues) == 11
    assert issues[-1] == "3 more duplicate mapping key(s) not shown."
    assert all(len(issue) < 300 for issue in issues)


def test_raw_guard_keeps_parser_errors():
    with pytest.raises(yaml.YAMLError):
        find_duplicate_mapping_key_issues("rules: [\n")


_KEY_SPELLINGS = (
    "a",
    '"a"',
    "'a'",
    "!!str a",
    "b",
    "1",
    "'1'",
    "0x1",
    "+1",
    "true",
    "yes",
    "1.0",
    "~",
    "null",
    "2024-01-01",
    "'2024-01-01'",
)
_ENTRIES = st.recursive(
    st.just(None),
    lambda children: st.lists(
        st.tuples(st.sampled_from(_KEY_SPELLINGS), children), min_size=1, max_size=5
    ),
    max_leaves=12,
)


def _render(entries, indent=0):
    lines = []
    for key, value in entries:
        if value is None:
            lines.append(" " * indent + f"{key}: 0")
        else:
            lines.append(" " * indent + f"{key}:")
            lines.extend(_render(value, indent + 2))
    return lines


def _raw_entry_count(node):
    if isinstance(node, yaml.MappingNode):
        return len(node.value) + sum(_raw_entry_count(v) for _k, v in node.value)
    return 0


def _loaded_entry_count(value):
    if isinstance(value, dict):
        return len(value) + sum(_loaded_entry_count(v) for v in value.values())
    return 0


@settings(max_examples=400, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(_KEY_SPELLINGS), _ENTRIES), min_size=1))
def test_raw_guard_matches_the_unique_key_loader_and_safe_load(entries):
    """Invariant: the guard reports exactly the documents safe_load collapses.

    Differential against the validator's constructor-based unique-key loader,
    the implementation that already rejects these files at compile time, and
    against entry counts: with no report, loading keeps every raw entry.
    """

    content = "\n".join(_render(entries)) + "\n"
    issues = find_duplicate_mapping_key_issues(content)
    try:
        _safe_load_unique_keys(content)
    except yaml.constructor.ConstructorError as exc:
        assert "duplicate key" in str(exc)
        loader_rejects = True
    else:
        loader_rejects = False

    assert bool(issues) == loader_rejects
    collapsed = _raw_entry_count(yaml.compose(content)) != _loaded_entry_count(
        yaml.safe_load(content)
    )
    assert bool(issues) == collapsed


def _candidate(tmp_path, content, test_content=None):
    output_root = tmp_path / "out"
    generated = output_root / "codex-test-model/statutes/26/1.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text(content)
    if test_content is not None:
        generated.with_name("1.test.yaml").write_text(test_content)
    result = SimpleNamespace(
        output_file=str(generated),
        runner="codex-test-model",
        backend="codex",
        source_attestation=_complete_source_attestation(APPLY_CITATION),
        context_manifest_file=str(
            _write_test_context_manifest(
                tmp_path / "context", citation_path=APPLY_CITATION
            )
        ),
    )
    return result, generated, output_root


def test_real_stamping_keeps_every_rule_without_duplicates(tmp_path):
    result, generated, _ = _candidate(tmp_path, BASE)

    _stamp_generated_source_attestation_for_apply(result, generated)

    stamped = yaml.safe_load(generated.read_text())
    assert [rule["name"] for rule in stamped["rules"]] == ["first_rate", "second_rate"]
    assert stamped["module"]["source_verification"]["corpus_citation_path"] == (
        APPLY_CITATION
    )


@pytest.mark.parametrize(("content", "diagnostic"), DUPLICATES)
def test_real_stamping_rejects_duplicate_keys_before_rewrite(
    tmp_path, content, diagnostic
):
    result, generated, _ = _candidate(tmp_path, content)
    original = generated.read_bytes()

    with pytest.raises(RuntimeError) as exc_info:
        _stamp_generated_source_attestation_for_apply(result, generated)

    assert str(exc_info.value).startswith(
        "Cannot apply RuleSpec YAML with duplicate mapping keys: "
    )
    assert diagnostic in str(exc_info.value)
    assert generated.read_bytes() == original


@pytest.mark.parametrize(("content", "diagnostic"), DUPLICATES)
@pytest.mark.parametrize(
    "error",
    [
        "Generated RuleSpec failed compile validation",
        "Generated RuleSpec failed CI validation",
    ],
)
def test_apply_preflight_blocks_duplicate_keys_before_repairs(
    tmp_path, content, diagnostic, error
):
    result, generated, output_root = _candidate(tmp_path, content)
    result.success = False
    result.error = error
    assert _can_attempt_apply(result)
    original = generated.read_bytes()

    issue = _generated_rulespec_yaml_nonparser_issue(
        result, generated, output_root=output_root
    )

    assert issue is not None
    assert issue.startswith("Cannot apply RuleSpec YAML with duplicate mapping keys: ")
    assert diagnostic in issue
    assert generated.read_bytes() == original


@pytest.mark.parametrize(("content", "diagnostic"), DUPLICATES)
def test_apply_overlay_blocks_duplicate_keys_with_real_stamping(
    tmp_path, content, diagnostic
):
    result, generated, output_root = _candidate(tmp_path, content)
    original = generated.read_bytes()

    ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
        result,
        output_root=output_root,
        policy_repo_path=tmp_path / "rulespec-us/us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=MagicMock(),
        validate_dependents=False,
    )

    assert not ok
    assert any(diagnostic in issue for issue in issues), issues
    assert supplemental == {}
    assert generated.read_bytes() == original


TESTS = """- name: first_rate_applies
  period: 2024-01-01
  input: {}
  output:
    first_rate: 0.1
"""
DUPLICATE_TESTS = TESTS + "  output:\n    second_rate: 0.2\n"


def test_apply_preflight_blocks_companion_test_duplicate_keys(tmp_path):
    result, generated, output_root = _candidate(tmp_path, BASE, DUPLICATE_TESTS)
    result.success = False
    result.error = "Generated RuleSpec failed CI validation"
    test_file = generated.with_name("1.test.yaml")

    issue = _generated_rulespec_yaml_nonparser_issue(
        result, generated, output_root=output_root
    )

    assert issue == (
        "Cannot apply companion test YAML with duplicate mapping keys: "
        "document root[0] has duplicate key `output` at line 6, column 3 "
        "(first at line 4, column 3)."
    )
    assert generated.read_text() == BASE
    assert test_file.read_text() == DUPLICATE_TESTS


def test_apply_preflight_passes_clean_module_and_tests(tmp_path):
    result, generated, output_root = _candidate(tmp_path, BASE, TESTS)

    assert (
        _generated_rulespec_yaml_nonparser_issue(
            result, generated, output_root=output_root
        )
        is None
    )


# Generated `module.inputs` makes materialization lift them through a load and
# dump, so these documents exercise the rewrite the guard must skip.
MISPLACED_INPUTS = """  inputs:
    - name: supplied_review_count
      entity: Person
      dtype: Count
      period: Month
"""


def _materialize(tmp_path, content, delivery, test_content=None):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    target = artifact_root / "section.yaml"
    workspace_root = None
    files = {"section.yaml": content}
    if test_content is not None:
        files["section.test.yaml"] = test_content
    if delivery == "response":
        response = content
    elif delivery == "bundle":
        response = "".join(
            f"=== FILE: {name} ===\n{text}" for name, text in files.items()
        )
    else:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        for name, text in files.items():
            (workspace_root / name).write_text(text, encoding="utf-8")
        response = "Wrote the requested files."
    assert _materialize_eval_artifact(
        response, target, workspace_root=workspace_root, artifact_root=artifact_root
    )
    test_target = target.with_name("section.test.yaml")
    return (
        target.read_text(encoding="utf-8"),
        test_target.read_text(encoding="utf-8") if test_target.exists() else None,
    )


@pytest.mark.parametrize(("content", "diagnostic"), DUPLICATES)
@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_keeps_raw_duplicate_keys(
    tmp_path, content, diagnostic, delivery
):
    content = content.replace("module:\n", "module:\n" + MISPLACED_INPUTS, 1)

    materialized, _tests = _materialize(tmp_path, content, delivery)

    assert materialized == content
    assert any(
        diagnostic.split(" at line")[0] in issue
        for issue in find_duplicate_mapping_key_issues(materialized)
    )


def test_normalization_still_refuses_non_rulespec_text_with_duplicates():
    """Keeping raw duplicate keys never lets text that is not RuleSpec through."""

    with pytest.raises(ValueError, match="must start with `format: rulespec/v1`"):
        _normalize_main_eval_content(
            "entries:\n  - note: first\n    note: second\n",
            target_path=Path("section.yaml"),
            single_amount_table_slice=False,
        )
    duplicate = BASE + "rules:\n" + SHADOW_RULES
    assert (
        _normalize_main_eval_content(
            "\n" + duplicate + "\n\n",
            target_path=Path("section.yaml"),
            single_amount_table_slice=False,
        )
        == duplicate
    )


@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_still_repairs_modules_without_duplicates(tmp_path, delivery):
    content = BASE.replace("module:\n", "module:\n" + MISPLACED_INPUTS, 1)

    materialized, _tests = _materialize(tmp_path, content, delivery)

    payload = yaml.safe_load(materialized)
    assert "inputs" not in payload["module"]
    assert payload["inputs"][0]["name"] == "supplied_review_count"


@pytest.mark.parametrize("delivery", ["bundle", "workspace"])
def test_materialization_keeps_raw_duplicate_test_keys(tmp_path, delivery):
    _main, tests = _materialize(tmp_path, BASE, delivery, DUPLICATE_TESTS)

    assert tests == DUPLICATE_TESTS


@pytest.mark.parametrize("delivery", ["bundle", "workspace"])
def test_materialization_still_normalizes_tests_without_duplicates(tmp_path, delivery):
    _main, tests = _materialize(tmp_path, BASE, delivery, TESTS)

    # Test normalization rewrites the cases (here, the period's spelling).
    assert tests != TESTS
    assert yaml.safe_load(tests)[0]["output"] == {"first_rate": 0.1}


@pytest.mark.parametrize(("content", "diagnostic"), DUPLICATES)
@pytest.mark.parametrize("duplicate_side", ["generated", "preserved"])
def test_retry_overlay_rejects_duplicate_keys_before_rewriting(
    tmp_path, content, diagnostic, duplicate_side
):
    generated = content if duplicate_side == "generated" else BASE
    preserved = content if duplicate_side == "preserved" else BASE
    target = tmp_path / "section.yaml"
    target.write_text(generated, encoding="utf-8")
    candidate = ValidationRetryCandidate(rulespec=preserved)

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target, artifact_root=tmp_path, candidate=candidate
        )

    assert str(exc_info.value).startswith(
        f"{duplicate_side} repair overlay has duplicate YAML mapping keys: "
    )
    assert diagnostic in str(exc_info.value)
    assert target.read_text(encoding="utf-8") == generated


@pytest.mark.parametrize("duplicate_side", ["generated", "preserved"])
def test_retry_overlay_rejects_duplicate_test_keys_before_rewriting(
    tmp_path, duplicate_side
):
    target = tmp_path / "section.yaml"
    target.write_text(BASE, encoding="utf-8")
    test_file = tmp_path / "section.test.yaml"
    generated_tests = DUPLICATE_TESTS if duplicate_side == "generated" else TESTS
    test_file.write_text(generated_tests, encoding="utf-8")
    candidate = ValidationRetryCandidate(
        rulespec=BASE,
        tests=DUPLICATE_TESTS if duplicate_side == "preserved" else TESTS,
    )

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target, artifact_root=tmp_path, candidate=candidate
        )

    assert str(exc_info.value).startswith(
        f"{duplicate_side} repair overlay tests have duplicate YAML mapping keys: "
    )
    assert "duplicate key `output`" in str(exc_info.value)
    assert target.read_text(encoding="utf-8") == BASE
    assert test_file.read_text(encoding="utf-8") == generated_tests


def test_retry_overlay_still_merges_without_duplicates(tmp_path):
    target = tmp_path / "section.yaml"
    target.write_text(BASE, encoding="utf-8")
    preserved = BASE.replace("module:\n", "imports:\n  - us:statutes/26/2\nmodule:\n")

    repairs = _overlay_validation_retry_candidate(
        target,
        artifact_root=tmp_path,
        candidate=ValidationRetryCandidate(rulespec=preserved),
    )

    assert repairs == ("import:us:statutes/26/2",)
    assert yaml.safe_load(target.read_text())["imports"] == ["us:statutes/26/2"]


UNUSED_IMPORT = "us:statutes/26/2#lantern_base_rate"
WITH_UNUSED_IMPORT = BASE.replace(
    "module:\n", f"imports:\n  - {UNUSED_IMPORT}\nmodule:\n", 1
)


def _eval_repairs(tmp_path, content, test_content=None):
    rules_file = tmp_path / "section.yaml"
    rules_file.write_text(content)
    if test_content is not None:
        rules_file.with_name("section.test.yaml").write_text(test_content)
    repairs = _apply_generated_eval_repairs(
        rulespec_file=rules_file,
        policy_repo_root=tmp_path,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        issues=[f"Unused import `{UNUSED_IMPORT}`"],
        local_corpus_release=MagicMock(),
    )
    return repairs, rules_file.read_text()


def test_eval_repairs_still_run_without_duplicates(tmp_path):
    repairs, repaired = _eval_repairs(tmp_path, WITH_UNUSED_IMPORT, TESTS)

    assert repairs == [f"unused_import:{UNUSED_IMPORT}"]
    assert UNUSED_IMPORT not in repaired


@pytest.mark.parametrize("duplicate_file", ["module", "tests"])
def test_eval_repairs_skip_files_with_duplicate_keys(tmp_path, duplicate_file):
    content = WITH_UNUSED_IMPORT + (
        "rules:\n" + SHADOW_RULES if duplicate_file == "module" else ""
    )
    test_content = DUPLICATE_TESTS if duplicate_file == "tests" else TESTS

    repairs, repaired = _eval_repairs(tmp_path, content, test_content)

    assert repairs == []
    assert repaired == content


def test_overlay_result_names_the_output_file_for_retry(tmp_path):
    """The overlay issue carries the file and the location the model must fix."""

    result, _generated, output_root = _candidate(
        tmp_path, BASE + "rules:\n" + SHADOW_RULES
    )

    ok, issues, _ = _validate_generated_encoding_in_policy_overlay(
        result,
        output_root=output_root,
        policy_repo_path=tmp_path / "rulespec-us/us",
        axiom_rules_path=Path(tmp_path / "axiom-rules-engine"),
        local_corpus_release=MagicMock(),
        validate_dependents=False,
    )

    assert not ok
    assert issues == [
        "statutes/26/1.yaml: Cannot apply RuleSpec YAML with duplicate mapping "
        "keys: document root has duplicate key `rules` at line 17, column 1 "
        "(first at line 4, column 1)."
    ]
