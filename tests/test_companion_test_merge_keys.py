"""Companion tests may override keys they merge in, as PyYAML allows (#1943).

Companion tests are read only by PyYAML-based harnesses, and each of them
applies YAML merge keys: an explicit key overrides a merged one, and an earlier
merge source wins over a later one. The validator's strict companion-test
reader must accept those documents and load them exactly as ``yaml.safe_load``
does, while still rejecting two equal keys written in one mapping. Modules are
different: the engine does not apply merge keys, so module loads stay as strict
as before.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import (
    _parse_deferred_output_review_contract_json,
    _required_deferred_output_contract_issues,
)
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    _safe_load_unique_keys,
)

MODULE = """format: rulespec/v1
module:
  summary: Standard utility allowance.
rules:
  - name: standard_utility_allowance
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 451
"""

# Five cases share one anchored input; the New York City case overrides a
# merged key with an explicit one. This is the shape of the rulespec-us
# companion test that issue #1943 reports.
NEW_YORK_CITY_OVERRIDE_TESTS = """- name: upstate household
  period: 2026-01
  input: &upstate_household
    household_size: 3
    resides_in_new_york_city: false
    heats_with_utilities: true
  output:
    standard_utility_allowance: 451
- name: larger upstate household
  period: 2026-01
  input:
    <<: *upstate_household
    household_size: 5
  output:
    standard_utility_allowance: 451
- name: New York City household
  period: 2026-01
  input:
    <<: *upstate_household
    resides_in_new_york_city: true
  output:
    standard_utility_allowance: 451
- name: household without utility heating
  period: 2026-01
  input:
    <<: *upstate_household
    heats_with_utilities: false
  output:
    standard_utility_allowance: 451
- name: single person
  period: 2026-01
  input:
    <<: *upstate_household
    household_size: 1
  output:
    standard_utility_allowance: 451
"""


def _companion_load(content: str):
    return validator_pipeline._safe_load_companion_test(content)


def _ci_issues(tmp_path: Path, monkeypatch, tests: str) -> list[str]:
    rules_file = tmp_path / "rulespec-us" / "policies" / "example" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(MODULE)
    rules_file.with_name("a.test.yaml").write_text(tests)
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    def fake_compile(rules_file, output_path):
        return (
            subprocess.CompletedProcess(["axiom-rules-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        )

    monkeypatch.setattr(pipeline, "_compile_rulespec_to_artifact", fake_compile)
    return list(pipeline._run_rulespec_ci(rules_file).issues)


def test_ci_reads_a_companion_test_whose_case_overrides_a_merged_key(
    tmp_path, monkeypatch
):
    issues = _ci_issues(tmp_path, monkeypatch, NEW_YORK_CITY_OVERRIDE_TESTS)

    assert not [issue for issue in issues if "a.test.yaml YAML parse" in issue]


def test_ci_still_rejects_two_equal_keys_written_in_one_companion_case(
    tmp_path, monkeypatch
):
    tests = NEW_YORK_CITY_OVERRIDE_TESTS.replace(
        "    resides_in_new_york_city: true\n",
        "    resides_in_new_york_city: true\n    resides_in_new_york_city: false\n",
    )

    issues = _ci_issues(tmp_path, monkeypatch, tests)

    assert [
        issue
        for issue in issues
        if issue.startswith("a.test.yaml YAML parse failed")
        and "found duplicate key 'resides_in_new_york_city'" in issue
        and "line 21, column 5" in issue
    ]


def test_the_apply_test_case_contract_reads_a_merge_override(tmp_path):
    single = "us-la:statutes/47/294#input.filing_status_is_single"
    joint = "us-la:statutes/47/294#input.filing_status_is_joint"
    principal = "us-la:statutes/47/294#standard_deduction"
    period = {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}
    contract = _parse_deferred_output_review_contract_json(
        json.dumps(
            {
                "schema": "axiom-encode/review-contract/v2",
                "citation": "us-la/statute/47:294",
                "rulespec_path": "us-la/statutes/47/294.yaml",
                "required_deferred_outputs": [],
                "required_test_cases": [
                    {
                        "name": "single filer",
                        "period": period,
                        "input": {single: True, joint: False},
                        "required_output": {principal: 12500},
                    }
                ],
            }
        )
    )
    rulespec = tmp_path / "294.yaml"
    rulespec.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    rulespec.with_name("294.test.yaml").write_text(
        "- name: joint filers\n"
        "  period: &period\n"
        "    period_kind: tax_year\n"
        "    start: '2025-01-01'\n"
        "    end: '2025-12-31'\n"
        "  input: &joint\n"
        f"    {single}: false\n"
        f"    {joint}: true\n"
        "  output:\n"
        f"    {principal}: 25000\n"
        "- name: single filer\n"
        "  period: *period\n"
        "  input:\n"
        "    <<: *joint\n"
        f"    {single}: true\n"
        f"    {joint}: false\n"
        "  output:\n"
        f"    {principal}: 12500\n"
    )

    assert (
        _required_deferred_output_contract_issues(
            rulespec,
            contract,
            citation="us-la/statute/47:294",
            rulespec_path="us-la/statutes/47/294.yaml",
        )
        == []
    )


def test_the_new_york_city_override_loads_as_pyyaml_reads_it():
    loaded = _companion_load(NEW_YORK_CITY_OVERRIDE_TESTS)

    assert loaded == yaml.safe_load(NEW_YORK_CITY_OVERRIDE_TESTS)
    assert loaded[2]["input"] == {
        "household_size": 3,
        "resides_in_new_york_city": True,
        "heats_with_utilities": True,
    }


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("base: &b {a: 1}\ncase: {<<: *b, a: 2}\n", id="explicit-wins"),
        pytest.param(
            "x: &x {a: 1, b: 1}\ny: &y {a: 2, c: 2}\ncase: {<<: [*x, *y], a: 3}\n",
            id="earlier-source-wins-then-explicit",
        ),
        pytest.param(
            "x: &x {a: 1, b: 1}\ny: &y {a: 2, b: 2}\ncase: {<<: [*x, *y]}\n",
            id="sources-share-keys",
        ),
        pytest.param(
            "x: &x {a: 1}\ny: &y {<<: *x, a: 2, b: 2}\ncase: {<<: *y, b: 3}\n",
            id="nested-merge-with-overrides",
        ),
        pytest.param(
            "case: {<<: {a: 1, b: 1}, a: 2}\n",
            id="inline-source",
        ),
        pytest.param(
            "x: &x {a: 1}\ncase:\n  <<: *x\n  <<: {a: 2, b: 2}\n",
            id="two-merge-keys",
        ),
        pytest.param(
            "case: {'<<': literal, a: 1}\n",
            id="quoted-merge-spelling-is-a-plain-key",
        ),
        pytest.param(
            "y: &y {<<: &x {a: 1}, a: 2}\ncases: [{<<: *y}, {<<: *x, a: 3}]\n",
            id="source-flattened-before-it-is-merged-again",
        ),
        pytest.param("plain: {a: 1, b: [1, 2]}\n", id="no-merge-keys"),
    ],
)
def test_the_companion_loader_applies_merges_as_pyyaml_does(content):
    assert _companion_load(content) == yaml.safe_load(content)


@pytest.mark.parametrize(
    ("content", "key", "line"),
    [
        pytest.param("case: {a: 1, a: 2}\n", "a", 1, id="plain"),
        pytest.param(
            "base: &b {a: 1}\ncase:\n  <<: *b\n  b: 1\n  b: 2\n",
            "b",
            5,
            id="beside-a-merge-key",
        ),
        pytest.param(
            "case: {<<: {a: 1, a: 2}, b: 1}\n", "a", 1, id="inside-an-inline-source"
        ),
        pytest.param(
            "cases:\n- <<: &b\n    a: 1\n    a: 2\n  c: 1\nother: *b\n",
            "a",
            4,
            id="inside-an-anchored-source-merged-first",
        ),
        pytest.param(
            "x: &x {a: 1}\ny: &y\n  <<: *x\n  c: 1\n  c: 2\ncase: {<<: *y}\n",
            "c",
            5,
            id="inside-a-nested-source",
        ),
        pytest.param(
            "case: {<<: [{a: 1}, {b: 1, b: 2}]}\n", "b", 1, id="inside-a-source-list"
        ),
        pytest.param("case: {1: a, 1.0: b}\n", 1.0, 1, id="equal-loaded-keys"),
        pytest.param("case: {a: 1, 'a': 2}\n", "a", 1, id="quoted-spelling"),
    ],
)
def test_the_companion_loader_rejects_two_equal_keys_in_one_mapping(content, key, line):
    with pytest.raises(yaml.constructor.ConstructorError) as raised:
        _companion_load(content)

    assert f"found duplicate key {key!r}" in str(raised.value)
    assert raised.value.problem_mark.line + 1 == line


def test_module_loads_still_reject_an_explicit_key_over_a_merged_key():
    # The engine does not apply merge keys, so a module that relies on one is
    # read differently there; module loads keep refusing this shape.
    content = "base: &b {a: 1}\nrule: {<<: *b, a: 2}\n"

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        _safe_load_unique_keys(content)


_KEYS = ("a", "b", "c")
_SPELLINGS = {"a": ("a", "'a'", '"a"'), "b": ("b", "'b'"), "c": ("c", "!!str c")}


@st.composite
def _merge_documents(draw):
    """A document of anchored sources and cases, and whether it plants two
    equal keys in one mapping."""

    planted = False
    lines: list[str] = []

    def mapping_body(indent: str, anchors: list[str]) -> list[str]:
        nonlocal planted
        body: list[str] = []
        if anchors and draw(st.booleans()):
            chosen = draw(st.lists(st.sampled_from(anchors), min_size=1, max_size=3))
            if len(chosen) == 1 and draw(st.booleans()):
                body.append(f"{indent}<<: *{chosen[0]}")
            else:
                body.append(f"{indent}<<: [{', '.join('*' + a for a in chosen)}]")
        keys = draw(st.lists(st.sampled_from(_KEYS), max_size=3, unique=True))
        written = list(keys)
        if keys and draw(st.integers(0, 4)) == 0:
            written.append(draw(st.sampled_from(keys)))
            planted = True
        for index, key in enumerate(written):
            spelling = draw(st.sampled_from(_SPELLINGS[key]))
            body.append(f"{indent}{spelling}: {index}")
        if not body:
            body.append(f"{indent}d: 0")
        return body

    anchors: list[str] = []
    lines.append("sources:")
    for index in range(draw(st.integers(0, 3))):
        name = f"s{index}"
        body = mapping_body("    ", anchors)
        lines.append(f"  - &{name}")
        lines.extend(body)
        anchors.append(name)
    if not anchors:
        lines[-1] = "sources: []"
    lines.append("cases:")
    for _case in range(draw(st.integers(1, 3))):
        body = mapping_body("      ", anchors)
        lines.append("  - input:")
        lines.extend(body)
    return "\n".join(lines) + "\n", planted


@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_merge_documents())
def test_the_companion_loader_matches_safe_load_unless_a_mapping_repeats_a_key(
    document,
):
    content, planted = document

    if planted:
        with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
            _companion_load(content)
    else:
        assert _companion_load(content) == yaml.safe_load(content)
