"""The complete-source analysis fails closed on payloads it cannot analyse.

Issue #1927: the analysis used to wrap the whole RuleSpec analysis in
``contextlib.suppress``, so a payload that made it raise returned zero issues.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures" / "source_completeness"
UNANALYZABLE = "[complete-source-unit:unanalyzable]"
CITATION = "us/statute/26/21"
SOURCE = "(a) The credit is $40 per taxpayer.\n"
HEADER = "format: rulespec/v1\nmodule:\n  summary: Credit.\n"


def _analyze(content: str, source: str = SOURCE, citation: str = CITATION, cases=None):
    return analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _unanalyzable(result) -> list[str]:
    return [issue for issue in result.issues if issue.startswith(UNANALYZABLE)]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(HEADER + "inputs: null\nrules: []\n", id="inputs-null"),
        pytest.param(HEADER + "rules: null\n", id="rules-null"),
        pytest.param("format: rulespec/v1\nrules: [\n", id="yaml-error"),
        pytest.param("- format: rulespec/v1\n", id="top-level-list"),
        pytest.param("format: rulespec/v2\nrules: []\n", id="unknown-format"),
        pytest.param("", id="empty-document"),
        # PyYAML raises ValueError, not YAMLError, for an impossible date.
        pytest.param(HEADER + "rules: []\nx: 2026-02-30\n", id="timestamp-value-error"),
        pytest.param(
            HEADER + "inputs:\n  - just_a_string\nrules: []\n", id="bare-string-input"
        ),
        pytest.param(HEADER + "inputs: 7\nrules: []\n", id="scalar-inputs"),
        pytest.param(HEADER + "rules:\n  - just_a_string\n", id="bare-string-rule"),
        pytest.param(HEADER + "rules: {}\n", id="mapping-rules"),
    ],
)
def test_malformed_payload_reports_unanalyzable(content):
    result = _analyze(content)

    assert _unanalyzable(result), result.issues


@pytest.mark.parametrize(
    "source",
    [SOURCE, "A credit is allowed.\n"],
    ids=["numbered-source", "source-without-numbers"],
)
def test_bare_string_input_item_is_reported_whatever_the_source(source):
    """The issue's third shape does not raise; the shape check reports it.

    With a source that has no numbers, main returned no issues at all for this
    payload; the old test passed on main because its source had a number.
    """

    result = _analyze(HEADER + "inputs:\n  - just_a_string\nrules: []\n", source)

    assert result.issues == (
        f"{UNANALYZABLE} The RuleSpec payload could not be analysed against its "
        "authoritative source (`inputs` is not a mapping or a list of mappings); "
        "fix the payload shape before completeness can pass.",
    )


def test_inner_exception_becomes_an_issue(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise TypeError("'NoneType' object is not iterable")

    monkeypatch.setattr(completeness_module, "_analyze_rulespec_payload", _raise)

    result = _analyze(HEADER + "rules: []\n")

    assert result.issues == (
        f"{UNANALYZABLE} The RuleSpec payload could not be analysed against "
        "its authoritative source (TypeError: 'NoneType' object is not "
        "iterable); fix the payload shape before completeness can pass.",
    )


def test_inner_value_error_becomes_an_issue(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise ValueError("could not convert string to float: 'x'")

    monkeypatch.setattr(completeness_module, "_analyze_rulespec_payload", _raise)

    result = _analyze(HEADER + "rules: []\n")

    assert result.issues == (
        f"{UNANALYZABLE} The RuleSpec payload could not be analysed against "
        "its authoritative source (ValueError: could not convert string to "
        "float: 'x'); fix the payload shape before completeness can pass.",
    )


def test_inner_yaml_error_becomes_an_issue(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise yaml.YAMLError("nested document")

    monkeypatch.setattr(completeness_module, "_analyze_rulespec_payload", _raise)

    assert _unanalyzable(_analyze(HEADER + "rules: []\n"))


def test_unexpected_exception_types_still_propagate(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(completeness_module, "_analyze_rulespec_payload", _raise)

    with pytest.raises(RuntimeError):
        _analyze(HEADER + "rules: []\n")


_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(max_size=8),
)
_SHAPES = st.one_of(
    _SCALARS,
    st.lists(_SCALARS, max_size=3),
    st.dictionaries(st.text(max_size=6), _SCALARS, max_size=3),
)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    fields=st.dictionaries(
        st.sampled_from(["module", "inputs", "rules", "units", "imports"]),
        _SHAPES,
        max_size=4,
    )
)
def test_property_raising_analysis_is_never_an_empty_pass(monkeypatch, fields):
    """The gate never passes a payload it could not read.

    A shape the analysis cannot read is reported before the inner analysis
    runs. Otherwise the spy records what the inner analysis did: the gate's
    verdict equals the inner verdict when it returned, and is an explicit
    unanalyzable issue when it raised.
    """

    inner = completeness_module._analyze_rulespec_payload
    seen: list[object] = []

    def _spy(*args, **kwargs):
        try:
            outcome = inner(*args, **kwargs)
        except Exception as exc:
            seen.append(exc)
            raise
        seen.append(outcome)
        return outcome

    payload = {"format": "rulespec/v1", **fields}
    content = yaml.safe_dump(payload, sort_keys=False)
    with monkeypatch.context() as patch:
        patch.setattr(completeness_module, "_analyze_rulespec_payload", _spy)
        try:
            result = _analyze(content)
        except Exception as exc:
            # Only exception types outside the analysed set may escape, and
            # escaping is itself a loud failure, not a pass.
            assert not isinstance(exc, (TypeError, ValueError, yaml.YAMLError))
            return

    if completeness_module._rulespec_payload_shape_problem(payload) is not None:
        assert seen == []
        assert _unanalyzable(result)
        return
    assert len(seen) == 1
    if isinstance(seen[0], BaseException):
        assert _unanalyzable(result)
    else:
        assert result == seen[0]
        assert not _unanalyzable(result)


def _fixture_cases():
    for fixture in sorted(FIXTURES.glob("signed_reencode_*")):
        recorded = json.loads((fixture / "issues.json").read_text())
        yield pytest.param(
            (fixture / "candidate.yaml").read_text(),
            (fixture / "source.txt").read_text(),
            recorded["citation"],
            yaml.safe_load((fixture / "candidate.test.yaml").read_text()),
            id=fixture.name,
        )


@pytest.mark.parametrize("content,source,citation,cases", list(_fixture_cases()))
def test_differential_fixture_verdicts_are_unchanged(
    monkeypatch, content, source, citation, cases
):
    """Every recorded candidate is analysed exactly as the inner analysis says.

    Main returned the inner analysis unchanged whenever it did not raise, so
    equality with the spy's return value is equality with main's verdict.
    """

    inner = completeness_module._analyze_rulespec_payload
    returned: list[object] = []

    def _spy(*args, **kwargs):
        outcome = inner(*args, **kwargs)
        returned.append(outcome)
        return outcome

    monkeypatch.setattr(completeness_module, "_analyze_rulespec_payload", _spy)

    result = _analyze(content, source, citation, cases)

    assert returned == [result]
    assert not _unanalyzable(result)


def _repository_rulespec_files():
    root = Path(__file__).parents[1]
    paths = []
    for path in sorted(root.rglob("*.yaml")):
        if {".venv", ".git", "node_modules"} & set(path.parts):
            continue
        try:
            payload = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("format") == "rulespec/v1":
            paths.append(path)
    return paths


def test_differential_no_repository_rulespec_file_is_unanalyzable():
    """Every `format: rulespec/v1` file in the repository passes the shape check.

    The shape check is the only new refusal for a payload that parses as a
    RuleSpec mapping, so no file main analysed is now refused as unanalyzable.
    """

    paths = _repository_rulespec_files()
    assert len(paths) >= 300
    for path in paths:
        payload = yaml.safe_load(path.read_text())
        assert completeness_module._rulespec_payload_shape_problem(payload) is None, (
            path
        )
