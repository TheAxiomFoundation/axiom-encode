"""Closed locator evidence must survive both actual numeric recall passes."""

import functools
from pathlib import Path

import pytest

from axiom_encode.form_numeric_structure import form_numeric_locator_spans
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness import validator_pipeline as v

SOURCE = (Path(__file__).parent / "fixtures/multipage_t936_source.txt").read_text()
CONTENT = """format: rulespec/v1
module:
  name: diagnostic
rules:
- name: diagnostic_value
  kind: derived
  dtype: Decimal
  entity: Person
  period: Year
  versions:
  - effective_from: 2025-01-01
    formula: '0'
"""


def analyze(source=SOURCE):
    return s.analyze_complete_source_unit(
        CONTENT,
        source,
        corpus_citation_path="ca/policy/cra/t1-2025/cumulative-net-investment-loss",
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
    )


def numeric_issues(source=SOURCE):
    return {
        str(x)
        for x in analyze(source).issues
        if "[complete-source-unit:numeric-recall]" in str(x)
    }


def test_exact_locators_do_not_remove_caps_rates_or_dates():
    spans = form_numeric_locator_spans(SOURCE)
    assert len(spans) == len(set(spans)) == 54
    assert (8395, 8397) in spans
    assert not any(a <= 3592 < b for a, b in spans)
    masked = s._mask_numeric_spans(SOURCE, spans)
    assert len(masked) == len(SOURCE)
    for i, ch in enumerate(SOURCE):
        if not any(a <= i < b for a, b in spans):
            assert masked[i] == ch
    for text in ["$30,000", "$55,000", "2022", "2023", "2024", "March 18, 2019", "50%"]:
        assert masked.count(text) == SOURCE.count(text)
    assert form_numeric_locator_spans(SOURCE.replace("$30,000", "$31,000", 1)) == ()


@pytest.mark.parametrize(
    "mask", ["represented_annual_spans", "represented_measurement_spans"]
)
def test_actual_secondary_numeric_pass_preserves_raw_locator_evidence(
    monkeypatch, mask
):
    before = numeric_issues()
    original = s._companion_test_issues
    pos = SOURCE.index("2025", SOURCE.index("Use this form"))

    calls = []

    def wrapped(*args, **kwargs):
        calls.append(mask)
        result = original(*args, **kwargs)
        kwargs[mask].add((pos, pos + 4))
        return result

    monkeypatch.setattr(s, "_companion_test_issues", wrapped)
    after = numeric_issues()
    # Covering the first year occurrence must relocate its repair context to
    # the next uncovered occurrence, without changing any other evidence.
    old_location = (
        "2025 at 241:245: `... if you had any investment income or investment "
        "expenses for 2025.`"
    )
    new_location = (
        "2025 at 495:499: `Even if you are not claiming a capital gains deduction "
        "in 2025, you should still complete this form if you had any investment`"
    )
    assert sum(old_location in issue for issue in before) == 1
    assert sum(new_location in issue for issue in after) == 1
    assert after == {issue.replace(old_location, new_location) for issue in before}
    assert calls == [mask]
    assert any("value 30000 " in x for x in before)
    assert not any("value 68130 " in x for x in before)
    assert any(
        "value 68130 " in x
        for x in numeric_issues(SOURCE.replace("$30,000", "$31,000", 1))
    )


def test_actual_recall_origin_map_preserves_rate_context():
    mapping = s._NumericRecallSourceMap(list(range(len(SOURCE))))
    text = s.authoritative_numeric_recall_text(SOURCE, _source_map=mapping)
    found = []
    for occurrence in v._tokenize_numeric_occurrences_from_text(
        text, profile="legacy"
    ).inventory:
        origins = mapping.offsets[occurrence.start : occurrence.end]
        if any(x in origins for x in [1446, 2466, 7852, 8475]):
            assert occurrence.has_rate_context
            assert v.numeric_value_is_grounded(0.5, (occurrence,))
            found.append(occurrence)
    assert len(found) == 4
