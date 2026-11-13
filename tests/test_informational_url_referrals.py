"""Closed referral classification; no numeric, condition or source waiver."""

from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as source
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

FIXTURE = Path(__file__).parent / "fixtures" / "informational_referral"
REFERRAL = (
    "For more information about these credits, including credit eligibility "
    "requirements, go to canada.ca/nl-tax-info."
)


@pytest.mark.parametrize(
    "text",
    [
        REFERRAL,
        REFERRAL.replace("canada.ca/nl-tax-info", "example.org/other-credit-guide"),
        REFERRAL.replace("canada.ca/nl-tax-info", "https://guidance.example.org/a/b"),
        REFERRAL.replace("eligibility requirements", "eligibility\nrequirements"),
        "For more information, go to example.org/reference-guide.",
    ],
)
def test_complete_information_referral_is_not_computation(text):
    assert source._is_informational_url_referral(text)
    assert not source.source_states_explicit_computation(text)
    assert (
        source._source_formula_branches(
            text, branches=(), active_branches=(), deferred_paths=set()
        )
        == ()
    )


@pytest.mark.parametrize(
    "text",
    [
        REFERRAL[:-1],
        REFERRAL.replace("nl-tax-info", "a" * 1024),
        REFERRAL[4:],
        REFERRAL + " Credit = income - assistance.",
        REFERRAL[:-1] + " if eligible.",
        REFERRAL[:-1] + " unless retired.",
        REFERRAL[:-1] + " for $200.",
        REFERRAL[:-1] + " + 200.",
        REFERRAL.replace("these credits", "credits equal to income minus assistance"),
        REFERRAL.replace("canada.ca/", "canada..ca/"),
        REFERRAL.replace("canada.ca/", "canada.ca//"),
        REFERRAL.replace("canada.ca/", "canada.ca/-"),
        REFERRAL.replace("canada.ca/", "canada.ca:80/"),
        REFERRAL.replace("nl-tax-info", "nl-tax-info?amount=200"),
        REFERRAL.replace("nl-tax-info", "nl-tax-info#condition"),
        "Credit = income - assistance (go to canada.ca/nl-tax-info).",
        "If applicable, " + REFERRAL,
    ],
)
def test_clipped_malformed_or_operative_referrals_are_not_excluded(text):
    assert not source._is_informational_url_referral(text)


@pytest.mark.parametrize(
    "text",
    [
        "Line 3: Line 1 minus line 2 (maximum $2,000) = ^",
        "Credit = income - assistance.",
        REFERRAL + " Credit = expenses * 0.174.",
        "Line 7: Credit = expenses * 0.174 (go to example.org/credit-guide).",
        "Credit = income / expenses. For more information, go to example.org/guide.",
    ],
)
def test_genuine_arithmetic_including_url_bearing_arithmetic_survives(text):
    assert source.source_states_explicit_computation(text)
    assert source._source_formula_branches(
        text, branches=(), active_branches=(), deferred_paths=set()
    )


def test_numeric_table_and_referral_digits_are_not_masked():
    text = "Income | Credit\n5 | 14\n10 | 28\n" + REFERRAL.replace(
        "nl-tax-info", "guide-2025"
    )
    recall = source.authoritative_numeric_recall_text(text)
    for number in ("5", "14", "10", "28", "2025"):
        assert number in recall


def test_actual_analyzer_removes_only_two_url_findings(monkeypatch):
    content = (FIXTURE / "diagnostic.yaml").read_text()
    body = (FIXTURE / "source.txt").read_text()
    cases = yaml.safe_load((FIXTURE / "diagnostic.test.yaml").read_text())
    pipeline = ValidatorPipeline(
        Path("/tmp/rulespec-ca"),
        Path("/tmp/axiom-rules-engine"),
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )

    def analyze():
        return pipeline._complete_source_unit_issues(
            content,
            validation_source_texts={
                "ca/policy/cra/t1-2025/nl479-credits/primary-document": body
            },
            test_cases=cases,
        )

    after = analyze()
    # Disable only this new recognition to reproduce the previous analyzer.
    with monkeypatch.context() as old:
        old.setattr(source, "_is_informational_url_referral", lambda _: False)
        before = analyze()
    removed = [issue for issue in before if issue not in after]
    assert len(before) == 8
    assert len(after) == 6
    assert len(removed) == 2
    assert all(REFERRAL in issue for issue in removed)
    assert any("[complete-source-unit:formula-output]" in issue for issue in removed)
    assert any("[complete-source-unit:tests]" in issue for issue in removed)
    assert after == [issue for issue in before if issue not in removed]
    assert any("numeric value 2025 has no named scalar" in issue for issue in after)
    # The form recognizer owns layout counts; this older fixture intentionally
    # lacks entered-row bindings, so each arithmetic row must still be refused.
    for number in (5, 14):
        assert not any(
            f"numeric value {number} has no named scalar" in issue for issue in after
        )
    for locator in ("1251:1299", "1375:1448", "1481:1569", "1693:1803"):
        assert any(f"characters {locator}" in issue for issue in after)
    assert any(
        "Source-stated exceptions or applicability conditions" in issue
        for issue in after
    )
    assert body[392:505].replace("\n", " ") == REFERRAL
