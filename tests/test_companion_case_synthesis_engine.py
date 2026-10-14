"""Real-engine check: synthesized companion cases carry engine-computed values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.companion_case_synthesis import SYNTHESIZED_CASE_PREFIX
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

CITATION = "us/statute/7/9999"
SOURCE = (
    "(a) A household whose countable income is less than $500 is eligible.\n"
    "(b) The countable shortfall is the household's unmet need, but not less "
    "than zero.\n"
)
MODULE = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CITATION}
inputs:
  - name: countable_income
    entity: Household
    dtype: Money
    period: Month
  - name: unmet_need
    entity: Household
    dtype: Money
    period: Month
rules:
  - name: income_limit
    kind: parameter
    dtype: Money
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: less than $500
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
  - name: household_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: A household whose countable income is less than $500 is eligible.
    versions:
      - effective_from: '2026-01-01'
        formula: countable_income < income_limit
  - name: countable_shortfall
    kind: derived
    entity: Household
    dtype: Money
    unit: USD
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: >-
                The countable shortfall is the household's unmet need, but not
                less than zero.
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, unmet_need)
"""
TARGET = "us:statutes/7/9999"
TESTS = [
    {
        "name": "low income",
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": 100,
            f"{TARGET}#input.unmet_need": 40,
        },
        "output": {
            f"{TARGET}#household_eligible": "holds",
            f"{TARGET}#countable_shortfall": 40,
        },
    },
    {
        "name": "high income",
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": 900,
            f"{TARGET}#input.unmet_need": 0,
        },
        "output": {
            f"{TARGET}#household_eligible": "not_holds",
            f"{TARGET}#countable_shortfall": 0,
        },
    },
]


@pytest.fixture
def staged_module(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path.resolve() / "rulespec-us" / "us"
    module = root / "statutes" / "7" / "9999.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(MODULE)
    module.with_name("9999.test.yaml").write_text(
        yaml.safe_dump(TESTS, sort_keys=False)
    )
    return root, module


def _pipeline(root: Path, engine_root: Path, monkeypatch) -> ValidatorPipeline:
    pipeline = ValidatorPipeline(
        policy_repo_path=root,
        axiom_rules_path=engine_root,
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
        source_citation_path=CITATION,
    )
    monkeypatch.setattr(
        pipeline,
        "_source_texts_for_rulespec_content",
        lambda _content: {CITATION: SOURCE},
    )
    monkeypatch.setattr(
        pipeline,
        "_proof_source_texts_for_rulespec_content",
        lambda _content, source_texts: dict(source_texts or {}),
    )
    # Production binds a local corpus release around synthesis; serve the
    # fixture body the same way to the static companion checks.
    monkeypatch.setattr(
        validator_pipeline,
        "_fetch_corpus_source_text",
        lambda citation_path: SOURCE if citation_path == CITATION else None,
    )
    return pipeline


def _tests_issues(pipeline: ValidatorPipeline, module: Path, cases) -> list[str]:
    return [
        issue
        for issue in pipeline._complete_source_unit_issues(
            module.read_text(),
            validation_source_texts={CITATION: SOURCE},
            test_cases=cases,
            rules_file=module,
        )
        if issue.startswith("[complete-source-unit:tests]")
    ]


def test_synthesized_cases_close_boundary_and_clamp_with_engine_values(
    real_engine, staged_module, monkeypatch
):
    root, module = staged_module
    pipeline = _pipeline(root, real_engine.root, monkeypatch)
    before = _tests_issues(pipeline, module, TESTS)
    assert any("boundary input; missing: (a)=500" in issue for issue in before)
    assert any("`max(0, unmet_need)`" in issue for issue in before)

    synthesized = pipeline.synthesize_complete_source_companion_cases(module)

    by_input = {
        tuple(sorted(case["input"].items())): case["output"] for case in synthesized
    }
    assert all(case["name"].startswith(SYNTHESIZED_CASE_PREFIX) for case in synthesized)
    at_boundary = tuple(
        sorted(
            {
                f"{TARGET}#input.countable_income": 500,
                f"{TARGET}#input.unmet_need": 40,
            }.items()
        )
    )
    negative_need = tuple(
        sorted(
            {
                f"{TARGET}#input.countable_income": 100,
                f"{TARGET}#input.unmet_need": -1,
            }.items()
        )
    )
    # The engine, not the synthesizer, supplies each expected value.
    assert by_input[at_boundary] == {
        f"{TARGET}#household_eligible": "not_holds",
        f"{TARGET}#countable_shortfall": 40,
    }
    assert by_input[negative_need] == {
        f"{TARGET}#household_eligible": "holds",
        f"{TARGET}#countable_shortfall": 0,
    }
    assert _tests_issues(pipeline, module, [*TESTS, *synthesized]) == []
    # The companion file is left for the caller to update.
    assert yaml.safe_load(module.with_name("9999.test.yaml").read_text()) == TESTS


def test_synthesis_is_off_outside_complete_source_mode(staged_module, monkeypatch):
    root, module = staged_module
    pipeline = _pipeline(root, Path("/nonexistent/axiom-rules-engine"), monkeypatch)
    pipeline.require_complete_source_unit = False

    assert pipeline.synthesize_complete_source_companion_cases(module) == []
