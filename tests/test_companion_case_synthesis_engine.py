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


def _pipeline(
    root: Path, engine_root: Path, monkeypatch, source: str = SOURCE
) -> ValidatorPipeline:
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
        lambda _content: {CITATION: source},
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
        lambda citation_path: source if citation_path == CITATION else None,
    )
    return pipeline


def _tests_issues(
    pipeline: ValidatorPipeline, module: Path, cases, source: str = SOURCE
) -> list[str]:
    return [
        issue
        for issue in pipeline._complete_source_unit_issues(
            module.read_text(),
            validation_source_texts={CITATION: source},
            test_cases=cases,
            rules_file=module,
        )
        if issue.startswith("[complete-source-unit:tests]")
    ]


def test_synthesized_boundary_case_carries_engine_values_and_clamp_stays(
    real_engine, staged_module, monkeypatch
):
    root, module = staged_module
    pipeline = _pipeline(root, real_engine.root, monkeypatch)
    before = _tests_issues(pipeline, module, TESTS)
    assert any("boundary input; missing: (a)=500" in issue for issue in before)
    assert any("`max(0, unmet_need)`" in issue for issue in before)

    synthesized = pipeline.synthesize_complete_source_companion_cases(module)

    [boundary] = synthesized
    assert boundary["name"].startswith(f"{SYNTHESIZED_CASE_PREFIX}boundary_")
    assert boundary["input"] == {
        f"{TARGET}#input.countable_income": 500,
        f"{TARGET}#input.unmet_need": 40,
    }
    # The engine, not the synthesizer, supplies the expected value, and the
    # clone asserts only the covering rule and what it reads.
    assert boundary["output"] == {f"{TARGET}#household_eligible": "not_holds"}
    after = _tests_issues(pipeline, module, [*TESTS, *synthesized])
    assert not any("boundary input" in issue for issue in after)
    # The clamp demand needs a model-stated value; it is never synthesized.
    assert any("`max(0, unmet_need)`" in issue for issue in after)
    # The companion file is left for the caller to update.
    assert yaml.safe_load(module.with_name("9999.test.yaml").read_text()) == TESTS


def test_synthesis_is_off_outside_complete_source_mode(staged_module, monkeypatch):
    root, module = staged_module
    pipeline = _pipeline(root, Path("/nonexistent/axiom-rules-engine"), monkeypatch)
    pipeline.require_complete_source_unit = False

    assert pipeline.synthesize_complete_source_companion_cases(module) == []


# Adversarial controls (review of #1901): wrong encodings must never be pinned
# by synthesized cases, and dead-path clones must never close a clamp demand.
GUARDED_CLAMP = (
    "formula: |-\n          if household_eligible: max(0, unmet_need)\n"
    "          else: 0"
)
WRONG_GUARDED_CLAMP = (
    "formula: |-\n          if household_eligible: max(0, unmet_need) + 7\n"
    "          else: 0"
)
INELIGIBLE = {
    **TESTS[1],
    "input": {
        f"{TARGET}#input.countable_income": 900,
        f"{TARGET}#input.unmet_need": 30,
    },
}
INVERTED_UNASSERTED_OUTPUT = f"""\
  - name: household_ineligible
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
        formula: household_eligible
"""
AT_BOUNDARY = {
    "name": "at",
    "period": "2026-01",
    "input": {
        f"{TARGET}#input.countable_income": 500,
        f"{TARGET}#input.unmet_need": -5,
    },
    "output": {
        f"{TARGET}#household_eligible": "not_holds",
        f"{TARGET}#countable_shortfall": 0,
    },
}


def _run_control(tmp_path, engine_root, monkeypatch, module_text, tests, source=SOURCE):
    root = tmp_path.resolve() / "rulespec-us" / "us"
    module = root / "statutes" / "7" / "9999.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(module_text)
    module.with_name("9999.test.yaml").write_text(
        yaml.safe_dump(tests, sort_keys=False)
    )
    pipeline = _pipeline(root, engine_root, monkeypatch, source)
    synthesized = pipeline.synthesize_complete_source_companion_cases(module)
    return synthesized, _tests_issues(pipeline, module, [*tests, *synthesized], source)


@pytest.mark.parametrize(
    "comparison",
    [
        # V1: the source says "less than $500"; `<=` includes 500.
        "countable_income <= income_limit",
        # V2: inverted polarity.
        "countable_income >= income_limit",
    ],
)
def test_wrong_boundary_encoding_is_never_pinned(
    real_engine, tmp_path, monkeypatch, comparison
):
    synthesized, after = _run_control(
        tmp_path,
        real_engine.root,
        monkeypatch,
        MODULE.replace("countable_income < income_limit", comparison),
        TESTS,
    )

    assert not any("boundary" in case["name"] for case in synthesized)
    assert any("boundary input; missing: (a)=500" in issue for issue in after)


def test_clamp_demand_is_never_synthesized_even_with_a_live_case(
    real_engine, tmp_path, monkeypatch
):
    """V3: a live case reaches the clamp, but the clamped value is a source
    value the model must state; only the boundary demand is synthesized."""

    synthesized, after = _run_control(
        tmp_path,
        real_engine.root,
        monkeypatch,
        MODULE.replace("formula: max(0, unmet_need)", GUARDED_CLAMP),
        [INELIGIBLE, TESTS[0]],
    )

    assert synthesized
    assert all(
        case["name"].startswith(f"{SYNTHESIZED_CASE_PREFIX}boundary_")
        and f"{TARGET}#countable_shortfall" not in case["output"]
        for case in synthesized
    )
    assert any("`max(0, unmet_need)`" in issue for issue in after)


def test_wrong_clamped_branch_without_a_live_case_keeps_its_demand(
    real_engine, tmp_path, monkeypatch
):
    """V4: no case reaches the (wrong) clamped branch, so nothing closes it."""

    synthesized, after = _run_control(
        tmp_path,
        real_engine.root,
        monkeypatch,
        MODULE.replace("formula: max(0, unmet_need)", WRONG_GUARDED_CLAMP),
        [INELIGIBLE],
    )

    assert not any("clamp" in case["name"] for case in synthesized)
    assert any("`max(0, unmet_need)`" in issue for issue in after)


def test_unasserted_output_is_left_for_the_model(real_engine, tmp_path, monkeypatch):
    """V5: an output encoded as the opposite of the source is never asserted
    on the model's behalf."""

    synthesized, after = _run_control(
        tmp_path,
        real_engine.root,
        monkeypatch,
        MODULE + INVERTED_UNASSERTED_OUTPUT,
        [*TESTS, AT_BOUNDARY],
    )

    assert synthesized == []
    assert any(
        "Principal output `household_ineligible` is never asserted" in issue
        for issue in after
    )


# Round-2 review controls: a clamp or formula-branch demand exists to make
# the model state a source value, so synthesis must leave it for the model
# even when the engine could execute the wrong encoding cleanly.
SOURCE_CLAMP_OF_DIFFERENCE = (
    "(a) A household whose countable income is less than $500 is eligible.\n"
    "(b) The countable shortfall is the household's unmet need minus the "
    "household contribution, but not less than zero.\n"
)


def _with_contribution(cases):
    return [
        {
            **case,
            "input": {**case["input"], f"{TARGET}#input.household_contribution": 10},
            "output": {
                **case["output"],
                f"{TARGET}#countable_shortfall": max(
                    0, case["input"][f"{TARGET}#input.unmet_need"] - 10
                ),
            },
        }
        for case in cases
    ]


def test_clamp_on_the_wrong_operand_is_never_pinned(real_engine, tmp_path, monkeypatch):
    """A1: `max(0, need) - contribution` clamps before subtracting; the
    model's cases agree with it, so only a model-stated clamp value can
    expose it."""

    module_text = (
        MODULE.replace(
            "  - name: unmet_need\n",
            "  - name: household_contribution\n    entity: Household\n"
            "    dtype: Money\n    period: Month\n  - name: unmet_need\n",
        )
        .replace(
            "The countable shortfall is the household's unmet need, but not\n"
            "                less than zero.",
            "The countable shortfall is the household's unmet need minus the\n"
            "                household contribution, but not less than zero.",
        )
        .replace(
            "formula: max(0, unmet_need)",
            "formula: max(0, unmet_need) - household_contribution",
        )
    )
    synthesized, after = _run_control(
        tmp_path,
        real_engine.root,
        monkeypatch,
        module_text,
        _with_contribution(TESTS[:1]),
        SOURCE_CLAMP_OF_DIFFERENCE,
    )

    assert all(
        f"{TARGET}#countable_shortfall" not in case["output"] for case in synthesized
    )
    assert any("`max(0, unmet_need)`" in issue for issue in after)


SOURCE_TOGGLE = (
    "(a) A household whose countable income is less than $500 is eligible.\n"
    "(b) The allowance of an elderly household is 20 percent of the countable "
    "income. The allowance of any other household is 10 percent of the "
    "countable income.\n"
)


def _rate_rule(name: str, formula: str, excerpt: str) -> str:
    return f"""\
  - name: {name}
    kind: parameter
    dtype: Rate
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: {excerpt}
    versions:
      - effective_from: '2026-01-01'
        formula: '{formula}'
"""


def _toggle_module(elderly_branch: str) -> str:
    head, _, _ = MODULE.partition("  - name: countable_shortfall\n")
    head = head.replace(
        "  - name: unmet_need\n    entity: Household\n    dtype: Money\n"
        "    period: Month\n",
        "  - name: unmet_need\n    entity: Household\n    dtype: Money\n"
        "    period: Month\n  - name: household_is_elderly\n"
        "    entity: Household\n    dtype: Boolean\n    period: Month\n",
    )
    return (
        head
        + _rate_rule("elderly_rate", "0.2", "20 percent")
        + _rate_rule("other_rate", "0.1", "10 percent")
        + f"""\
  - name: allowance
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
                The allowance of an elderly household is 20 percent of the
                countable income. The allowance of any other household is 10
                percent of the countable income.
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if household_is_elderly: {elderly_branch}
          else: countable_income * other_rate
"""
    )


def _toggle_case(name, income, eligible, allowance):
    return {
        "name": name,
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": income,
            f"{TARGET}#input.unmet_need": 40,
            f"{TARGET}#input.household_is_elderly": False,
        },
        "output": {
            f"{TARGET}#household_eligible": eligible,
            f"{TARGET}#allowance": allowance,
        },
    }


def test_untested_toggle_branch_is_never_pinned(real_engine, tmp_path, monkeypatch):
    """A13: the elderly branch reads the wrong operand and no model case
    sets the toggle; synthesis must not assert the engine's elderly value."""

    synthesized, after = _run_control(
        tmp_path,
        real_engine.root,
        monkeypatch,
        _toggle_module("unmet_need * elderly_rate"),
        [
            _toggle_case("young", 100, "holds", 10),
            _toggle_case("high", 900, "not_holds", 90),
        ],
        SOURCE_TOGGLE,
    )

    assert all(
        case["input"].get(f"{TARGET}#input.household_is_elderly") is False
        and f"{TARGET}#allowance" not in case["output"]
        for case in synthesized
    )
    assert any(
        "Companion tests do not demonstrate formula branch (b)" in issue
        for issue in after
    )
