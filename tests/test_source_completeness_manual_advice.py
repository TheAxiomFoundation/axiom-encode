"""US manual outcome suggestions leave the pinned main completeness gate intact.

The baseline module is read from the worktree's original main commit, rather
than reconstructed from the current implementation. Only the unchanged numeric
adapters are shared. Loading it once keeps the generated differential inexpensive.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

BASELINE_COMMIT = "85845da38962e2ccbb222afdf8917f3339604aec"
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures/source_completeness"
MANUAL = "us-or/manual/odhs/open/page-644"
EMPTY_CANDIDATE = "format: rulespec/v1\nmodule:\n  summary: probe\nrules: []\n"
DERIVED_CANDIDATE = """format: rulespec/v1
module:
  summary: probe
inputs:
- name: residence_verified
  entity: Person
  dtype: Boolean
  period: Month
rules:
- name: household_eligible
  kind: derived
  entity: Person
  dtype: Judgment
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: residence_verified
"""
OUTCOME_SENTENCE = "Households are eligible if residence is verified."
PROCEDURE = (
    "When child care hours are requested due to schooling, verification of "
    "registration and current class schedule are required."
)


def _analyze(module, content, source, citation, test_cases=()):
    return module.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=test_cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


@pytest.fixture(scope="module")
def baseline_completeness():
    """Use the immutable main module, including its original exception filter."""
    source = subprocess.run(
        [
            "git",
            "show",
            f"{BASELINE_COMMIT}:src/axiom_encode/harness/source_completeness.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    name = "axiom_encode.harness._manual_outcome_advice_baseline"
    module = types.ModuleType(name)
    module.__file__ = f"{BASELINE_COMMIT}:source_completeness.py"
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        yield module
    finally:
        del sys.modules[name]


@pytest.mark.parametrize(
    "outcome",
    (
        "the household is eligible",
        "eligibility continues",
        "the household is ineligible",
        "benefits stop",
        "the allotment changes",
        "income is counted",
        "the resource is excluded",
        "a payment is withheld",
        "the sanction applies",
        "disqualification follows",
        "assistance terminates",
        "deny the application",
        "a denial follows",
        "approve the application",
        "the benefit is reduced",
        "the allotment increases",
        "the household is entitled to assistance",
        "the household is paid",
        "deduct the expense",
    ),
)
def test_manual_outcome_words_produce_optional_specific_advice(outcome):
    sentence = f"If the condition applies, {outcome}."

    advice = completeness.manual_outcome_exception_advice(
        sentence, corpus_citation_path=MANUAL
    )

    assert len(advice) == 1
    record = advice[0]
    assert record["corpus_citation_path"] == MANUAL
    assert record["excerpt"] == sentence
    assert sentence[record["source_start"] : record["source_end"]] == sentence
    assert record["source_sha256"] == hashlib.sha256(sentence.encode()).hexdigest()
    assert sentence in record["message"]
    assert "operative condition" in record["message"]
    assert "derived output" in record["message"]
    assert "paired" in record["message"] and "tests" in record["message"]
    assert "not required" in record["message"].lower()


@pytest.mark.parametrize(
    ("citation", "source", "expected_count"),
    (
        (MANUAL, OUTCOME_SENTENCE, 1),
        ("us/manual/fns/snap-quality-control", OUTCOME_SENTENCE, 1),
        ("us-ut/manual/dws/eligibility-manual/510", OUTCOME_SENTENCE, 1),
        ("us/statute/7/2015", OUTCOME_SENTENCE, 0),
        ("us-or/regulation/461/135/0750", OUTCOME_SENTENCE, 0),
        ("uk-harrow/manual/council-tax-reduction", OUTCOME_SENTENCE, 0),
        ("de/manual/x", OUTCOME_SENTENCE, 0),
        ("", OUTCOME_SENTENCE, 0),
        (MANUAL, PROCEDURE, 0),
        (MANUAL, "The household receives benefits.", 0),
        (MANUAL, "Households are eligible if income is less than $500.", 0),
        (MANUAL, "A child is eligible if under age 18.", 0),
        (MANUAL, "A person is exempt if age 55 or older.", 0),
        (MANUAL, "If one or more of the conditions applies, deny the request.", 1),
        (MANUAL, "The resource is excluded under WAC 388-450-0140 if verified.", 1),
        (MANUAL, "A noncitizen is eligible under §243 of the INA if verified.", 1),
        (MANUAL, "When income is verified, record the document.", 0),
        (MANUAL, "When resources are verified, record the document.", 0),
    ),
)
def test_manual_advice_scope(citation, source, expected_count):
    assert (
        len(
            completeness.manual_outcome_exception_advice(
                source, corpus_citation_path=citation
            )
        )
        == expected_count
    )


def test_advice_preserves_raw_offsets_source_order_and_is_deterministic():
    first = "If residence is verified, approve the application."
    second = "When the household fails to cooperate, benefits are reduced."
    source = f"\n\n{first}\n\n{PROCEDURE}\n\n{second}\n"

    advice = completeness.manual_outcome_exception_advice(
        source, corpus_citation_path=MANUAL
    )

    assert advice == completeness.manual_outcome_exception_advice(
        source, corpus_citation_path=MANUAL
    )
    assert [record["excerpt"] for record in advice] == [first, second]
    assert [record["source_start"] for record in advice] == sorted(
        record["source_start"] for record in advice
    )
    for record in advice:
        assert (
            source[record["source_start"] : record["source_end"]] == record["excerpt"]
        )
        assert record["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()


def test_repeated_sentences_keep_distinct_replay_offsets():
    source = f"{OUTCOME_SENTENCE}\n\n{OUTCOME_SENTENCE}"

    advice = completeness.manual_outcome_exception_advice(
        source, corpus_citation_path=MANUAL
    )

    assert [record["excerpt"] for record in advice] == [OUTCOME_SENTENCE] * 2
    assert [record["source_start"] for record in advice] == [
        0,
        len(OUTCOME_SENTENCE) + 2,
    ]


def test_long_sentence_advice_is_bounded_but_retains_full_source_span():
    source = (
        "If " + "the verified household condition and " * 40 + "benefits terminate."
    )

    advice = completeness.manual_outcome_exception_advice(
        source, corpus_citation_path=MANUAL
    )

    assert len(advice) == 1
    record = advice[0]
    assert len(record["excerpt"]) <= 400
    assert record["excerpt"].startswith("If the verified household")
    assert record["excerpt"].endswith("benefits terminate.")
    assert record["source_start"] == 0
    assert record["source_end"] == len(source)


@st.composite
def source_examples(draw):
    country = draw(st.sampled_from(("us", "us-or", "us-ut", "us-ca", "uk", "de")))
    kind = draw(st.sampled_from(("manual", "statute", "regulation")))
    slug = draw(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=1, max_size=15)
    )
    citation = f"{country}/{kind}/agency/{slug}"
    sentences = draw(
        st.lists(
            st.sampled_from(
                (
                    OUTCOME_SENTENCE,
                    "Deny the application when identity cannot be verified.",
                    "Benefits are reduced unless the household cooperates.",
                    "Households are eligible if income is less than $500.",
                    "The resource is excluded under WAC 388-450-0140 if verified.",
                    "A person is exempt if age 55 or older.",
                    PROCEDURE,
                    "When the household moves, update the address.",
                    "When income is verified, record the document.",
                    "The household receives benefits.",
                )
            ),
            min_size=1,
            max_size=4,
        )
    )
    separator = draw(st.sampled_from((" ", "\n", "\n\n")))
    prefix = draw(st.sampled_from(("", "Eligibility\n\n", "\n\n")))
    source = prefix + separator.join(sentences)
    candidate = draw(st.sampled_from((EMPTY_CANDIDATE, DERIVED_CANDIDATE)))
    return citation, source, candidate


@given(source_examples())
@settings(max_examples=100, deadline=None)
def test_generated_advice_does_not_change_main_blocking_issues(
    baseline_completeness, example
):
    citation, source, candidate = example
    baseline = _analyze(baseline_completeness, candidate, source, citation)
    before_advice = _analyze(completeness, candidate, source, citation)

    advice = completeness.manual_outcome_exception_advice(
        source, corpus_citation_path=citation
    )
    after_advice = _analyze(completeness, candidate, source, citation)

    assert before_advice.issues == after_advice.issues == baseline.issues
    assert (
        bool(before_advice.issues) == bool(after_advice.issues) == bool(baseline.issues)
    )
    assert advice == completeness.manual_outcome_exception_advice(
        source, corpus_citation_path=citation
    )
    if not (citation.split("/", 1)[0].startswith("us") and "/manual/" in citation):
        assert not advice
    for record in advice:
        excerpt = record["excerpt"]
        assert not baseline_completeness._STATED_NUMERIC_LIMIT.search(excerpt)
        original = source[record["source_start"] : record["source_end"]]
        assert " ".join(original.split()) == excerpt


def test_byte_exact_utah_replay_keeps_main_gate_and_has_no_outcome_advice(
    baseline_completeness,
):
    fixture = FIXTURES / "signed_reencode_36880457452"
    recorded = json.loads((fixture / "issues.json").read_text())
    content = (fixture / "candidate.yaml").read_text()
    test_content = (fixture / "candidate.test.yaml").read_text()
    source = (fixture / "source.txt").read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == recorded["rulespec_sha256"]
    assert hashlib.sha256(test_content.encode()).hexdigest() == recorded["tests_sha256"]
    tests = yaml.safe_load(test_content)

    baseline = _analyze(
        baseline_completeness, content, source, recorded["citation"], tests
    )
    actual = _analyze(completeness, content, source, recorded["citation"], tests)

    assert actual.issues == baseline.issues == ()
    assert (
        completeness.manual_outcome_exception_advice(
            source, corpus_citation_path=recorded["citation"]
        )
        == ()
    )
