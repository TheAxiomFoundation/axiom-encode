"""The ordinal of a `See Example N` cross-reference is not a policy value.

A policy manual points at its worked examples with `(See Example 1 )`. The
ordinal names the example. Numeric recall read it as a value of the rule and
told the encoder to add a parameter for it, so a correct encoding could not
pass. Only the ordinal of a closed reference is masked; every other number,
and every other use of the word `example`, keeps its demand.

All source text here is synthetic.
"""

from __future__ import annotations

import functools
import itertools
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.evals import evaluate_artifact
from axiom_encode.harness.validator_pipeline import (
    ValidationResult,
    ValidatorPipeline,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_evals import (
    _canonical_rulespec_content_root,
    _write_test_corpus_provision,
)

MANUAL = "us-zz/manual/agency/handbook/shelter-costs/block-3"
STATUTE = "us-zz/statute/12/12-345"
PROFILES = ("legacy", "en-US")

# (source, values that stay in the recall inventory)
MASKED = (
    ("The cost is allowed for the whole period. (See Example 1 )", []),
    ("The cost is allowed for the whole period. (See Example 1)", []),
    ("The cost is allowed for the whole period. (See Example 12 )", []),
    ("The cost is allowed for the whole period. (See Example 12)", []),
    ("The cost is allowed for the whole period (See Example 12).", []),
    ("The cost is allowed for the whole period. See Example 12.", []),
    ("The cost is allowed for the whole period. (See Example 12.)", []),
    ("The cost is allowed for the whole period; see example 3.", []),
    ("The cost is allowed for the whole period. (SEE EXAMPLE 4)", []),
    ("See Example 2; the cost is allowed for the whole period.", []),
    ("See Example 2, where the cost is allowed for the whole period.", []),
    ("The cost is allowed [See Example 7] for the whole period.", []),
    ("The cost is allowed. (See Examples 1 and 2)", []),
    ("The cost is allowed. See Examples 1 and 2.", []),
    ("The cost is allowed. (See Examples 1, 2, and 3 )", []),
    ("The cost is allowed. (see examples 4 or 15)", []),
    ("The unit pays $50 for 3 months. (See Example 1 )", [50.0, 3.0]),
    ("(See Example 1 ) The unit pays $1 for 1 month.", [1.0, 1.0]),
    ("(See Example 2 ) Allow $20 when 2 persons pay.", [20.0, 2.0]),
    # A closed reference ends its sentence; the next sentence keeps its number.
    ("See Example 2. 3 months of costs are allowed.", [3.0]),
    ("See Example 2.\n3 months of costs are allowed.", [3.0]),
    ('The cost is allowed. "See Example 2."', []),
    ("The cost is allowed. \u201cSee Example 2.\u201d", []),
    ("The cost is allowed. (See Example 2;)", []),
    ("The cost is allowed. 'See Example 2.'", []),
    ("The cost is allowed. \u2018See Example 2.\u2019", []),
    ("The cost is allowed. [See Example 2.]", []),
    ("The cost is allowed; see Example 2,", []),
    ("See Example 2,\nwhere the cost is allowed.", []),
)

# Nothing in these is a closed reference to an example: every number stays.
KEPT = (
    ("The unit pays $1.", [1.0]),
    ("The cost is allowed for 1 month.", [1.0]),
    ("Example 2: the household pays $1.", [2.0, 1.0]),
    ("Example 2\nThe household pays $1 for 3 persons.", [2.0, 1.0, 3.0]),
    ("Example: 3 persons share the cost.", [3.0]),
    ("For example, 3 persons share the cost.", [3.0]),
    ("See, for example, 3 persons who share the cost.", [3.0]),
    ("See Example 1 and pay 2.", [1.0, 2.0]),
    ("See Example 1 month of costs.", [1.0]),
    ("See Example 1 and 2.", [1.0, 2.0]),
    ("See Examples 1 and 2 months of costs.", [1.0, 2.0]),
    ("See Examples 1, 2 and 3 persons.", [1.0, 2.0, 3.0]),
    ("See Examples 4)", [4.0]),
    ("See Example 4, 3 persons share the cost.", [4.0, 3.0]),
    ("See Example 4\n3 persons share the cost.", [4.0, 3.0]),
    ("See Example 4: 3 persons share the cost.", [4.0, 3.0]),
    ("See Example 4 of the appendix.", [4.0]),
    ("See Example 123)", [123.0]),
    ("See Example 4.5)", [4.5]),
    ("See the Example 4)", [4.0]),
    ("See Table 4)", [4.0]),
    ("Oversee Example 4)", [4.0]),
    ("See Example $4.", [4.0]),
)

# The reference recognizer leaves these alone too. Other, older rules decide
# how their numbers are read, so only the recognizer itself is checked.
NOT_A_CLOSED_REFERENCE = (
    "See Example\n4)",
    "See Example 4a)",
    "See Example 4 percent.",
    "See Example4)",
    "See Example 0)",
    "See Example 04)",
    "See Example -4)",
    "See Example 4-5)",
    "See Example 4/5)",
    "See Example 4.5.",
    "See Example 4,500.",
    "See Example 4;5",
    "See Examples 4 through 6)",
    "See Examples 4, 5 6)",
    "Example 4)",
    "Example 4.",
    # A Unicode numeral after the closer continues the number.
    "Pay $50. See Example 1.\uff15 end",
    "See Example 1.\u0665",
    "See Example 1;\uff15",
    "See Example 1,\uff15",
    "See Example 1, \uff15",
    "See Example 12.\uff15 end",
    "See Example 12,\uff15 end",
    "See Examples 1 and 2.\uff15 end",
    "See Examples 1 and 2,\uff15 end",
    "See Example 1\uff12)",
    "See Example 1.\u00bd",
    "See Example 1,\u00bd",
    # A grouped number that wraps after its comma.
    "See Example 1,\n000 dollars",
    "See Example 1, \n 000 dollars",
    "See Example 1,\u00a0000 dollars",
    "See Examples 1 and 2,\n000 dollars",
    # Boundaries of the words.
    "5See Example 1.",
    "Seee Example 1.",
    "\u017fee Example 1.",
    "See\nExample 1.",
    "See Examples\n1 and 2)",
    "See Examples1 and 2)",
    "See Example 4\n)",
    "See Examples 1 and 2\n)",
    # After `.` or `;` only whitespace, a closing bracket or quote, or the end.
    "See Example 1.Note(2) end",
    "See Example 1.a",
    "See Example 1;a",
    "See Example 1.,5",
    "See Example 1.-5",
    "See Example 1;-5",
    "See Example 1.(5)",
    "See Example 1.$5",
    "See Example 1..5",
    # An opening quote does not end the reference.
    "See Example 1.\u201c5",
    "See Example 1.\u2018000",
    # After `,` only an ASCII letter or the end: the tokenizer joins a number
    # across a run of separators.
    "Pay $50. See Example 1,,000 end",
    "See Example 1,.5",
    "See Example 1,,5",
    "See Example 1,,.5",
    "See Example 1,,,000",
    "See Example 1,.\uff15",
    "See Example 1, ,000",
    "See Example 1,-5",
    "See Example 1, $5",
    "See Example 1,(5)",
    "See Example 1,)",
    "See Example 1, \u00e9lectricit\u00e9",
    "See Examples 1 and 2,,000",
    # A list comma needs a blank after it: `1,5` can be one number.
    "See Examples 1,5)",
    "See Examples 1,2)",
    "See Examples 1 ,2 and 3)",
    "See Examples 4\n,5)",
    "See Examples 4\n, 5)",
    "See Examples 4, and\n5)",
    # Only `,`, `and` and `or` join a list, on one line.
    "See Examples 4 to 6)",
    "See Examples 4-5)",
    "See Examples 4 & 5)",
    "See Examples 4,\n5)",
    "See Examples 4\nand 5)",
    "See Examples 4 and\n5)",
)


def _recall(source: str, citation: str) -> str:
    return sc.authoritative_numeric_recall_text(source, corpus_citation_path=citation)


def _inventory(source: str, citation: str, profile: str) -> list[float]:
    return [
        item.value
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            _recall(source, citation), profile=profile
        )
    ]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (MANUAL, STATUTE))
@pytest.mark.parametrize(("source", "kept"), MASKED)
def test_closed_example_reference_ordinal_leaves_the_inventory(
    source, kept, citation, profile
):
    assert _inventory(source, citation, profile) == kept
    assert len(_recall(source, citation)) == len(source)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (MANUAL, STATUTE))
@pytest.mark.parametrize(("source", "kept"), KEPT)
def test_number_outside_a_closed_example_reference_stays(
    source, kept, citation, profile
):
    assert sc.mask_example_cross_references(source) == source
    assert sorted(_inventory(source, citation, profile)) == sorted(kept)


@pytest.mark.parametrize("source", NOT_A_CLOSED_REFERENCE)
def test_recognizer_requires_a_closed_reference(source):
    assert sc.mask_example_cross_references(source) == source


def test_only_the_ordinal_is_blanked_in_place():
    source = "Pay $1. (See Example 1 ) Pay 1 more. See Examples 2, 3 and 14."
    assert sc.mask_example_cross_references(source) == (
        "Pay $1. (See Example   ) Pay 1 more. See Examples  ,   and   ."
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "source",
    (
        "Pay $50. See Example 1.\uff15 end",
        "Pay $50. See Example 12,\uff15 end",
        "Pay $50. See Examples 1 and 2.\uff15 end",
        "Pay $50. See Example 1,\n000 dollars",
    ),
)
def test_number_that_continues_after_the_closer_keeps_its_inventory(
    source, profile, monkeypatch
):
    with_mask = _inventory(source, MANUAL, profile)
    monkeypatch.setattr(
        sc, "_example_cross_reference_numeric_recall_spans", lambda text: ()
    )
    assert with_mask == _inventory(source, MANUAL, profile)


def test_recall_masks_examples_after_subdivisions_through_the_source_map(
    monkeypatch,
):
    source = "Pay $50 under subdivision 3 of this section. (See Example 7 )"
    calls = []
    for name in (
        "_english_subdivision_numeric_recall_spans",
        "_example_cross_reference_numeric_recall_spans",
        "_contiguous_states_numeric_recall_spans",
    ):
        real = getattr(sc, name)

        def recorded(text, _name=name, _real=real):
            calls.append((_name, text))
            return tuple(_real(text))

        monkeypatch.setattr(sc, name, recorded)
    source_map = sc._NumericRecallSourceMap(list(range(len(source))))
    recall = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=STATUTE, _source_map=source_map
    )

    assert [name for name, _text in calls] == [
        "_english_subdivision_numeric_recall_spans",
        "_example_cross_reference_numeric_recall_spans",
        "_contiguous_states_numeric_recall_spans",
    ]
    ordinal = source.index("7")
    subdivision = source.index("3")
    # The example recognizer reads the text after the subdivision mask.
    assert calls[1][1][subdivision] == " " and calls[1][1][ordinal] == "7"
    assert calls[2][1][ordinal] == " "
    assert recall[ordinal] == " "
    # The masked ordinal has no source origin; its neighbors keep theirs.
    assert source_map.offsets[ordinal] is None
    assert source_map.source_span(ordinal, ordinal + 1) is None
    assert source_map.offsets[ordinal - 1] == ordinal - 1
    assert source_map.offsets[ordinal + 1] == ordinal + 1
    amount = source.index("50")
    assert source_map.source_span(amount, amount + 2) == (amount, amount + 2)


def test_reference_closed_by_a_bracket_keeps_the_number_after_it():
    assert sc.mask_example_cross_references("(See Example 1)5") == "(See Example  )5"
    assert sc.mask_example_cross_references("See Example 2. 3 months") == (
        "See Example  . 3 months"
    )


_DIFFERENTIAL_ALPHABET = tuple("05\uff15\u0665,.;-+$%()[]aN \n")
_DIFFERENTIAL_HEADS = (
    "Pay $50. See Example 1",
    "Pay $50. (See Example 12",
    "Pay $50. See Examples 1 and 2",
    "Pay $50. (See Examples 3, 4, or 7",
)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("head", _DIFFERENTIAL_HEADS)
@pytest.mark.parametrize("closer", (")", "]", ".", ";", ",", " )", " .", " ,"))
def test_mask_removes_exactly_the_ordinals_from_the_inventory(head, closer, profile):
    """Whatever follows the closer, the mask never cuts or creates a number."""

    def values(text):
        return Counter(
            item.value
            for item in extract_typed_numeric_inventory_occurrences_from_text(
                text, profile=profile
            )
        )

    masked_count = 0
    for depth in range(3):
        for suffix in itertools.product(_DIFFERENTIAL_ALPHABET, repeat=depth):
            for tail in ("", " end"):
                text = head + closer + "".join(suffix) + tail
                masked = sc.mask_example_cross_references(text)
                if masked == text:
                    continue
                masked_count += 1
                ordinals = Counter(
                    float(text[start:end])
                    for start, end in sc._example_cross_reference_numeric_recall_spans(
                        text
                    )
                )
                assert values(text) - values(masked) == ordinals, text
                assert not values(masked) - values(text), text
    assert masked_count


def test_recall_view_agrees_with_the_public_mask():
    source = "Allow $20 when 2 persons pay. (See Example 2 )"
    assert _recall(source, MANUAL) == sc.mask_example_cross_references(source)


def _issues(source: str, *, scalars: tuple[float, ...], profile: str = "legacy"):
    candidate = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {},
            "rules": [
                {"name": f"supplied_{index}", "versions": [{"formula": value}]}
                for index, value in enumerate(scalars)
            ],
        }
    )
    return sc.analyze_complete_source_unit(
        candidate,
        source,
        corpus_citation_path=MANUAL,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues


def _demanded(issues) -> set[float]:
    prefix = "[complete-source-unit:numeric-recall] Authoritative corpus numeric value "
    return {
        float(issue[len(prefix) :].split(" ", 1)[0])
        for issue in issues
        if issue.startswith(prefix)
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_complete_source_recall_does_not_demand_the_reference_ordinal(profile):
    source = "The unit pays $50 for the whole approval period. (See Example 1 )"
    assert _demanded(_issues(source, scalars=(50,), profile=profile)) == set()


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    ("source", "demanded"),
    (
        ("The unit pays $50 and then $1. (See Example 1 )", {1.0}),
        ("The unit pays $50 for 1 month. (See Example 1 )", {1.0}),
        ("Example 1: the household pays $50 and then $1.", {1.0}),
        ("Example 3\nThe household pays $50.", {3.0}),
        ("Example: 3 persons pay $50.", {3.0}),
        ("The unit pays $50. See Example 1 and pay 2.", {1.0, 2.0}),
        ("The unit pays $50. (See Example 1 ) Then pay 2.", {2.0}),
    ),
)
def test_complete_source_recall_still_demands_real_values(source, demanded, profile):
    assert _demanded(_issues(source, scalars=(50,), profile=profile)) == demanded


def _eval_metrics(tmp_path: Path, source_text: str, *, complete: bool):
    corpus_release = _write_test_corpus_provision(
        tmp_path, citation_path=MANUAL, body=source_text
    )
    rulespec_file = tmp_path / "policies" / "agency" / "handbook" / "block-3.yaml"
    rulespec_file.parent.mkdir(parents=True)
    rulespec_file.write_text(
        f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {MANUAL}
rules:
  - name: monthly_cost_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 50
"""
    )
    passed = ValidationResult("compile", True, issues=[])
    with (
        patch.object(ValidatorPipeline, "_run_compile_check", return_value=passed),
        patch.object(
            ValidatorPipeline,
            "_run_ci",
            return_value=ValidationResult("ci", True, issues=[]),
        ),
    ):
        return evaluate_artifact(
            rulespec_file=rulespec_file,
            policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us-zz"),
            axiom_rules_path=Path("/tmp/axiom-rules-engine"),
            source_text=source_text,
            local_corpus_release=corpus_release,
            source_citation_path=MANUAL,
            require_complete_source_unit=complete,
        )


@pytest.mark.parametrize("complete", [False, True])
def test_eval_numeric_coverage_skips_the_reference_ordinal(tmp_path, complete):
    metrics = _eval_metrics(
        tmp_path,
        "The unit pays $50 for the whole approval period. (See Example 1 )",
        complete=complete,
    )

    assert metrics.numeric_occurrence_issues == []
    assert metrics.source_numeric_occurrence_count == 1


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize(
    "source",
    (
        "The unit pays $50 and then $1. (See Example 1 )",
        "Example 1: the household pays $50.",
        "The unit pays $50. See Example 1 and pay it.",
    ),
)
def test_eval_numeric_coverage_keeps_real_values(tmp_path, source, complete):
    metrics = _eval_metrics(tmp_path, source, complete=complete)

    assert any(
        issue.startswith("Source numeric value 1 appears 1 time(s)")
        for issue in metrics.numeric_occurrence_issues
    )
