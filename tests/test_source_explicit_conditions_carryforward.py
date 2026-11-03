"""A carryforward consequent is an effect, not a second condition (#1832).

Va. Code § 58.1-322.03 7 a says "If the purchase price or annual contribution
to a college savings trust account exceeds $ 4,000, the remainder may be
carried forward and subtracted in future taxable years until the purchase
price or college savings trust contribution has been fully deducted". The
conditional body ran past the antecedent's comma into the consequent, and the
gate split the consequent on its `and` into a second factual condition.

Only a closed carryforward consequent is recognized, and only when cutting it
off loses no antecedent gate. The negative controls come first: every
condition the antecedent states, and any condition the consequent states, is
still demanded.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

REPOSITORY = Path(__file__).parents[1]
FIXTURES = REPOSITORY / "tests/fixtures/source_completeness"
VIRGINIA = (
    "If the purchase price or annual contribution to a college savings trust "
    "account exceeds $ 4,000, the remainder may be carried forward and "
    "subtracted in future taxable years until the purchase price or college "
    "savings trust contribution has been fully deducted"
)
CARRYFORWARD = (
    "the remainder may be carried forward and subtracted in future taxable "
    "years until the purchase price has been fully deducted"
)
GATE = "[complete-source-unit:source-explicit-conditions]"


def _without_carryforward_recognition(monkeypatch):
    """Main's semantics: no consequent is ever recognized.

    `raising=False` lets the same baseline run on main, where the helper does
    not exist yet.
    """

    monkeypatch.setattr(
        sc, "_carryforward_consequent_antecedent", lambda _body: None, raising=False
    )


def _main_gates(text):
    with pytest.MonkeyPatch.context() as patched:
        _without_carryforward_recognition(patched)
        return sc._source_conjunctive_fact_gates(text)


# Negative controls: main's gates stand wherever a condition could be lost.
NEGATIVE_CONTROLS = (
    # A genuine conjunction in the antecedent, with an ordinary consequent.
    (
        "If the taxpayer is a resident and the taxpayer is blind, the taxpayer "
        "may claim the credit.",
        2,
    ),
    # The antecedent's second condition reads as a gate only through the
    # consequent's `has been`, which shares its segment: cutting the
    # consequent would lose the election.
    (
        "If the purchase price exceeds $ 4,000 and the taxpayer elects, the "
        "remainder may be carried forward until the purchase price has been fully "
        "deducted",
        2,
    ),
    # Conditions inside the consequent are conditions.
    (
        "If the purchase price exceeds $ 4,000, the remainder may be carried "
        "forward and subtracted in future taxable years if the purchaser is a "
        "resident and is blind",
        3,
    ),
    (
        "If the purchase price exceeds $ 4,000, " + CARRYFORWARD + " and the "
        "purchaser remains a resident",
        3,
    ),
    (
        "If the purchase price exceeds $ 4,000, the remainder may be carried "
        "forward and subtracted until the credit of a resident who is blind has "
        "been fully deducted",
        2,
    ),
    # A factual `until` clause is not exhaustion of a carried amount.
    (
        "If the child is disabled, the benefit may be paid and continued until "
        "the child attains age 18",
        2,
    ),
    # No carryforward consequent: comma-listed and relative-clause antecedents.
    (
        "If the taxpayer is a resident, the spouse is disabled, and the child is "
        "blind, the credit is allowed.",
        2,
    ),
    (
        "If an individual, who must be a resident, is blind and is employed, the "
        "individual may claim the credit.",
        2,
    ),
    # A postposed condition has no consequent after it.
    (
        "The remainder may be carried forward and subtracted if the price exceeds "
        "$4,000 and the contribution is deducted.",
        2,
    ),
    # Review inputs. A condition inside the exhausted amount's words.
    (
        "If the contribution exceeds the annual limit, the remainder may be "
        "carried forward and subtracted in future taxable years until the "
        "contribution, so long as the contributor remains a resident, has been "
        "fully deducted.",
        2,
    ),
    (
        "If the credit exceeds the tax liability, the unused portion of the credit "
        "may be carried forward and applied in future taxable years until the "
        "credit of a taxpayer residing in an enterprise zone has been fully used.",
        2,
    ),
    (
        "If the purchase price or annual contribution to a college savings trust "
        "account exceeds $ 4,000, the remainder may be carried forward and "
        "subtracted in future taxable years until the purchase price or "
        "contribution of a purchaser under age 70 has been fully deducted.",
        2,
    ),
    # An antecedent condition with no gate of its own, whose only demand was
    # main's consequent gate.
    (
        "If the credit exceeds the tax liability and the taxpayer continues to "
        "operate the facility in the Commonwealth, the unused portion of the "
        "credit may be carried forward and applied in future taxable years until "
        "the credit for the facility has been fully used.",
        2,
    ),
    (
        "If the purchase price exceeds $ 4,000 and the taxpayer elects, "
        + CARRYFORWARD,
        2,
    ),
    (
        "If the household's allotment is not used and the household remains "
        "certified, the balance may be carried over and used until the balance "
        "has been fully used.",
        2,
    ),
    # Two conditions in one segment: a relative clause or a nested conditional.
    (
        "If an individual, who is a resident, contributes more than the annual "
        "limit, the remainder may be carried forward and subtracted in future "
        "taxable years until the contribution has been fully deducted.",
        2,
    ),
    (
        "If the taxpayer is a resident, then if the contribution exceeds the annual "
        "limit, the remainder may be carried forward and subtracted in future "
        "taxable years until the contribution has been fully deducted.",
        2,
    ),
)


@pytest.mark.parametrize(("text", "count"), NEGATIVE_CONTROLS)
def test_conditions_stay_demanded(text, count):
    gates = sc._source_conjunctive_fact_gates(text)
    assert len(gates) == count
    assert gates == _main_gates(text)


@pytest.mark.parametrize(
    "text",
    (
        VIRGINIA,
        VIRGINIA + ";",
        "If the purchase price exceeds $ 4,000, then " + CARRYFORWARD,
        "If the contribution exceeds the limit, the excess may be carried over and "
        "deducted in succeeding taxable years until the contribution has been "
        "fully deducted.",
        "If the credit exceeds the tax, the unused portion of the credit may be "
        "carried forward and applied in future taxable years until the credit has "
        "been fully used",
        VIRGINIA.upper(),
        VIRGINIA.replace(" ", " ").replace(", the", ",\nthe"),
    ),
)
def test_a_single_antecedent_condition_is_not_split(text):
    assert len(_main_gates(text)) == 2
    assert sc._source_conjunctive_fact_gates(text) == ()


@pytest.mark.parametrize(
    "antecedent",
    (
        "the taxpayer is a resident and the taxpayer is blind",
        "the purchase price exceeds $ 4,000 and the purchaser is a resident",
        "the purchaser is a resident and is blind and is employed",
    ),
)
def test_antecedent_conditions_keep_their_gates(antecedent):
    """`If A and B, <carryforward>` demands exactly the antecedent's gates."""

    with_consequent = sc._source_conjunctive_fact_gates(
        f"If {antecedent}, {CARRYFORWARD}"
    )
    alone = sc._source_conjunctive_fact_gates(f"If {antecedent}")
    assert len(alone) >= 2
    assert with_consequent == alone
    assert len(_main_gates(f"If {antecedent}, {CARRYFORWARD}")) > len(alone)


CONDITIONS = (
    "the taxpayer is a resident",
    "the taxpayer is blind",
    "the purchaser has attained age 70",
    "the contribution exceeds $ 4,000",
    "the spouse is employed",
    "the account is owned by the purchaser",
)
CONSEQUENTS = (
    CARRYFORWARD,
    "the remainder may be carried forward",
    "the excess may be carried over to succeeding taxable years",
    "then the unused portion of the credit may be carried forward and applied "
    "in future taxable years until fully used",
)


@given(
    conditions=st.lists(st.sampled_from(CONDITIONS), min_size=1, max_size=3),
    consequent=st.sampled_from(CONSEQUENTS),
)
@example(conditions=["the contribution exceeds $ 4,000"], consequent=CARRYFORWARD)
@settings(max_examples=120, deadline=None)
def test_carryforward_consequent_never_changes_antecedent_gates(conditions, consequent):
    antecedent = " and ".join(conditions)
    assert sc._source_conjunctive_fact_gates(
        f"If {antecedent}, {consequent}"
    ) == sc._source_conjunctive_fact_gates(f"If {antecedent}")


UNGATED = (
    "the taxpayer elects",
    "the taxpayer continues to operate the facility",
    "the household remains certified",
)
RESTRICTIONS = (
    ", so long as the contributor remains a resident,",
    " of a purchaser under age 70",
    " of a taxpayer residing in an enterprise zone",
    " for taxpayers age 65 or older",
    ", where the purchaser is a resident,",
    " that the taxpayer elects to carry",
)


@given(
    gated=st.lists(st.sampled_from(CONDITIONS), min_size=1, max_size=2),
    ungated=st.lists(st.sampled_from(UNGATED), min_size=1, max_size=2),
    order=st.permutations(range(4)),
)
@settings(max_examples=120, deadline=None)
def test_an_ungated_antecedent_condition_keeps_mains_gates(gated, ungated, order):
    """A condition the splitter cannot gate keeps main's consequent demand."""

    conditions = [*gated, *ungated]
    conditions = [conditions[index] for index in order if index < len(conditions)]
    text = f"If {' and '.join(conditions)}, {CARRYFORWARD}"
    assert sc._source_conjunctive_fact_gates(text) == _main_gates(text)


@given(
    conditions=st.lists(st.sampled_from(CONDITIONS), min_size=1, max_size=2),
    restriction=st.sampled_from(RESTRICTIONS),
)
@settings(max_examples=80, deadline=None)
def test_a_restricted_exhausted_amount_keeps_mains_gates(conditions, restriction):
    text = (
        f"If {' and '.join(conditions)}, the remainder may be carried forward and "
        f"subtracted in future taxable years until the contribution{restriction} "
        "has been fully deducted"
    )
    assert sc._source_conjunctive_fact_gates(text) == _main_gates(text)


@given(
    conditions=st.lists(st.sampled_from(CONDITIONS), min_size=1, max_size=2),
    nested=st.lists(st.sampled_from(CONDITIONS), min_size=1, max_size=2),
    # `unless` and `except` are left out: main already cuts the body at a
    # trailing exception, which the exception gates handle separately.
    cue=st.sampled_from(("if", "when", "provided that", "while", "where")),
)
@settings(max_examples=120, deadline=None)
def test_a_condition_in_the_consequent_keeps_mains_gates(conditions, nested, cue):
    """The same conditions, moved into the consequent, are not cut off."""

    text = (
        f"If {' and '.join(conditions)}, the remainder may be carried forward "
        f"and subtracted {cue} {' and '.join(nested)}"
    )
    assert sc._source_conjunctive_fact_gates(text) == _main_gates(text)


# The public gate on the replayed Virginia candidate and on synthetic modules.
def _public_issues(content, source, citation, tests):
    return sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=tests,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues


def test_replayed_carryforward_formula_with_federal_selector_passes(monkeypatch):
    fixture = FIXTURES / "signed_reencode_37936341740"
    recorded = json.loads((fixture / "issues.json").read_text())
    args = (
        (fixture / "candidate.yaml").read_text(),
        (fixture / "source.txt").read_text(),
        recorded["citation"],
        yaml.safe_load((fixture / "candidate.test.yaml").read_text()),
    )
    with monkeypatch.context() as patched:
        _without_carryforward_recognition(patched)
        main_issues = _public_issues(*args)
    flagged = [issue for issue in main_issues if issue.startswith(GATE)]
    assert len(flagged) == 1
    assert "`college_savings_account_carryforward` versions[0]" in flagged[0]
    issues = _public_issues(*args)
    assert not [issue for issue in issues if issue.startswith(GATE)]
    assert sorted(set(main_issues) - set(issues)) == flagged
    assert set(issues) <= set(main_issues)


CITATION = "us-va/statute/58.1/58.1-322.03"


def _module(clause, formula, inputs):
    rule = {
        "name": "deduction_carryforward",
        "kind": "derived",
        "entity": "Person",
        "dtype": "Money",
        "unit": "USD",
        "period": "Year",
        "source": "Va. Code § 58.1-322.03",
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {"corpus_citation_path": CITATION, "excerpt": clause},
                    }
                ]
            }
        },
    }
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"summary": "Carryforward."},
            "inputs": [
                {"name": name, "entity": "Person", "dtype": dtype, "period": "Year"}
                for name, dtype in inputs
            ],
            "rules": [rule],
        },
        sort_keys=False,
    )


TWO_CONDITIONS = (
    "If the taxpayer is a resident and the taxpayer is blind, the remainder "
    "may be carried forward and subtracted in future taxable years until the "
    "purchase price has been fully deducted."
)


OPAQUE = _module(
    TWO_CONDITIONS,
    "if taxpayer_qualifies: max(0, contribution - limit) else: 0",
    [("taxpayer_qualifies", "Boolean"), ("contribution", "Money"), ("limit", "Money")],
)


def _flagged(content):
    return [
        issue
        for issue in _public_issues(content, TWO_CONDITIONS, CITATION, [])
        if issue.startswith(GATE)
    ]


def test_a_genuine_two_condition_antecedent_still_needs_two_terminals(monkeypatch):
    """The issue's control: `if A and B, ...` with one opaque selector fails."""

    assert len(_flagged(OPAQUE)) == 1
    with monkeypatch.context() as patched:
        _without_carryforward_recognition(patched)
        assert len(_flagged(OPAQUE)) == 1


def test_the_two_condition_antecedent_demands_exactly_its_two_gates(monkeypatch):
    with monkeypatch.context() as patched:
        _without_carryforward_recognition(patched)
        assert "for 3 source gates" in _flagged(OPAQUE)[0]
    assert "for 2 source gates" in _flagged(OPAQUE)[0]


def test_a_genuine_two_condition_antecedent_passes_with_two_terminals():
    content = _module(
        TWO_CONDITIONS,
        "if taxpayer_is_resident and taxpayer_is_blind: max(0, contribution - limit) "
        "else: 0",
        [
            ("taxpayer_is_resident", "Boolean"),
            ("taxpayer_is_blind", "Boolean"),
            ("contribution", "Money"),
            ("limit", "Money"),
        ],
    )
    issues = _public_issues(content, TWO_CONDITIONS, CITATION, [])
    assert not [issue for issue in issues if issue.startswith(GATE)]


def _repository_texts():
    for root in (REPOSITORY / "tests/fixtures", REPOSITORY / "benchmarks"):
        for path in sorted(root.rglob("*.txt")):
            yield path.relative_to(REPOSITORY), path.read_text()


def test_fixture_differential_changes_only_the_named_clause():
    """Over every conditional clause in every repository source text, only the
    Virginia carry-forward sentence changes its gates.

    Clauses are the spans between sentence and semicolon boundaries, which is
    where the gate's clause ownership cuts them.
    """

    changed = []
    clauses = 0
    for path, text in _repository_texts():
        for clause in re.split(r"(?<=[.;])\s+|\n\s*\n", text):
            if not re.search(r"\b(?:if|when|unless|provided\s+that)\b", clause, re.I):
                continue
            clauses += 1
            clause = clause.strip(" ;,")
            current = sc._source_conjunctive_fact_gates(clause)
            main = _main_gates(clause)
            if current != main:
                changed.append((path, clause[:60], len(main), len(current)))
    assert clauses > 1000
    assert changed == [
        (
            FIXTURES.relative_to(REPOSITORY) / "signed_reencode_37936341740/source.txt",
            "If the purchase price or annual contribution to a college sa",
            2,
            0,
        )
    ]


class _CountingFullmatch:
    def __init__(self, pattern, meter):
        self._pattern = pattern
        self._meter = meter

    def fullmatch(self, string, pos=0):
        self._meter["fullmatch"] += 1
        return self._pattern.fullmatch(string, pos)


@pytest.mark.parametrize("copies", (1, 8, 64, 512))
def test_consequent_scan_is_one_anchored_match_per_top_level_comma(monkeypatch, copies):
    meter = {"fullmatch": 0, "segment_gates": 0}
    monkeypatch.setattr(
        sc,
        "_CARRYFORWARD_CONSEQUENT",
        _CountingFullmatch(sc._CARRYFORWARD_CONSEQUENT, meter),
    )
    segment_gates = sc._source_gate_segment_gates

    def counting_segment_gates(*args, **kwargs):
        meter["segment_gates"] += 1
        return segment_gates(*args, **kwargs)

    monkeypatch.setattr(sc, "_source_gate_segment_gates", counting_segment_gates)
    # `4,000` is a grouped number, not a clause boundary; every other comma is.
    clause = (
        "If the purchase price exceeds $ 4,000"
        + ", the spouse is blind" * copies
        + ", "
        + CARRYFORWARD
    )
    sc._source_conjunctive_fact_gates(clause)
    assert meter["fullmatch"] == copies + 1
    # A comma-listed antecedent keeps main's gates, so the antecedent's own
    # gates are never computed.
    assert meter["segment_gates"] == 1
    meter.update(fullmatch=0, segment_gates=0)
    clean = (
        "If the purchase price exceeds $ 4,000"
        + " and the spouse is blind" * copies
        + ", "
        + CARRYFORWARD
    )
    sc._source_conjunctive_fact_gates(clean)
    assert meter["fullmatch"] == 1
    assert meter["segment_gates"] == 2
