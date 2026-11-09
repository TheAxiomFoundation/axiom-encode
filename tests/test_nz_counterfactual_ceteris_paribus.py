"""A counterfactual rate's ceteris paribus clause is not a second factual gate.

Excerpt: Social Security Act 2018 (2018/32), Schedule 4, Part 7, clause 6A, as
replaced by the amending Act 2026/54, section 15, as quoted in
`tests/test_bare_structure_nz_harness.py`, from the `nz-rulespec-2026-10-20`
release (see `tests/fixtures/nz_bare_labels/README.md`).
Source: New Zealand Legislation (legislation.govt.nz), Crown copyright, CC BY 4.0

"The rate ... that would apply to them if they were single (and their other
circumstances were unchanged)" sets a hypothetical: everything except the
named status stays as it is. The parenthesis holds the comparison fixed; no
person must be shown to have unchanged circumstances. Every other shape keeps
base's two gates.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from tests.test_bare_structure_nz_harness import REPLACEMENT_TABLE

CITATION = "nz/statute/example/1"
FIXTURES = Path(__file__).parent / "fixtures" / "nz_bare_labels"
SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
S6A = next(
    line.split(" | ", 1)[1]
    for line in REPLACEMENT_TABLE.splitlines()
    if line.startswith("6A | ")
)
UNCHANGED = "(and their other circumstances were unchanged)"


def opaque_issues(source: str, module=sc) -> list[str]:
    rule = {
        "name": "claimant_rate",
        "kind": "derived",
        "dtype": "Judgment",
        "source": CITATION,
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {"corpus_citation_path": CITATION, "excerpt": source},
                    }
                ]
            }
        },
        "versions": [
            {"effective_from": "2026-01-01", "formula": "claimant_conditions_hold"}
        ],
    }
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "claimant_conditions_hold",
                "dtype": "Boolean",
                "description": "The claimant conditions hold.",
            }
        ],
        "rules": [rule],
    }
    return module._opaque_same_source_condition_input_issues(
        payload,
        source_text=source,
        branches=(),
        principal_rules={"claimant_rate": rule},
        corpus_citation_path=CITATION,
    )


@pytest.fixture(scope="module")
def receipts():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import nz_harness_defect_receipts as module
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    return module


@pytest.fixture(scope="module")
def base(receipts):
    return receipts.load_base_completeness()


def test_fixture_is_the_quoted_clause_6a():
    assert S6A.startswith("The rate of accommodation supplement payable")
    assert S6A.endswith(f"if they were single {UNCHANGED}.")


# Admitted examples. -------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    (
        S6A,
        "The rate is the rate in clauses 1 to 6AA that would apply to them "
        f"if they were single {UNCHANGED}.",
        "The amount is the amount that would otherwise apply to him if he were "
        "single (and his other circumstances were unchanged).",
        "The rate is the rate that would apply to her if she were a sole parent "
        "(and her other circumstances were unchanged)",
    ),
)
def test_ceteris_paribus_counterfactual_owes_no_unchanged_fact(base, source):
    # Base demands a second input for the held-fixed clause.
    assert len(base._source_conjunctive_fact_gates(source)) == 2
    assert len(opaque_issues(source, base)) == 1
    assert sc._source_conjunctive_fact_gates(source) == ()
    assert opaque_issues(source) == []


# Negative controls: base's gates stay. ------------------------------------------


@pytest.mark.parametrize(
    "source",
    (
        # The #1992 review input: `would apply` belongs to another clause.
        "Although the ordinary rate would apply, the claimant was eligible only "
        f"if the claimant was single {UNCHANGED}.",
        # Indicative mood: an actual condition, not a hypothetical.
        f"The rate is the rate that would apply to them if they are single {UNCHANGED}.",
        # `would apply` does not govern the condition directly.
        f"The rate that would apply to them applies if they were single {UNCHANGED}.",
        f"The rate would apply to them, if they were single {UNCHANGED}.",
        # Negated or modal variants.
        f"The rate is the rate that would not apply to them if they were single "
        f"{UNCHANGED}.",
        f"The rate is the rate that could apply to them if they were single "
        f"{UNCHANGED}.",
        # The parenthesis is not terminal, so more conditions may follow it.
        f"The rate is the rate that would apply to them if they were single "
        f"{UNCHANGED} and resident.",
        # The held-fixed clause names something narrower than all else.
        "The rate is the rate that would apply to them if they were single "
        "(and their income were unchanged).",
        # A disjunction is a second alternative, not a ceteris paribus clause.
        "The rate is the rate that would apply to them if they were single "
        "(or their other circumstances were unchanged).",
        # The pronouns name different people.
        f"The rate is the rate that would apply to the claimant if they were single "
        f"{UNCHANGED}.",
        f"The rate is the rate that would apply to them if he were single {UNCHANGED}.",
        "The rate is the rate that would apply to them if they were single "
        "(and his other circumstances were unchanged).",
        # A second stated condition beside the hypothetical status.
        "The rate is the rate that would apply to them if they were single and "
        f"resident {UNCHANGED}.",
        "The rate is the rate that would apply to them if they were single, "
        f"resident {UNCHANGED}.",
        *(
            f"The rate is the rate that would apply to them if they were single "
            f"{joiner} {UNCHANGED}."
            for joiner in (
                "yet employed",
                "whilst resident in New Zealand",
                "as well as in hardship",
                "plus resident",
                "& resident",
            )
        ),
        # Third-review inputs: a status that carries its own condition.
        *(
            f"The rate is the rate that would apply to them if they were {status} "
            f"{UNCHANGED}."
            for status in (
                "single provided they are resident",
                "single if resident",
                "single where resident",
                "single who were resident",
                "single when aged 65",
                "single having a dependent child",
                "single with 2 children",
                "single excluding any partner who is resident",
                "single or a sole parent",
            )
        ),
        # The cue, the governing `that`, and the held-fixed wording all matter.
        f"The rate is the rate that would apply to them unless they were single "
        f"{UNCHANGED}.",
        f"The rate would apply to them if they were single {UNCHANGED}.",
        "The rate is the rate that would apply to them if they were single "
        "(and their other circumstances were exceptional).",
        # Base's ordinary two-gate conditions.
        "The payment applies if the applicant is single and has no income.",
    ),
)
def test_other_shapes_keep_base_gates(base, source):
    assert sc._source_conjunctive_fact_gates(
        source
    ) == base._source_conjunctive_fact_gates(source)
    assert opaque_issues(source) == opaque_issues(source, base)


STATUS_STOPWORDS = (
    "provided if where wherever when whenever while whilst who whom whose which "
    "that until because since after before with without "
    "aged and or nor but yet plus as than for "
    "under in on at by of to from not rather only also"
).split()


def status_clause(status):
    return (
        f"The rate is the rate that would apply to them if they were {status} "
        f"{UNCHANGED}."
    )


@pytest.mark.parametrize("word", STATUS_STOPWORDS)
def test_each_status_stopword_keeps_base_gates(base, word):
    # Word level: `single <stopword>` fits the two-word shape, so only the
    # stopword itself keeps base's gates.
    source = status_clause(f"single {word}")
    assert sc._source_conjunctive_fact_gates(
        source
    ) == base._source_conjunctive_fact_gates(source)
    assert len(base._source_conjunctive_fact_gates(source)) >= 2


@pytest.mark.parametrize(
    "status",
    (
        "single pending",  # a participle
        "Single",  # upper case: a proper noun or heading, not a bare status
        "single resident overseas",  # three words: the cap
        "a single resident adult",  # article plus three words
    ),
)
def test_status_shape_limits_keep_base_gates(base, status):
    source = status_clause(status)
    assert sc._source_conjunctive_fact_gates(
        source
    ) == base._source_conjunctive_fact_gates(source)
    assert len(base._source_conjunctive_fact_gates(source)) >= 2


@pytest.mark.parametrize("status", ("single resident", "a resident", "an adult"))
def test_two_word_statuses_are_owned(base, status):
    # Paired with the limits: two words and an article stay hypothetical.
    assert sc._source_conjunctive_fact_gates(status_clause(status)) == ()
    assert len(base._source_conjunctive_fact_gates(status_clause(status))) >= 2


def test_review_input_keeps_its_exact_base_gates():
    source = (
        "Although the ordinary rate would apply, the claimant was eligible only "
        f"if the claimant was single {UNCHANGED}."
    )
    assert sc._source_conjunctive_fact_gates(source) == (
        (frozenset({"claimant"}), frozenset({"single"})),
        (frozenset({"claimant"}), frozenset({"circumstance", "other", "unchanged"})),
    )


# Invariant: only the exact ceteris paribus shape changes the gates. ------------

_PREFIX = st.sampled_from(
    (
        "The rate is the rate that would apply to them ",
        "The rate is the rate that would otherwise apply to them ",
        "Although the ordinary rate would apply, the claimant was eligible only ",
        "The rate that would apply to them applies ",
        "The claimant is eligible ",
    )
)
_CONDITION = st.sampled_from(
    (
        "if they were single",
        "if they are single",
        "if they were single and resident",
        "if the claimant was single",
        "if he were single",
    )
)
_TAIL = st.sampled_from(
    (
        f" {UNCHANGED}.",
        f" {UNCHANGED} and resident.",
        " (and their income were unchanged).",
        " and their other circumstances were unchanged.",
        " (or their other circumstances were unchanged).",
    )
)


@settings(max_examples=300, deadline=None)
@given(prefix=_PREFIX, condition=_CONDITION, tail=_TAIL)
def test_gates_change_only_for_the_owned_counterfactual(base, prefix, condition, tail):
    source = prefix + condition + tail
    gates = sc._source_conjunctive_fact_gates(source)
    owned = bool(
        re.search(r"\bthat would (?:otherwise )?apply to them $", prefix)
        and condition == "if they were single"
        and tail == f" {UNCHANGED}."
    )
    if owned:
        # The gates equal base's for the clause without the parenthesis. Both
        # gates go: with it gone, `single` alone falls below base's two-conjunct
        # threshold, as it does on main for the same clause.
        without_clause = source.replace(f" {UNCHANGED}", "")
        assert gates == base._source_conjunctive_fact_gates(without_clause) == ()
        assert len(base._source_conjunctive_fact_gates(source)) == 2
    else:
        assert gates == base._source_conjunctive_fact_gates(source)


def test_fixture_differential_changes_only_clause_6a(receipts, base):
    fresh = receipts.build_receipts(base_completeness=base)
    assert receipts.changed_gate_fixtures(fresh) == receipts.EXPECTED_GATE_CHANGES
    (row,) = fresh["gate_changes"]
    assert row["clause"] == S6A
    assert row["gates"] == []
    assert row["base_gates"] == [
        [[], ["single"]],
        [[], ["circumstance", "other", "unchanged"]],
    ]
    assert (FIXTURES / "amendment_2026_0054_s15_clause6a.txt").read_text() == (
        S6A + "\n"
    )
