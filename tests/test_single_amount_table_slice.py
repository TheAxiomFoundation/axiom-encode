"""Only a real table row or atomic amount leaf takes the single-row path (#1838).

The single-row path tells the model that a base case is enough, drops
"alternate" companion cases, and used to keep only numeric outputs in a mixed
case. A whole statute that happened to contain one currency amount took that
path, and staging deleted the Judgment assertions its branch gates needed.
The path now needs table structure or a one-sentence amount leaf, and the
normalizer keeps every assertion: a mixed case keeps its numeric outputs under
its own name and gains a twin, with the same facts, for the other outputs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import evals as ev

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "source",
    [
        # The reduced Virginia 58.1-339.8 section from #1838.
        "A. Eligibility is determined separately.\nB. The credit is $300.",
        # New Mexico 7-2-39: lettered branches around one $4,000 amount.
        "A. A taxpayer may claim the exemption for each dependent.\n"
        "B. The exemption is $4,000 for each qualifying dependent.\n"
        "C. The department shall prescribe the forms.",
        # Committed statutes with one threshold amount and no table.
        (
            ROOT / "benchmarks/verifier/fixtures/real_defects_example/cases/rd-0001"
            "/provision.txt"
        ).read_text(),
        (
            ROOT / "tests/fixtures/source_completeness/signed_reencode_37936331759"
            "/source.txt"
        ).read_text(),
    ],
)
def test_statute_with_one_currency_amount_is_not_a_table_row(source):
    assert not ev._is_single_amount_table_slice(source)


@pytest.mark.parametrize(
    "source",
    [
        "Editorial note: current text valid from 2025-04-07.\n\n"
        "Structured table:\n"
        "Element | Amount for each assessment period\n"
        "single claimant aged under 25 | £316.98",
        "£22,020 for joint claimants not resident in Greater London.",
        "(a) £20 in respect of each child.",
    ],
)
def test_table_row_and_atomic_amount_leaf_keep_the_single_row_path(source):
    assert ev._is_single_amount_table_slice(source)


def test_conditional_amount_leaf_stays_off_the_single_row_path():
    assert not ev._is_single_amount_table_slice(
        "£20 is disregarded if the claimant is in receipt of Scottish adult "
        "disability living allowance."
    )


ALLOWED = "us-va:statutes/example#credit_allowed"
AMOUNT = "us-va:statutes/example#credit_amount"


def test_mixed_case_keeps_its_judgment_assertion_in_a_twin_case():
    tests = yaml.safe_dump(
        [
            {
                "name": "eligible control",
                "period": 2026,
                "input": {"us-va:statutes/example#input.agi": 0},
                "output": {ALLOWED: "holds", AMOUNT: 300},
            }
        ]
    )
    normalized = yaml.safe_load(
        ev._normalize_single_amount_row_test_content(
            tests, source_text="£300 for each eligible claimant."
        )
    )
    assert normalized == [
        {
            "name": "eligible control",
            "period": 2026,
            "input": {"us-va:statutes/example#input.agi": 0},
            "output": {AMOUNT: 300},
        },
        {
            "name": "eligible control (non-amount outputs)",
            "period": 2026,
            "input": {"us-va:statutes/example#input.agi": 0},
            "output": {ALLOWED: "holds"},
        },
    ]


def test_twin_case_name_never_collides_with_an_existing_case():
    tests = yaml.safe_dump(
        [
            {"name": "base", "output": {ALLOWED: "holds", AMOUNT: 300}},
            {"name": "base (non-amount outputs)", "output": {AMOUNT: 1}},
        ]
    )
    names = [
        case["name"]
        for case in yaml.safe_load(ev._normalize_single_amount_row_test_content(tests))
    ]
    assert len(names) == len(set(names)) == 3


OUTPUT_VALUE = st.one_of(
    st.integers(min_value=-5, max_value=5000),
    st.sampled_from(["holds", "not_holds", True, False, None, 12.5]),
)


@settings(max_examples=150, deadline=None)
@given(
    cases=st.lists(
        st.fixed_dictionaries(
            {
                "name": st.sampled_from(
                    ["base", "other", "alternate branch", "x (non-amount outputs)"]
                ),
                "input": st.dictionaries(
                    st.sampled_from(["#input.a", "#input.b"]),
                    st.integers(0, 3),
                    max_size=2,
                ),
                "output": st.dictionaries(
                    st.sampled_from([ALLOWED, AMOUNT, "#other", "#rate"]),
                    OUTPUT_VALUE,
                    min_size=1,
                    max_size=4,
                ),
            }
        ),
        max_size=5,
        unique_by=lambda case: case["name"],
    )
)
def test_normalizer_keeps_every_assertion_with_its_own_facts(cases):
    normalized = (
        yaml.safe_load(
            ev._normalize_single_amount_row_test_content(yaml.safe_dump(cases))
        )
        or []
    )

    def assertions(items):
        return Counter(
            (repr(sorted(case.get("input", {}).items())), key, repr(value))
            for case in items
            for key, value in case.get("output", {}).items()
        )

    kept = [case for case in cases if "alternate" not in case["name"]]
    assert assertions(normalized) == assertions(kept)
    names = [case["name"] for case in normalized]
    assert len(names) == len(set(names))
