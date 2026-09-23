"""Printed row labels must not join independent form instructions."""

import pytest

from axiom_encode.harness import source_completeness as completeness

SOURCE = """Enter the amount from line 1 of Form T2209. 1
Enter the amount from line 3 of Form T2209, unless you have to pay minimum tax.(1) – 2
Line 1 minus line 2 = 3
Net foreign non-business income (2) Provincial or territorial
× = 4
Net income (3) tax otherwise payable (4)
Enter whichever amount is less: line 3 or line 4.
"""


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        (
            "Enter the amount from line 1 of Form T2209.",
            "Enter the amount from line 1 of Form T2209.",
        ),
        (
            "unless you have to pay minimum tax",
            " 1\nEnter the amount from line 3 of Form T2209, unless you have to pay minimum tax.",
        ),
        (
            "Line 1 minus line 2 = 3",
            "(1) – 2\nLine 1 minus line 2 = 3\nNet foreign non-business income (2) Provincial or territorial\n× = 4\nNet income (3) tax otherwise payable (4)\nEnter whichever amount is less: line 3 or line 4.",
        ),
    ],
)
def test_worksheet_sentence_ownership_stops_before_next_row(excerpt, expected):
    start = SOURCE.index(excerpt)
    a, b = completeness._source_proposition_bounds(SOURCE, start, start + len(excerpt))
    assert SOURCE[a:b].rstrip() == expected
    assert not completeness._source_conjunctive_fact_gates(SOURCE[a:b])


@pytest.mark.parametrize(
    "source",
    [
        "Multiply by 1.5 if the claimant is employed and disabled.",
        "Apply section 3.(1) if the claimant is employed and disabled.",
        "The amount is $3. 1 dollar is added if employed and disabled.",
        "The rule applies under Art. 1 of the statute if employed and disabled.",
    ],
)
def test_nonworksheet_numeric_prose_remains_one_proposition(source):
    start = source.index("if")
    assert completeness._source_proposition_bounds(source, start, len(source)) == (
        0,
        len(source),
    )


def test_proof_spanning_two_rows_remains_intact():
    excerpt = SOURCE[: SOURCE.index("Line 1 minus")].strip()
    a, b = completeness._source_proposition_bounds(SOURCE, 0, len(excerpt))
    assert SOURCE[a:b].startswith(excerpt)
    assert "unless you have to pay minimum tax" in SOURCE[a:b]
