"""Printed labels may finish proofs without changing subsequent clause starts."""

import pytest

from axiom_encode.harness.source_completeness import _source_proposition_bounds


@pytest.mark.parametrize("ending", [". 1", "! 12", "? 123", ".(2) – 13"])
def test_exact_proof_includes_printed_label(ending):
    proof = "If the amount is zero, enter zero" + ending
    text = proof + "\nAmount from another form – 2\nIf the total is zero, enter zero."
    start, end = _source_proposition_bounds(text, 0, len(proof))
    assert (start, end) == (0, len(proof))


def test_later_proof_retains_previous_leading_label():
    first = "If the amount is zero, enter zero. 1"
    next_proof = "Amount from another form – 2"
    text = first + "\n" + next_proof
    start, end = _source_proposition_bounds(text, len(first) + 1, len(text))
    assert text[start:end].lstrip().startswith("1\n")
    assert end == len(text)


@pytest.mark.parametrize(
    "prefix",
    [
        "If resident and ",
        "Subject to approval, ",
        "If resident then ",
    ],
)
def test_existing_same_proposition_prefix_is_retained(prefix):
    proof = "if the amount is zero, enter zero. 1"
    text = prefix + proof + "\nAmount from another form – 2"
    start, end = _source_proposition_bounds(text, len(prefix), len(prefix) + len(proof))
    assert start == 0
    assert end == len(prefix) + len(proof)
    assert text[start:end].startswith(prefix)


@pytest.mark.parametrize(
    "suffix",
    [
        ". 1 next instruction",
        ". 1\nlowercase continuation",
        ". 1.5\nAmount next",
        ". 1234\nAmount next",
    ],
)
def test_no_new_boundary_for_nonprinted_row_suffix(suffix):
    text = "Enter the amount" + suffix + " and require approval"
    proof_end = len("Enter the amount" + suffix.split("\n")[0])
    _, end = _source_proposition_bounds(text, 0, proof_end)
    assert end == len(text)
