"""U.S. Code citation lists are locators, not SNAP policy amounts."""

from axiom_encode.harness.source_completeness import authoritative_numeric_recall_text
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)


def _recall_values(source: str, citation: str) -> set[float]:
    cleaned = authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    )
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            cleaned, profile="legacy"
        )
    }


def test_bracketed_usc_et_seq_list_does_not_demand_named_scalars():
    source = (
        "receives disability payments under the Social Security Act "
        "[ 42 U.S.C. 301 et seq., 401 et seq., 1201 et seq., "
        "1351 et seq., 1381 et seq.] and receives a $45 payment"
    )

    assert _recall_values(source, "us/statute/7/2012/j") == {45}


def test_usc_locator_mask_does_not_hide_a_non_citation_amount():
    source = (
        "[42 U.S.C. 301 et seq., 401 dollars] "
        "The household receives $1201."
    )

    values = _recall_values(source, "us/statute/7/2012/j")

    assert {401, 1201} <= values
