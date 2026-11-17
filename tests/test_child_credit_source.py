"""Source-only records cannot certify the policy or related-provider execution."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.child_credit_source import recognize_child_credit_form

SOURCE = (Path(__file__).parent / "fixtures/child_credit/source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/ns479-credits"


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("crlf", [True, False])
def test_exact_raw_partition_and_binding(wrapped, crlf):
    source = SOURCE if wrapped else SOURCE.split("\n\n", 1)[1]
    if crlf:
        source = source.replace("\n", "\r\n")
    form = recognize_child_credit_form(source, CITATION)
    assert form is not None and form.year == 2025
    assert form.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    spans = sorted([r.span for r in form.records] + list(form.unowned_spans))
    assert spans[0][0] == 0 and spans[-1][1] == len(source)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert "".join(source[a:b] for a, b in spans) == source
    assert len(form.records) == 22
    assert form.bound(source, CITATION)
    assert not form.bound(source + " ", CITATION)
    assert not replace(form, year=2024).bound(source, CITATION)
    assert not replace(form, records=form.records[:-1]).bound(source, CITATION)
    first = form.records[0]
    assert source[first.span[0] : first.span[1]].startswith("\ufeff5003-TC")


@pytest.mark.parametrize("index", range(22))
@pytest.mark.parametrize(
    "extra",
    [
        "\n\nDo not claim this credit.",
        "\n\nunless you receive assistance.",
        "\n\nSubtract an additional $100.",
    ],
)
def test_extra_operative_text_after_every_record_refused(index, extra):
    form = recognize_child_credit_form(SOURCE, CITATION)
    assert form is not None
    at = form.records[index].span[1]
    assert (
        recognize_child_credit_form(SOURCE[:at] + extra + SOURCE[at:], CITATION) is None
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("2025 Nova", "2024 Nova"),
        ("Form NS479\n", "Form NS478\n"),
        ("six\ncolumns", "five\ncolumns"),
        ("possibility of six entries", "legal maximum of six children"),
        ("Column B minus column C", "Column C minus column B"),
        ("Maximum $500 per child", "Maximum $500 per family"),
        ("received or receivable", "received only"),
        ("claimed by others", "claimed by you"),
        ("8.79%", "8.78%"),
        ("line 47900", "line 47800"),
        ("If applicable, complete", "If ineligible, complete"),
        ("Do not include your receipts", "Include your receipts"),
        ("one child per line", "one family per line"),
        ("keep them", "discard them"),
    ],
)
def test_changed_source_roles_refused(old, new):
    assert old in SOURCE
    assert recognize_child_credit_form(SOURCE.replace(old, new, 1), CITATION) is None


@pytest.mark.parametrize("index", range(22))
def test_duplicated_clause_refused(index):
    form = recognize_child_credit_form(SOURCE, CITATION)
    assert form is not None
    a, b = form.records[index].span
    assert (
        recognize_child_credit_form(
            SOURCE[:b] + "\n\n" + SOURCE[a:b] + SOURCE[b:], CITATION
        )
        is None
    )


def test_no_authority_and_layout_kept_raw():
    form = recognize_child_credit_form(SOURCE, CITATION)
    assert form is not None
    roles = {r.role: SOURCE[slice(*r.span)] for r in form.records}
    assert "six entries" in roles["transcriber_table"]
    assert "receivable" in roles["table_applicability_and_assistance"]
    assert "canada.ca/ns-tax-info" in roles["eligibility_referral"]
    assert "may get a refund" in roles["refund"]
    assert not hasattr(form, "owns_condition") and not hasattr(
        form, "legal_child_limit"
    )
    assert recognize_child_credit_form("Do not:\n\n" + SOURCE, CITATION) is None
    assert recognize_child_credit_form(SOURCE, "") is None
    assert recognize_child_credit_form(None, CITATION) is None
