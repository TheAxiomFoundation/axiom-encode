"""Condition-only partitions require complete, unquoted publisher row context."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as c

SOURCE = (Path(__file__).parent / "fixtures/t2036_operand_gates/source.txt").read_text()
START = SOURCE.index("Enter the amount from line 1 of Form T2209.")
END = SOURCE.index("Form 428.", START) + len("Form 428.")
MAINLINE = SOURCE[START:END]


def partition(source, start=0, end=None):
    end = len(source) if end is None else end
    clause = c._SourceConditionClause((), start, end, source[start:end])
    return c._partition_condition_clause_at_rows(
        clause, source_text=source, row_ends=c._corroborated_unless_row_ends(source)
    )


def test_actual_full_source_separates_operands_without_changing_any_bytes():
    parts = partition(SOURCE, START, END)
    assert len(parts) == 2
    assert "unless you have to pay minimum tax.(1) – 2" in parts[0].text
    assert parts[1].text.startswith("\nLine 1 minus line 2")
    assert len(c._source_conjunctive_fact_gates(MAINLINE)) == 4
    assert all(not c._source_conjunctive_fact_gates(p.text) for p in parts)
    assert "".join(SOURCE[p.start : p.end] for p in parts) == MAINLINE
    assert SOURCE[parts[0].end : parts[0].end + 1] == "\n"


@pytest.mark.parametrize(
    "old,new",
    [
        ("× = 4", "x = 4"),
        ("Line 1 minus line 2 = 3", "Line 1 minus line 2 = 9"),
        ("Enter the total from line 5", "Enter the total from line 6"),
        ("minimum tax.(1) – 2", "minimum tax if eligible.(1) – 2"),
        ("minimum tax.(1) – 2", "minimum tax only for residents.(1) – 2"),
        ("minimum tax.(1) – 2", "minimum tax for spouses who are disabled.(1) – 2"),
        ("minimum tax.(1) – 2", "minimum tax subject to approval.(1) – 2"),
        ("minimum tax.(1) – 2", "minimum tax, and enter another amount.(1) – 2"),
        ("minimum tax.(1) – 2", "minimum tax; unless resident.(1) – 2"),
    ],
)
def test_partial_unknown_or_restrictive_layout_does_not_partition(old, new):
    assert not c._corroborated_unless_row_ends(SOURCE.replace(old, new))


@pytest.mark.parametrize(
    "prefix,suffix", [('"', '"'), ("“", "”"), ('"', ""), ("“", '"'), ("[", ")")]
)
def test_quoted_or_malformed_context_is_not_admitted(prefix, suffix):
    assert not c._corroborated_unless_row_ends(prefix + "\n" + SOURCE + suffix)


def test_partial_excerpt_without_full_publisher_context_cannot_supply_boundary():
    assert not c._corroborated_unless_row_ends(MAINLINE)
    assert len(partition(MAINLINE)) == 1


@pytest.mark.parametrize("cue", ["If", "Unless"])
def test_later_genuine_condition_is_retained_as_an_independent_obligation(cue):
    later = f"\n{cue} the claimant is resident and the spouse is disabled, enter zero."
    source = SOURCE[:END] + later + SOURCE[END:]
    parts = partition(source, START, END + len(later))
    assert len(parts) == 1
    assert later in parts[0].text
    assert c._source_conjunctive_fact_gates(parts[0].text)


def condition_issues(source, excerpt):
    path = "ca/policy/example"
    rule = {
        "name": "credit",
        "kind": "derived",
        "dtype": "Money",
        "versions": [{"formula": "if minimum_tax_is_payable: 1 else: 2"}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {"corpus_citation_path": path, "excerpt": excerpt},
                    }
                ]
            }
        },
    }
    payload = {
        "rules": [rule],
        "inputs": [
            {"name": "minimum_tax_is_payable", "description": "Minimum tax is payable"}
        ],
    }
    return c._opaque_same_source_condition_input_issues(
        payload,
        source_text=source,
        branches=c.recognize_source_structure(source),
        principal_rules={"credit": rule},
        corpus_citation_path=path,
    )


def test_actual_condition_collector_no_longer_requires_ratio_operands_as_facts():
    assert condition_issues(SOURCE, MAINLINE) == []
    partial = MAINLINE
    assert condition_issues(partial, partial)


def test_actual_condition_collector_still_rejects_later_missing_factual_gates():
    later = "\nIf the claimant is resident and the spouse is disabled, enter zero."
    source = SOURCE[:END] + later + SOURCE[END:]
    assert condition_issues(source, MAINLINE + later)


@pytest.mark.parametrize(
    "tail",
    [
        " Only claimants who are residents and whose spouses are disabled may claim.",
        " Except for claimants who are resident and disabled.",
        " Provided the claimant is resident and the spouse is disabled, enter zero.",
        " Where the claimant is resident and the spouse is disabled, enter zero.",
        " Until the claimant is resident and the spouse is disabled, enter zero.",
        " Claimants must be resident and have a disabled spouse.",
        " Enter another amount.",
    ],
)
def test_unknown_or_restrictive_transfer_tail_keeps_original_scan(tail):
    source = SOURCE[:END] + tail + SOURCE[END:]
    assert len(partition(source, START, END + len(tail))) == 1
    assert condition_issues(source, MAINLINE + tail)


def test_wrong_transfer_field_is_not_accepted_as_complete_row_ownership():
    source = SOURCE.replace(
        "on the line for the provincial or territorial foreign tax credit of",
        "on the line for the credit for disabled claimants of",
    )
    end = source.index("Form 428.", START) + len("Form 428.")
    assert len(partition(source, START, end)) == 1


@pytest.mark.parametrize(
    "old,new",
    [
        ("Net foreign", "Only resident claimants"),
        ("Net foreign", "Claimants must be resident"),
        (
            "non-business income (2) Provincial or territorial",
            "non-business income (2) Disabled spouse required",
        ),
        (
            "Net income (3) tax otherwise payable (4)",
            "Only residents (3) tax otherwise payable (4)",
        ),
    ],
)
def test_broad_numeric_caption_is_not_condition_admission(old, new):
    source = SOURCE.replace(old, new)
    end = source.index("Form 428.", START) + len("Form 428.")
    assert c._corroborated_form_output_label_spans(source)
    assert len(partition(source, START, end)) == 1
    assert condition_issues(source, source[START:end])


@pytest.mark.parametrize(
    "title", ["other tax credit", "credit only for disabled claimants"]
)
def test_matching_unrelated_heading_does_not_authorize_a_different_transfer(title):
    source = (
        title
        + "\n"
        + SOURCE.replace(
            "on the line for the provincial or territorial foreign tax credit of",
            f"on the line for the {title} of",
        )
    )
    start = source.index("Enter the amount from line 1 of Form T2209.")
    end = source.index("Form 428.", start) + len("Form 428.")
    assert c._corroborated_form_output_label_spans(source)
    assert len(partition(source, start, end)) == 1
    assert condition_issues(source, source[start:end])
