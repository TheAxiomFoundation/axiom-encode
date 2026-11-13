"""Start date of a date-silent source unit (#2212).

The cases mirror the local encode of
``us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-16`` on
2026-11-12: the unit prints no date, the corpus snapshot is 2026-07-17,
attempt 1 copied ``2025-10-01`` from a neighbouring module and attempt 2 used
the evidence floor ``2026-08-01``. The module text here is synthetic; the
attempt-1 candidate itself is not public.
"""

from __future__ import annotations

import dataclasses
import functools
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

from axiom_encode import checks
from axiom_encode.checks import ValidationIssue
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import start_date_evidence as sde
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.eval_prompt_surface import (
    render_date_silent_scaffold_guidance,
    render_date_silent_start_guidance,
)
from axiom_encode.harness.start_date_evidence import (
    CHECK_START_UNSUPPORTED,
    StartDateEvidence,
    date_silent_start_issues,
    evidence_floor,
    start_date_evidence_from_metadata,
    text_prints_any_date,
    text_prints_date,
)

CITATION = "us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-16"
CHECK_ID = "complete-source-unit:effective-date/start-unsupported"
SOURCE = (
    "A budgetary unit that pays a telephone expense and no other utility "
    "expense is allowed the telephone allowance. The allowance is not "
    "prorated when the expense is shared."
)
SNAPSHOT = date(2026, 7, 17)
EVIDENCE = StartDateEvidence(source_as_of=SNAPSHOT, expression_date=SNAPSHOT)


def metadata(
    source_as_of: object = "2026-07-17",
    expression_date: object = "2026-07-17",
    *,
    row: dict | None = None,
) -> dict:
    attestation: dict = {
        "requested_corpus_citation_path": CITATION,
        "row": {"citation_path": CITATION, "document_class": "manual"},
    }
    for name, value in (
        ("source_as_of", source_as_of),
        ("expression_date", expression_date),
    ):
        if value is not None:
            attestation[name] = value
            attestation["row"][name] = value
    if row is not None:
        attestation["row"].update(row)
    return {"source_attestation": attestation}


def module(*starts: object, excerpt: str | None = None, **top: object) -> dict:
    """A one-rule module per start date, in the block-16 shape."""

    rules = []
    for index, start in enumerate(starts or ("2025-10-01",)):
        rule: dict = {
            "name": f"telephone_allowance_applies_{index}",
            "kind": "derived",
            "dtype": "Boolean",
            "entity": "Household",
            "period": "Month",
            "source": CITATION,
            "versions": [{"effective_from": start, "formula": "pays_telephone"}],
        }
        if excerpt is not None:
            rule["metadata"] = {
                "proof": {
                    "atoms": [
                        {
                            "path": "versions[0].formula",
                            "kind": "effective_period",
                            "source": {
                                "corpus_citation_path": CITATION,
                                "excerpt": excerpt,
                            },
                        }
                    ]
                }
            }
        rules.append(rule)
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "pays_telephone",
                "dtype": "Boolean",
                "entity": "Household",
                "period": "Month",
            }
        ],
        "rules": rules,
        **top,
    }


def issues(
    payload: dict,
    *,
    evidence: StartDateEvidence = EVIDENCE,
    source: str = SOURCE,
    citation: str = CITATION,
) -> list[ValidationIssue]:
    return date_silent_start_issues(
        payload, source, corpus_citation_path=citation, evidence=evidence
    )


# --------------------------------------------------------------------------
# Evidence floor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source_as_of", "floor"),
    [
        (date(2026, 7, 17), date(2026, 8, 1)),  # block 16
        (date(2026, 7, 1), date(2026, 7, 1)),  # the 1st covers its own month
        (date(2026, 7, 2), date(2026, 8, 1)),
        (date(2026, 7, 31), date(2026, 8, 1)),
        (date(2026, 12, 17), date(2027, 1, 1)),  # December rolls the year
        (date(2026, 12, 1), date(2026, 12, 1)),
        (date(2026, 12, 31), date(2027, 1, 1)),
        (date(2026, 1, 31), date(2026, 2, 1)),
        (date(2026, 1, 30), date(2026, 2, 1)),
        (date(2024, 2, 29), date(2024, 3, 1)),
        (date(2026, 11, 30), date(2026, 12, 1)),
    ],
)
def test_evidence_floor_is_the_first_full_month_on_or_after_the_snapshot(
    source_as_of, floor
):
    assert evidence_floor(source_as_of) == floor
    assert StartDateEvidence(source_as_of, None).evidence_floor == floor


def test_evidence_floor_needs_a_snapshot():
    assert StartDateEvidence(None, SNAPSHOT).evidence_floor is None


# --------------------------------------------------------------------------
# Block 16
# --------------------------------------------------------------------------


def test_block_16_attempt_1_copied_precedent_date_is_refused():
    found = issues(module("2025-10-01"))
    assert len(found) == 1
    issue = found[0]
    assert issue.check == CHECK_ID
    assert issue.startswith("[complete-source-unit:effective-date] ")
    assert issue.found == "2025-10-01"
    assert issue.subject.citation == CITATION
    assert issue.subject.rule == "telephone_allowance_applies_0"
    assert issue.subject.field == "versions[0].effective_from"
    # The message names the field and every permitted source of a date, so a
    # retry can fix it.
    for needle in (
        "`rules[].versions[].effective_from` is `2025-10-01`",
        "it is not the evidence floor `2026-08-01`, the corpus `source_as_of` "
        "`2026-07-17` or the corpus `expression_date`",
        "no proof-atom excerpt prints it",
        "A date copied from a neighbouring module is not support",
        "a date printed in the unit text or in a proof-atom excerpt you cite",
        "the corpus `expression_date` `2026-07-17`",
        "the corpus `source_as_of` `2026-07-17`",
        "the evidence floor `2026-08-01`",
        "not a claimed legal commencement date",
        "the `0001-01-01` scaffold",
    ):
        assert needle in issue, needle
    # The retry text names no way out of the check's scope.
    assert "import" not in issue.lower()


def test_block_16_attempt_2_evidence_floor_passes():
    assert issues(module("2026-08-01")) == []


@pytest.mark.parametrize(
    ("start", "refused"),
    [
        ("2026-07-16", True),  # one day before the snapshot
        ("2026-07-17", False),  # the snapshot day itself
        ("2026-07-18", True),  # after the snapshot, but not the floor
        ("2026-07-01", True),  # the month of the snapshot is not covered
        ("2026-07-31", True),
        ("2026-08-01", False),  # the evidence floor
        ("2026-08-02", True),
        ("2026-09-01", True),  # a later month start is not the floor
        ("2027-01-01", True),
        ("2099-01-01", True),  # an arbitrary later date is not support
        ("0001-01-02", True),  # only the exact scaffold date is the scaffold
    ],
)
def test_start_must_equal_a_supported_date(start, refused):
    # An expression_date apart from the snapshot, so it supports none of these.
    evidence = StartDateEvidence(SNAPSHOT, date(2020, 1, 1))
    assert bool(issues(module(start), evidence=evidence)) is refused


def test_floor_of_a_snapshot_on_the_first_is_that_day_only():
    evidence = StartDateEvidence(date(2026, 12, 1), None)
    assert issues(module("2026-12-01"), evidence=evidence) == []
    assert len(issues(module("2027-01-01"), evidence=evidence)) == 1
    december = StartDateEvidence(date(2026, 12, 17), None)
    assert issues(module("2027-01-01"), evidence=december) == []
    assert len(issues(module("2026-12-01"), evidence=december)) == 1


@pytest.mark.parametrize("scaffold", ["0001-01-01", date(1, 1, 1)])
def test_scaffold_start_passes(scaffold):
    assert issues(module(scaffold)) == []
    # Also without any corpus date: the scaffold claims no start.
    assert issues(module(scaffold), evidence=StartDateEvidence(None, None)) == []


def test_expression_date_passes():
    evidence = StartDateEvidence(SNAPSHOT, date(2025, 10, 1))
    assert issues(module("2025-10-01"), evidence=evidence) == []
    assert len(issues(module("2025-10-02"), evidence=evidence)) == 1
    assert len(issues(module("2025-09-30"), evidence=evidence)) == 1


@pytest.mark.parametrize(
    "excerpt",
    [
        "Effective 10/01/2025, the allowance is not prorated.",
        "Effective October 1, 2025, the allowance is not prorated.",
        "For the benefit month of 10/2025 and ongoing",
        "This policy applies from 2025-10-01.",
        "Beginning 1 Oct 2025 the allowance applies.",
        "Beginning October 2025 the allowance applies.",
    ],
)
def test_date_printed_in_a_proof_excerpt_passes(excerpt):
    assert issues(module("2025-10-01", excerpt=excerpt)) == []


@pytest.mark.parametrize(
    "excerpt",
    [
        "The allowance is not prorated.",
        "Effective 10/02/2025, the allowance is not prorated.",
        "Effective 10/01/2024, the allowance is not prorated.",
        "Effective 15 October 2025, the allowance is not prorated.",
        "Effective 11/10/2025, the allowance is not prorated.",
        "Effective 2025-10-011.",
    ],
)
def test_another_date_in_a_proof_excerpt_is_not_support(excerpt):
    assert len(issues(module("2025-10-01", excerpt=excerpt))) == 1


def test_date_printed_in_the_unit_text_takes_the_unit_out_of_the_rule():
    source = SOURCE + " This policy is effective 10/01/2025."
    assert issues(module("2025-10-01"), source=source) == []
    # Any printed date does: the unit is then not date-silent.
    assert issues(module("2025-10-01"), source=SOURCE + " Revised 03/2024.") == []


# --------------------------------------------------------------------------
# The narrow scope
# --------------------------------------------------------------------------


IMPORT = "us-az:policies/des/faa6/utility-allowance-current-amount#amount"


def provider(*starts: str, name: str = "amount") -> str:
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "rules": [
                {
                    "name": name,
                    "kind": "parameter",
                    "versions": [
                        {"effective_from": start, "formula": "1"} for start in starts
                    ],
                },
                {
                    "name": "other",
                    "kind": "parameter",
                    "versions": [{"effective_from": "2023-01-01", "formula": "2"}],
                },
            ],
        }
    )


def import_issues(start: str, contents, imports=(IMPORT,)) -> list[ValidationIssue]:
    return date_silent_start_issues(
        module(start, imports=list(imports)),
        SOURCE,
        corpus_citation_path=CITATION,
        evidence=EVIDENCE,
        imported_symbol_contents=contents,
    )


def test_start_of_an_imported_export_is_not_examined():
    contents = [("amount", provider("2024-10-01", "2025-10-01"))]
    assert import_issues("2025-10-01", contents) == []
    assert import_issues("2024-10-01", contents) == []
    assert import_issues("2026-08-01", contents) == []


def test_an_import_does_not_take_other_dates_out_of_the_check():
    contents = [("amount", provider("2024-10-01"))]
    # Not a start of the imported export: not of the module, and not of
    # another rule of the provider.
    assert len(import_issues("2025-10-01", contents)) == 1
    assert len(import_issues("2023-01-01", contents)) == 1
    found = import_issues("2025-10-01", contents)[0]
    assert "import" not in found.lower()


GHOST_IMPORT = (
    "us-az:policies/des/faa5/na-utility-expenses-and-allowances/block-13#pays_telephone"
)
FILE_IMPORT = "us-az:policies/des/faa5/na-utility-expenses-and-allowances/block-13"


@pytest.mark.parametrize(
    ("contents", "imports"),
    [
        ([], (IMPORT,)),  # provider not supplied
        ([("another", provider("2025-10-01"))], (IMPORT,)),
        ([("amount", "rules: [")], (IMPORT,)),  # unreadable provider
        ([("amount", "- not a mapping")], (IMPORT,)),
        # A "ghost" symbol: the provider is supplied but has no such rule.
        ([("pays_telephone", provider("2025-10-01"))], (GHOST_IMPORT,)),
        ([("amount", provider("2025-10-01", name="renamed"))], (IMPORT,)),
        # Several rules of the imported name in one provider.
        ([("other", provider("2025-10-01", name="other"))], (IMPORT + "x#other",)),
        ([], (FILE_IMPORT,)),  # file-level import, no `#symbol`
        ([("amount", provider("2025-10-01"))], (FILE_IMPORT,)),
        ([("amount", provider("2025-10-01"))], ("no-symbol", 7, None, ["x#amount"])),
        ([("", provider("2025-10-01", name=""))], ("#",)),
    ],
)
def test_unverifiable_import_adds_nothing_and_the_module_is_examined(contents, imports):
    # Refused at the copied date, passes at the floor: the module is examined
    # exactly as a module without imports.
    assert len(import_issues("2025-10-01", contents, imports)) == 1
    assert import_issues("2026-08-01", contents, imports) == []
    payload = module("2025-10-01", imports=list(imports))
    assert sde.imported_export_starts(payload, contents) == frozenset()


@pytest.mark.parametrize("imports", ["x#amount", {"a": IMPORT}, 7, True])
def test_imports_that_are_not_a_list_add_nothing(imports):
    contents = [("amount", provider("2025-10-01"))]
    found = date_silent_start_issues(
        module("2025-10-01", imports=imports),
        SOURCE,
        corpus_citation_path=CITATION,
        evidence=EVIDENCE,
        imported_symbol_contents=contents,
    )
    assert len(found) == 1


def test_verifiable_import_beside_unverifiable_ones_still_adds_its_starts():
    contents = [("amount", provider("2024-10-01"))]
    mixed = (FILE_IMPORT, GHOST_IMPORT, 7, IMPORT)
    assert import_issues("2024-10-01", contents, mixed) == []
    assert len(import_issues("2025-10-01", contents, mixed)) == 1
    # Two supplied providers of one symbol name: the readable one counts.
    two = [("amount", "rules: ["), ("amount", provider("2025-10-01"))]
    assert import_issues("2025-10-01", two) == []


def test_import_shapes():
    assert sde.imported_export_starts(module("2025-10-01"), ()) == frozenset()
    assert sde.imported_export_starts(module(imports=[]), ()) == frozenset()
    assert sde.imported_export_starts(
        module(imports=[IMPORT]), [("amount", provider("2024-10-01", "bad-date"))]
    ) == frozenset({date(2024, 10, 1)})
    # An `import` proof atom alone is no import.
    with_import_atom = module("2025-10-01")
    with_import_atom["rules"][0]["metadata"] = {
        "proof": {"atoms": [{"kind": "import", "import": IMPORT}]}
    }
    assert len(issues(with_import_atom)) == 1


def test_only_proof_atom_excerpts_count():
    for metadata_block in (
        {"note": {"excerpt": "Effective 10/01/2025"}},
        {"proof": {"excerpt": "Effective 10/01/2025"}},
        {"proof": {"atoms": [{"excerpt": "Effective 10/01/2025"}]}},
        {"proof": {"atoms": [{"source": "Effective 10/01/2025"}]}},
        {"proof": {"atoms": [{"source": {"note": "Effective 10/01/2025"}}]}},
    ):
        payload = module("2025-10-01")
        payload["rules"][0]["metadata"] = metadata_block
        assert len(issues(payload)) == 1, metadata_block
    payload = module("2025-10-01")
    payload["excerpt"] = "Effective 10/01/2025"
    payload["rules"][0]["summary"] = {"excerpt": "Effective 10/01/2025"}
    assert len(issues(payload)) == 1
    assert sde.proof_excerpts(module("2025-10-01", excerpt="Effective 10/01/2025")) == (
        "Effective 10/01/2025",
    )


@pytest.mark.parametrize(
    "citation",
    [
        "us/statute/7/2014/e",
        "us/regulation/7/273/9",
        "us-md/guidance/dhs/fia/snap-manual-214/page-5",
        "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
        "manual",
        "",
    ],
)
def test_only_manual_documents_are_in_scope(citation):
    assert issues(module("2025-10-01"), citation=citation) == []


@pytest.mark.parametrize(
    "citation",
    [CITATION, "us-wv/manual/bfa/income-maintenance-manual/page-3", "uk/manual/x/1"],
)
def test_manual_documents_of_any_jurisdiction_are_in_scope(citation):
    assert len(issues(module("2025-10-01"), citation=citation)) == 1


def test_amendment_context_is_out_of_scope():
    evidence = dataclasses.replace(EVIDENCE, amendment_context=True)
    assert issues(module("2025-10-01"), evidence=evidence) == []


def test_one_issue_per_refused_date_with_distinct_fingerprints():
    found = issues(module("2025-10-01", "2023-01-01", "2025-10-01", "2026-08-01"))
    assert [issue.found for issue in found] == ["2023-01-01", "2025-10-01"]
    assert "in 2 version(s)" in found[1]
    assert (
        "`telephone_allowance_applies_0`, `telephone_allowance_applies_2`" in (found[1])
    )
    assert found[0].fingerprint != found[1].fingerprint
    assert found[1].fingerprint == issues(module("2025-10-01"))[0].fingerprint


@pytest.mark.parametrize(
    "start",
    [
        "2025-10-1",  # the engine compiles a start without zero padding
        "2025-10-01 ",
        datetime(2025, 10, 1),
        datetime(2025, 10, 1, 12, 30),
        date(2025, 10, 1),
    ],
)
def test_lenient_start_forms_are_read_as_their_calendar_day(start):
    found = issues(module(start))
    assert len(found) == 1
    assert found[0].found == "2025-10-01"
    assert found[0].fingerprint == issues(module("2025-10-01"))[0].fingerprint


@pytest.mark.parametrize(
    "start", ["2026-8-1", "2026-08-1", datetime(2026, 8, 1), "0001-1-1", "2026-7-17"]
)
def test_lenient_forms_of_a_supported_start_pass(start):
    assert issues(module(start)) == []


@pytest.mark.parametrize(
    ("start", "found"),
    [
        ("October 2025", "October 2025"),
        ("2025-10-01T00:00:00", "2025-10-01T00:00:00"),
        ("2025-13-01", "2025-13-01"),
        ("20251001", "20251001"),
        (20251001, "20251001"),
        ("", ""),
        (True, "True"),
    ],
)
def test_unreadable_start_in_scope_is_refused_not_skipped(start, found):
    result = issues(module(start))
    assert len(result) == 1
    assert f"`rules[].versions[].effective_from` is `{found}`" in result[0]
    # Out of scope it stays with the other validators.
    assert issues(module(start), citation="us/statute/7/2014/e") == []


def test_version_without_a_start_is_left_to_other_checks():
    assert issues(module(None)) == []
    payload = module("2026-08-01")
    del payload["rules"][0]["versions"][0]["effective_from"]
    assert issues(payload) == []


# --------------------------------------------------------------------------
# Missing or unreadable corpus metadata never counts as support
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source_metadata", "problem"),
    [
        (None, "source_attestation is missing"),
        ({}, "source_attestation is missing"),
        ({"source_attestation": "x"}, "source_attestation is missing"),
        (metadata(source_as_of=None), "source_attestation.source_as_of is missing"),
        (
            metadata(source_as_of="2026-07-17T00:00:00Z"),
            "source_attestation.source_as_of is not a YYYY-MM-DD date",
        ),
        (
            metadata(source_as_of="2026-02-30"),
            "source_attestation.source_as_of is not a YYYY-MM-DD date",
        ),
        (
            metadata(source_as_of=20260717),
            "source_attestation.source_as_of is not a YYYY-MM-DD date",
        ),
        (
            metadata(row={"source_as_of": "2026-07-18"}),
            "source_attestation.source_as_of and "
            "source_attestation.row.source_as_of differ",
        ),
    ],
)
def test_missing_or_unreadable_snapshot_refuses_an_unsupported_start(
    source_metadata, problem
):
    evidence = start_date_evidence_from_metadata(source_metadata)
    assert evidence.source_as_of is None
    assert problem in evidence.problems
    # Even a start after the real snapshot: without a readable snapshot it
    # cannot be shown to be on or after it.
    for start in ("2025-10-01", "2026-08-01"):
        found = issues(module(start), evidence=evidence)
        assert len(found) == 1, start
        assert "the corpus `source_as_of` is unavailable" in found[0]
        assert problem in found[0]
        assert "the evidence floor `" not in found[0]
        assert "no evidence floor to equal" in found[0]
    assert issues(module("0001-01-01"), evidence=evidence) == []
    assert (
        issues(module("2025-10-01", excerpt="Effective 10/01/2025."), evidence=evidence)
        == []
    )


def test_unreadable_expression_date_supports_nothing():
    evidence = start_date_evidence_from_metadata(
        metadata(expression_date="October 1, 2025")
    )
    assert evidence.source_as_of == SNAPSHOT
    assert evidence.expression_date is None
    assert evidence.problems == (
        "source_attestation.expression_date is not a YYYY-MM-DD date",
    )
    assert len(issues(module("2025-10-01"), evidence=evidence)) == 1
    assert issues(module("2026-08-01"), evidence=evidence) == []


def test_preferred_start():
    def preferred(source_as_of, expression_date):
        return StartDateEvidence(source_as_of, expression_date).preferred_start

    assert preferred(SNAPSHOT, SNAPSHOT) == date(2026, 8, 1)
    assert preferred(SNAPSHOT, None) == date(2026, 8, 1)
    assert preferred(SNAPSHOT, date(2025, 10, 1)) == date(2026, 8, 1)
    assert preferred(SNAPSHOT, date(2026, 8, 1)) == date(2026, 8, 1)
    # IRS notice 2025-67 page 4 and the SNAP FY2026 COLA memo: the expression
    # date is after the floor, so the floor would start the rule too early.
    assert preferred(date(2025, 11, 13), date(2026, 1, 1)) == date(2026, 1, 1)
    assert preferred(date(2025, 8, 14), date(2025, 10, 1)) == date(2025, 10, 1)
    assert preferred(SNAPSHOT, date(2026, 8, 2)) == date(2026, 8, 2)
    assert preferred(None, date(2025, 10, 1)) == date(2025, 10, 1)
    assert preferred(None, None) == date(1, 1, 1)


def test_readable_metadata_has_no_problem_and_reads_both_copies():
    evidence = start_date_evidence_from_metadata(
        metadata(expression_date="2025-10-01"), amendment_context=True
    )
    assert evidence == StartDateEvidence(SNAPSHOT, date(2025, 10, 1), (), True)
    only_row = {"source_attestation": {"row": {"source_as_of": "2026-07-17"}}}
    assert start_date_evidence_from_metadata(only_row).source_as_of == SNAPSHOT


# --------------------------------------------------------------------------
# Dates in text
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Effective 2025-10-01.",
        "Revised 10/01/2025.",
        "Revised 10-01-25.",
        "For the benefit month of 03/2026 and ongoing",
        "Effective October 1, 2025.",
        "each year on October 1",
        "Rev. 11 Aug 25",
        "as of 1st July 2026",
        "in September 2025",
        "Sept. 2025 update",
        "due May 15 of each year",
        "May 2026 change notice",
        "FY26 standards",
        "FY 2026 standards",
        "the fiscal year 2026 table",
        "enacted in 1996.",
    ],
)
def test_text_that_prints_a_date_is_not_date_silent(text):
    assert text_prints_any_date(text)


@pytest.mark.parametrize(
    "text",
    [
        SOURCE,
        "Any 2 may be excluded and the unit may 1) apply or 2) withdraw.",
        "The unit may receive $2000 or $2,000 or 2000.50 dollars.",
        "The limit is 200% of the FPL; 130 percent applies to 3 or more people.",
        "See form FAA-1249A and section 12000, item 20145.",
        "Count 1/2 of the amount for 30 days, or 12 months.",
        "A march of 10 steps; the second of the month.",
    ],
)
def test_text_without_a_date_is_date_silent(text):
    assert not text_prints_any_date(text)


@pytest.mark.parametrize(
    ("text", "value", "printed"),
    [
        ("Effective 3/1/2026.", date(2026, 3, 1), True),
        ("Effective 03/01/26.", date(2026, 3, 1), True),
        ("Effective 13/1/2026.", date(2026, 3, 1), False),
        ("Effective 3/11/2026.", date(2026, 3, 1), False),
        ("Effective 3/1/2025.", date(2026, 3, 1), False),
        ("Effective March 1st, 2026.", date(2026, 3, 1), True),
        ("Effective Mar. 1, 2026.", date(2026, 3, 1), True),
        ("Effective March 11, 2026.", date(2026, 3, 1), False),
        ("benefit month of 03/2026", date(2026, 3, 1), True),
        ("benefit month of 03/2026", date(2026, 3, 2), False),
        ("revised 10/03/2026", date(2026, 3, 1), False),
        ("in March 2026", date(2026, 3, 1), True),
        ("on 15 March 2026", date(2026, 3, 1), False),
        ("on 15 March 2026", date(2026, 3, 15), True),
        ("Rev. 11 Aug 25", date(2025, 8, 11), True),
        ("Rev. 11 Aug 25", date(2025, 8, 1), False),
        ("Sept. 1, 2025", date(2025, 9, 1), True),
        ("may 1, 2026", date(2026, 5, 1), True),
        ("2026-03-01", date(2026, 3, 1), True),
        ("12026-03-01", date(2026, 3, 1), False),
        ("2026-03-015", date(2026, 3, 1), False),
    ],
)
def test_text_prints_exactly_the_date(text, value, printed):
    assert text_prints_date(text, value) is printed


# --------------------------------------------------------------------------
# Registration and wiring
# --------------------------------------------------------------------------


def test_check_is_declared_and_registered():
    registered = checks.load_checks()[CHECK_ID]
    assert registered is CHECK_START_UNSUPPORTED
    assert registered.category == "complete-source-unit:effective-date"
    assert registered.gate == "ci" and registered.cause == "encoder_output"
    assert "axiom_encode.harness.start_date_evidence" in checks.CHECK_MODULES
    feedback = "\n".join(checks.issue_fix_feedback(issues(module("2025-10-01"))))
    assert f"[{CHECK_ID}]" in feedback
    assert "Do not copy the start date of a neighbouring module" in feedback
    assert "import" not in feedback.lower()
    assert "found 2025-10-01" in feedback


def _analyze(payload: dict, evidence: StartDateEvidence | None):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
        start_date_evidence=evidence,
    )


def _start_issues(found) -> list[str]:
    return [str(issue) for issue in found if getattr(issue, "check", None) == CHECK_ID]


def test_analyzer_runs_the_check_only_with_start_date_evidence():
    attempt_1 = module("2025-10-01")
    without = _analyze(attempt_1, None)
    with_evidence = _analyze(attempt_1, EVIDENCE)
    assert _start_issues(without.issues) == []
    assert len(_start_issues(with_evidence.issues)) == 1
    # The check adds its issue and changes nothing else.
    assert with_evidence.issues[:-1] == without.issues
    assert dataclasses.replace(with_evidence, issues=without.issues) == without
    attempt_2 = _analyze(module("2026-08-01"), EVIDENCE)
    assert attempt_2 == _analyze(module("2026-08-01"), None)


def _pipeline(**kwargs) -> vp.ValidatorPipeline:
    return vp.ValidatorPipeline(
        Path("/tmp/rulespec-us/us-az"),
        Path("/tmp/unused"),
        enable_oracles=False,
        local_corpus_release=None,
        **kwargs,
    )


def _pipeline_start_issues(pipeline: vp.ValidatorPipeline, payload: dict) -> list[str]:
    return _start_issues(
        pipeline._complete_source_unit_issues(
            yaml.safe_dump(payload, sort_keys=False),
            validation_source_texts={CITATION: SOURCE},
            test_cases=[],
        )
    )


def test_pipeline_refuses_attempt_1_and_passes_attempt_2():
    pipeline = _pipeline(require_complete_source_unit=True, source_metadata=metadata())
    assert len(_pipeline_start_issues(pipeline, module("2025-10-01"))) == 1
    assert _pipeline_start_issues(pipeline, module("2026-08-01")) == []


def test_pipeline_with_metadata_that_has_no_attestation_refuses():
    pipeline = _pipeline(require_complete_source_unit=True, source_metadata={})
    found = _pipeline_start_issues(pipeline, module("2025-10-01"))
    assert len(found) == 1
    assert "source_attestation is missing" in found[0]
    # Without a readable snapshot even the real floor has no support.
    assert len(_pipeline_start_issues(pipeline, module("2026-08-01"))) == 1
    assert _pipeline_start_issues(pipeline, module("0001-01-01")) == []


def test_pipeline_without_any_corpus_date_channel_does_not_run_the_check():
    # No metadata object and no bound corpus release: the caller gave nothing
    # to read dates from, as distinct from metadata without readable dates.
    pipeline = _pipeline(require_complete_source_unit=True)
    assert pipeline._complete_source_start_date_evidence(CITATION) is None
    assert _pipeline_start_issues(pipeline, module("2025-10-01")) == []


def _validate_pipeline(tmp_path: Path, **row) -> vp.ValidatorPipeline:
    """The pipeline `validate --require-complete-source-unit` builds.

    `cmd_validate` passes the bound corpus release and no `source_metadata`.
    """
    from tests.test_evals import _write_test_corpus_release

    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": CITATION,
                "body": SOURCE,
                "source_as_of": "2026-07-17",
                "expression_date": "2026-07-17",
                **row,
            }
        ],
    )
    return vp.ValidatorPipeline(
        Path("/tmp/rulespec-us/us-az"),
        Path("/tmp/unused"),
        enable_oracles=False,
        local_corpus_release=release,
        require_complete_source_unit=True,
    )


def test_validate_reads_the_row_dates_from_the_bound_corpus_release(tmp_path):
    pipeline = _validate_pipeline(tmp_path)
    assert pipeline.source_metadata is None
    assert pipeline._complete_source_start_date_evidence(CITATION) == EVIDENCE
    assert _pipeline_start_issues(pipeline, module("2026-08-01")) == []
    found = _pipeline_start_issues(pipeline, module("2025-10-01"))
    assert len(found) == 1
    assert "the evidence floor `2026-08-01`" in found[0]
    assert "is unavailable" not in found[0]


def test_validate_with_an_unreadable_row_date_refuses(tmp_path):
    pipeline = _validate_pipeline(tmp_path, source_as_of="17 July 2026")
    found = _pipeline_start_issues(pipeline, module("2026-08-01"))
    assert len(found) == 1
    assert "source_attestation.source_as_of is not a YYYY-MM-DD date" in found[0]
    assert _pipeline_start_issues(pipeline, module("2026-07-17")) == []  # expression


def test_validate_with_a_citation_the_release_does_not_resolve_refuses(tmp_path):
    pipeline = _validate_pipeline(tmp_path)
    other = CITATION.replace("block-16", "block-99")
    evidence = pipeline._complete_source_start_date_evidence(other)
    assert evidence == StartDateEvidence(
        None, None, ("the bound corpus release does not resolve the citation",)
    )


def test_pipeline_runs_no_start_date_check_outside_complete_source_mode():
    # Existing modules: `validate` without --require-complete-source-unit.
    pipeline = _pipeline(source_metadata=metadata())
    assert (
        pipeline._complete_source_unit_issues(
            yaml.safe_dump(module("2025-10-01")),
            validation_source_texts={CITATION: SOURCE},
            test_cases=[],
        )
        == []
    )


def test_pipeline_passes_amendment_context(monkeypatch):
    from types import SimpleNamespace

    seen = []

    def capture(content, source, **kwargs):
        seen.append(kwargs["start_date_evidence"])
        return SimpleNamespace(issues=())

    monkeypatch.setattr(vp, "analyze_complete_source_unit", capture)
    content = yaml.safe_dump(module("2025-10-01"))
    for kwargs in ({}, {"amendment_source_texts": {"us/statute/1/1": "Amended."}}):
        _pipeline(
            require_complete_source_unit=True, source_metadata=metadata(), **kwargs
        )._complete_source_unit_issues(
            content, validation_source_texts={CITATION: SOURCE}, test_cases=[]
        )
    assert seen == [EVIDENCE, dataclasses.replace(EVIDENCE, amendment_context=True)]


def test_replay_passes_the_fixture_snapshot_dates():
    from scripts.replay_us_failure_bundles import (
        DEFAULT_FIXTURE_ROOT,
        load_fixture,
        replay_case,
    )

    # West Virginia manual page: date-silent, no imports, source_as_of
    # 2026-07-21, expression_date 2026-07-01, candidate starts on 2026-07-01.
    case = load_fixture(DEFAULT_FIXTURE_ROOT / "36596173514")
    assert _start_issues(replay_case(case)["issues"]) == []
    copied = dataclasses.replace(
        case, candidate=case.candidate.replace("'2026-07-01'", "'2025-10-01'")
    )
    assert copied.candidate != case.candidate
    found = _start_issues(replay_case(copied)["issues"])
    assert len(found) == 1
    assert "the evidence floor `2026-08-01`" in found[0]
    assert "the corpus `expression_date` `2026-07-01`" in found[0]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


def _guidance(source_as_of, expression_date) -> str:
    evidence = StartDateEvidence(source_as_of, expression_date)
    return render_date_silent_start_guidance(
        source_as_of=evidence.source_as_of,
        expression_date=evidence.expression_date,
        evidence_floor=evidence.evidence_floor,
        preferred_start=evidence.preferred_start,
    )


def test_prompt_guidance_states_the_rule_with_the_computed_floor():
    text = _guidance(SNAPSHOT, SNAPSHOT)
    for needle in (
        "when `./source.txt` prints no effective date for a rule",
        "1. a date printed in `./source.txt`, or printed in a proof-atom excerpt",
        "2. the evidence floor `2026-08-01`: the first day of the first full "
        "calendar month that starts on or after the corpus `source_as_of` "
        "(`2026-07-17`)",
        "3. the corpus `expression_date` (`2026-07-17`) or `source_as_of` "
        "(`2026-07-17`) itself",
        "4. the neutral scaffold `0001-01-01`",
        "Without one, use the evidence floor `2026-08-01`.",
        "A date copied from a precedent module or a sibling module is not support",
        "The evidence floor is an evidence-coverage floor, not a claimed legal "
        "commencement date",
        "Put every companion test period on or after the start date you choose.",
    ):
        assert needle in text, needle
    assert "Prefer the earliest scaffold date" not in text
    assert "import" not in text.lower()


def test_prompt_guidance_prefers_an_expression_date_after_the_floor():
    text = _guidance(date(2025, 11, 13), date(2026, 1, 1))
    assert "2. the evidence floor `2025-12-01`" in text
    assert (
        "Without one, use the corpus `expression_date` `2026-01-01`: it is "
        "later than the evidence floor"
    ) in text
    assert "use the evidence floor" not in text
    earlier = _guidance(SNAPSHOT, date(2025, 10, 1))
    assert "Without one, use the evidence floor `2026-08-01`." in earlier
    assert "use the corpus `expression_date`" not in earlier


def test_prompt_guidance_without_a_snapshot_names_no_floor():
    text = _guidance(None, None)
    assert "1. a date printed in `./source.txt`" in text
    assert "2. the neutral scaffold `0001-01-01`" in text
    assert "there is no evidence floor" in text
    assert "use the neutral scaffold `0001-01-01`." in text
    assert "the evidence floor `" not in text
    assert "the corpus `" not in text
    only_expression = _guidance(None, date(2025, 10, 1))
    assert "2. the corpus `expression_date` (`2025-10-01`) itself" in only_expression
    assert "3. the neutral scaffold `0001-01-01`" in only_expression
    assert "use the corpus `expression_date` `2025-10-01`." in only_expression


def test_scaffold_fallback_for_a_manual_names_no_precedent_date():
    default = render_date_silent_scaffold_guidance()
    manual = render_date_silent_scaffold_guidance(precedent_dates=False)
    assert "no copied precedent file provides a scaffold date" in default
    assert "precedent" not in manual
    assert (
        "is silent on effective dates, and no supplied corpus metadata or "
        "amendment/commencement context supplies a date, you may use "
        "`effective_from: '0001-01-01'`"
    ) in manual
    assert default.splitlines()[2:] == manual.splitlines()[2:]


def test_citation_document_class_is_the_second_path_segment():
    assert sde.SCAFFOLD_START == date(1, 1, 1)
    assert sde.citation_document_class(CITATION) == "manual"
    assert sde.citation_document_class("us-az/manual") is None
    assert sde.citation_document_class(None) is None
    assert sde.is_manual_citation(CITATION)
    assert not sde.is_manual_citation("us/statute/7/2014/a")
    assert not sde.is_manual_citation("26 USC 24(a)")
    assert not sde.is_manual_citation(None)


STATUTE = "us/statute/7/2014/a"
START_RULE = "Start date of a date-silent unit"
MAIN_PRECEDENT_LINE = (
    "Prefer the earliest scaffold date that is relevant to the copied precedent "
    "when `./source.txt` lacks its own effective date."
)
MANUAL_PRECEDENT_LINE = (
    "A date copied from a precedent or a sibling module is not evidence of "
    "this source unit's own start date"
)


def _attestation(citation: str, source_as_of: str, expression_date: str) -> dict:
    return {
        "source_attestation": {
            "requested_corpus_citation_path": citation,
            "source_as_of": source_as_of,
            "expression_date": expression_date,
            "row": {"source_as_of": source_as_of, "expression_date": expression_date},
        }
    }


def _eval_prompt(
    tmp_path: Path,
    citation: str,
    source_metadata: dict | None,
    *,
    precedent: bool = False,
    amendment: bool = False,
) -> str:
    from axiom_encode.harness import evals
    from axiom_encode.repo_routing import monorepo_checkout_name

    jurisdiction = citation.split("/")[0]
    content_root = tmp_path / monorepo_checkout_name(jurisdiction) / jurisdiction
    content_root.mkdir(parents=True)
    workspace = evals.prepare_eval_workspace(
        citation=citation,
        runner=evals.parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text=SOURCE,
        axiom_rules_path=content_root,
        mode="cold",
        source_metadata_payload=source_metadata,
        extra_context_paths=[],
    )
    context_files = []
    if precedent:
        copied = workspace.root / "precedent.yaml"
        copied.write_text("versions:\n  - effective_from: '2025-10-01'\n")
        context_files.append(
            evals.EvalContextFile(
                source_path=str(copied),
                workspace_path=copied.name,
                import_path=f"{jurisdiction}:policies/precedent",
                kind="implementation_precedent",
            )
        )
    if amendment:
        workspace = dataclasses.replace(
            workspace,
            amendment_documents=(
                evals.CorpusAmendmentDocument(
                    citation_path=f"{jurisdiction}/statute/amending-act",
                    title="Amending act",
                    expression_date="2026-03-01",
                    metadata={},
                    body="The allowance is amended.",
                ),
            ),
        )
    return evals._build_eval_prompt(
        citation,
        "cold",
        workspace,
        context_files,
        target_file_name="target.yaml",
        include_tests=True,
        runner_backend="codex",
    )


def test_eval_prompt_for_a_manual_names_the_block_16_floor(tmp_path):
    prompt = _eval_prompt(
        tmp_path, CITATION, _attestation(CITATION, "2026-07-17", "2026-07-17")
    )
    assert START_RULE in prompt
    assert "Without one, use the evidence floor `2026-08-01`." in prompt
    assert "(`2026-07-17`)" in prompt
    assert "not a claimed legal commencement date" in prompt
    assert "no copied precedent file provides a scaffold date" not in prompt
    assert "harness-only fallback" in prompt


def test_eval_prompt_for_a_manual_with_expression_after_the_floor(tmp_path):
    prompt = _eval_prompt(
        tmp_path, CITATION, _attestation(CITATION, "2025-08-14", "2025-10-01")
    )
    assert "the evidence floor `2025-09-01`" in prompt
    assert "Without one, use the corpus `expression_date` `2025-10-01`" in prompt
    assert "use the evidence floor" not in prompt


def test_eval_prompt_for_a_manual_with_a_precedent_does_not_offer_its_date(tmp_path):
    prompt = _eval_prompt(
        tmp_path,
        CITATION,
        _attestation(CITATION, "2026-07-17", "2026-07-17"),
        precedent=True,
    )
    assert "`2025-10-01`" in prompt  # the copied date is still listed
    assert MANUAL_PRECEDENT_LINE in prompt
    assert MAIN_PRECEDENT_LINE not in prompt
    assert START_RULE in prompt
    # The scaffold fallback renders for a manual also beside copied dates.
    assert "harness-only fallback" in prompt
    assert "no copied precedent file provides a scaffold date" not in prompt


def test_eval_prompt_for_a_manual_without_corpus_dates_has_no_floor(tmp_path):
    prompt = _eval_prompt(tmp_path, CITATION, None)
    assert START_RULE in prompt
    assert "there is no evidence floor" in prompt
    assert "the evidence floor `" not in prompt


@pytest.mark.parametrize("precedent", [False, True])
def test_eval_prompt_for_a_statute_has_no_start_date_block(tmp_path, precedent):
    # us/statute/7/2014/a: date-silent, snapshot 2026-06-26. A floor of
    # 2026-07-01 would contradict the 2022-01 and 2024-01 oracle periods.
    prompt = _eval_prompt(
        tmp_path,
        STATUTE,
        _attestation(STATUTE, "2026-06-26", "2026-06-26"),
        precedent=precedent,
    )
    assert START_RULE not in prompt
    assert "evidence floor" not in prompt
    assert "2026-07-01" not in prompt
    assert MANUAL_PRECEDENT_LINE not in prompt
    assert (MAIN_PRECEDENT_LINE in prompt) is precedent
    # Main's fallback text, with its precedent clause, when no date is copied.
    assert (
        "and no copied precedent file provides a scaffold date" in prompt
    ) is not precedent


def test_eval_prompt_with_amendment_documents_has_no_start_date_block(tmp_path):
    metadata = _attestation(CITATION, "2026-07-17", "2026-07-17")
    with_amendment = _eval_prompt(
        tmp_path / "a", CITATION, metadata, amendment=True, precedent=True
    )
    assert START_RULE not in with_amendment
    assert "evidence floor" not in with_amendment
    assert "harness-only fallback" not in with_amendment
    # With amendment context the manual prompt is main's prompt.
    assert MAIN_PRECEDENT_LINE in with_amendment
    assert MANUAL_PRECEDENT_LINE not in with_amendment
    without = _eval_prompt(tmp_path / "b", CITATION, metadata)
    assert START_RULE in without
