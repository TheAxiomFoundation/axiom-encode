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
from datetime import date
from pathlib import Path

import pytest
import yaml

from axiom_encode import checks
from axiom_encode.checks import ValidationIssue
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import start_date_evidence as sde
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.eval_prompt_surface import (
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
        "earlier than the corpus snapshot `2026-07-17` (`source_as_of`)",
        "A date copied from a precedent or an import is not support",
        "a date printed in the unit text or in a proof excerpt you cite",
        "the corpus `expression_date` `2026-07-17`",
        "the corpus `source_as_of` `2026-07-17`",
        "the evidence floor `2026-08-01`",
        "not a claimed legal commencement date",
        "the `0001-01-01` scaffold",
    ):
        assert needle in issue, needle


def test_block_16_attempt_2_evidence_floor_passes():
    assert issues(module("2026-08-01")) == []


@pytest.mark.parametrize(
    ("start", "refused"),
    [
        ("2026-07-16", True),  # one day before the snapshot
        ("2026-07-17", False),  # the snapshot day
        ("2026-07-18", False),
        ("2026-07-01", True),  # the month of the snapshot is not covered
        ("2026-08-01", False),
        ("2027-01-01", False),  # after the snapshot is not examined
        ("0001-01-02", True),  # only the exact scaffold date is the scaffold
    ],
)
def test_start_is_compared_with_the_snapshot_day(start, refused):
    # An expression_date apart from the snapshot, so it supports none of these.
    evidence = StartDateEvidence(SNAPSHOT, date(2020, 1, 1))
    assert bool(issues(module(start), evidence=evidence)) is refused


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


def test_unit_with_imports_is_out_of_the_narrow_rule():
    imports = ["us-az:policies/des/faa6/utility-allowance-current-amount#amount"]
    assert issues(module("2025-10-01", imports=imports)) == []
    assert len(issues(module("2025-10-01", imports=[]))) == 1
    with_import_atom = module("2025-10-01")
    with_import_atom["rules"][0]["metadata"] = {
        "proof": {"atoms": [{"kind": "import", "import": imports[0]}]}
    }
    assert issues(with_import_atom) == []


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


def test_unreadable_start_values_are_left_to_other_checks():
    assert issues(module("October 2025")) == []
    assert issues(module(None)) == []
    assert issues(module(20251001)) == []


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
        assert "evidence floor" not in found[0]
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
    assert "Do not copy the start date of a precedent" in feedback
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


def test_pipeline_without_corpus_metadata_refuses_instead_of_passing():
    pipeline = _pipeline(require_complete_source_unit=True)
    found = _pipeline_start_issues(pipeline, module("2025-10-01"))
    assert len(found) == 1
    assert "source_attestation is missing" in found[0]
    assert _pipeline_start_issues(pipeline, module("0001-01-01")) == []


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


def test_prompt_guidance_states_the_rule_with_the_computed_floor():
    text = render_date_silent_start_guidance(
        source_as_of=SNAPSHOT,
        expression_date=SNAPSHOT,
        evidence_floor=evidence_floor(SNAPSHOT),
    )
    for needle in (
        "when `./source.txt` prints no effective date for a rule",
        "1. a date printed in `./source.txt`, or printed in a proof excerpt",
        "2. the evidence floor `2026-08-01`: the first day of the first full "
        "calendar month that starts on or after the corpus `source_as_of` "
        "(`2026-07-17`)",
        "3. the corpus `expression_date` (`2026-07-17`) or `source_as_of` "
        "(`2026-07-17`) itself",
        "4. the neutral scaffold `0001-01-01`",
        "Without one, prefer the evidence floor `2026-08-01`.",
        "A date copied from a precedent module, a sibling module or an import "
        "is not support",
        "The evidence floor is an evidence-coverage floor, not a claimed legal "
        "commencement date",
        "Put every companion test period on or after the start date you choose.",
    ):
        assert needle in text, needle
    assert "Prefer the earliest scaffold date" not in text


def test_prompt_guidance_without_a_snapshot_names_no_floor():
    text = render_date_silent_start_guidance(
        source_as_of=None, expression_date=None, evidence_floor=None
    )
    assert "1. a date printed in `./source.txt`" in text
    assert "2. the neutral scaffold `0001-01-01`" in text
    assert "there is no evidence floor" in text
    assert "the evidence floor `" not in text
    assert "the corpus `" not in text
    only_expression = render_date_silent_start_guidance(
        source_as_of=None, expression_date=date(2025, 10, 1), evidence_floor=None
    )
    assert "2. the corpus `expression_date` (`2025-10-01`) itself" in only_expression
    assert "3. the neutral scaffold `0001-01-01`" in only_expression


def test_citation_document_class_is_the_second_path_segment():
    assert sde.SCAFFOLD_START == date(1, 1, 1)
    assert sde.MANUAL_DOCUMENT_CLASS == "manual"
    assert sde.citation_document_class(CITATION) == "manual"
    assert sde.citation_document_class("us-az/manual") is None


def _eval_prompt(tmp_path: Path, source_metadata: dict | None) -> str:
    from axiom_encode.harness import evals
    from axiom_encode.repo_routing import monorepo_checkout_name

    content_root = tmp_path / monorepo_checkout_name("us-az") / "us-az"
    content_root.mkdir(parents=True)
    workspace = evals.prepare_eval_workspace(
        citation=CITATION,
        runner=evals.parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text=SOURCE,
        axiom_rules_path=content_root,
        mode="cold",
        source_metadata_payload=source_metadata,
        extra_context_paths=[],
    )
    return evals._build_eval_prompt(
        CITATION,
        "cold",
        workspace,
        [],
        target_file_name="block-16.yaml",
        include_tests=True,
        runner_backend="codex",
    )


def test_eval_prompt_names_the_block_16_floor_from_the_attestation(tmp_path):
    prompt = _eval_prompt(tmp_path, metadata())
    assert "Start date of a date-silent unit" in prompt
    assert "the evidence floor `2026-08-01`" in prompt
    assert "(`2026-07-17`)" in prompt
    assert "not a claimed legal commencement date" in prompt
    assert "Prefer the earliest scaffold date" not in prompt


def test_eval_prompt_without_corpus_or_copied_dates_keeps_the_scaffold_fallback(
    tmp_path,
):
    prompt = _eval_prompt(tmp_path, None)
    assert "Start date of a date-silent unit" not in prompt
    assert "harness-only fallback" in prompt
