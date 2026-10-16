"""Structure findings behind an imprecise deferral are reported in it.

A branch that only a rejected (imprecise) deferral would cover used to get its
own `[complete-source-unit:structure]` finding next to that deferral's
finding: two findings for one fix. The branch is now listed inside the
deferral finding. It is reported on its own again when the deferral is
removed, and it is covered once the deferral is precise.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from types import SimpleNamespace

from axiom_encode import cli
from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CFR = "us/regulation/7/273/4"
SOURCE = """\
(a) Household members meeting citizenship or alien status requirements. No person is eligible unless that person is:
(1) A U.S. citizen;
(2) A qualified alien.
(b) Reporting illegal aliens. The State agency must inform USCIS.
(c) Sponsored aliens. The State agency must count sponsor income.
"""


def _analyze(content: str, source: str = SOURCE, citation: str = CFR):
    return analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _module(deferrals: str = "", citation: str = CFR, rule_source: str = "") -> str:
    deferred = f"  deferred_outputs:\n{deferrals}" if deferrals else ""
    rules = (
        f"""\
  - name: sponsor_income_counted
    kind: derived
    dtype: Judgment
    source: {rule_source}
    versions:
      - effective_from: '2025-01-01'
        formula: sponsor_income_present
"""
        if rule_source
        else ""
    )
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation}
{deferred}rules:
{rules or "  []"}
"""


IMPRECISE_A = """\
    - output: us:regulations/7-cfr/273/4/a#citizenship_status_eligible
      reason: Not encoded yet.
"""


def _findings(result, category: str) -> list[str]:
    return [
        issue
        for issue in result.issues
        if issue.startswith(f"[complete-source-unit:{category}]")
    ]


def _structure_paths(result) -> set[str]:
    """Branch paths (`a/1`) of the structure findings, whatever the labels."""

    paths = set()
    for issue in _findings(result, "structure"):
        labels = re.search(r"\[([^\]]+)\] is neither", issue).group(1)
        paths.add(
            "/".join(
                component.strip().rsplit(maxsplit=1)[-1]
                for component in labels.split(",")
            )
        )
    return paths


def test_branches_behind_an_imprecise_deferral_are_listed_in_it():
    result = _analyze(_module(IMPRECISE_A, rule_source="7 CFR 273.4(c)"))

    deferral = _findings(result, "deferral")
    assert len(deferral) == 1, result.issues
    own_text, pending = completeness.split_imprecise_deferral_branches(deferral[0])
    assert pending == ("a", "a/1", "a/2")
    assert (
        "Until it is precise, 3 source branches that it would cover"
        in (deferral[0].split("\n", 1)[0])
    )
    assert "identifies source branch (a) (`a`)" in own_text
    assert "Required shape" in own_text
    # Negative control: (b) is not behind the deferral and keeps its own
    # structure finding; nothing under (a) is repeated.
    assert _structure_paths(result) == {"b"}


def test_removing_the_deferral_brings_every_structure_finding_back():
    with_deferral = _analyze(_module(IMPRECISE_A, rule_source="7 CFR 273.4(c)"))
    without = _analyze(_module(rule_source="7 CFR 273.4(c)"))

    assert not _findings(without, "deferral")
    restored = _structure_paths(without) - _structure_paths(with_deferral)
    _own, pending = completeness.split_imprecise_deferral_branches(
        _findings(with_deferral, "deferral")[0]
    )
    assert restored == set(pending) == {"a", "a/1", "a/2"}
    # Every other finding is the same with or without the deferral.
    others = [
        issue
        for issue in with_deferral.issues
        if not issue.startswith("[complete-source-unit:deferral]")
    ]
    assert [
        issue
        for issue in without.issues
        if not issue.startswith("[complete-source-unit:structure]")
        or _structure_paths(SimpleNamespace(issues=[issue])) == {"b"}
    ] == others


def test_a_precise_deferral_covers_the_branches_it_listed():
    imprecise, source = _louisiana_fixture("Not encoded yet.")
    precise, _source = _louisiana_fixture(
        "R.S. 47:295(C) cannot be encoded because the runtime lacks a "
        "document-filing and document-incorporation capability for a filed "
        "federal income tax return becoming part of the required state return."
    )

    rejected = _analyze(imprecise, source, "us-la/statute/47:295")
    accepted = _analyze(precise, source, "us-la/statute/47:295")

    _own, pending = completeness.split_imprecise_deferral_branches(
        _findings(rejected, "deferral")[0]
    )
    assert pending == ("c",)
    assert not _findings(rejected, "structure")
    assert not _findings(accepted, "deferral")
    assert not _findings(accepted, "structure")


def test_a_repeated_branch_is_listed_once_like_its_single_finding(monkeypatch):
    # Flattened text can yield the same (path, label) twice; without a
    # deferral both render one identical, de-duplicated structure finding.
    recognize = completeness.recognize_source_structure

    def doubled(source_text, **kwargs):
        branches = recognize(source_text, **kwargs)
        return (*branches, *(b for b in branches if b.path == ("a", "1")))

    monkeypatch.setattr(completeness, "recognize_source_structure", doubled)
    with_deferral = _analyze(_module(IMPRECISE_A, rule_source="7 CFR 273.4(c)"))
    without = _analyze(_module(rule_source="7 CFR 273.4(c)"))

    _own, pending = completeness.split_imprecise_deferral_branches(
        _findings(with_deferral, "deferral")[0]
    )
    assert pending == ("a", "a/1", "a/2")
    assert len(_findings(without, "structure")) == 4


def test_nested_imprecise_deferrals_each_hold_their_own_branches():
    nested = IMPRECISE_A + (
        "    - output: us:regulations/7-cfr/273/4/a/2#qualified_alien_eligible\n"
        "      reason: Not encoded yet.\n"
    )
    result = _analyze(_module(nested, rule_source="7 CFR 273.4(c)"))

    pending = {
        own.split("`module.deferred_outputs[", 1)[1][0]: branches
        for own, branches in map(
            completeness.split_imprecise_deferral_branches,
            _findings(result, "deferral"),
        )
    }
    assert pending == {"0": ("a", "a/1"), "1": ("a/2",)}


def test_root_imprecise_deferral_holds_only_the_source_unit_branch():
    root = """\
    - output: us:regulations/7-cfr/273/4#citizenship_status_eligible
      reason: Not encoded yet.
"""
    result = _analyze(_module(root, rule_source="7 CFR 273.4(c)"))

    # A root deferral covers only the root, never its descendants, so every
    # branch keeps its own structure finding.
    for own, pending in map(
        completeness.split_imprecise_deferral_branches,
        _findings(result, "deferral"),
    ):
        assert pending == (), own
    assert _structure_paths(result) == {"a", "a/1", "a/2", "b"}


def test_other_deferral_rejections_keep_their_structure_findings():
    # A non-source root is rejected for its anchor, not its reason; fixing
    # the anchor does not by itself make it precise.
    non_source_root = """\
    - output: us:policies/snap/eligibility/a#citizenship_status_eligible
      reason: Not encoded yet.
"""
    result = _analyze(_module(non_source_root, rule_source="7 CFR 273.4(c)"))

    assert _findings(result, "deferral")
    for own, pending in map(
        completeness.split_imprecise_deferral_branches,
        _findings(result, "deferral"),
    ):
        assert pending == (), own
    assert _structure_paths(result) == {"a", "a/1", "a/2", "b"}


def test_us_deferral_retry_shape_uses_a_us_example():
    result = _analyze(_module(IMPRECISE_A))
    deferral = _findings(result, "deferral")[0]

    assert "output: us:statutes/26/24/c/1#ctc_qualifying_child" in deferral
    assert "26 U.S.C. 152(c)" in deferral
    assert "de:statutes" not in deferral
    assert "EStG" not in deferral

    german = completeness._imprecise_deferral_retry_shape(
        corpus_citation_path="de/statute/estg/32a", path=("6",)
    )
    assert "output: de:statutes/estg/32a/6#surviving_spouse_splitting_tax" in german
    assert "us:statutes" not in german


def test_retry_ranking_counts_each_branch_behind_a_deferral():
    with_deferral = _analyze(_module(IMPRECISE_A, rule_source="7 CFR 273.4(c)"))
    without = _analyze(_module(rule_source="7 CFR 273.4(c)"))

    paired = cli._validation_residual_obligation_count(with_deferral.issues)
    # Before pairing the same candidate had the deferral finding plus one
    # structure finding per branch it would cover: `without` plus the
    # deferral's own text.
    own, pending = completeness.split_imprecise_deferral_branches(
        _findings(with_deferral, "deferral")[0]
    )
    unpaired = cli._validation_residual_obligation_count([*without.issues, own])
    assert len(pending) == 3
    assert paired == unpaired == len(without.issues) + 1


def _validation(*issues: str):
    return SimpleNamespace(
        results={"ci": SimpleNamespace(passed=False, issues=list(issues), error=None)}
    )


def test_dependent_regression_counts_branches_behind_a_deferral_one_by_one():
    full = _findings(
        _analyze(_module(IMPRECISE_A, rule_source="7 CFR 273.4(c)")), "deferral"
    )[0]
    own, pending = completeness.split_imprecise_deferral_branches(full)
    fewer = full.replace(
        "3 source branches that it would cover are",
        "2 source branches that it would cover are",
    ).replace("`a`, `a/1`, `a/2`", "`a`, `a/1`")
    assert completeness.split_imprecise_deferral_branches(fewer) == (
        own,
        ("a", "a/1"),
    )

    baseline = cli._failed_validation_issue_counts(_validation(full))["ci"]
    improved = cli._failed_validation_issue_counts(_validation(fewer))["ci"]
    assert improved <= baseline
    assert not baseline <= improved
    assert sum(baseline.values()) == 1 + len(pending)
    assert baseline == Counter(
        {own: 1, **{f"{own}\npending source branch `{path}`": 1 for path in pending}}
    )


def _louisiana_fixture(reason: str) -> tuple[str, str]:
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:295
  deferred_outputs:
    - output: us-la:statutes/47/295/c#federal_return_copy_becomes_part_of_state_return
      reason: >-
        {reason}
rules:
  - name: annual_report_filing_record
    kind: derived
    dtype: Judgment
    period: Day
    source: us-la/statute/47:295(A)
    versions:
      - formula: 'true'
  - name: audit_record_retention
    kind: derived
    dtype: Judgment
    period: Day
    source: us-la/statute/47:295(B)
    versions:
      - formula: 'true'
"""
    source = """\
A. The secretary shall file an annual administration report.

B. The secretary shall retain the audit record.

C. The secretary may require that a complete copy of the taxpayer's federal income
tax return, or any part thereof, be filed. When the return is filed, the federal
income tax return, or part thereof, shall constitute and become part of the return
required to be filed under this Part.
"""
    return content, source


def _attempt(issues, label):
    return cli._FailedEncodeAttempt(
        result=SimpleNamespace(),
        error="Generated RuleSpec failed CI validation",
        candidate=cli.ValidationRetryCandidate(rulespec=f"# {label}\n", tests="[]\n"),
        validation_issues=list(issues),
        validation_issue_count=len(issues),
        full_validation_issues=list(issues),
    )


def test_ranking_tie_break_counts_folded_branches_as_messages():
    folded = _analyze(_module(IMPRECISE_A, rule_source="7 CFR 273.4(c)")).issues
    # An older candidate with the same obligations written as separate
    # structure findings (as before folding) and one more message elsewhere.
    own, pending = completeness.split_imprecise_deferral_branches(
        _findings(SimpleNamespace(issues=folded), "deferral")[0]
    )
    unfolded = [
        own,
        *(f"[complete-source-unit:structure] branch `{path}`" for path in pending),
        *(i for i in folded if not i.startswith("[complete-source-unit:deferral]")),
    ]
    older, newer = _attempt(unfolded, "older"), _attempt(folded, "newer")
    assert cli._validation_residual_obligation_count(
        folded
    ) == cli._validation_residual_obligation_count(unfolded)
    # Equal obligations and equal pre-folding message counts: newer wins, as
    # it would have before folding.
    assert cli._best_validation_retry_attempt([older, newer]) is newer
    assert cli._best_validation_retry_attempt([newer, older]) is older
