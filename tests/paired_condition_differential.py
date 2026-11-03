"""Compare analyzer verdicts with and without the per-part condition witness.

The per-part witness only adds demands to the clause witness. These helpers run
a fixture both ways, so a test can state which verdicts the part witness changes
and check that it changes nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from axiom_encode.harness import paired_condition_witness as pcw
from axiom_encode.harness import source_completeness as sc

PAIRED = "paired positive/blocking"


@dataclass(frozen=True)
class ComparedAnalysis:
    """The analyzer result, and the issues the clause witness alone reports."""

    actual: sc.CompleteSourceUnitAnalysis
    clause_witness_issues: tuple[str, ...]
    part_gaps: tuple[pcw.ConditionPartGap, ...] = ()

    def __getattr__(self, name):
        return getattr(self.actual, name)

    @property
    def changed(self) -> bool:
        return tuple(self.actual.issues) != tuple(self.clause_witness_issues)


def clause_witness_only(verdict, *args, **kwargs):
    """Run ``verdict`` with the per-part witness switched off."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pcw, "condition_part_gaps", lambda *_args, **_kwargs: ())
        patch.setattr(pcw, "proviso_condition_branches", lambda *_args, **_kwargs: ())
        return verdict(*args, **kwargs)


def assert_refusal_only(actual_issues, clause_witness_issues) -> None:
    """The part witness may add a paired demand; every other issue is unchanged."""

    def others(issues):
        return [issue for issue in issues if PAIRED not in issue]

    assert others(actual_issues) == others(clause_witness_issues)
    if any(PAIRED in issue for issue in clause_witness_issues):
        assert any(PAIRED in issue for issue in actual_issues)


def analyze_with_part_witness_differential(content, source, **kwargs):
    """Analyze once with the part witness and once with the clause witness only.

    ``part_gaps`` keeps what the part witness itself found, including on
    clauses the clause witness already refused.
    """

    gaps: list[pcw.ConditionPartGap] = []
    part_gaps = pcw.condition_part_gaps

    def recording(*args, **kwargs):
        found = part_gaps(*args, **kwargs)
        gaps.extend(found)
        return found

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pcw, "condition_part_gaps", recording)
        actual = sc.analyze_complete_source_unit(content, source, **kwargs)
    baseline = clause_witness_only(
        sc.analyze_complete_source_unit, content, source, **kwargs
    )
    assert_refusal_only(actual.issues, baseline.issues)
    return ComparedAnalysis(actual, tuple(baseline.issues), tuple(gaps))
