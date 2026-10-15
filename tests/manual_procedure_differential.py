"""Compare manual-procedure probes with the closed exemption list disabled."""

import hashlib
import re
from dataclasses import dataclass

import pytest

from axiom_encode.harness import source_completeness as sc

# Independent reviewed keys: widening production membership must not widen these.
MANUAL_CITATION = (
    "us-fl/manual/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-8"
)
REVIEWED_CLAUSES = frozenset(
    {
        "chapter: 2600 calculation of benefits program: fs 1 2610.0000 food stamps "
        "when the eligibility specialist has determined available income as per "
        "chapters 1800 and 2400, the policy in this chapter must be used to perform "
        "the budgets and tests to determine eligibility for benefits and the benefit "
        "amount. 2610.0100 budgets and test calculations (fs) each program has "
        "budgets and tests that must be executed in order to determine eligibility.",
        "refer to appendix a-1 for the 165% need standard table. 2610.0104.03 gross "
        "income test (fs) when gross income is determined, that income is compared "
        "to the allowable gross income limit for the assistance group's size.",
    }
)
REVIEWED_SOURCE_SHA256 = (
    "e17c82b40e09f545b80f2cedfad3bfc91daea3eea87801f9cb3c10188fccf0a0"
)


@dataclass(frozen=True)
class _ComparedAnalysis:
    actual: sc.CompleteSourceUnitAnalysis
    empty_list_issues: tuple[str, ...]

    def __getattr__(self, name):
        return getattr(self.actual, name)


def _normalized(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_reviewed_key(text, *, corpus_citation_path="", source_text="", **_):
    clause = _normalized(text)
    body = _normalized(source_text or text)
    return (
        corpus_citation_path == MANUAL_CITATION
        and clause in REVIEWED_CLAUSES
        and (
            body == clause
            or hashlib.sha256(body.encode("utf-8")).hexdigest()
            == REVIEWED_SOURCE_SHA256
        )
    )


def unchanged_without_manual_procedure_exemptions(verdict, *args, **kwargs):
    """Retain parity for direct witness and extraction probes too."""

    actual = verdict(*args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline = verdict(*args, **kwargs)
    assert actual == baseline
    return actual


def manual_procedure_requires_paired_witness(text, **kwargs):
    """Only an independently enumerated exact key may remove a base demand."""

    actual = sc._source_exception_requires_paired_witness(text, **kwargs)
    if not sc._MANUAL_PROCEDURE_CLAUSES:
        return actual
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline = sc._source_exception_requires_paired_witness(text, **kwargs)
    assert actual is (baseline and not _is_reviewed_key(text, **kwargs))
    return actual


def analyze_with_manual_procedure_differential(content, source, **kwargs):
    """Compare every issue, including passing and generated adversarial probes."""

    actual = sc.analyze_complete_source_unit(content, source, **kwargs)
    if not sc._MANUAL_PROCEDURE_CLAUSES:
        return _ComparedAnalysis(actual, actual.issues)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline = sc.analyze_complete_source_unit(content, source, **kwargs)
    if _is_reviewed_key(source, corpus_citation_path=kwargs["corpus_citation_path"]):
        expected = tuple(
            issue
            for issue in baseline.issues
            if "paired positive/blocking" not in issue
        )
        assert len(baseline.issues) - len(expected) <= 1
        assert actual.issues == expected
    else:
        assert actual.issues == baseline.issues
    return _ComparedAnalysis(actual, baseline.issues)
