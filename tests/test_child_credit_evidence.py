"""Whole-analyzer allocation with actual records and synthetic closure identities."""

from dataclasses import replace

import pytest

from axiom_encode.harness.child_credit_evidence import _child_credit_context
from axiom_encode.harness.source_completeness import (
    SourceStructureBranch,
    _SourceConditionClause,
)
from tests.test_child_credit_execution import CITATION, data, evidence


def context(tmp_path, d=None, *, observed=True):
    d = data() if d is None else d
    return _child_credit_context(
        d["source"],
        CITATION,
        d["content"],
        evidence(tmp_path, d) if observed else None,
        d["cases"],
    )


def analyzer(tmp_path, d, *, observed=True):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    return pipeline._complete_source_unit_issues(
        d["content"],
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
        execution_evidence=evidence(tmp_path, d) if observed else None,
    )


def branch(c, role):
    a, b = c.owned_spans()[role]
    return SourceStructureBranch(
        (),
        "formula-clause" if role.startswith("formula_") else "exception-clause",
        "source unit",
        c.source.source[a:b],
        a,
        b,
    )


def test_full_analyzer_actual_complete_graph(tmp_path):
    assert analyzer(tmp_path, data()) == []


def test_full_analyzer_missing_observation_cannot_borrow_generic_pairs(tmp_path):
    text = "\n".join(map(str, analyzer(tmp_path, data(), observed=False)))
    assert "source-explicit-conditions" in text
    assert "numeric-recall" in text
    assert "Column D: Column B" in text
    assert "If the total" in text and "receipts" in text and "paper return" in text


def test_exact_distinct_maps_and_multiplicity(tmp_path):
    c = context(tmp_path)
    assert c.observation is not None
    assert c.owned_spans()["refund"] == (201, 321)
    assert c.owned_spans()["table_completion"] == (536, 924)
    b = branch(c, "refund")
    assert c.unmatched([b]) == ()
    assert c.unmatched([b, b]) == (b,)
    assert not c.witnesses(replace(b, start=0, text=c.source.source[:321]))
    assert not c.witnesses(replace(b, path=("foreign",)))
    assert not c.witnesses(replace(b, text=b.text + " unless eligible"))
    assert c.witnesses(branch(c, "volunteer")).isdisjoint(
        c.witnesses(branch(c, "table_completion"))
    )


def test_refund_gate_exact_owner_version_clause(tmp_path):
    import json

    c = context(tmp_path)
    name = dict(c.binding.rule_roles)["conditional_excess"]
    rule = next(
        r for r in json.loads(c.binding.declaration_json)["rules"] if r["name"] == name
    )
    v = rule["versions"][0]
    clause = _SourceConditionClause((), 0, 321, c.source.source[:321])
    args = [
        name,
        0,
        v["formula"],
        v["effective_from"],
        v["effective_to"],
        clause,
        False,
    ]
    assert c.owns_refund_condition(*args)
    for index, value in [
        (0, "other"),
        (1, True),
        (1, 1),
        (2, "0"),
        (3, "2024-01-01"),
        (4, "2026-12-31"),
        (5, replace(clause, start=201, text=c.source.source[201:321])),
        (5, replace(clause, branch_path=("other",))),
        (5, replace(clause, start=201)),
        (5, replace(clause, end=320)),
        (6, True),
    ]:
        changed = args.copy()
        changed[index] = value
        assert not c.owns_refund_condition(*changed)
    assert not replace(c, observation=None).owns_refund_condition(*args)


@pytest.mark.parametrize("defect", ["case", "content", "artifact", "missing_pair"])
def test_full_analyzer_invalid_evidence_no_generic_rescue(tmp_path, defect):
    d = data()
    if defect == "case":
        d["cases"][-1]["name"] += " changed"
    elif defect == "content":
        d["content"] = d["content"].replace(
            "Column B minus column C", "Column C minus column B"
        )
    elif defect == "artifact":
        d["artifact"]["program"]["units"][0]["name"] = "unknown_unit"
    else:
        d["cases"] = d["cases"][:-1]
        d["records"] = d["records"][:-1]
    issues = analyzer(tmp_path, d)
    assert issues
    assert any("Source-stated exceptions" in str(i) for i in issues)


@pytest.mark.parametrize(
    "addition",
    [
        "unless eligible",
        "Do not claim these credits.",
        "six dollars must be deducted",
        "Add 2025 to this amount",
    ],
)
def test_coherent_extra_source_cannot_acquire_ownership(tmp_path, addition):
    d = data()
    d["source"] += "\n\n" + addition
    assert context(tmp_path, d) is None
    assert analyzer(tmp_path / "analyzer", d)


def test_numeric_only_observed_exact_layout_and_header(tmp_path):
    c = context(tmp_path)
    spans = c.numeric_spans()
    assert sorted(c.source.source[a:b] for a, b in spans) == [
        "2025",
        "2025",
        "six",
        "six",
    ]
    assert replace(c, observation=None).numeric_spans() == set()


def test_same_body_trailing_whitespace_keeps_raw_identity(tmp_path):
    d = data()
    e = evidence(tmp_path, d)
    c = _child_credit_context(
        d["source"].rstrip(),
        CITATION,
        d["content"],
        e,
        d["cases"],
        raw_source=d["source"],
    )
    assert c.observation is not None
    assert (
        _child_credit_context(
            d["source"],
            CITATION,
            d["content"],
            e,
            d["cases"],
            raw_source=" " + d["source"],
        )
        is None
    )


def test_owned_branches_never_enter_generic_allocation(tmp_path, monkeypatch):
    from axiom_encode.harness import source_completeness as sc

    seen_conditions = []
    seen_formulas = []
    original_conditions = sc._exception_witnesses_for_branch
    original_formulas = sc._unwitnessed_formula_branches

    def condition(branch, **kwargs):
        seen_conditions.append((branch.start, branch.end))
        return original_conditions(branch, **kwargs)

    def formula(branches, **kwargs):
        seen_formulas.extend((b.start, b.end) for b in branches)
        return original_formulas(branches, **kwargs)

    monkeypatch.setattr(sc, "_exception_witnesses_for_branch", condition)
    monkeypatch.setattr(sc, "_unwitnessed_formula_branches", formula)
    assert analyzer(tmp_path, data(), observed=False)
    assert seen_conditions == []
    assert seen_formulas == []
