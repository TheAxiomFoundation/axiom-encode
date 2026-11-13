"""Whole-source allocation; actual records, synthetic test freshness identities."""

import json
from dataclasses import replace

import pytest

from axiom_encode.harness.retirement_transfer_evidence import (
    _retirement_transfer_context,
)
from axiom_encode.harness.source_completeness import (
    SourceStructureBranch,
    _SourceConditionClause,
)
from tests.test_retirement_transfer_capture import CITATION, data, evidence


def context(tmp_path, d=None, observed=True):
    d = data() if d is None else d
    return _retirement_transfer_context(
        d["source"],
        CITATION,
        d["content"],
        evidence(tmp_path, d) if observed else None,
        d["cases"],
    )


def analyzer(tmp_path, d=None, observed=True, captured=None):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    d = data() if d is None else d
    v = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    return v._complete_source_unit_issues(
        d["content"],
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
        execution_evidence=(captured or evidence(tmp_path, d)) if observed else None,
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


def test_actual_whole_form_all26_records(tmp_path):
    assert analyzer(tmp_path) == []


def test_unobserved_owners_cannot_borrow_generic_pairs(tmp_path):
    issues = "\n".join(map(str, analyzer(tmp_path, observed=False)))
    assert "source-explicit-conditions" in issues
    assert "numeric-recall" in issues
    assert "Direct nonnegative clamp" in issues
    assert "born before" in issues and "box E" in issues
    assert "enclose" in issues and "social insurance" in issues
    assert "copy for your records" in issues


def test_exact_source_owners_and_multiplicity(tmp_path):
    c = context(tmp_path)
    for role in c.owned_spans():
        b = branch(c, role)
        assert c.witnesses(b), role
        assert c.unmatched([b]) == ()
        assert c.unmatched([b, b]) == (b,)
        assert not c.witnesses(replace(b, start=b.start - 1))
        assert not c.witnesses(replace(b, end=b.end + 1))
        assert not c.witnesses(replace(b, path=("foreign",)))
        assert not c.witnesses(replace(b, text=b.text + " unless eligible"))
    assert c.witnesses(branch(c, "copy_guide")).isdisjoint(
        c.witnesses(branch(c, "permission"))
    )


def test_gate_coordinate_is_not_paired_coordinate(tmp_path):
    c = context(tmp_path)
    owner = dict(c.binding.rule_roles)["permission"]
    rule = next(
        r for r in json.loads(c.binding.declaration_json)["rules"] if r["name"] == owner
    )
    v = rule["versions"][0]
    a, b = c.owned_spans()["permission"]
    gate = _SourceConditionClause((), a - 1, b, c.source.source[a:b])

    def accepted(clause, **kwargs):
        return c.owns_gate(
            owner,
            kwargs.get("version", 0),
            kwargs.get("formula", v["formula"]),
            v["effective_from"],
            v["effective_to"],
            clause,
            kwargs.get("ambiguous", False),
        )

    assert (gate.start, a) == (85, 86)
    assert accepted(gate)
    assert not accepted(replace(gate, start=a))
    assert not accepted(replace(gate, start=a - 2))
    assert not accepted(replace(gate, end=b + 1))
    assert not accepted(replace(gate, text="Do not: " + gate.text))
    assert not accepted(replace(gate, text=gate.text + " unless eligible"))
    assert not accepted(gate, ambiguous=True)
    assert not accepted(gate, version=True)
    assert not accepted(gate, formula="true")


@pytest.mark.parametrize("defect", ["case_name", "raw_date", "artifact", "proof"])
def test_invalid_whole_evidence_keeps_source_findings(tmp_path, defect):
    d = data()
    captured = (
        evidence(tmp_path / "original-capture", d) if defect == "case_name" else None
    )
    if defect == "case_name":
        d["cases"][0]["name"] += " changed"
    elif defect == "raw_date":
        k = next(
            k for k in d["cases"][0]["input"] if k.endswith("transferor_birth_date")
        )
        d["cases"][0]["input"][k] = "1959-01-01"
    elif defect == "artifact":
        d["artifact"]["program"]["units"][-1]["minor_units"] = 9
    else:
        d["content"] = d["content"].replace(
            "Add lines 12 through 16.", "Nonexistent proof."
        )
    assert analyzer(tmp_path, d, captured=captured)


def omit(d, indices):
    """Coherent synthetic omission; not an additional real runtime capture."""
    retained = [
        (c, r)
        for i, (c, r) in enumerate(zip(d["cases"], d["records"], strict=True), 1)
        if i not in indices
    ]
    d["cases"] = [c for c, _ in retained]
    d["records"] = [r for _, r in retained]
    for i, r in enumerate(d["records"], 1):
        r["case_index"] = i
    return d


@pytest.mark.parametrize("indices", [{4}, {6}, {8}, {10}, {14}, {20}, {22}, {15, 23}])
def test_coherent_missing_role_cannot_use_other_pairs(tmp_path, indices):
    assert analyzer(tmp_path, omit(data(), indices))


def test_recipient_negative_context_independently_required(tmp_path):
    d = omit(data(), {24})
    c = context(tmp_path / "context", d)
    assert c.observation is not None  # reaches the allocation-specific guard
    issues = "\n".join(map(str, analyzer(tmp_path / "analyzer", d)))
    assert "Direct nonnegative clamp" in issues


@pytest.mark.parametrize(
    "before,after",
    [
        ("+ 13", "+ 15"),
        ("maximum: 50%", "maximum: 60%"),
        ("= 20", "= 21"),
        ("If you were born", "Unless you were born"),
        ("NOTE\n", "Do not:\nNOTE\n"),
    ],
)
def test_unknown_changed_source_no_certificate(tmp_path, before, after):
    d = data()
    d["source"] = d["source"].replace(before, after)
    assert context(tmp_path, d) is None
    assert analyzer(tmp_path / "analyzer", d)


def test_numeric_roles_do_not_credit_footer_identifier(tmp_path):
    c = context(tmp_path)
    spans = c.numeric_spans()
    assert {c.source.source[a:b] for a, b in spans} >= {
        "3",
        "4",
        "6",
        "10",
        "12",
        "13",
        "14",
        "16",
        "20",
        "22",
        "50",
        "52",
        "53",
        "58",
    }
    assert not any("89528149" in c.source.source[a:b] for a, b in spans)
    assert replace(c, observation=None).numeric_spans() == set()


@pytest.mark.parametrize(
    "role,label",
    [
        ("permission", "permission_birth"),
        ("permission", "recipient_income:spouse_reference_date"),
        (
            "permission",
            "recipient_withholding:jointly_decided_retirement_income_transfer",
        ),
        ("withholding", "withholding_duty"),
        (
            "withholding",
            "recipient_withholding:schedule_q_requested_retirement_income_transfer_line_22",
        ),
        ("withholding_inclusion", "withholding_inclusion"),
        ("compensation_guide", "compensation_guide"),
        ("copy_guide", "copy_guide"),
        ("enclosure", "enclosure"),
        ("identification", "identification"),
    ],
)
def test_role_certificate_requires_each_observed_component(tmp_path, role, label):
    # Guard-isolation mutation of a valid observation, not a minted certificate
    # or new runtime capture. Coherent case/record omission is tested above.
    c = context(tmp_path)
    assert any(p[0] == label for p in c.observation.witness_pairs)
    o = replace(
        c.observation,
        witness_pairs=tuple(p for p in c.observation.witness_pairs if p[0] != label),
    )
    assert not replace(c, observation=o).witnesses(branch(c, role))


def test_formula_certificate_requires_selected_actual_output(tmp_path):
    c = context(tmp_path)
    o = replace(c.observation, executions=())
    for role in ("formula_deductions", "formula_selection"):
        assert not replace(c, observation=o).witnesses(branch(c, role))
