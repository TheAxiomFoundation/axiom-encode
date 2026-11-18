"""Composition isolation seams; these do not admit runtime/source evidence."""

from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import (
    actual_floor_evidence,
    actual_local_relation_evidence,
    bound_form_operands,
    student_loan_schedule_evidence,
)
from axiom_encode.harness import (
    local_relation_source_evidence as local,
)
from axiom_encode.harness import (
    same_case_execution as execution,
)
from axiom_encode.harness import (
    source_completeness as source,
)
from tests.test_bound_form_snapshot import local_world
from tests.test_local_signed_reservation import BODY, BRANCH, CITE


@pytest.mark.parametrize("context_present", [False, True])
@pytest.mark.parametrize("student_owned", [False, True])
def test_independent_reservations_cannot_borrow_other_family(
    monkeypatch, context_present, student_owned
):
    seen = []
    student_seen = []
    student = SimpleNamespace(
        span=lambda role: (BRANCH.start, BRANCH.end),
        formula_owned=lambda branch: False,
        branch_role=lambda branch: "owned" if student_owned else None,
        witnesses=lambda branch: student_seen.append(branch) or set(),
    )
    monkeypatch.setattr(
        student_loan_schedule_evidence,
        "_student_loan_context",
        lambda *a, **k: student if student_owned else None,
    )
    monkeypatch.setattr(source, "_source_exception_branches", lambda *a, **k: (BRANCH,))
    monkeypatch.setattr(
        source, "_source_exception_requires_paired_witness", lambda *a, **k: True
    )

    def permissive(branches, **kwargs):
        seen.extend(branches)
        return ()

    monkeypatch.setattr(source, "_unwitnessed_exception_branches", permissive)
    for name in ("certify_reporting_routes", "certify_partnership_reporting"):
        monkeypatch.setattr(
            source, name, lambda *a, **k: SimpleNamespace(unmatched=permissive)
        )
    for name in ("numeric_pair_witnesses", "boolean_pair_witnesses"):
        monkeypatch.setattr(local, name, lambda *a, **k: set())

    def paired(context, branch, **kwargs):
        seen.append(branch)
        return {("wrong-family", (1, 2))}

    monkeypatch.setattr(local, "paired_branch_witnesses", paired)
    # A synthetic overlap tests that clearing T936 cannot erase student duty.
    for name in (
        "signed_branch_witnesses",
        "chart_duty_witnesses",
        "remaining_chart_witnesses",
        "expense_group_witnesses",
    ):
        monkeypatch.setattr(
            local,
            name,
            lambda *a, **k: {("t936", (1, 2, 3))} if student_owned else set(),
        )
    issues = source._companion_test_issues(
        {},
        parameter_rules={},
        principal_rule_paths={},
        principal_formula_clause_rules={},
        formula_branches=(),
        branches=(),
        source_text=BODY,
        corpus_citation_path=CITE,
        deferred_paths=set(),
        test_cases=[{"name": "seam", "input": {}, "output": {}}],
        extract_numeric_occurrences=lambda *a, **k: (),
        numeric_value_is_grounded=lambda *a, **k: False,
        formula_environment={},
        source_bound_constant_occurrences={},
        declared_input_names=set(),
        partnership_content="unchanged",
        local_relation_context=object() if context_present else None,
    )
    assert BRANCH not in seen
    assert student_seen == ([BRANCH] if student_owned else [])
    assert any("paired positive/blocking" in str(issue) for issue in issues)


@pytest.mark.parametrize("arm", ["form", "floor", "relation", "none", "imports"])
def test_snapshot_composition_recording_admission(tmp_path, monkeypatch, arm):
    pipe, target = local_world(tmp_path)
    if arm == "imports":
        payload = yaml.safe_load(target.read_text())
        payload["imports"] = ["ca:policies/provider#out"]
        target.write_text(yaml.safe_dump(payload))

    def admitted(name):
        def check(*args, **kwargs):
            assert arm != "imports", "imported route must not consult local admission"
            return arm == name

        return check

    monkeypatch.setattr(
        bound_form_operands, "bind_local_form_snapshot", admitted("form")
    )
    monkeypatch.setattr(
        actual_floor_evidence, "bind_same_page_floors", admitted("floor")
    )
    monkeypatch.setattr(
        actual_local_relation_evidence,
        "local_relation_snapshot_eligible",
        admitted("relation"),
    )
    with execution.compilation_snapshot(pipe, target) as capture:
        if arm in {"none", "imports"}:
            assert capture is None
        else:
            assert capture is not None and capture.fresh()
            assert capture.evidence() is None
