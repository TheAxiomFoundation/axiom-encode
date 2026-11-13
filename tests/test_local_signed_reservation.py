"""Source-only reservation and fallback isolation, not runtime admission proof."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as local
from axiom_encode.harness import source_completeness as source
from axiom_encode.harness.local_pair_source_evidence import reserves_signed_copy_branch
from tests.test_local_vehicle_date_evidence import BODY, CITE

BRANCH = source.SourceStructureBranch(
    (), "condition", "mixed", BODY[7587:7779], 7587, 7779
)


@pytest.mark.parametrize(
    "mutation", ["none", "source", "citation", "span", "text", "boolspan"]
)
def test_source_only_reservation(mutation):
    body, cite, branch = BODY, CITE, BRANCH
    if mutation == "source":
        body += "changed"
    elif mutation == "citation":
        cite += "/other"
    elif mutation == "span":
        branch = replace(branch, start=7588)
    elif mutation == "text":
        branch = replace(branch, text="gate only")
    elif mutation == "boolspan":
        branch = replace(branch, start=True)
    assert reserves_signed_copy_branch(branch, body, cite) is (mutation == "none")


@pytest.mark.parametrize("context_present", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_reserved_duty_bypasses_permissive_fallbacks(
    monkeypatch, context_present, composite
):
    seen = []

    def generic(branches, **kwargs):
        seen.extend(branches)
        return ()  # Deliberately accepts everything it sees.

    def unmatched(branches):
        seen.extend(branches)
        return ()

    monkeypatch.setattr(source, "_source_exception_branches", lambda *a, **k: (BRANCH,))
    monkeypatch.setattr(
        source, "_source_exception_requires_paired_witness", lambda *a, **k: True
    )
    monkeypatch.setattr(source, "_unwitnessed_exception_branches", generic)
    monkeypatch.setattr(
        source,
        "certify_reporting_routes",
        lambda *a, **k: SimpleNamespace(unmatched=unmatched),
    )
    monkeypatch.setattr(
        source,
        "certify_partnership_reporting",
        lambda *a, **k: SimpleNamespace(unmatched=unmatched),
    )
    monkeypatch.setattr(local, "numeric_pair_witnesses", lambda *a, **k: set())
    monkeypatch.setattr(local, "boolean_pair_witnesses", lambda *a, **k: set())
    signed_seen = []

    def signed(context, branch, **kwargs):
        signed_seen.append(branch)
        return {("owned", (1, 2, 3))} if composite else set()

    monkeypatch.setattr(local, "signed_branch_witnesses", signed)
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
    assert signed_seen == ([BRANCH] if context_present else [])
    assert any("paired positive/blocking" in str(i) for i in issues) is not (
        context_present and composite
    )
