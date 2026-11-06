"""Source-owner consumer tests; saved runtime fixture is not fresh authority."""

import copy
import json
from dataclasses import replace
from pathlib import Path

from axiom_encode.harness.bound_investment_adjustment_structure import CITATION
from axiom_encode.harness.investment_adjustment_coverage import FORMULAS
from axiom_encode.harness.investment_adjustment_evidence import (
    InvestmentAdjustmentEvidence,
    certify_investment_adjustment,
)
from axiom_encode.harness.source_completeness import SourceStructureBranch
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)
from tests import test_investment_adjustment_capture as capture_fixture
from tests.test_investment_adjustment_refusal import CONTENT, PARENT


def branch(start=78, end=850):
    return SourceStructureBranch(
        (), "formula-clause", "source unit", PARENT[start:end], start, end
    )


def test_known_source_without_capture_stays_owned_without_generic_authority():
    context = certify_investment_adjustment(CONTENT, PARENT, CITATION, [], None)
    assert context.recognized and context.owns_branch(branch(), "formula")
    assert not context.covers_branch(branch(), "formula")
    assert context.missing()
    assert not certify_investment_adjustment(
        CONTENT, PARENT, "other", [], None
    ).recognized
    assert not InvestmentAdjustmentEvidence(True, PARENT, True).live()
    assert not InvestmentAdjustmentEvidence(True, PARENT, True).owns_branch(
        branch(), "formula"
    )


def test_partial_synthetic_capture_cannot_supply_full_source_authority(
    tmp_path, monkeypatch
):
    cap, cases, calls = capture_fixture.n_capture.__wrapped__(tmp_path, monkeypatch)
    capture_fixture.stage_all(cap, cases, calls)
    context = certify_investment_adjustment(
        CONTENT, PARENT, CITATION, cases, cap.evidence()
    )
    assert context.recognized and not context.live() and context.missing()


def test_complete_saved_actual_custody_and_stale_mutations(tmp_path, monkeypatch):
    full = json.loads(
        (Path(__file__).parent / "fixtures/schedule-n-full-capture.json").read_text()
    )
    # Build the existing materializer-shaped fixture first, then replace only
    # its raw companion before activation so immutable ordinals include all34.
    original = capture_fixture.DATA
    import weakref
    from types import SimpleNamespace

    import yaml

    from axiom_encode.harness import investment_adjustment_capture as c
    from axiom_encode.harness import same_case_execution as s

    cap, _, _ = capture_fixture.n_capture.__wrapped__(tmp_path, monkeypatch)
    c._CAPTURES.pop(id(cap))
    cases, calls = full["cases"], full["calls"]
    relative = cap.root_relative.removesuffix(".yaml") + ".test.yaml"
    raw = yaml.safe_dump(cases, sort_keys=False).encode()
    (cap.original_root / relative).write_bytes(raw)
    (cap.snapshot_root / relative).write_bytes(raw)
    cap.files = tuple(
        s.ClosureFile(f.relative, f.original, raw if f.relative == relative else f.raw)
        for f in cap.files
    )
    monkeypatch.setattr(
        s, "_materialized_capture", lambda obj, ref=weakref.ref(cap): obj is ref()
    )
    assert c.activate(SimpleNamespace(local_corpus_release=object()), cap)
    cap.compiled(cap.artifact)
    capture_fixture.stage_all(cap, cases, calls)
    evidence = cap.evidence()
    context = certify_investment_adjustment(CONTENT, PARENT, CITATION, cases, evidence)
    assert context.live() and not context.missing()
    from axiom_encode.harness import validator_pipeline as v
    from axiom_encode.harness.bound_investment_adjustment_operands import (
        GUIDE as GUIDE_CITATION,
    )
    from axiom_encode.harness.source_completeness import analyze_complete_source_unit
    from tests.test_investment_adjustment_refusal import GUIDE

    def analyze(e):
        return analyze_complete_source_unit(
            CONTENT,
            PARENT,
            corpus_citation_path=CITATION,
            test_cases=cases,
            extract_numeric_occurrences=v.extract_typed_numeric_inventory_occurrences_from_text,
            extract_numeric_grounding_occurrences=v.extract_typed_numeric_occurrences_from_text,
            extract_named_scalars=v.extract_named_scalar_occurrences,
            numeric_value_is_grounded=v.numeric_value_is_grounded,
            source_context={CITATION: PARENT, GUIDE_CITATION: GUIDE},
            execution_evidence=e,
        )

    result = analyze(evidence)
    assert not result.issues, "\n".join(result.issues)
    assert analyze(None).issues
    assert analyze(replace(evidence)).issues
    remaining, missing = context.partition((branch(), branch()), "formula")
    assert remaining == () and len(missing) == 2
    assert context.partition((replace(branch(), start=77),), "formula")[0]
    from axiom_encode.harness.source_completeness import _SourceConditionClause

    rules = {r["name"]: r for r in yaml.safe_load(CONTENT)["rules"]}
    for name in (
        "schedule_n_investment_income_line_36",
        "schedule_n_resource_deduction_recovery_line_28",
    ):
        rule = rules[name]
        formula = rule["versions"][0]["formula"]
        clause = _SourceConditionClause((), 77, 850, PARENT[78:850])
        assert PARENT[77:78] == " "
        assert context.covers_gate(
            name, rule, 0, formula, "2025-01-01", "2025-12-31", clause
        )
        for bad in (
            replace(clause, start=78),
            replace(clause, text=PARENT[77:850]),
            replace(clause, branch_path=("1",)),
        ):
            assert not context.covers_gate(
                name, rule, 0, formula, "2025-01-01", "2025-12-31", bad
            )
        assert not context.covers_gate(
            name, rule, 1, formula, "2025-01-01", "2025-12-31", clause
        )
        assert not context.covers_gate(
            name, rule, 0, formula + " + 0", "2025-01-01", "2025-12-31", clause
        )
    from axiom_encode.harness import source_completeness as sc

    original_branches = sc._source_formula_branches

    def duplicate_owned(*args, **kwargs):
        found = original_branches(*args, **kwargs)
        owned = next(b for b in found if context.owns_branch(b, "formula"))
        return (*found, owned)

    with monkeypatch.context() as mutation:
        mutation.setattr(sc, "_source_formula_branches", duplicate_owned)
        assert any("duplicate" in str(x) for x in analyze(evidence).issues)
    for start, end, _ in FORMULAS:
        assert context.covers_branch(branch(start, end), "formula")
    for changed in [
        replace(branch(), start=77),
        replace(branch(), kind="other"),
        replace(branch(), required_principal_output="other"),
        replace(branch(), path=("1",)),
    ]:
        assert not context.owns_branch(changed, "formula")
    for occurrence in extract_typed_numeric_inventory_occurrences_from_text(PARENT):
        assert context.numeric_owner(occurrence) is not None
    assert not replace(context).live()
    wrong_cases = copy.deepcopy(cases)
    wrong_cases[0]["name"] += "borrowed"
    assert not certify_investment_adjustment(
        CONTENT, PARENT, CITATION, wrong_cases, evidence
    ).live()
    assert not certify_investment_adjustment(
        CONTENT, PARENT + " ", CITATION, cases, evidence
    ).live()
    assert not certify_investment_adjustment(
        CONTENT, PARENT, CITATION, cases, replace(evidence)
    ).live()
    cap.artifact.write_text("{}")
    assert not context.live() and context.missing()
    assert analyze(evidence).issues
    assert (
        context.numeric_owner(
            extract_typed_numeric_inventory_occurrences_from_text(PARENT)[0]
        )
        is None
    )
    assert capture_fixture.DATA is original
