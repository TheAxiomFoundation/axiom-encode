"""Bare-label excerpts retain physical proof ownership after the main merge."""

import functools
from dataclasses import replace

import pytest

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_source_completeness_bare_markers import _analyze, _source_rule

CITATION = "nz/statute/act/public/2018/0032/section/71"
CHAPEAU = "The payment must be calculated as:—"
OTHER_CHAPEAU = "The supplement must be calculated as:—"
FIRST = "the income multiplied by 2; or"
SECOND = "the income multiplied by 3."
SOURCE = (
    f"1\n{CHAPEAU}\na\n{FIRST}\nb\n{SECOND}\n"
    f"2\n{OTHER_CHAPEAU}\na\nthe savings divided by 4; or\n"
    "b\nthe savings divided by 5.\n"
)


def _branch(source, raw, *, chapeaux=()):
    start = source.index(raw)
    end = start + len(raw)
    return sc.SourceStructureBranch(
        path=("1",),
        kind="formula-clause",
        label="payment computation",
        text=sc.mask_bare_labels(source)[start:end],
        start=start,
        end=end,
        computation_chapeaux=chapeaux,
    )


def _cue(source, text):
    start = source.index(text)
    return (start, start + len(text), text)


def _bindings(source, branches, excerpt, **binding_kwargs):
    # Several computations share this path, so a structural path alone cannot
    # satisfy the binding and every case exercises physical excerpt ownership.
    assert len(branches) > 1
    assert {branch.path for branch in branches} == {("1",)}
    rule = {
        "name": "payment",
        "kind": "derived",
        "source": f"{CITATION}(1)",
        "versions": [{"effective_from": "2026-01-01", "formula": "income * 2"}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
    }
    return sc._principal_formula_clause_rules(
        branches,
        source_text=source,
        principal_rules={"payment": rule},
        principal_rule_paths={"payment": {("1",)}},
        named_rules={"payment": rule},
        corpus_citation_path=CITATION,
        **binding_kwargs,
    )


def test_raw_proof_covers_certified_masked_punctuation_computation():
    assert sc.bare_outline(SOURCE) is not None
    raw = f"{CHAPEAU}\na\n{FIRST}"
    first = _branch(SOURCE, raw)
    second = _branch(SOURCE, SECOND)
    assert first.text != raw
    result = _bindings(SOURCE, (first, second), raw)
    assert result[first] == {"payment"}
    assert not result[second]


def test_unique_physical_chapeau_can_bind_its_inheriting_child():
    first = _branch(SOURCE, FIRST)
    second = _branch(SOURCE, SECOND, chapeaux=(_cue(SOURCE, CHAPEAU),))
    result = _bindings(SOURCE, (first, second), CHAPEAU)
    assert not result[first]
    assert result[second] == {"payment"}


def test_repeated_physical_chapeau_remains_ambiguous_with_bare_labels():
    source = SOURCE.replace(OTHER_CHAPEAU, CHAPEAU)
    assert sc.bare_outline(source) is not None
    first = _branch(source, FIRST, chapeaux=(_cue(source, CHAPEAU),))
    second = _branch(source, SECOND, chapeaux=(_cue(source, CHAPEAU),))
    assert all(
        not owners for owners in _bindings(source, (first, second), CHAPEAU).values()
    )


@pytest.mark.parametrize("change", ["start", "end", "text", "uncertified-mask"])
def test_masked_clause_still_requires_exact_physical_coordinates_and_text(change):
    raw = f"{CHAPEAU}\na\n{FIRST}"
    first = _branch(SOURCE, raw)
    if change == "text":
        first = replace(first, text=first.text.replace("income", "savings"))
    elif change == "uncertified-mask":
        first = replace(first, text=first.text.replace("\n \n", "\nX\n"))
    else:
        first = replace(first, **{change: getattr(first, change) + 1})
    assert not _bindings(SOURCE, (first, _branch(SOURCE, SECOND)), raw)[first]


@pytest.mark.parametrize("change", ["start", "end", "text", "non-integer"])
def test_inherited_chapeau_requires_its_own_physical_coordinates_and_text(change):
    start, end, text = _cue(SOURCE, CHAPEAU)
    if change == "start":
        start += 1
    elif change == "end":
        end += 1
    elif change == "text":
        text = text.replace("payment", "supplement")
    else:
        start = True
    second = _branch(SOURCE, SECOND, chapeaux=((start, end, text),))
    assert not _bindings(SOURCE, (_branch(SOURCE, FIRST), second), CHAPEAU)[second]


def test_other_sibling_context_cannot_claim_a_child_with_its_own_chapeau():
    first = _branch(SOURCE, FIRST)
    second = _branch(SOURCE, SECOND, chapeaux=(_cue(SOURCE, CHAPEAU),))
    assert all(
        not owners
        for owners in _bindings(SOURCE, (first, second), OTHER_CHAPEAU).values()
    )


@pytest.mark.parametrize(
    "case",
    [
        "authenticated-deferral",
        "missing-deferral",
        "other-sibling-deferral",
        "other-computation-proof",
        "forged-text",
        "forged-start",
        "forged-end",
    ],
)
def test_deferred_first_child_binding_requires_authenticated_active_source(case):
    other_computation = "The supplement must be calculated as income divided by 2."
    source = (
        f"1\n{other_computation}\n{CHAPEAU}\n"
        "a\nthe amount stated in the federal return copy that must be filed; or\n"
        "b\n25 dollars for the active payment.\n"
    )
    branches = sc.recognize_source_structure(source)
    covered = {("1", "a")}
    formulas = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=[branch for branch in branches if branch.path not in covered],
        deferred_paths=covered,
    )
    assert len(formulas) == 2
    active = next(branch for branch in formulas if "25 dollars" in branch.text)
    other = next(branch for branch in formulas if branch != active)
    assert active.path == other.path == ("1",)
    assert "federal return" not in active.text
    assert active.text != sc.mask_bare_labels(source)[active.start : active.end]
    assert active.computation_chapeau is not None

    excerpt = CHAPEAU
    binding_kwargs = {"deferred_paths": covered}
    if case == "missing-deferral":
        binding_kwargs = {}
    elif case == "other-sibling-deferral":
        binding_kwargs = {"deferred_paths": {("1", "b")}}
    elif case == "other-computation-proof":
        excerpt = other_computation
    elif case == "forged-text":
        active = replace(active, text=active.text.replace("25 dollars", "26 dollars"))
    elif case == "forged-start":
        active = replace(active, start=active.start + 1)
    elif case == "forged-end":
        active = replace(active, end=active.end - 1)

    result = _bindings(source, (other, active), excerpt, **binding_kwargs)
    assert bool(result[active]) == (case == "authenticated-deferral")
    if case == "other-computation-proof":
        assert result[other] == {"payment"}


@pytest.mark.parametrize("dishonest", [False, True])
def test_analyzer_forwards_authenticated_deferral_for_multiple_parent_computations(
    dishonest,
):
    supplement = "The supplement must be calculated as 10 dollars."
    source = (
        f"1\n{supplement}\n{CHAPEAU}\n"
        "a\nthe amount stated in the federal return copy that must be filed; or\n"
        "b\n25 dollars for the active payment.\n"
    )
    active_amount = _source_rule(
        "active_amount",
        "25",
        "25 dollars for the active payment.",
        f"{CITATION}(1)(b)",
        dtype="Money",
    )
    active_amount["kind"] = "parameter"
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": CITATION},
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/1/a#return_amount",
                    "reason": (
                        f"{CITATION}(1)(a) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ],
        },
        "rules": [
            _source_rule(
                "active_payment",
                "0" if dishonest else "active_amount",
                CHAPEAU,
                f"{CITATION}(1)",
                dtype="Money",
            ),
            active_amount,
            _source_rule(
                "supplement", "10", supplement, f"{CITATION}(1)", dtype="Money"
            ),
        ],
    }
    branches = sc.recognize_source_structure(source)
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("1", "a")}
    assert not issues
    formulas = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=[branch for branch in branches if branch.path not in covered],
        deferred_paths=covered,
    )
    assert len(formulas) == 2
    assert {branch.path for branch in formulas} == {("1",)}
    result = _analyze(
        source,
        payload,
        [
            {
                "period": "2026-07-01",
                "input": {},
                "output": {"active_payment": 0 if dishonest else 25, "supplement": 10},
            }
        ],
    )
    assert any("formula branch" in issue for issue in result.issues) == dishonest


def test_operation_override_takes_precedence_over_inherited_computation_prose():
    chapeau = "The payment is the sum of:—"
    source = f"1\n{chapeau}\na\nthe person's income; and\nb\nthe person's savings.\n"
    assert sc.bare_outline(source) is not None
    branch = _branch(source, "the person's savings.", chapeaux=(_cue(source, chapeau),))
    assert sc._formula_operation_kinds(branch.text) == set()
    assert sc._formula_operation_kinds(branch.computation_text) == {"add"}
    execution = sc._FormulaExecution(
        trace=(),
        leaf="income * 2",
        evaluated_value=None,
        evaluates_to_zero=False,
        constant_environment={},
    )
    kwargs = {
        "interval": None,
        "formula_environment": {},
        "extract_numeric_occurrences": functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        "numeric_value_is_grounded": numeric_value_is_grounded,
    }
    assert not sc._formula_execution_matches_source_branch(execution, branch, **kwargs)
    assert sc._formula_execution_matches_source_branch(
        execution, branch, source_operation_kinds=frozenset({"multiply"}), **kwargs
    )
