import hashlib
import json

import pytest
import yaml

from axiom_encode.harness.coverage_index import format_coverage_index
from axiom_encode.harness.evals import (
    EvalContextFile,
    EvalWorkspace,
    ValidationRetryCandidate,
    _build_rulespec_eval_prompt,
    _openai_prompt_cache_parts,
)

SOURCE = "The credit equals income multiplied by 0.1. The second credit equals income multiplied by 0.2."
CITATION = "ca/policy/example"


def candidate():
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "rules": [
                {
                    "name": "credit",
                    "kind": "derived",
                    "source": CITATION,
                    "dtype": "Decimal",
                    "versions": [
                        {"effective_from": "2025-01-01", "formula": "income * 0.1"}
                    ],
                    "metadata": {
                        "proof": {
                            "atoms": [
                                {
                                    "path": "versions[0].formula",
                                    "kind": "formula",
                                    "source": {
                                        "corpus_citation_path": CITATION,
                                        "excerpt": SOURCE.split(". The")[0] + ".",
                                    },
                                }
                            ]
                        }
                    },
                }
            ],
        }
    )


def rows(index):
    return [json.loads(line) for line in index.splitlines() if line.startswith("{")]


def test_late_coordinates_and_candidate_binding_are_visible():
    initial = format_coverage_index(SOURCE, CITATION)
    repaired = format_coverage_index(SOURCE, CITATION, candidate=candidate())
    assert hashlib.sha256(SOURCE.encode()).hexdigest() in initial
    assert [r["span"] for r in rows(initial)] == [r["span"] for r in rows(repaired)]
    assert rows(initial)[-1]["span"][1] == len(SOURCE)
    assert all(r["bound_rules"] is None for r in rows(initial))
    assert rows(repaired)[0]["bound_rules"] == ["credit"]
    assert rows(repaired)[-1]["bound_rules"] == []


def test_changed_source_identity_and_malformed_candidate():
    a = format_coverage_index(SOURCE, CITATION, candidate="rules: [null]")
    b = format_coverage_index(SOURCE + " ", CITATION)
    assert hashlib.sha256((SOURCE + " ").encode()).hexdigest() in b
    assert all(r["bound_rules"] is None for r in rows(a))
    assert "not clearance" in a


def test_baseline_is_name_reconciliation_not_union():
    baseline = yaml.safe_dump({"rules": [{"name": "credit"}, {"name": "other"}]})
    output = format_coverage_index(
        SOURCE, CITATION, candidate=candidate(), baseline=baseline
    )
    assert rows(output)[-2:] == [
        {"baseline_name": "credit", "candidate_name_present": True},
        {"baseline_name": "other", "candidate_name_present": False},
    ]
    assert "not automatic copying" in output


def test_size_bound_reports_omissions(monkeypatch):
    from axiom_encode.harness import coverage_index

    monkeypatch.setattr(coverage_index, "_MAX_INDEX_CHARS", 1800)
    output = format_coverage_index(SOURCE * 20, CITATION)
    assert len(output) <= 1800
    assert "Index rows omitted by size bound: 0;" not in output


def workspace(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(SOURCE)
    return EvalWorkspace(tmp_path, source, tmp_path / "manifest.json")


def prompt(ws, contexts=(), complete=True, retry=None):
    return _build_rulespec_eval_prompt(
        CITATION,
        "repo-augmented",
        ws,
        list(contexts),
        "example.yaml",
        "ca:policies/example",
        True,
        "openai",
        None,
        require_complete_source_unit=complete,
        validation_retry_candidate=retry,
    )


def test_index_is_dynamic_and_ordinary_mode_unchanged(tmp_path):
    ws = workspace(tmp_path)
    first = prompt(ws)
    retry = prompt(ws, retry=ValidationRetryCandidate(candidate(), None))
    prefix, suffix = _openai_prompt_cache_parts(first)
    retry_prefix, retry_suffix = _openai_prompt_cache_parts(retry)
    assert prefix == retry_prefix
    assert "Complete-source coverage index" not in prefix
    assert "Complete-source coverage index" in suffix
    assert "bound_rules" in retry_suffix
    assert "Complete-source coverage index" not in prompt(ws, complete=False)


@pytest.mark.parametrize("bad", ["duplicate", "mismatch", "symlink", "escape"])
def test_existing_target_admission_fails_closed(tmp_path, bad):
    ws = workspace(tmp_path)
    context = tmp_path / "context.yaml"
    context.write_text(candidate())
    item = EvalContextFile(
        str(context), "context.yaml", "ca:policies/example", "existing_target"
    )
    items = [item]
    if bad == "duplicate":
        items.append(item)
    elif bad == "mismatch":
        item.import_path = "ca:policies/sibling"
    elif bad == "symlink":
        link = tmp_path / "link.yaml"
        link.symlink_to(context)
        item.workspace_path = "link.yaml"
    else:
        item.workspace_path = "../outside.yaml"
    with pytest.raises((ValueError, OSError)):
        prompt(ws, items)


def test_unavailable_candidate_does_not_claim_baseline_names_missing():
    output = format_coverage_index(SOURCE, CITATION, baseline=candidate())
    assert rows(output)[-1]["candidate_name_present"] is None


def test_tests_only_repair_has_no_formula_generation_index(tmp_path):
    ws = workspace(tmp_path)
    output = _build_rulespec_eval_prompt(
        CITATION,
        "repo-augmented",
        ws,
        [],
        "example.yaml",
        "ca:policies/example",
        True,
        "openai",
        None,
        require_complete_source_unit=True,
        validation_retry_candidate=ValidationRetryCandidate(candidate(), "cases: []"),
        repair_candidate_tests_only=True,
    )
    assert "Complete-source coverage index" not in output


@pytest.mark.parametrize(
    "content",
    [
        "rules: [{name: lost}]\nrules: [{name: credit}]",
        "rules: [{name: credit, metadata: {proof: {atoms: [], atoms: []}}}]",
        "rules: [{name: credit}, {name: credit}]",
    ],
)
def test_ambiguous_candidate_and_baseline_are_unavailable(content):
    output = format_coverage_index(
        SOURCE, CITATION, candidate=content, baseline=content
    )
    assert all(row["bound_rules"] is None for row in rows(output))
    assert "baseline_names=0" in output
