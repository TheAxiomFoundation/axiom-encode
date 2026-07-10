"""Tests for money proof-obligation derivation and the ratchet surface.

Strict proof validation is per-file opt-in; these tests cover the
unconditional money-atom obligation derivation that closes the gap where a
repository ships monetary parameters with no proof atoms at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode.harness.proof_validator import (
    MoneyAtomRatchet,
    emit_money_atom_ratchet,
    evaluate_money_atoms,
    find_missing_money_proof_atoms,
    load_money_atom_ratchet,
)

MONEY_PARAM_ATOM_LESS = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: be/statute/x
rules:
  - name: benefit_amount
    kind: parameter
    dtype: Money
    unit: EUR
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          186.51
"""

MONEY_PARAM_WITH_ATOM = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: be/statute/x
rules:
  - name: benefit_amount
    kind: parameter
    dtype: Money
    unit: EUR
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: be/statute/x
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          186.51
"""


def test_money_param_without_atom_creates_missing_obligation():
    report = find_missing_money_proof_atoms(MONEY_PARAM_ATOM_LESS)

    assert report.obligation_count == 1
    assert report.missing_count == 1
    obligation = report.missing[0]
    assert obligation.rule_name == "benefit_amount"
    assert obligation.path == "versions[0].formula"
    assert obligation.kind == "parameter"


def test_money_param_with_citing_atom_is_satisfied():
    report = find_missing_money_proof_atoms(MONEY_PARAM_WITH_ATOM)

    assert report.obligation_count == 1
    assert report.missing_count == 0


def test_atom_without_provision_does_not_satisfy_obligation():
    # A proof atom that names the path but cites no provision must NOT count.
    content = MONEY_PARAM_WITH_ATOM.replace(
        """            source:
              corpus_citation_path: be/statute/x
""",
        "",
    )

    report = find_missing_money_proof_atoms(content)

    assert report.obligation_count == 1
    assert report.missing_count == 1


def test_claim_backed_atom_satisfies_obligation():
    content = MONEY_PARAM_ATOM_LESS.replace(
        """    versions:""",
        """    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            claim:
              id: claims:be/statute/x#sets-amount
    versions:""",
    )

    report = find_missing_money_proof_atoms(content)

    assert report.missing_count == 0


def test_unit_only_currency_marks_rule_monetary():
    # dtype is not Money, but the currency unit still makes the value monetary.
    content = MONEY_PARAM_ATOM_LESS.replace("dtype: Money", "dtype: Decimal")

    report = find_missing_money_proof_atoms(content)

    assert report.obligation_count == 1
    assert report.missing_count == 1


def test_non_monetary_rule_creates_no_obligation():
    content = """format: rulespec/v1
rules:
  - name: region
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          42
"""

    report = find_missing_money_proof_atoms(content)

    assert report.obligation_count == 0


@pytest.mark.parametrize("sentinel", ["-1", "0", "1", "2", "3"])
def test_sentinel_money_formula_creates_no_obligation(sentinel):
    content = f"""format: rulespec/v1
rules:
  - name: flag
    kind: parameter
    dtype: Money
    unit: EUR
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          {sentinel}
"""

    report = find_missing_money_proof_atoms(content)

    assert report.obligation_count == 0


def test_half_up_rounding_literal_is_not_a_money_obligation():
    # 0.5 inside a half-up rounding expression is structural, not a policy amount.
    content = """format: rulespec/v1
rules:
  - name: rounded_amount
    kind: derived
    dtype: Money
    unit: EUR
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          floor(base_amount + 0.5)
"""

    report = find_missing_money_proof_atoms(content)

    assert report.obligation_count == 0


def test_money_parameter_table_obligation_and_satisfaction():
    table = """format: rulespec/v1
rules:
  - name: allotment_table
    kind: parameter
    dtype: Money
    unit: EUR
    indexed_by: household_size
    versions:
      - effective_from: '2025-01-01'
        values:
          "1": "186.51"
          "2": "198.94"
"""

    report = find_missing_money_proof_atoms(table)
    assert report.obligation_count == 1
    assert report.missing_count == 1
    assert report.missing[0].path == "versions[0].values"
    assert report.missing[0].kind == "parameter_table"

    # A single table-level atom covers the whole table.
    table_with_atom = table.replace(
        """    indexed_by: household_size""",
        """    indexed_by: household_size
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              corpus_citation_path: be/statute/x""",
    )
    satisfied = find_missing_money_proof_atoms(table_with_atom)
    assert satisfied.obligation_count == 1
    assert satisfied.missing_count == 0


def test_table_cell_suffix_atom_path_folds_to_table():
    # A cell-specific atom path still satisfies the whole-table obligation.
    table = """format: rulespec/v1
rules:
  - name: allotment_table
    kind: parameter
    dtype: Money
    unit: EUR
    indexed_by: household_size
    metadata:
      proof:
        atoms:
          - path: versions[0].values.1
            kind: parameter_table
            source:
              corpus_citation_path: be/statute/x
    versions:
      - effective_from: '2025-01-01'
        values:
          "1": "186.51"
"""

    report = find_missing_money_proof_atoms(table)

    assert report.missing_count == 0


def test_versions_path_parameter_table_atom_covers_each_table_version():
    table = """format: rulespec/v1
rules:
  - name: allotment_table
    kind: parameter
    dtype: Money
    unit: EUR
    indexed_by: household_size
    metadata:
      proof:
        atoms:
          - path: versions
            kind: parameter_table
            source:
              corpus_citation_path: be/statute/x
    versions:
      - effective_from: '2025-01-01'
        values:
          "1": "186.51"
      - effective_from: '2026-01-01'
        values:
          "1": "198.94"
"""

    report = find_missing_money_proof_atoms(table)

    assert report.obligation_count == 2
    assert report.missing_count == 0


def test_derived_formula_with_currency_literal_creates_obligation():
    content = """format: rulespec/v1
rules:
  - name: top_up
    kind: derived
    dtype: Money
    unit: EUR
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          min(base_benefit, 1367.74)
"""

    report = find_missing_money_proof_atoms(content)

    assert report.obligation_count == 1
    assert report.missing[0].path == "versions[0].formula"


def test_bare_versions_formula_path_matches_indexed_location():
    # An author may omit the [0] index on a single-version rule.
    content = MONEY_PARAM_WITH_ATOM.replace(
        "path: versions[0].formula", "path: versions.formula"
    )

    report = find_missing_money_proof_atoms(content)

    assert report.missing_count == 0


# --- Ratchet -------------------------------------------------------------


def test_ratchet_allows_seeded_count_and_fails_on_one_more():
    one_missing = [("be/x.yaml", MONEY_PARAM_ATOM_LESS)]
    two_missing = [
        ("be/x.yaml", MONEY_PARAM_ATOM_LESS),
        ("be/y.yaml", MONEY_PARAM_ATOM_LESS),
    ]

    ratchet_one = load_money_atom_ratchet("total_allowed: 1")
    assert evaluate_money_atoms(one_missing, ratchet_one).passed is True
    # One more atom-less monetary value pushes the repo over the seeded budget.
    assert evaluate_money_atoms(two_missing, ratchet_one).passed is False


def test_absent_ratchet_is_strict_zero_allowance():
    one_missing = [("be/x.yaml", MONEY_PARAM_ATOM_LESS)]

    run = evaluate_money_atoms(one_missing, MoneyAtomRatchet.empty())

    assert run.passed is False
    assert run.total_over_budget == (1, 0)


def test_clean_repo_passes_zero_allowance():
    clean = [("be/x.yaml", MONEY_PARAM_WITH_ATOM)]

    run = evaluate_money_atoms(clean, MoneyAtomRatchet.empty())

    assert run.passed is True
    assert run.total_missing == 0


def test_per_path_ratchet_budget():
    files = [("be/x.yaml", MONEY_PARAM_ATOM_LESS)]

    assert (
        evaluate_money_atoms(
            files, load_money_atom_ratchet("paths:\n  be/x.yaml: 1")
        ).passed
        is True
    )
    over = evaluate_money_atoms(
        files, load_money_atom_ratchet("paths:\n  be/x.yaml: 0")
    )
    assert over.passed is False
    assert over.over_budget_paths == {"be/x.yaml": (1, 0)}


def test_malformed_ratchet_raises():
    with pytest.raises(ValueError):
        load_money_atom_ratchet("total_allowed: not-a-number")
    with pytest.raises(ValueError):
        load_money_atom_ratchet("paths: [1, 2, 3]")


def test_emit_ratchet_captures_current_backlog():
    files = [
        ("be/x.yaml", MONEY_PARAM_ATOM_LESS),
        ("be/y.yaml", MONEY_PARAM_ATOM_LESS),
        ("be/clean.yaml", MONEY_PARAM_WITH_ATOM),
    ]

    emitted = emit_money_atom_ratchet(files)
    ratchet = load_money_atom_ratchet(emitted)

    assert ratchet.total_allowed == 2
    # The freshly seeded ratchet exactly permits the current backlog.
    assert evaluate_money_atoms(files, ratchet).passed is True
    # The clean file is not listed in the informational per-file breakdown.
    assert "be/clean.yaml" not in emitted


# --- CLI ------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "axiom_encode.cli", *args],
        capture_output=True,
        text=True,
    )


def _write_active_corpus(root: Path, bodies: list[str]) -> Path:
    selector = root / "manifests" / "releases" / "current.json"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        json.dumps(
            {
                "name": "current",
                "scopes": [
                    {
                        "jurisdiction": "be",
                        "document_class": "statute",
                        "version": "v1",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provision = root / "data/corpus/provisions/be/statute/v1.jsonl"
    provision.parent.mkdir(parents=True)
    rows = [
        {
            "id": f"be-x-{index}",
            "citation_path": "be/statute/x",
            "body": body,
            "jurisdiction": "be",
            "document_class": "statute",
            "version": "v1",
            "source_path": f"sources/be-x-{index}.xml",
            "source_as_of": "2026-01-01",
            "expression_date": "2026-01-01",
        }
        for index, body in enumerate(bodies, start=1)
    ]
    provision.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def _write_corpus_claim(root: Path, *, quote: str) -> None:
    claims = root / "data/corpus/claims/be/claims.jsonl"
    claims.parent.mkdir(parents=True, exist_ok=True)
    claims.write_text(
        json.dumps(
            {
                "id": "claim.test",
                "status": "accepted",
                "kind": "sets",
                "subject": {"id": "be:statutes/x", "type": "legal"},
                "evidence": [
                    {
                        "corpus_citation_path": "be/statute/x",
                        "quote": quote,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _claim_backed_proof_content() -> str:
    return """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: be/statute/x
  source_claims:
    - claim.test
rules:
  - name: benefit_amount
    kind: parameter
    dtype: Money
    unit: EUR
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            claim:
              id: claim.test
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          186.51
"""


def test_cli_proof_validate_rejects_excerpt_absent_from_active_source(tmp_path):
    corpus = _write_active_corpus(
        tmp_path / "axiom-corpus",
        ["The benefit amount is 186.51."],
    )
    content = MONEY_PARAM_WITH_ATOM.replace(
        "              corpus_citation_path: be/statute/x",
        "              corpus_citation_path: be/statute/x\n"
        "              excerpt: This text was invented.",
    )
    module = _write(tmp_path / "be" / "x.yaml", content)

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 1
    assert "source.excerpt" in result.stdout
    assert "does not appear" in result.stdout


def test_cli_proof_validate_rejects_ambiguous_active_source(tmp_path):
    corpus = _write_active_corpus(
        tmp_path / "axiom-corpus",
        ["first active body", "second active body"],
    )
    module = _write(tmp_path / "be" / "x.yaml", MONEY_PARAM_WITH_ATOM)

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-root",
        str(corpus),
    )

    assert result.returncode == 1
    assert "Proof source unresolved" in result.stdout
    assert "AmbiguousCorpusSourceError" in result.stdout


def test_cli_proof_validate_binds_claim_lookup_to_explicit_corpus(
    tmp_path,
    monkeypatch,
):
    explicit_corpus = _write_active_corpus(
        tmp_path / "explicit-corpus",
        ["EXPLICIT BODY WITHOUT CLAIM QUOTE"],
    )
    ambient_corpus = _write_active_corpus(
        tmp_path / "ambient-corpus",
        ["AMBIENT QUOTE"],
    )
    _write_corpus_claim(ambient_corpus, quote="AMBIENT QUOTE")
    monkeypatch.setenv("AXIOM_CORPUS_REPO", str(ambient_corpus))
    module = _write(tmp_path / "be" / "x.yaml", _claim_backed_proof_content())

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(explicit_corpus),
    )

    assert result.returncode == 1
    assert "Source claim missing" in result.stdout
    assert "claim.test" in result.stdout


def test_cli_proof_validate_binds_claim_quote_to_explicit_corpus(
    tmp_path,
    monkeypatch,
):
    explicit_corpus = _write_active_corpus(
        tmp_path / "explicit-corpus",
        ["EXPLICIT BODY WITHOUT CLAIM QUOTE"],
    )
    _write_corpus_claim(explicit_corpus, quote="AMBIENT QUOTE")
    ambient_corpus = _write_active_corpus(
        tmp_path / "ambient-corpus",
        ["AMBIENT QUOTE"],
    )
    _write_corpus_claim(ambient_corpus, quote="AMBIENT QUOTE")
    monkeypatch.setenv("AXIOM_CORPUS_REPO", str(ambient_corpus))
    module = _write(tmp_path / "be" / "x.yaml", _claim_backed_proof_content())

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(explicit_corpus),
    )

    assert result.returncode == 1
    assert "Source claim quote not found" in result.stdout
    assert "claim.test" in result.stdout


def test_cli_require_money_atoms_fails_without_ratchet(tmp_path):
    module = _write(tmp_path / "be" / "x.yaml", MONEY_PARAM_ATOM_LESS)

    result = _run_cli(
        "proof-validate",
        str(module),
        "--require-money-atoms",
        "--money-atom-root",
        str(tmp_path),
    )

    assert result.returncode == 1
    assert "missing money proof atom" in result.stdout
    assert "over ratchet allowance" in result.stdout


def test_cli_require_money_atoms_passes_with_seed_ratchet(tmp_path):
    module = _write(tmp_path / "be" / "x.yaml", MONEY_PARAM_ATOM_LESS)
    ratchet = _write(tmp_path / "known-missing-money-atoms.yaml", "total_allowed: 1\n")

    result = _run_cli(
        "proof-validate",
        str(module),
        "--require-money-atoms",
        "--ratchet-file",
        str(ratchet),
        "--money-atom-root",
        str(tmp_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "within ratchet allowance" in result.stdout


def test_cli_proof_validate_allows_corpus_free_rulespec_with_missing_root(tmp_path):
    module = _write(
        tmp_path / "be" / "x.yaml",
        "format: rulespec/v1\nrules: []\n",
    )

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(tmp_path / "missing-corpus"),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Result: ✓ PASSED" in result.stdout


def test_cli_emit_ratchet_round_trips(tmp_path):
    _write(tmp_path / "be" / "x.yaml", MONEY_PARAM_ATOM_LESS)
    module_b = _write(tmp_path / "be" / "y.yaml", MONEY_PARAM_ATOM_LESS)

    emit = _run_cli(
        "proof-validate",
        str(tmp_path / "be" / "x.yaml"),
        str(module_b),
        "--emit-ratchet",
        "--money-atom-root",
        str(tmp_path),
    )
    assert emit.returncode == 0
    ratchet_path = _write(tmp_path / "known-missing-money-atoms.yaml", emit.stdout)

    check = _run_cli(
        "proof-validate",
        str(tmp_path / "be" / "x.yaml"),
        str(module_b),
        "--require-money-atoms",
        "--ratchet-file",
        str(ratchet_path),
        "--money-atom-root",
        str(tmp_path),
    )
    assert check.returncode == 0, check.stdout + check.stderr


# A module whose money value is fully covered, but that carries a malformed
# base proof block on another rule (empty atoms list). Base proof validation
# fails on it; the money-atoms-only gate must ignore that and pass.
MONEY_OK_BUT_BASE_PROOF_BROKEN = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: be/statute/x
rules:
  - name: benefit_amount
    kind: parameter
    dtype: Money
    unit: EUR
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: be/statute/x
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          186.51
  - name: some_flag
    kind: derived
    dtype: Judgment
    metadata:
      proof:
        atoms: []
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          household_qualifies
"""


def test_money_atoms_only_ignores_base_proof_failures(tmp_path):
    module = _write(tmp_path / "be" / "x.yaml", MONEY_OK_BUT_BASE_PROOF_BROKEN)

    # Plain proof validation fails on the malformed base proof block.
    base = _run_cli("proof-validate", str(module))
    assert base.returncode == 1

    # Money-atoms-only passes: the monetary value is covered and the malformed
    # base proof block is out of scope for the money gate.
    money_only = _run_cli(
        "proof-validate",
        str(module),
        "--money-atoms-only",
        "--money-atom-root",
        str(tmp_path),
    )
    assert money_only.returncode == 0, money_only.stdout + money_only.stderr
    assert "PASSED" in money_only.stdout


def test_money_atoms_only_fails_on_uncovered_money(tmp_path):
    module = _write(tmp_path / "be" / "x.yaml", MONEY_PARAM_ATOM_LESS)

    result = _run_cli(
        "proof-validate",
        str(module),
        "--money-atoms-only",
        "--money-atom-root",
        str(tmp_path),
    )

    assert result.returncode == 1
    assert "missing money proof atom" in result.stdout


def test_money_atoms_only_json_shape(tmp_path):
    module = _write(tmp_path / "be" / "x.yaml", MONEY_PARAM_ATOM_LESS)

    result = _run_cli(
        "proof-validate",
        str(module),
        "--money-atoms-only",
        "--json",
        "--money-atom-root",
        str(tmp_path),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["money_atoms_only"] is True
    assert payload["passed"] is False
    assert payload["total_missing"] == 1
    assert payload["files"][0]["missing_atoms"][0]["rule"] == "benefit_amount"
