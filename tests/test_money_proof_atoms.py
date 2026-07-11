"""Tests for money proof-obligation derivation and the ratchet surface.

Strict proof validation is per-file opt-in; these tests cover the
unconditional money-atom obligation derivation that closes the gap where a
repository ships monetary parameters with no proof atoms at all.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode import entrypoint
from axiom_encode.harness.proof_validator import (
    MoneyAtomRatchet,
    emit_money_atom_ratchet,
    evaluate_money_atoms,
    find_missing_money_proof_atoms,
    find_noncanonical_corpus_citation_path_issues,
    load_money_atom_ratchet,
)
from tests.release_object_fixtures import bind_test_corpus_release

TEST_CORPUS_RELEASE = "money-proof-test-release"

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


def test_claim_backed_atom_does_not_satisfy_obligation():
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

    assert report.missing_count == 1


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
    if "format: rulespec/v1" in content:
        checkout_root = next(
            parent for parent in path.parents if parent.name.startswith("rulespec-")
        )
        release_objects = sorted(
            (
                checkout_root.parent / "axiom-corpus" / "releases" / TEST_CORPUS_RELEASE
            ).glob("*.json")
        )
        content_sha256 = (
            release_objects[0].stem if len(release_objects) == 1 else "e" * 64
        )
        _write_test_toolchain(checkout_root, content_sha256=content_sha256)
    return path


def _write_test_toolchain(checkout_root: Path, *, content_sha256: str) -> None:
    toolchain = checkout_root / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    waiver = checkout_root / "known-validation-gaps.yaml"
    if not waiver.exists():
        waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{TEST_CORPUS_RELEASE}"\n'
        f'axiom_corpus_release_content_sha256 = "{content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n',
        encoding="utf-8",
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Drive the public entrypoint with the non-inheritable test broker attached."""

    stdout = io.StringIO()
    stderr = io.StringIO()
    returncode = 0
    with (
        patch.object(sys, "argv", ["axiom-encode", *args]),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            result = entrypoint.main()
            returncode = int(result or 0)
        except SystemExit as exc:
            returncode = int(exc.code or 0)
        except Exception:
            traceback.print_exc(file=stderr)
            returncode = 1
    return subprocess.CompletedProcess(
        [sys.executable, "-m", "axiom_encode.entrypoint", *args],
        returncode,
        stdout.getvalue(),
        stderr.getvalue(),
    )


@pytest.mark.parametrize("mode_flag", ["--emit-ratchet", "--money-atoms-only"])
def test_proof_shortcuts_reject_legacy_flat_module_layout(tmp_path, mode_flag):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "x.yaml",
        MONEY_PARAM_ATOM_LESS,
    )

    result = _run_cli(
        "proof-validate",
        str(module),
        mode_flag,
        "--corpus-path",
        str(tmp_path / "unused-corpus"),
    )

    assert result.returncode == 1
    assert "primary .yaml file" in result.stderr


def test_noncanonical_corpus_identity_finder_checks_nested_fields():
    payload = {
        "module": {"source_verification": {"corpus_citation_path": "be:statute/x"}},
        "rules": [
            {
                "metadata": {
                    "proof": {
                        "atoms": [{"source": {"corpus_citation_path": "be/statute/x "}}]
                    }
                }
            }
        ],
    }

    issues = find_noncanonical_corpus_citation_path_issues(payload)

    assert len(issues) == 2
    assert all("Noncanonical corpus citation path" in issue for issue in issues)


@pytest.mark.parametrize("mode_flag", [None, "--emit-ratchet", "--money-atoms-only"])
def test_proof_modes_reject_noncanonical_corpus_machine_identity(
    tmp_path,
    mode_flag,
):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_WITH_ATOM.replace("be/statute/x", "be:statute/x"),
    )
    args = ["proof-validate", str(module)]
    if mode_flag is not None:
        args.append(mode_flag)
    args.extend(("--corpus-path", str(tmp_path / "unused-corpus")))

    result = _run_cli(*args)

    assert result.returncode == 1
    assert "Noncanonical corpus citation path" in result.stderr


def _write_active_corpus(root: Path, bodies: list[str]) -> Path:
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
    release = bind_test_corpus_release(
        root,
        TEST_CORPUS_RELEASE,
        [("be", "statute", "v1")],
    )
    for rulespec_root in root.parent.glob("rulespec-*"):
        if rulespec_root.is_dir():
            _write_test_toolchain(
                rulespec_root,
                content_sha256=release.content_sha256,
            )
    return root


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
    module = _write(tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml", content)

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
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_WITH_ATOM,
    )

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 1
    assert "Proof source unresolved" in result.stdout
    assert "AmbiguousCorpusSourceError" in result.stdout


@pytest.mark.parametrize(
    "content",
    [
        MONEY_PARAM_WITH_ATOM.replace(
            "    corpus_citation_path: be/statute/x\n",
            "    corpus_citation_path: be/statute/x\n"
            "    corpus_citation_paths:\n"
            "      - be/statute/x\n",
            1,
        ),
        MONEY_PARAM_WITH_ATOM.replace(
            "            kind: parameter\n            source:\n",
            "            kind: parameter\n"
            "            corpus_citation_paths:\n"
            "              - be/statute/x\n"
            "            source:\n",
            1,
        ),
        MONEY_PARAM_WITH_ATOM.replace(
            "              corpus_citation_path: be/statute/x\n",
            "              corpus_citation_path: be/statute/x\n"
            "              corpus_citation_paths:\n"
            "                - be/statute/x\n",
            1,
        ),
    ],
    ids=("module", "proof-atom", "proof-source"),
)
def test_cli_proof_validate_rejects_single_item_plural_corpus_paths(
    tmp_path,
    content,
):
    corpus = _write_active_corpus(
        tmp_path / "axiom-corpus",
        ["The benefit amount is 186.51."],
    )
    module = _write(tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml", content)

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 1
    assert "Plural corpus source paths are not supported" in result.stdout


def test_cli_proof_validate_rejects_retired_claim_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AXIOM_CORPUS_REPO", str(tmp_path / "ambient-corpus"))
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        _claim_backed_proof_content(),
    )
    explicit_corpus = _write_active_corpus(
        tmp_path / "explicit-corpus",
        ["EXPLICIT BODY WITHOUT CLAIM QUOTE"],
    )

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(explicit_corpus),
    )

    assert result.returncode == 1
    assert "Proof claim references are not supported" in result.stdout
    assert "Source claims are not supported" in result.stdout


def test_cli_require_money_atoms_fails_without_ratchet(tmp_path):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_ATOM_LESS,
    )
    corpus = _write_active_corpus(tmp_path / "axiom-corpus", ["source text"])

    result = _run_cli(
        "proof-validate",
        str(module),
        "--require-money-atoms",
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 1
    assert "missing money proof atom" in result.stdout
    assert "over ratchet allowance" in result.stdout


def test_cli_rejects_removed_money_atom_root_override(tmp_path):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_ATOM_LESS,
    )
    corpus = _write_active_corpus(tmp_path / "axiom-corpus", ["source text"])

    result = _run_cli(
        "proof-validate",
        str(module),
        "--money-atom-root",
        str(tmp_path),
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --money-atom-root" in result.stderr


def test_cli_require_money_atoms_passes_with_seed_ratchet(tmp_path):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_ATOM_LESS,
    )
    corpus = _write_active_corpus(tmp_path / "axiom-corpus", ["source text"])
    ratchet = _write(tmp_path / "known-missing-money-atoms.yaml", "total_allowed: 1\n")

    result = _run_cli(
        "proof-validate",
        str(module),
        "--require-money-atoms",
        "--ratchet-file",
        str(ratchet),
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "within ratchet allowance" in result.stdout


def test_cli_proof_validate_requires_named_release_for_corpus_free_rulespec(tmp_path):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        "format: rulespec/v1\nrules: []\n",
    )

    result = _run_cli(
        "proof-validate",
        str(module),
        "--corpus-path",
        str(tmp_path / "missing-corpus"),
    )

    assert result.returncode == 1
    assert "corpus root does not exist" in result.stderr


def test_cli_emit_ratchet_round_trips(tmp_path):
    _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_ATOM_LESS,
    )
    module_b = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "y.yaml",
        MONEY_PARAM_ATOM_LESS,
    )
    corpus = _write_active_corpus(tmp_path / "axiom-corpus", ["source text"])

    emit = _run_cli(
        "proof-validate",
        str(tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml"),
        str(module_b),
        "--emit-ratchet",
        "--corpus-path",
        str(corpus),
    )
    assert emit.returncode == 0
    assert "be/statutes/x.yaml" in emit.stdout
    assert "rulespec-be" not in emit.stdout
    ratchet_path = _write(tmp_path / "known-missing-money-atoms.yaml", emit.stdout)

    check = _run_cli(
        "proof-validate",
        str(tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml"),
        str(module_b),
        "--require-money-atoms",
        "--ratchet-file",
        str(ratchet_path),
        "--corpus-path",
        str(corpus),
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
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_OK_BUT_BASE_PROOF_BROKEN,
    )
    corpus = _write_active_corpus(
        tmp_path / "axiom-corpus", ["The benefit amount is 186.51."]
    )

    # Plain proof validation fails on the malformed base proof block.
    base = _run_cli("proof-validate", str(module), "--corpus-path", str(corpus))
    assert base.returncode == 1

    # Money-atoms-only passes: the monetary value is covered and the malformed
    # base proof block is out of scope for the money gate.
    money_only = _run_cli(
        "proof-validate",
        str(module),
        "--money-atoms-only",
        "--corpus-path",
        str(corpus),
    )
    assert money_only.returncode == 0, money_only.stdout + money_only.stderr
    assert "PASSED" in money_only.stdout


def test_money_atoms_only_fails_on_uncovered_money(tmp_path):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_ATOM_LESS,
    )
    corpus = _write_active_corpus(tmp_path / "axiom-corpus", ["source text"])

    result = _run_cli(
        "proof-validate",
        str(module),
        "--money-atoms-only",
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 1
    assert "missing money proof atom" in result.stdout


def test_money_atoms_only_json_shape(tmp_path):
    module = _write(
        tmp_path / "rulespec-be" / "be" / "statutes" / "x.yaml",
        MONEY_PARAM_ATOM_LESS,
    )
    corpus = _write_active_corpus(tmp_path / "axiom-corpus", ["source text"])

    result = _run_cli(
        "proof-validate",
        str(module),
        "--money-atoms-only",
        "--json",
        "--corpus-path",
        str(corpus),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["money_atoms_only"] is True
    assert payload["passed"] is False
    assert payload["total_missing"] == 1
    assert payload["files"][0]["missing_atoms"][0]["rule"] == "benefit_amount"
