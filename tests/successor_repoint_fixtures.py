"""A real-shaped rulespec-us checkout for successor-repoint end-to-end tests.

The layout follows rulespec-us c654250f: a legacy v1 module with six indexed
tables and one scalar (here two tables and the scalar) and no ``effective_to``;
a signed-v5 successor at an unrelated canonical path whose window ends
2026-12-31; one dependent, ``us/statutes/26/32.yaml``, whose two v1 manifests
disagree about its companion digest; one ProgramSpec; and all six metadata
files a repoint reconciles.  Nothing here vendors rulespec-us bytes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from axiom_encode import __version__
from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    _sign_applied_encoding_manifest,
    cmd_repoint_legacy_successor,
)
from axiom_encode.harness.evals import resolve_corpus_source_unit
from axiom_encode.successor_repoint import ENVELOPE_SCHEMA
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
    install_test_eval_evidence_keys,
)
from tests.signing_broker_fixtures import SigningBrokerFixture
from tests.test_cli import _bind_test_corpus_release, _signed_manifest_payload

LEGACY = "us/policies/irs/rev-proc-2025-32/earned-income-credit.yaml"
LEGACY_COMPANION = "us/policies/irs/rev-proc-2025-32/earned-income-credit.test.yaml"
LEGACY_IDENTITY = "us:policies/irs/rev-proc-2025-32/earned-income-credit"
SUCCESSOR = "us/policies/irs/rev-proc-2025-32/page-15.yaml"
SUCCESSOR_COMPANION = "us/policies/irs/rev-proc-2025-32/page-15.test.yaml"
SUCCESSOR_IDENTITY = "us:policies/irs/rev-proc-2025-32/page-15"
SUCCESSOR_CITATION = "us/guidance/irs/rev-proc-2025-32/page-15"
DEPENDENT = "us/statutes/26/32.yaml"
DEPENDENT_COMPANION = "us/statutes/26/32.test.yaml"
TRANSITIVE = "us/statutes/26/24/d.yaml"
PROGRAM_SPEC = "programs/us/fiit/fy-2026.yaml"
LEGACY_V1_MANIFEST = (
    ".axiom/encoding-manifests/policies/irs/rev-proc-2025-32/earned-income-credit.json"
)
RETIRED_MANIFEST = (
    ".axiom/encoding-manifests/us/policies/irs/rev-proc-2025-32/"
    "earned-income-credit.json"
)
SUCCESSOR_MANIFEST = (
    ".axiom/encoding-manifests/us/policies/irs/rev-proc-2025-32/page-15.json"
)
DEPENDENT_RELATIVE_V1 = ".axiom/encoding-manifests/statutes/26/32.json"
DEPENDENT_MANIFEST = ".axiom/encoding-manifests/us/statutes/26/32.json"
METADATA_FILES = (
    "known-validation-gaps.yaml",
    ".axiom/toolchain.toml",
    ".axiom/index/provisions_to_rules.json",
    ".axiom/pending-validation-fingerprints.json",
    ".axiom/upstream-source-check-baseline.txt",
    "known-missing-money-atoms.yaml",
)
CONCEPT_MAP = [
    {
        "from": "eitc_earned_income_amounts",
        "to": "earned_income_credit_earned_income_amounts",
    },
    {
        "from": "eitc_maximum_credit_amounts",
        "to": "earned_income_credit_maximum_credit_amounts",
    },
    {
        "from": "eitc_maximum_investment_income",
        "to": "earned_income_credit_maximum_investment_income",
    },
]
ENVELOPE = {
    "schema": ENVELOPE_SCHEMA,
    "legacy_primary": LEGACY,
    "successor_primary": SUCCESSOR,
    "dependents": [DEPENDENT],
    "concept_map": CONCEPT_MAP,
    "program_scope_updates": [{"program_spec": PROGRAM_SPEC, "scope": "federal"}],
}
BROKER = SigningBrokerFixture(
    apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
    apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
)
ENCODER_PROVENANCE = {
    "root": "/repo/axiom-encode",
    "commit": "a" * 40,
    "dirty_tracked": False,
    "version": __version__,
    "version_commit": "b" * 40,
    "identity_source": "git",
}
HISTORICAL_ENCODER = {"commit": "c" * 40, "version": "0.2.1985"}

LEGACY_TEXT = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_paths:
      - us/guidance/irs/rev-proc-2025-32/page-14
      - us/guidance/irs/rev-proc-2025-32/page-15
rules:
  - name: eitc_earned_income_amounts
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 8680
          1: 13020
          2: 18290
          3: 18290
  - name: eitc_maximum_credit_amounts
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 664
          1: 4427
          2: 7316
          3: 8231
  - name: eitc_maximum_investment_income
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: 12200
"""
LEGACY_COMPANION_TEXT = f"""\
- name: base case
  period: 2026
  outputs:
    {LEGACY_IDENTITY}#eitc_earned_income_amounts: 8680
    {LEGACY_IDENTITY}#eitc_maximum_investment_income: 12200
"""
SUCCESSOR_TEMPLATE = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/irs/rev-proc-2025-32/page-15
    source_sha256: {source_sha256}
rules:
  - name: earned_income_credit_earned_income_amounts
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: qualifying_child_count_category
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        values:
          0: 8680
          1: 13020
          2: 18290
          3: 18290
  - name: earned_income_credit_maximum_credit_amounts
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: qualifying_child_count_category
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        values:
          0: 664
          1: 4427
          2: 7316
          3: 8231
  - name: earned_income_credit_maximum_investment_income
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: 12200
"""
SUCCESSOR_COMPANION_TEXT = f"""\
- name: page 15 amounts
  period: 2026
  outputs:
    {SUCCESSOR_IDENTITY}#earned_income_credit_maximum_investment_income: 12200
"""
DEPENDENT_TEMPLATE = """\
format: rulespec/v1
imports:
  - us:statutes/26/152/c
  - us:policies/irs/rev-proc-2025-32/earned-income-credit
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/32
  deferred_outputs:
    - output: us:statutes/26/32/f#eitc_table_based_credit_determination
      blocked_by:
        - us:policies/irs/rev-proc-2025-32/earned-income-credit#eitc_maximum_credit_amounts
      reason: >-
        Subsection (f) directs the Secretary to prescribe earned-income-credit
        tables rather than this module materializing each table row.
rules:
  - name: eitc_earned_income_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:policies/irs/rev-proc-2025-32/earned-income-credit#eitc_earned_income_amounts
              output: eitc_earned_income_amounts
              hash: sha256:{legacy_sha256}
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match capped_qualifying_children:
              0 => eitc_earned_income_amounts[0]
              1 => eitc_earned_income_amounts[1]
              2 => eitc_earned_income_amounts[2]
              3 => eitc_earned_income_amounts[3]
  - name: eitc_maximum_credit_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:policies/irs/rev-proc-2025-32/earned-income-credit#eitc_maximum_credit_amounts
              output: eitc_maximum_credit_amounts
              hash: sha256:{legacy_sha256}
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match capped_qualifying_children:
              0 => eitc_maximum_credit_amounts[0]
              1 => eitc_maximum_credit_amounts[1]
              2 => eitc_maximum_credit_amounts[2]
              3 => eitc_maximum_credit_amounts[3]
  - name: eitc_investment_income_eligible
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:policies/irs/rev-proc-2025-32/earned-income-credit#eitc_maximum_investment_income
              output: eitc_maximum_investment_income
              hash: sha256:{legacy_sha256}
    versions:
      - effective_from: '2026-01-01'
        formula: aggregate_investment_income <= eitc_maximum_investment_income
"""
DEPENDENT_COMPANION_TEXT = """\
- name: no children
  period: 2026
  inputs:
    us:statutes/26/32#input.capped_qualifying_children: 0
  outputs:
    us:statutes/26/32#eitc_earned_income_amount: 8680
"""
TRANSITIVE_TEXT = """\
format: rulespec/v1
imports:
  - us:statutes/26/32
rules: []
"""
PROGRAM_SPEC_TEXT = """\
program: us/fiit
scope:
  federal:
    # Rev. Proc. 2025-32 inflation adjustments
    - policies/irs/rev-proc-2025-32/earned-income-credit
    - statutes/26/24/d
    - statutes/26/32
"""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _write(root: Path, relative: str, raw: bytes | str) -> bytes:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.chmod(0o644)
    return data


def _waiver_entry(fingerprint: str) -> str:
    expires = (date.today() + timedelta(days=30)).isoformat()
    return (
        "    active:\n"
        f'      fingerprint: "sha256:{fingerprint}"\n'
        '      owner: "@MaxGhenis"\n'
        '      issue: "https://github.com/TheAxiomFoundation/rulespec-us/issues/782"\n'
        f'      expires: "{expires}"\n'
    )


def _v1_manifest(
    applied_files: list[dict[str, str]],
    *,
    tool: str,
    backend: str,
    runner: str,
    citation: str,
    manual_exception: str | None = None,
) -> str:
    payload: dict[str, object] = {
        "applied_files": applied_files,
        "axiom_encode_version": "0.2.64",
        "backend": backend,
        "citation": citation,
        "context_manifest_file": None,
        "context_manifest_sha256": None,
        "generated_at": "2026-05-11T12:38:16.848631+00:00",
        "generated_output_file": None,
        "generated_output_root": "/repo/rulespec-us",
        "generated_output_sha256": None,
        "generation_prompt_sha256": None,
        "model": "gpt-5.5",
        "run_id": None,
        "runner": runner,
        "schema_version": "axiom-encode/applied-rulespec/v1",
        "signature": {
            "algorithm": "hmac-sha256",
            "key_id": "axiom-encode-apply-v1",
            "value": "1" * 64,
        },
        "tool": tool,
        "trace_file": None,
        "trace_sha256": None,
    }
    if manual_exception is not None:
        payload["manual_exception"] = manual_exception
    return json.dumps(payload, indent=2) + "\n"


@dataclass
class RepointFixture:
    """One committed fixture checkout plus everything needed to repoint it."""

    tmp_path: Path
    repo: Path
    corpus: Path
    engine: Path
    request: Path
    base: str
    preimages: dict[str, bytes]
    successor_manifest_bytes: bytes

    def args(self) -> SimpleNamespace:
        return SimpleNamespace(
            request=self.request,
            policy_repo_path=self.repo,
            axiom_rules_path=self.engine,
            corpus_path=self.corpus,
        )


def install_repoint_signing(monkeypatch) -> None:
    """Install the test apply broker and the pinned clean encoder provenance."""

    install_test_eval_evidence_keys(
        monkeypatch,
        apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
        apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
    )
    monkeypatch.setattr(
        "axiom_encode.cli._require_applied_encoding_manifest_signer",
        lambda: BROKER,
    )
    monkeypatch.setattr(
        "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
        lambda: dict(ENCODER_PROVENANCE),
    )
    monkeypatch.setattr(
        "axiom_encode.cli._current_guard_encoder_execution_identity",
        lambda: {
            "repository": "github.com/TheAxiomFoundation/axiom-encode",
            "commit": ENCODER_PROVENANCE["commit"],
            "version": __version__,
        },
    )


def build_repoint_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    envelope: dict | None = None,
    before_commit=None,
    successor_encoder: dict | None = HISTORICAL_ENCODER,
) -> RepointFixture:
    """Create and commit one canonical rulespec-us checkout ready to repoint.

    ``before_commit(repo)`` may adjust the checkout before the base commit, and
    ``envelope`` replaces the dispatched request.
    """

    install_repoint_signing(monkeypatch)

    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")
    corpus = tmp_path / "axiom-corpus"
    engine = tmp_path / "axiom-rules-engine"
    engine.mkdir()

    legacy = _write(repo, LEGACY, LEGACY_TEXT)
    legacy_companion = _write(repo, LEGACY_COMPANION, LEGACY_COMPANION_TEXT)
    dependent = _write(
        repo, DEPENDENT, DEPENDENT_TEMPLATE.format(legacy_sha256=_sha256(legacy))
    )
    dependent_companion = _write(repo, DEPENDENT_COMPANION, DEPENDENT_COMPANION_TEXT)
    _write(repo, TRANSITIVE, TRANSITIVE_TEXT)
    _write(repo, PROGRAM_SPEC, PROGRAM_SPEC_TEXT)

    # Metadata: every file a repoint reconciles, each naming the legacy module
    # the way rulespec-us does.
    _write(
        repo,
        "known-validation-gaps.yaml",
        "# Modules failing encode validate.\n"
        "validate_failures:\n"
        f'  "{LEGACY}":\n'
        + _waiver_entry("1" * 64)
        + f'  "{TRANSITIVE}":\n'
        + _waiver_entry("2" * 64),
    )
    release = _bind_test_corpus_release(
        repo,
        corpus,
        citation_path=SUCCESSOR_CITATION,
        body="Rev. Proc. 2025-32 section 3.07 earned income credit amounts.\n",
    )
    _write(
        repo,
        ".axiom/index/provisions_to_rules.json",
        json.dumps(
            {
                "schema": "axiom.rulespec.provisions_to_rules/v1",
                "description": "Reverse index (fixture).",
                "provisions": {
                    SUCCESSOR_CITATION: [
                        {"module": LEGACY, "via": ["module"]},
                        {"module": SUCCESSOR, "via": ["module"]},
                    ],
                    "us/statute/26/32": [{"module": DEPENDENT, "via": ["module"]}],
                },
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        repo,
        ".axiom/pending-validation-fingerprints.json",
        json.dumps(
            {
                "generated_from": {
                    "divergence_note": (
                        "branch bytes differ for "
                        f"{LEGACY}, {TRANSITIVE} and 91 other modules"
                    )
                },
                "modules": {
                    LEGACY: {"fingerprint": "sha256:" + "3" * 64},
                    TRANSITIVE: {"fingerprint": "sha256:" + "4" * 64},
                },
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        repo,
        ".axiom/upstream-source-check-baseline.txt",
        f"# Modules exempt from the upstream source check.\n{LEGACY}\n{TRANSITIVE}\n",
    )
    _write(
        repo,
        "known-missing-money-atoms.yaml",
        "# Money-atom proof-obligation ratchet.\n"
        "total_allowed: 12\n"
        "# Current per-file backlog (informational):\n"
        f"#   {LEGACY}: 7\n"
        f"#   {TRANSITIVE}: 5\n",
    )

    # The successor and its signed-v5 model manifest from a historical encoder.
    attestation = resolve_corpus_source_unit(
        SUCCESSOR_CITATION, release
    ).source_attestation
    attestation["generation_input_sha256"] = attestation["resolved_text_sha256"]
    attestation["rulespec_root"] = "rulespec-us/us"
    successor = _write(
        repo,
        SUCCESSOR,
        SUCCESSOR_TEMPLATE.format(source_sha256=attestation["source_sha256"]),
    )
    successor_companion = _write(repo, SUCCESSOR_COMPANION, SUCCESSOR_COMPANION_TEXT)
    waiver_sha256 = _sha256((repo / "known-validation-gaps.yaml").read_bytes())
    manifest = _signed_manifest_payload(
        {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "backend": "codex",
            "citation": SUCCESSOR_CITATION,
            "validation_waiver_set_sha256": waiver_sha256,
            "source_attestation": attestation,
            "applied_files": [
                {"path": SUCCESSOR, "sha256": _sha256(successor)},
                {"path": SUCCESSOR_COMPANION, "sha256": _sha256(successor_companion)},
            ],
        }
    )
    if successor_encoder is not None:
        manifest["axiom_encode_version"] = successor_encoder["version"]
        manifest["axiom_encode_git"]["commit"] = successor_encoder["commit"]
        manifest["axiom_encode_git"]["version"] = successor_encoder["version"]
        execution = manifest["validation_execution"]["axiom_encode"]
        execution["commit"] = successor_encoder["commit"]
        execution["version"] = successor_encoder["version"]
        manifest.pop("signature", None)
    _sign_applied_encoding_manifest(manifest, BROKER)
    successor_manifest_bytes = _write(
        repo, SUCCESSOR_MANIFEST, json.dumps(manifest, indent=2) + "\n"
    )

    # Historical v1 ownership, shaped like rulespec-us's: the legacy manifest
    # binds its exact bytes in the jurisdiction-less scope; the dependent has
    # a deterministic-repair manifest with a superseded companion digest and a
    # later manual re-attestation of the current companion.
    _write(
        repo,
        LEGACY_V1_MANIFEST,
        _v1_manifest(
            [
                {"path": LEGACY.removeprefix("us/"), "sha256": _sha256(legacy)},
                {
                    "path": LEGACY_COMPANION.removeprefix("us/"),
                    "sha256": _sha256(legacy_companion),
                },
            ],
            tool="axiom-encode encode --apply",
            backend="codex",
            runner="codex-gpt-5.5",
            citation="policies/irs/rev-proc-2025-32/earned-income-credit",
        ),
    )
    _write(
        repo,
        DEPENDENT_RELATIVE_V1,
        _v1_manifest(
            [
                {"path": DEPENDENT.removeprefix("us/"), "sha256": _sha256(dependent)},
                {"path": DEPENDENT_COMPANION.removeprefix("us/"), "sha256": "e" * 64},
            ],
            tool="axiom-encode deterministic/manual repair",
            backend="deterministic",
            runner="deterministic-repair",
            citation="us:statutes/26/32",
        ),
    )
    _write(
        repo,
        DEPENDENT_MANIFEST,
        _v1_manifest(
            [{"path": DEPENDENT_COMPANION, "sha256": _sha256(dependent_companion)}],
            tool="axiom-encode sign-applied-files",
            backend="manual",
            runner="manual-attestation",
            citation="us:statutes/26/32",
            manual_exception="repair",
        ),
    )

    if before_commit is not None:
        before_commit(repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fixture base")
    base = git(repo, "rev-parse", "HEAD").strip()
    request = tmp_path / "successor-repoint-request.json"
    request.write_text(json.dumps(envelope or ENVELOPE), encoding="utf-8")
    preimages = {
        path: (repo / path).read_bytes()
        for path in (
            LEGACY,
            LEGACY_COMPANION,
            DEPENDENT,
            DEPENDENT_COMPANION,
            PROGRAM_SPEC,
            *METADATA_FILES,
        )
    }
    return RepointFixture(
        tmp_path=tmp_path,
        repo=repo,
        corpus=release.root,
        engine=engine,
        request=request,
        base=base,
        preimages=preimages,
        successor_manifest_bytes=successor_manifest_bytes,
    )


def _passing_overlay_validation(
    _pipeline, *, dependent_pipeline, overlay_target, dependents
):
    passed = SimpleNamespace(all_passed=True, results={})
    return [(overlay_target, passed), *[(path, passed) for path in dependents]]


def run_repoint(fixture: RepointFixture) -> None:
    """Run the real command with the overlay validators stubbed to pass."""

    with (
        patch("axiom_encode.cli.ValidatorPipeline", MagicMock()),
        patch(
            "axiom_encode.cli._validate_overlay_files",
            side_effect=_passing_overlay_validation,
        ),
    ):
        cmd_repoint_legacy_successor(fixture.args())
