"""Failure-triage evidence bundle and probes (Stage 3, PR 1; no model)."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.concepts.registry import Concept, ConceptRegistry
from axiom_encode.harness.evals import _slugify
from axiom_encode.triage import (
    EVIDENCE_SCHEMA,
    AttemptMaterial,
    CandidateMaterial,
    EvidenceInputError,
    TriageMaterial,
    build_evidence,
    cli_commands,
    evidence_sha256,
    load_any,
    load_local_encode,
    load_local_run,
    load_workflow_bundle,
    resolve_ref,
)
from axiom_encode.triage.evidence import (
    _RESERVED_BYTES,
    EvidenceLimits,
    canonical_json,
    iter_run_inputs,
)
from axiom_encode.triage.issues import issue_signature, parse_issue
from axiom_encode.triage.probes import (
    probe_corpus,
    probe_gate,
    probe_import,
    probe_persistence,
    probe_registry,
)
from axiom_encode.triage.views import (
    CheckoutRulespecView,
    GitRefRulespecView,
    LocalCorpusIndex,
    RulespecSnapshot,
)

FIXTURES = Path(__file__).parent / "fixtures" / "triage_evidence"
# Pinned: any change to the bundle shape or the fixture changes this digest.
E2E_EVIDENCE_SHA256 = "4d9c619f8dad439dbf83c427a71696b7438d011641f4348ddea9c743d06a8c0e"
CITATION = "us/statute/7/2014/d"
MODULE = "statutes/7/2014/d.yaml"
OPERATIVE = (
    "(d) Income. Household income shall include all income from whatever "
    "source, as provided in 7 U.S.C. 2012.\n"
)
CANDIDATE = """format: rulespec/v1
imports:
  - us:statutes/7/2012/a#household_definition
  - us:statutes/7/2015/b#missing_symbol
  - us:statutes/7/9999#anything
module:
  deferred_outputs:
  - output: us:statutes/7/2014/d#excluded_income
    blocked_by:
    - us:statutes/7/2012/a#household_definition
    reason: defined elsewhere
rules:
- name: household_income
  kind: derived
  metadata:
    proof:
      atoms:
      - source:
          corpus_citation_path: us/statute/7/2014/d
  versions:
  - effective_from: '2026-01-01'
    formula: gross_income + household_definition
"""


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rulespec_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


RULESPEC_FILES = {
    "us/statutes/7/2012/a.yaml": (
        "format: rulespec/v1\nrules:\n- name: household_definition\n  kind: derived\n"
    ),
    "us/statutes/7/2015/b.yaml": (
        "format: rulespec/v1\nrules:\n- name: other_symbol\n  kind: derived\n"
    ),
    "us/statutes/7/2014/d.yaml": "format: rulespec/v1\nrules: []\n",
    "us/statutes/7/2012/x/y.yaml": "format: rulespec/v1\nrules: []\n",
}


def _corpus(tmp_path: Path, rows: list[dict], version: str = "2026-01-01-v") -> Path:
    root = tmp_path / "corpus"
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        jurisdiction, document_class = row["citation_path"].split("/")[:2]
        by_bucket.setdefault((jurisdiction, document_class), []).append(row)
    for (jurisdiction, document_class), items in by_bucket.items():
        directory = (
            root / "data" / "corpus" / "provisions" / jurisdiction / document_class
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{version}.jsonl").write_text(
            "".join(json.dumps({"version": version, **row}) + "\n" for row in items)
        )
    (root / "data" / "corpus" / "provisions").mkdir(parents=True, exist_ok=True)
    return root


def _material(**overrides) -> TriageMaterial:
    fields = dict(
        input_kind="local_encode",
        input_name="case",
        citation=CITATION,
        policy_prefix="us",
        source_text=OPERATIVE,
        source_attestation={
            "resolved_corpus_citation_path": CITATION,
            "row": {
                "jurisdiction": "us",
                "document_class": "statute",
                "version": "2026-01-01-v",
            },
        },
        candidate=CandidateMaterial(
            module_path=MODULE, rulespec=CANDIDATE, tests="[]\n"
        ),
        final_issues=("statutes/7/2014/d.yaml: ci: [complete-source-unit:tests] x",),
        attempts=(
            AttemptMaterial(attempt=1, issues=("a",), issue_count=1),
            AttemptMaterial(attempt=2, issues=("a", "b"), issue_count=2),
        ),
        attempt_count=2,
    )
    fields.update(overrides)
    return TriageMaterial(**fields)


def _view(tmp_path: Path) -> GitRefRulespecView:
    repo, sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    return GitRefRulespecView(repo, sha)


def _factory(view):
    return lambda: RulespecSnapshot(view, "us")


def _by_subject(result: dict) -> dict[str, dict]:
    return {finding["subject"]: finding for finding in result["findings"]}


# --------------------------------------------------------------------------
# Issues
# --------------------------------------------------------------------------


def test_parse_issue_reads_file_gate_and_tag():
    issue = parse_issue("statutes/7/a.yaml: ci: [complete-source-unit:tests] body")
    assert (issue.file, issue.gate, issue.tag) == (
        "statutes/7/a.yaml",
        "ci",
        "complete-source-unit:tests",
    )
    plain = parse_issue("Axiom rules engine compile failed: x")
    assert (plain.file, plain.gate, plain.tag) == (None, None, None)


def test_issue_signature_normalizes_volatile_parts():
    first = issue_signature(
        "yaml parse error at line 12 column 10 in "
        "`/home/runner/work/_temp/rulespec-us/us/statutes/7/a.yaml` "
        "offsets 59:63 digest 0123456789abcdef0123"
    )
    second = issue_signature(
        "yaml parse error at line 40 column 2 in "
        "`<rulespec-validation-root>/rulespec-us/us/statutes/7/a.yaml` "
        "offsets 1:9 digest fedcba9876543210fedc"
    )
    assert first == second
    assert issue_signature("value 2026 missing") != issue_signature(
        "value 2025 missing"
    )


# --------------------------------------------------------------------------
# Gate probe
# --------------------------------------------------------------------------


def test_gate_probe_reports_an_empty_body_and_an_existing_module(tmp_path):
    view = _view(tmp_path)
    result = probe_gate(_material(source_text="  \n"), snapshot_factory=_factory(view))
    kinds = {finding["kind"]: finding for finding in result["findings"]}
    assert kinds["no_operative_text"]["class"] == "corpus"
    assert kinds["no_operative_text"]["enforceable"] is True
    assert kinds["existing_module"]["evidence"] == ["statutes/7/2014/d.yaml"]
    assert result["checks_not_run"] == ["unencoded_reference"]


def test_gate_probe_finds_nothing_for_an_operative_replacement_target(tmp_path):
    view = _view(tmp_path)
    corpus = LocalCorpusIndex(
        _corpus(tmp_path, [{"citation_path": CITATION, "body": "x"}])
    )
    result = probe_gate(
        _material(replacement_target=True),
        snapshot_factory=_factory(view),
        corpus=corpus,
    )
    assert result["status"] == "ok"
    assert result["findings"] == []
    # The run's release-bound corpus scope is never available locally.
    assert result["checks_not_run"] == ["unencoded_reference"]


def test_gate_probe_reports_a_repealed_row_and_corpus_reports_the_reference(
    tmp_path,
):
    view = _view(tmp_path)
    body = "Benefits under 7 U.S.C. 2020 apply."
    corpus = LocalCorpusIndex(
        _corpus(
            tmp_path,
            [
                {
                    "citation_path": CITATION,
                    "body": body,
                    "metadata": {"status": "repealed"},
                },
                {"citation_path": "us/statute/7/2020", "body": "y"},
            ],
        )
    )
    result = probe_gate(
        _material(source_text=body, replacement_target=True),
        snapshot_factory=_factory(view),
        corpus=corpus,
    )
    kinds = {finding["kind"]: finding for finding in result["findings"]}
    assert set(kinds) == {"repealed_target"}
    assert kinds["repealed_target"]["class"] == "selection"
    corpus_result = probe_corpus(
        _material(source_text=body), corpus=corpus, snapshot_factory=_factory(view)
    )
    reference = _by_subject(corpus_result)["us/statute/7/2020"]
    assert (reference["status"], reference["rulespec_module"]) == ("present", "absent")
    encoded = probe_corpus(
        _material(source_text="See 7 U.S.C. 2012."),
        corpus=corpus,
        snapshot_factory=_factory(view),
    )
    # 7/2012 has modules under it at the ref (2012/a.yaml).
    assert _by_subject(encoded)["us/statute/7/2012"]["rulespec_module"] == "present"


def test_gate_probe_skips_without_source():
    assert probe_gate(_material(source_text=None))["status"] == "skipped"


# --------------------------------------------------------------------------
# Registry probe
# --------------------------------------------------------------------------


def _registry() -> ConceptRegistry:
    concept = Concept(
        id="snap_gross_income",
        canonical_name="snap_gross_income",
        producer_anchor="us:statutes/7/2014/c",
        blocked_synonyms=("gross_income",),
    )
    return ConceptRegistry(
        concepts_by_id={concept.id: concept},
        canonical_to_concept={concept.canonical_name: concept},
        synonym_to_concept={"gross_income": concept},
    )


def test_registry_probe_checks_imports_deferrals_and_registered_inputs(tmp_path):
    view = _view(tmp_path)
    result = probe_registry(
        _material(workspace_context=("us:statutes/7/2015/b",)),
        snapshot_factory=_factory(view),
        registry=_registry(),
    )
    findings = {(f["kind"], f["subject"]): f for f in result["findings"]}
    exported = findings[("import", "us:statutes/7/2012/a#household_definition")]
    assert exported["status"] == "exported"
    assert exported["module"] == "us/statutes/7/2012/a.yaml"
    assert exported["refs"] == ["candidate:L3"]
    missing_symbol = findings[("import", "us:statutes/7/2015/b#missing_symbol")]
    assert missing_symbol["status"] == "symbol_missing"
    assert missing_symbol["in_workspace_context"] is True
    assert findings[("import", "us:statutes/7/9999#anything")]["status"] == (
        "module_missing"
    )
    blocker = findings[
        ("deferral_blocked_by", "us:statutes/7/2012/a#household_definition")
    ]
    assert blocker["status"] == "exported"
    registered = findings[("input", "us:statutes/7/2014/c#snap_gross_income")]
    assert registered["status"] == "module_missing"
    assert registered["blocked_synonym"] is True
    assert registered["input_name"] == "gross_income"


def test_registry_probe_without_a_rulespec_state_is_not_checked():
    result = probe_registry(_material())
    assert result["reason"] == "no RuleSpec state at the run's ref is available locally"
    assert {f["status"] for f in result["findings"]} == {"not_checked"}


def test_registry_probe_skips_without_candidate():
    assert probe_registry(_material(candidate=None))["status"] == "skipped"


def test_registry_probe_on_a_checkout_view(tmp_path):
    repo, _sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    view = CheckoutRulespecView(repo)
    result = probe_registry(_material(), snapshot_factory=_factory(view))
    statuses = _by_subject(result)
    assert statuses["us:statutes/7/2012/a#household_definition"]["status"] == "exported"
    assert view.identity()["kind"] == "checkout"


def test_git_ref_view_refuses_unknown_and_unsafe_refs(tmp_path):
    repo, _sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    with pytest.raises(LookupError):
        GitRefRulespecView(repo, "0" * 40)
    with pytest.raises(ValueError):
        GitRefRulespecView(repo, "--upload-pack=x")


# --------------------------------------------------------------------------
# Corpus probe
# --------------------------------------------------------------------------


def test_corpus_probe_reports_present_absent_and_repealed(tmp_path):
    corpus = LocalCorpusIndex(
        _corpus(
            tmp_path,
            [
                {"citation_path": CITATION, "body": OPERATIVE, "heading": "Income"},
                {
                    "citation_path": "us/statute/7/2012",
                    "body": None,
                    "metadata": {"status": "repealed"},
                },
            ],
        )
    )
    result = probe_corpus(_material(), corpus=corpus)
    findings = _by_subject(result)
    assert findings[CITATION]["status"] == "present"
    assert findings[CITATION]["refs"] == ["target"]
    assert findings["us/statute/7/2012"]["status"] == "repealed"
    assert findings["us/statute/7/2012"]["refs"] == ["source:L1"]
    assert result["attested_version_present_locally"] is True


def test_corpus_probe_reports_an_absent_row_and_a_missing_attested_version(tmp_path):
    corpus = LocalCorpusIndex(_corpus(tmp_path, [], version="other"))
    result = probe_corpus(_material(), corpus=corpus)
    assert _by_subject(result)[CITATION]["status"] == "absent"
    assert result["attested_version_present_locally"] is False


def test_corpus_probe_skips_without_a_corpus():
    result = probe_corpus(_material())
    assert result["status"] == "skipped"
    assert result["subject_count"] == 2


# --------------------------------------------------------------------------
# Import probe
# --------------------------------------------------------------------------


def _compile_issue(path: str) -> str:
    return (
        "Axiom rules engine compile failed: failed to load RuleSpec module "
        f"`<rulespec-validation-root>/rulespec-us/{path}`: bad"
    )


def test_import_probe_places_each_compile_error():
    material = _material(
        final_issues=(
            _compile_issue("us/statutes/7/2014/d.yaml"),
            _compile_issue("us/statutes/7/2012/a.yaml"),
            _compile_issue("us/statutes/7/2015/b.yaml"),
            _compile_issue("us/statutes/7/2016.yaml"),
            "Axiom rules engine compile failed: duplicate derived rule `x`",
            "[complete-source-unit:tests] not a compile error",
        ),
        candidate=CandidateMaterial(
            module_path=MODULE,
            rulespec=CANDIDATE,
            sibling_modules=("statutes/7/2016.yaml",),
        ),
        run_outputs=("us:statutes/7/2015/b",),
        workspace_context=("us:statutes/7/2012/a",),
    )
    result = probe_import(material)
    statuses = [(f["subject"], f["status"], f["refs"]) for f in result["findings"]]
    assert statuses == [
        ("us:statutes/7/2014/d", "own_module", ["issue:final:0"]),
        ("us:statutes/7/2012/a", "workspace_context_module", ["issue:final:1"]),
        ("us:statutes/7/2015/b", "authored_sibling", ["issue:final:2"]),
        ("us:statutes/7/2016", "authored_sibling", ["issue:final:3"]),
        (None, "unlocated", ["issue:final:4"]),
    ]
    result = probe_import(
        replace(material, final_issues=(_compile_issue("us/statutes/26/32.yaml"),))
    )
    assert result["findings"][0]["status"] == "imported_not_authored"


def test_import_probe_without_compile_errors_has_no_findings():
    result = probe_import(_material())
    assert result["status"] == "ok"
    assert result["findings"] == []
    assert probe_import(_material(final_issues=()))["status"] == "skipped"


# --------------------------------------------------------------------------
# Persistence probe
# --------------------------------------------------------------------------


def test_persistence_probe_splits_persistent_and_changed_issues():
    material = _material(
        attempts=(
            AttemptMaterial(
                attempt=1, issues=("[a:b] parse error at line 3", "[c:d] gone")
            ),
            AttemptMaterial(attempt=2, issues=("[a:b] parse error at line 9", "new")),
        ),
        final_issues=("[a:b] parse error at line 9",),
    )
    result = probe_persistence(material)
    findings = {f["subject"]: f for f in result["findings"]}
    persistent = findings["[a:b] parse error at line #"]
    assert persistent["status"] == "persistent"
    assert persistent["refs"] == [
        "attempt:1:issue:0",
        "attempt:2:issue:0",
        "issue:final:0",
    ]
    assert persistent["attempts_present"] == 2
    assert findings["[c:d] gone"]["status"] == "changed"
    assert findings["new"]["status"] == "changed"
    assert result["persistent_tags"] == ["a:b"]


def test_persistence_probe_needs_two_recorded_attempts():
    material = _material(
        attempts=(
            AttemptMaterial(attempt=1, issues=None),
            AttemptMaterial(attempt=2, issues=("x",)),
        )
    )
    result = probe_persistence(material)
    assert result["status"] == "skipped"
    assert result["attempts_with_issues"] == 1


# --------------------------------------------------------------------------
# Bundle: determinism, truncation, refs
# --------------------------------------------------------------------------


def test_bundle_is_deterministic_and_location_independent(tmp_path):
    view = _view(tmp_path)
    first = build_evidence(_material(), rulespec_view=view)
    second = build_evidence(
        replace(_material(), input_name="moved"), rulespec_view=view
    )
    assert first["schema"] == EVIDENCE_SCHEMA
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert evidence_sha256(first) == first["evidence_sha256"]
    changed = build_evidence(
        _material(final_issues=("other issue",)), rulespec_view=view
    )
    assert changed["evidence_sha256"] != first["evidence_sha256"]
    assert json.dumps(first, sort_keys=True) == json.dumps(
        build_evidence(_material(), rulespec_view=view), sort_keys=True
    )


def test_bundle_records_truncation_and_drops_refs_it_cut(tmp_path):
    long_source = "line\n" * 50 + "Benefits under 7 U.S.C. 2020 apply.\n"
    limits = EvidenceLimits(source_chars=20, issues=1, issue_chars=5)
    bundle = build_evidence(
        _material(source_text=long_source, final_issues=("first issue", "second")),
        corpus=LocalCorpusIndex(_corpus(tmp_path, [])),
        limits=limits,
    )
    fields = {item["field"]: item for item in bundle["truncation"]}
    assert fields["target.source"]["original_chars"] == len(long_source)
    assert fields["target.source"]["kept_chars"] == 20
    assert fields["final_issues"] == {
        "field": "final_issues",
        "original_items": 2,
        "kept_items": 1,
    }
    assert fields["final_issues[0]"]["kept_chars"] == 5
    assert fields["probes.refs"]["dropped_unresolvable_refs"] >= 1
    reference = _by_subject(bundle["probes"]["corpus"])["us/statute/7/2020"]
    assert reference["refs"] == []
    assert bundle["target"]["source"]["sha256"] != ""


def test_bundle_shrinks_to_fit_its_byte_bound():
    material = _material(source_text="word " * 30_000)
    bundle = build_evidence(material, max_bytes=40_000)
    assert bundle["size_bytes"] <= 40_000
    assert {"field": "bundle", "shrunk_to_fit_bytes": 40_000} in bundle["truncation"]


def test_every_finding_ref_resolves_and_ids_are_stable(tmp_path):
    view = _view(tmp_path)
    bundle = build_evidence(_material(), rulespec_view=view, registry=_registry())
    for name, probe in bundle["probes"].items():
        for index, finding in enumerate(probe["findings"]):
            assert finding["id"] == f"probe:{name}:{index}"
            assert resolve_ref(bundle, finding["id"])
            assert all(resolve_ref(bundle, ref) for ref in finding["refs"])


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("target", True),
        ("source", True),
        ("source:L1", True),
        ("source:L2", False),
        ("candidate:L3", True),
        ("tests:L1", True),
        ("issue:final:0", True),
        ("issue:final:1", False),
        ("attempt:2", True),
        ("attempt:2:issue:1", True),
        ("attempt:2:issue:2", False),
        ("attempt:3", False),
        ("probe:corpus:0", False),
        ("source:L0", False),
        ("nonsense", False),
    ],
)
def test_resolve_ref(ref, expected):
    bundle = build_evidence(_material())
    assert resolve_ref(bundle, ref) is expected


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _workspace(root: Path, *, runner: str, citation: str, source: str) -> Path:
    workspace = root / "_eval_workspaces" / runner / _slugify(citation) / "workspace"
    (workspace / "context" / "statutes" / "7" / "2012").mkdir(parents=True)
    (workspace / "source.txt").write_text(source)
    (workspace / "source-metadata.json").write_text(
        json.dumps(
            {
                "source_attestation": {
                    "resolved_corpus_citation_path": citation,
                    "rulespec_root": "/home/runner/x",
                    "row": {"version": "2026-01-01-v"},
                }
            }
        )
    )
    (workspace / "provision-metadata.txt").write_text(json.dumps({"title": "7"}))
    (workspace / "context" / "statutes" / "7" / "2012" / "a.yaml").write_text("ctx\n")
    (workspace / "context-manifest.json").write_text(
        json.dumps(
            {
                "policy_prefix": "us",
                "context_files": [
                    {
                        "import_path": "us:statutes/7/2012/a",
                        "workspace_path": "context/statutes/7/2012/a.yaml",
                    },
                    {
                        "kind": "corpus_amendment_act",
                        "citation_path": "us/statute/pl-1",
                        "workspace_path": "context/amendment-act-1.txt",
                    },
                ],
                "dropped_amendment_documents": [{"citation_path": "us/statute/pl-2"}],
                "review_findings_files": [{"content": "Encode the income rule."}],
            }
        )
    )
    return workspace


def _output_root(root: Path, *, runner: str = "openai-gpt-6-sol") -> Path:
    module = root / runner / MODULE
    module.parent.mkdir(parents=True)
    module.write_text(CANDIDATE)
    context_copy = root / runner / "statutes" / "7" / "2012" / "a.yaml"
    context_copy.parent.mkdir(parents=True)
    context_copy.write_text("ctx\n")
    authored = root / runner / "statutes" / "7" / "2016.yaml"
    authored.write_text("format: rulespec/v1\n")
    _workspace(root, runner=runner, citation=CITATION, source=OPERATIVE)
    return root


def _candidate_dir(root: Path, issues: list[str]) -> Path:
    candidate = root / "final-rejected-candidate"
    (candidate / "statutes" / "7" / "2014").mkdir(parents=True)
    (candidate / MODULE).write_text(CANDIDATE)
    (candidate / "statutes" / "7" / "2014" / "d.test.yaml").write_text("[]\n")
    (candidate / "issues.json").write_text(
        json.dumps(
            {
                "schema": "axiom-encode/failed-encode-candidate/v1",
                "attempt_count": 3,
                "citation": CITATION,
                "encoder_version": "0.2.1",
                "issues": issues,
                "path": MODULE,
            }
        )
    )
    return candidate


def _workflow_bundle(tmp_path: Path, name: str = "123-1") -> Path:
    bundle = tmp_path / "bundles" / name
    root = bundle / "x"
    lane = root / "generated" / "target"
    lane.mkdir(parents=True)
    _output_root(lane)
    _candidate_dir(lane, ["statutes/7/2014/d.yaml: ci: [complete-source-unit:tests] x"])
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema": "axiom-encode/failed-reencode-diagnostics/v1",
                "citation": CITATION,
                "corpus_ref": "c" * 40,
                "rulespec_ref": "r" * 40,
                "encoder_commit": "e" * 40,
                "failed_steps": ["encode_apply"],
                "replace_rulespec_path": None,
                "workflow_run_id": "123",
            }
        )
    )
    return bundle


def test_workflow_bundle_loader_reads_the_lane(tmp_path):
    [material] = load_workflow_bundle(_workflow_bundle(tmp_path))
    assert material.input_kind == "workflow_bundle"
    assert (material.run_id, material.lane, material.policy_prefix) == (
        "123",
        "target",
        "us",
    )
    assert material.rulespec_ref == "r" * 40
    assert material.candidate.runner == "openai-gpt-6-sol"
    assert material.candidate.tests == "[]\n"
    assert material.attempt_count == 3
    # The selected candidate's attempt number is not recorded.
    assert material.attempts == ()
    assert material.workspace_context == ("us:statutes/7/2012/a",)
    # The runner's copy of its context module is not authored output.
    assert material.run_outputs == ("us:statutes/7/2014/d", "us:statutes/7/2016")
    assert material.amendment_documents == ("us/statute/pl-1",)
    assert material.dropped_amendment_documents == ("us/statute/pl-2",)
    assert material.review_findings == ("Encode the income rule.",)
    assert material.outcome == {"failed_steps": ["encode_apply"]}
    bundle = build_evidence(material)
    assert "rulespec_root" not in bundle["target"]["source_attestation"]
    assert bundle["probes"]["gate"]["amendment_context"] is True


def test_workflow_bundle_without_candidate_still_builds(tmp_path):
    bundle_dir = _workflow_bundle(tmp_path)
    shutil.rmtree(
        bundle_dir / "x" / "generated" / "target" / "final-rejected-candidate"
    )
    [material] = load_workflow_bundle(bundle_dir)
    assert material.candidate is None
    assert material.source_text == OPERATIVE
    bundle = build_evidence(material)
    assert bundle["candidate"] is None
    assert bundle["probes"]["registry"]["status"] == "skipped"
    assert "no final rejected candidate was recorded" in bundle["run"]["notes"]


def test_workflow_bundle_refuses_symlinked_files(tmp_path):
    bundle_dir = _workflow_bundle(tmp_path)
    source = next((bundle_dir / "x").rglob("source.txt"))
    source.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    source.symlink_to(outside)
    with pytest.raises(EvidenceInputError):
        load_workflow_bundle(bundle_dir)


def test_non_bundle_input_is_refused(tmp_path):
    with pytest.raises(EvidenceInputError):
        load_any(tmp_path)


def _local_run(tmp_path: Path, *, final_success: bool = False) -> Path:
    run = tmp_path / "runs" / "main" / "r1" / "case"
    generated = run / "generated"
    generated.mkdir(parents=True)
    _output_root(generated)
    _candidate_dir(run, ["[complete-source-unit:tests] at line 4"])
    (run / "run.json").write_text(
        json.dumps({"citation": CITATION, "rc": 1, "encoder_commit": "e" * 40})
    )
    (run / "failed_attempts.jsonl").write_text(
        json.dumps({"error": "e1", "validation_issues": ["[x:y] at line 1", "z"]})
        + "\n"
        + json.dumps(
            {
                "error": "e2",
                "validation_issue_count": 5,
                "validation_issues": ["[x:y] at line 7"],
            }
        )
        + "\n"
    )
    connection = sqlite3.connect(run / "encodings.db")
    connection.execute(
        "create table encoding_runs (id text, timestamp text, citation text, "
        "outcome_json text, iterations_json text, axiom_encode_version text)"
    )
    connection.execute(
        "insert into encoding_runs values (?, ?, ?, ?, ?, ?)",
        (
            "run1",
            "2026-10-20",
            CITATION,
            json.dumps(
                {
                    "final_success": final_success,
                    "status": "standalone_failed",
                    "primary_error": "Generated RuleSpec failed CI validation",
                    "applied_files": [],
                }
            ),
            json.dumps([{"model": "gpt-6-luna"}, {"model": "gpt-6-sol"}]),
            "0.2.2230",
        ),
    )
    connection.commit()
    connection.close()
    return run


def test_local_run_loader_reads_attempt_history(tmp_path):
    [material] = load_local_run(_local_run(tmp_path), rulespec_ref="abc")
    assert material.input_kind == "local_run"
    assert material.run_id == "run1"
    assert material.encoder_version == "0.2.2230"
    assert material.rulespec_ref_source == "argument"
    assert [(a.attempt, a.model, a.issue_count) for a in material.attempts] == [
        (1, "gpt-6-luna", 2),
        (2, "gpt-6-sol", 5),
    ]
    assert material.outcome["status"] == "standalone_failed"
    assert "applied_files" not in material.outcome
    bundle = build_evidence(material)
    persistence = bundle["probes"]["persistence"]
    # issues.json says 3 attempts and attempt 2 recorded 1 of its 5 issues, so
    # the history is incomplete: nothing is called persistent.
    assert persistence["history_complete"] is False
    assert persistence["persistent_tags"] == []
    assert persistence["tags_in_all_recorded"] == ["x:y"]
    assert persistence["counts"] == {"changed": 1, "in_all_recorded": 1}


def test_local_run_that_passed_gives_no_material(tmp_path):
    assert load_local_run(_local_run(tmp_path, final_success=True)) == []


def test_local_encode_loader_from_outcome_and_workspace(tmp_path):
    output = _output_root(tmp_path / "out")
    material = load_local_encode(
        output,
        citation=CITATION,
        outcome={"status": "standalone_failed", "final_success": False, "x": 1},
        attempts=[{"error": "e", "validation_issues": ["a"], "model": "m"}],
        run_id="r",
    )
    assert material.candidate is None
    assert material.outcome == {"status": "standalone_failed", "final_success": False}
    assert material.attempts[0].model == "m"
    assert material.source_text == OPERATIVE


def test_iter_run_inputs_finds_bundles_and_runs(tmp_path):
    _workflow_bundle(tmp_path)
    _local_run(tmp_path)
    found = [p.relative_to(tmp_path).as_posix() for p in iter_run_inputs(tmp_path)]
    assert found == ["bundles/123-1", "runs/main/r1/case"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _run_cli(monkeypatch, argv: list[str]) -> int:
    from axiom_encode import cli

    monkeypatch.setattr(sys, "argv", ["axiom-encode", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value.code


def test_cli_single_and_batch(tmp_path, monkeypatch, capsys):
    bundle_dir = _workflow_bundle(tmp_path)
    _local_run(tmp_path)
    corpus = _corpus(tmp_path, [{"citation_path": CITATION, "body": OPERATIVE}])
    out = tmp_path / "one.json"
    code = _run_cli(
        monkeypatch,
        [
            "triage-evidence",
            str(bundle_dir),
            "--out",
            str(out),
            "--corpus-root",
            str(corpus),
        ],
    )
    assert code == 0
    single = json.loads(out.read_text())
    assert single["views"]["rulespec"]["reason"] == "no RuleSpec repository was given"
    # The target row is present; the source's 7 U.S.C. 2012 is not.
    assert single["probes"]["corpus"]["counts"] == {"absent": 1, "present": 1}

    summary_path = tmp_path / "summary.json"
    code = _run_cli(
        monkeypatch,
        [
            "triage-evidence",
            str(tmp_path),
            "--batch",
            "--summary",
            str(summary_path),
            "--no-concept-registry",
        ],
    )
    assert code == 0
    lines = (tmp_path / "evidence.jsonl").read_text().splitlines()
    names = [json.loads(line)["input"]["name"] for line in lines]
    assert names == ["bundles/123-1", "runs/main/r1/case"]
    summary = json.loads(summary_path.read_text())
    assert summary["inputs"] == 2
    assert summary["evidence_bundles"] == 2
    assert summary["probe_status"]["persistence"] == {"ok": 1, "skipped": 1}
    first = (tmp_path / "evidence.jsonl").read_text()
    _run_cli(
        monkeypatch,
        ["triage-evidence", str(tmp_path), "--batch", "--no-concept-registry"],
    )
    assert (tmp_path / "evidence.jsonl").read_text() == first


def test_cli_reports_unreadable_input(tmp_path, monkeypatch, capsys):
    code = _run_cli(monkeypatch, ["triage-evidence", str(tmp_path)])
    assert code == 1
    assert "not a workflow failure bundle" in capsys.readouterr().err


def test_dispatch_rejects_unknown_command():
    class Args:
        command = "other"

    with pytest.raises(ValueError):
        cli_commands.dispatch(Args())


# --------------------------------------------------------------------------
# End to end on a real recorded failure bundle
# --------------------------------------------------------------------------


def test_end_to_end_on_a_real_failure_bundle(tmp_path):
    # Workflow run 36635442145 (us/regulation/42/457/800): a subset of the
    # real bundle's files, copied byte for byte. #1891 labels it `selection`
    # (scope and applicability paragraphs only).
    bundle_dir = tmp_path / "a" / "36635442145"
    shutil.copytree(FIXTURES / "36635442145", bundle_dir)
    moved = tmp_path / "b" / "elsewhere"
    shutil.copytree(FIXTURES / "36635442145", moved)
    [material] = load_any(bundle_dir)
    assert material.citation == "us/regulation/42/457/800"
    assert material.candidate.module_path == "regulations/42-cfr/457/800.yaml"
    assert material.candidate.runner == "openai-gpt-6-sol"
    assert material.policy_prefix == "us"
    # 344.yaml in the runner tree is a copy of its context, not authored.
    assert material.run_outputs == ("us:regulations/42-cfr/457/800",)
    assert material.workspace_context == (
        "us:regulations/42-cfr/457/340",
        "us:regulations/42-cfr/457/344",
        "us:regulations/42-cfr/457/960",
    )
    first = build_evidence(material)
    second = build_evidence(load_any(moved)[0])
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["evidence_sha256"] == E2E_EVIDENCE_SHA256
    assert [issue["tag"] for issue in first["final_issues"]] == [
        "complete-source-unit:structure",
        "complete-source-unit:structure",
    ]
    assert [issue["gate"] for issue in first["final_issues"]] == ["ci", "ci"]
    assert first["run"]["rulespec_ref"] == "5f7a38c20a3ed4dd217d9c5505a73cddd9152d9c"
    assert first["run"]["attempt_count"] == 4
    assert first["probes"]["persistence"]["status"] == "skipped"
    assert first["probes"]["import"]["findings"] == []
    assert first["probes"]["gate"]["status"] == "ok"
    assert first["truncation"] == []
    assert first["size_bytes"] < 512 * 1024


# --------------------------------------------------------------------------
# Review round 1 regressions
# --------------------------------------------------------------------------


def test_persistence_is_not_claimed_over_a_missing_attempt():
    material = _material(
        attempts=(
            AttemptMaterial(attempt=1, issues=("x",)),
            AttemptMaterial(attempt=2, issues=None),
            AttemptMaterial(attempt=3, issues=("x",)),
        ),
        attempt_count=3,
    )
    result = probe_persistence(material)
    assert result["history_complete"] is False
    assert result["counts"] == {"in_all_recorded": 1}
    complete = probe_persistence(
        _material(
            attempts=(
                AttemptMaterial(attempt=1, issues=("x",), issue_count=1),
                AttemptMaterial(attempt=2, issues=("x",), issue_count=1),
            ),
            attempt_count=2,
        )
    )
    assert complete["history_complete"] is True
    assert complete["counts"] == {"persistent": 1}
    assert complete["persistent_tags"] == ["untagged"]


def test_symlinked_workspace_directory_is_refused(tmp_path):
    bundle_dir = _workflow_bundle(tmp_path)
    runner = bundle_dir / "x/generated/target/_eval_workspaces/openai-gpt-6-sol"
    citation_dir = runner / CITATION.replace("/", "-")
    outside = tmp_path / "outside"
    shutil.move(str(citation_dir), str(outside))
    citation_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(EvidenceInputError, match="symlink"):
        load_workflow_bundle(bundle_dir)


def test_output_write_never_follows_a_temporary_file_link(tmp_path):
    target = tmp_path / "victim.txt"
    target.write_text("keep")
    out = tmp_path / "evidence.jsonl"
    (tmp_path / "evidence.jsonl.tmp").symlink_to(target)
    cli_commands._write(out, "new\n")
    assert out.read_text() == "new\n"
    assert target.read_text() == "keep"
    assert not out.is_symlink()


def test_git_reads_are_local_only():
    from axiom_encode.triage.views import _git_env

    env = _git_env()
    assert env["GIT_NO_LAZY_FETCH"] == "1"
    assert env["GIT_ALLOW_PROTOCOL"] == "none"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_free_form_fields_are_bounded_and_the_byte_bound_is_enforced():
    material = _material(
        attempts=(AttemptMaterial(attempt=1, error="e" * 600_000, issues=("a",)),),
        outcome={"primary_error": "p" * 600_000},
    )
    bundle = build_evidence(material)
    assert bundle["size_bytes"] <= 512 * 1024
    fields = {item["field"] for item in bundle["truncation"]}
    assert {"attempts[1].error", "run.outcome"} <= fields
    assert bundle["attempts"][0]["error"]["truncated"] is True
    with pytest.raises(EvidenceInputError, match="tightest"):
        build_evidence(_material(), max_bytes=1_000)


def test_shrink_tier_does_not_depend_on_the_input_name():
    material = _material(source_text="s" * 30_000)
    probe = build_evidence(material)
    # One byte short of fitting at the default limits.
    bound = probe["size_bytes"] + _RESERVED_BYTES - 1
    short = build_evidence(replace(material, input_name="a"), max_bytes=bound)
    long = build_evidence(replace(material, input_name="n" * 255), max_bytes=bound)
    assert short["evidence_sha256"] == long["evidence_sha256"]
    assert short["target"]["source"]["text"] == long["target"]["source"]["text"]


def test_another_version_never_stands_in_for_the_attested_row(tmp_path):
    corpus = LocalCorpusIndex(
        _corpus(
            tmp_path,
            [
                {
                    "citation_path": CITATION,
                    "body": "x",
                    "heading": "(d) Income.",
                    "metadata": {"status": "repealed"},
                }
            ],
            version="2025-other",
        )
    )
    result = probe_corpus(_material(), corpus=corpus)
    assert _by_subject(result)[CITATION]["status"] == "absent_at_attested_version"
    gate = probe_gate(_material(), corpus=corpus)
    assert gate["label_count"] == 0
    assert "repealed_target" not in {f["kind"] for f in gate["findings"]}


def test_registry_reads_flow_style_imports(tmp_path):
    view = _view(tmp_path)
    content = (
        "format: rulespec/v1\n"
        'imports: ["us:statutes/7/2012/a#household_definition", '
        '"us:statutes/7/9999#missing"]\n'
        "rules: []\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, snapshot_factory=_factory(view))
    statuses = {f["subject"]: f["status"] for f in result["findings"]}
    assert statuses == {
        "us:statutes/7/2012/a#household_definition": "exported",
        "us:statutes/7/9999#missing": "module_missing",
    }


def test_workspace_lookup_uses_the_encoder_slug(tmp_path):
    output = tmp_path / "out"
    runner = "openai-gpt-6-sol"
    _workspace(output, runner=runner, citation="26 USC 32", source="text")
    material = load_local_encode(output, citation="26 USC 32")
    assert material.source_text == "text"
    assert material.policy_prefix == "us"


def test_refs_point_at_the_matching_occurrence():
    source = "In 2012 the rule changed.\nSee 7 U.S.C. 2012 for households.\n"
    corpus_result = probe_corpus(_material(source_text=source), corpus=None)
    assert corpus_result["status"] == "skipped"
    from axiom_encode.triage.probes import _corpus_subjects

    subjects = {
        path: ref
        for path, _origin, ref in _corpus_subjects(_material(source_text=source))
    }
    assert subjects["us/statute/7/2012"] == "source:L2"
    content = (
        "format: rulespec/v1\nimports:\n  - us:statutes/7/2012/a#household_definition\n"
        "module:\n  deferred_outputs:\n  - output: us:statutes/7/2014/d#x\n"
        "    blocked_by:\n    - us:statutes/7/2012/a#household_definition\n"
        "rules: []\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    refs = {(f["kind"], f["refs"][0]) for f in probe_registry(material)["findings"]}
    assert refs == {("import", "candidate:L3"), ("deferral_blocked_by", "candidate:L8")}


def test_local_run_ignores_links_outside_what_it_reads(tmp_path):
    run = _local_run(tmp_path)
    (run / "tmp").mkdir()
    (run / "tmp" / "installation_id").symlink_to(tmp_path / "anything")
    assert len(load_local_run(run)) == 1
    (run / "generated" / "link.yaml").symlink_to(tmp_path / "anything")
    with pytest.raises(EvidenceInputError, match="symlink"):
        load_local_run(run)


# --------------------------------------------------------------------------
# Review round 2 regressions
# --------------------------------------------------------------------------


def test_local_run_with_more_iterations_than_recorded_attempts_is_incomplete(
    tmp_path,
):
    run = _local_run(tmp_path)
    (run / "final-rejected-candidate" / "issues.json").unlink()
    (run / "failed_attempts.jsonl").write_text(
        json.dumps({"validation_issues": ["x"], "validation_issue_count": 1})
        + "\n"
        + json.dumps({"validation_issues": ["x"], "validation_issue_count": 1})
        + "\n"
    )
    connection = sqlite3.connect(run / "encodings.db")
    connection.execute(
        "update encoding_runs set iterations_json = ?",
        (json.dumps([{}, {}, {}]),),
    )
    connection.commit()
    connection.close()
    [material] = load_local_run(run)
    assert material.attempt_count == 3
    result = probe_persistence(material)
    assert result["history_complete"] is False
    assert result["counts"] == {"in_all_recorded": 1}


def test_import_comments_and_alias_mappings_give_only_real_targets(tmp_path):
    view = _view(tmp_path)
    content = (
        "format: rulespec/v1\n"
        "imports:\n"
        "  - us:statutes/7/2012/a#household_definition # household symbol\n"
        "  - alias: us:statutes/7/2015/b#other_symbol\n"
        "rules: []\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, snapshot_factory=_factory(view))
    assert {(f["subject"], f["status"], f["refs"][0]) for f in result["findings"]} == {
        ("us:statutes/7/2012/a#household_definition", "exported", "candidate:L3"),
        ("us:statutes/7/2015/b#other_symbol", "exported", "candidate:L4"),
    }


def test_flow_mapping_citations_are_read_as_scalars():
    from axiom_encode.triage.probes import _corpus_subjects

    content = (
        "format: rulespec/v1\nrules:\n- name: x\n  metadata:\n"
        "    proof: {source: {corpus_citation_path: us/statute/7/2099}}\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    subjects = {path: ref for path, _origin, ref in _corpus_subjects(material)}
    assert subjects["us/statute/7/2099"] == "candidate:L5"


def test_blocker_refs_use_their_own_node_line():
    content = (
        "format: rulespec/v1\nmodule:\n  deferred_outputs:\n"
        "  - output: us:statutes/7/2014/d#a\n"
        "    blocked_by:\n    - us:statutes/7/2012/a#one\n"
        "    reason: see us:statutes/7/2012/a#two\n"
        "  - output: us:statutes/7/2014/d#b\n"
        "    blocked_by:\n    - us:statutes/7/2012/a#two\n"
        "    reason: r\n"
        "rules: []\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    refs = {f["subject"]: f["refs"] for f in probe_registry(material)["findings"]}
    assert refs == {
        "us:statutes/7/2012/a#one": ["candidate:L6"],
        "us:statutes/7/2012/a#two": ["candidate:L10"],
    }


def test_written_output_respects_the_byte_bound(tmp_path, monkeypatch):
    material = _material(source_text="word " * 5_000)
    bound = 12_000
    bundle = build_evidence(replace(material, input_name="n" * 600), max_bytes=bound)
    bundle["input"]["name"] = "n" * 600  # the CLI bounds it before writing
    from axiom_encode.triage.evidence import bounded_input_name

    bundle["input"]["name"] = bounded_input_name(bundle["input"]["name"])
    assert len((canonical_json(bundle) + "\n").encode()) <= bound


def test_a_ref_to_a_cut_last_line_does_not_resolve():
    material = _material(source_text="first line\nsecond line is long\n")
    bundle = build_evidence(material, limits=EvidenceLimits(source_chars=15))
    assert resolve_ref(bundle, "source:L1")
    assert not resolve_ref(bundle, "source:L2")


def test_long_issues_that_differ_late_are_not_merged():
    prefix = "[a:b] " + "x" * 400
    material = _material(
        attempts=(
            AttemptMaterial(attempt=1, issues=(prefix + " one",)),
            AttemptMaterial(attempt=2, issues=(prefix + " two",)),
        ),
        attempt_count=2,
    )
    result = probe_persistence(material)
    assert result["counts"] == {"changed": 2}
    assert len({f["signature_sha256"] for f in result["findings"]}) == 2


# --------------------------------------------------------------------------
# Review round 3 regressions
# --------------------------------------------------------------------------


def test_gate_label_order_does_not_depend_on_the_hash_seed():
    code = (
        "from axiom_encode.harness.pre_encode_gate import check_no_operative_text\n"
        "print(check_no_operative_text('abab', ('aba', 'bab'), 'us/x/a'))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3", "4", "5", "6")
    }
    assert len(outputs) == 1


class _UnreadableView(GitRefRulespecView):
    def __init__(self, base, unreadable):
        self.__dict__.update(base.__dict__)
        self._unreadable = unreadable

    def read(self, relative):
        from axiom_encode.triage.views import UnreadableRulespecFile

        if relative in self._unreadable:
            raise UnreadableRulespecFile(relative)
        return super().read(relative)


def test_unreadable_modules_are_never_reported_absent(tmp_path):
    view = _UnreadableView(
        _view(tmp_path),
        {"us/statutes/7/2012/a.yaml", "us/statutes/7/2014/d.yaml"},
    )
    registry = probe_registry(_material(), snapshot_factory=_factory(view))
    statuses = {f["subject"]: f["status"] for f in registry["findings"]}
    assert statuses["us:statutes/7/2012/a#household_definition"] == (
        "module_unreadable"
    )
    gate = probe_gate(_material(source_text=""), snapshot_factory=_factory(view))
    assert "existing_module" in gate["checks_not_run"]
    assert "existing_module" not in {f["kind"] for f in gate["findings"]}


def test_git_view_ignores_replace_objects(tmp_path):
    repo, sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    original = _git(repo, "rev-parse", f"{sha}:us/statutes/7/2012/a.yaml")
    other = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
        input="format: rulespec/v1\nrules: []\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _git(repo, "replace", original, other)
    data = GitRefRulespecView(repo, sha).read("us/statutes/7/2012/a.yaml")
    assert b"household_definition" in data


@pytest.mark.parametrize("parent", ["with#hash", "with?query"])
def test_local_run_db_is_read_under_any_directory_name(tmp_path, parent):
    run = _local_run(tmp_path / parent)
    [material] = load_local_run(run)
    assert material.run_id == "run1"
    assert material.outcome["status"] == "standalone_failed"


def test_symlinked_input_container_is_refused(tmp_path):
    bundle_dir = _workflow_bundle(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(bundle_dir, target_is_directory=True)
    with pytest.raises(EvidenceInputError, match="symlink"):
        load_any(link)


def test_registered_input_ref_points_at_the_formula():
    content = (
        "format: rulespec/v1\n"
        "rules:\n"
        "- name: x\n"
        "  description: gross_income is described here\n"
        "  versions:\n"
        "  - formula: gross_income + 1\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, registry=_registry())
    [finding] = [f for f in result["findings"] if f["kind"] == "input"]
    assert finding["refs"] == ["candidate:L6"]


def test_control_character_names_stay_within_the_bound():
    material = replace(_material(source_text="word " * 3_000), input_name="\x01" * 600)
    bound = 16_000
    bundle = build_evidence(material, max_bytes=bound)
    assert len((canonical_json(bundle) + "\n").encode()) <= bound


def test_clipped_attempt_signatures_are_recorded():
    material = _material(
        attempts=(
            AttemptMaterial(attempt=1, issues=("y" * 500,)),
            AttemptMaterial(attempt=2, issues=("short",)),
        )
    )
    bundle = build_evidence(material)
    [entry] = [
        item
        for item in bundle["truncation"]
        if item["field"] == "attempts[1].issue_signatures"
    ]
    assert entry["clipped_items"] == 1
    assert entry["max_original_chars"] == 500


# --------------------------------------------------------------------------
# Review round 4 regressions
# --------------------------------------------------------------------------


def test_recursive_yaml_alias_is_walked_once():
    from axiom_encode.triage.probes import yaml_scalars

    content = 'imports: &loop {target: "us:statutes/7/2012/a#x", recur: *loop}\n'
    scalars = yaml_scalars(content)
    assert [item.value for item in scalars] == ["us:statutes/7/2012/a#x"]


def test_checkout_head_is_only_its_own_repository(tmp_path):
    from axiom_encode.triage.views import git_head

    repo, sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    assert git_head(repo) == sha
    assert git_head(repo / "us") is None
    assert CheckoutRulespecView(repo / "us").identity() == {
        "kind": "checkout",
        "head": None,
    }


def test_unreadable_corpus_rows_make_absence_unknown(tmp_path):
    root = _corpus(tmp_path, [{"citation_path": "us/statute/7/1", "body": "x"}])
    bucket = root / "data/corpus/provisions/us/statute"
    (bucket / "broken.jsonl").write_text('{"citation_path": "us/statute/7/2014/d", \n')
    result = probe_corpus(_material(), corpus=LocalCorpusIndex(root))
    assert _by_subject(result)[CITATION]["status"] == "unavailable"


def test_replace_flags_apply_to_the_target_lane_only(tmp_path):
    bundle_dir = _workflow_bundle(tmp_path)
    root = bundle_dir / "x"
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["replace_rulespec_path"] = MODULE
    (root / "metadata.json").write_text(json.dumps(metadata))
    shutil.copytree(root / "generated" / "target", root / "generated" / "dependent")
    by_lane = {m.lane: m for m in load_workflow_bundle(bundle_dir)}
    assert by_lane["target"].replacement_target is True
    assert by_lane["dependent"].replacement_target is False


def test_tracked_symlink_module_is_unreadable_not_present(tmp_path):
    repo, _sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    link = repo / "us/statutes/7/2015/b.yaml"
    link.unlink()
    link.symlink_to("../2012/a.yaml")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "link")
    view = GitRefRulespecView(repo, _git(repo, "rev-parse", "HEAD"))
    result = probe_registry(_material(), snapshot_factory=_factory(view))
    statuses = {f["subject"]: f["status"] for f in result["findings"]}
    assert statuses["us:statutes/7/2015/b#missing_symbol"] == "module_unreadable"


def test_malformed_attempt_record_keeps_its_place(tmp_path):
    run = _local_run(tmp_path)
    (run / "final-rejected-candidate" / "issues.json").unlink()
    (run / "failed_attempts.jsonl").write_text(
        json.dumps({"validation_issues": ["x"]})
        + "\nnot json\n"
        + json.dumps({"validation_issues": ["x"]})
        + "\n"
    )
    connection = sqlite3.connect(run / "encodings.db")
    connection.execute("update encoding_runs set iterations_json = '[]'")
    connection.commit()
    connection.close()
    [material] = load_local_run(run)
    assert [a.issues for a in material.attempts] == [("x",), None, ("x",)]
    result = probe_persistence(material)
    assert result["history_complete"] is False
    assert result["counts"] == {"in_all_recorded": 1}


@pytest.mark.parametrize("indicator", ["|-", ">-"])
def test_input_refs_in_block_formulas_point_at_the_expression(indicator):
    content = (
        "format: rulespec/v1\n"
        "rules:\n"
        "- name: x\n"
        "  versions:\n"
        f"  - formula: {indicator}\n"
        "      1 +\n"
        "      gross_income\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, registry=_registry())
    [finding] = [f for f in result["findings"] if f["kind"] == "input"]
    assert finding["refs"] == ["candidate:L7"]


def test_invalid_utf8_import_is_unreadable(tmp_path):
    files = dict(RULESPEC_FILES)
    repo, _sha = _rulespec_repo(tmp_path, files)
    (repo / "us/statutes/7/2012/a.yaml").write_bytes(b"rules:\n- name: \xff\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bad bytes")
    view = GitRefRulespecView(repo, _git(repo, "rev-parse", "HEAD"))
    result = probe_registry(_material(), snapshot_factory=_factory(view))
    statuses = {f["subject"]: f["status"] for f in result["findings"]}
    assert statuses["us:statutes/7/2012/a#household_definition"] == (
        "module_unreadable"
    )


def test_unreadable_fallback_does_not_mask_the_direct_module(tmp_path):
    view = _UnreadableView(
        _view(tmp_path), {"us/statutes/7/2014/statutes/7/2012/a.yaml"}
    )
    content = "format: rulespec/v1\nimports:\n  - statutes/7/2012/a#household_definition\nrules: []\n"
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, snapshot_factory=_factory(view))
    assert [f["status"] for f in result["findings"]] == ["exported"]


# --------------------------------------------------------------------------
# Review round 5 regressions
# --------------------------------------------------------------------------


def test_target_preflight_lane_keeps_the_target_citation(tmp_path):
    bundle_dir = _workflow_bundle(tmp_path)
    root = bundle_dir / "x"
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["dependent_citation"] = "us/statute/7/9999"
    (root / "metadata.json").write_text(json.dumps(metadata))
    target = root / "generated" / "target"
    shutil.rmtree(target / "final-rejected-candidate")
    target.rename(root / "generated" / "target-preflight")
    [material] = load_workflow_bundle(bundle_dir)
    assert (material.lane, material.citation) == ("target-preflight", CITATION)
    assert material.source_text == OPERATIVE


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        '{"citation_path": null}',
        '{"\\u0063itation_path": "us/statute/7/2014/d", "body": "x"}',
    ],
)
def test_every_corpus_line_is_parsed_or_marks_the_bucket(tmp_path, line):
    root = _corpus(tmp_path, [{"citation_path": "us/statute/7/1", "body": "x"}])
    bucket = root / "data/corpus/provisions/us/statute"
    (bucket / "extra.jsonl").write_text(line + "\n")
    status = _by_subject(probe_corpus(_material(), corpus=LocalCorpusIndex(root)))[
        CITATION
    ]["status"]
    # The escaped key is a real row (at another version); the others leave
    # the bucket incomplete.
    assert status in {"unavailable", "absent_at_attested_version"}
    assert status != "absent"


def test_deeply_nested_yaml_does_not_abort_evidence(tmp_path):
    deep = "format: rulespec/v1\nimports: " + "[" * 1000 + "]" * 1000 + "\n"
    material = _material(candidate=CandidateMaterial(module_path=MODULE, rulespec=deep))
    bundle = build_evidence(material)
    assert bundle["probes"]["registry"]["status"] == "ok"


def test_checkout_view_reports_symlinks_as_unreadable(tmp_path):
    repo, _sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    link = repo / "us/statutes/7/2015/b.yaml"
    link.unlink()
    link.symlink_to("../2012/a.yaml")
    broken = repo / "us/statutes/7/9999.yaml"
    broken.symlink_to("missing.yaml")
    view = CheckoutRulespecView(repo)
    result = probe_registry(_material(), snapshot_factory=_factory(view))
    statuses = {f["subject"]: f["status"] for f in result["findings"]}
    assert statuses["us:statutes/7/2015/b#missing_symbol"] == "module_unreadable"
    assert statuses["us:statutes/7/9999#anything"] == "module_unreadable"


def test_reused_yaml_anchor_is_read_at_each_path():
    from axiom_encode.triage.probes import yaml_scalars

    content = (
        "module:\n  deferred_outputs:\n  - output: x\n"
        "    blocked_by: &shared\n    - us:statutes/7/2012/a#household_definition\n"
        "imports: *shared\n"
    )
    paths = [item.path for item in yaml_scalars(content)]
    assert ("module", "deferred_outputs", 0, "blocked_by", 0) in paths
    assert ("imports", 0) in paths


def test_module_only_import_with_invalid_utf8_is_unreadable(tmp_path):
    repo, _sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    (repo / "us/statutes/7/2012/a.yaml").write_bytes(b"rules: \xff\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bad")
    view = GitRefRulespecView(repo, _git(repo, "rev-parse", "HEAD"))
    content = "format: rulespec/v1\nimports:\n  - us:statutes/7/2012/a\nrules: []\n"
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, snapshot_factory=_factory(view))
    assert [f["status"] for f in result["findings"]] == ["module_unreadable"]


def test_input_ref_skips_string_literals():
    content = (
        "format: rulespec/v1\n"
        "rules:\n"
        "- name: x\n"
        "  versions:\n"
        "  - formula: |-\n"
        '      label("gross_income") +\n'
        "      gross_income\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, registry=_registry())
    [finding] = [f for f in result["findings"] if f["kind"] == "input"]
    assert finding["refs"] == ["candidate:L7"]


# --------------------------------------------------------------------------
# Review round 6 regressions
# --------------------------------------------------------------------------


def test_routed_amendment_finds_the_principal_workspace(tmp_path):
    # The encoder's layout: the workspace is under the principal's slug, its
    # attestation requests the principal, and the provision metadata names
    # the original amendment request.
    output = tmp_path / "out"
    principal = "us/statute/7/2014"
    workspace = _workspace(
        output, runner="openai-gpt-6-sol", citation=principal, source="s"
    )
    metadata = json.loads((workspace / "source-metadata.json").read_text())
    metadata["source_attestation"]["requested_corpus_citation_path"] = principal
    (workspace / "source-metadata.json").write_text(json.dumps(metadata))
    (workspace / "provision-metadata.txt").write_text(
        json.dumps({"title": "7", "amendment_request": "us/statute/pl-9/sec-1"})
    )
    material = load_local_encode(output, citation="us/statute/pl-9/sec-1")
    assert material.source_text == "s"
    assert material.citation == principal
    assert material.amendment_request == "us/statute/pl-9/sec-1"
    bundle = build_evidence(material)
    assert bundle["target"]["amendment_context"]["request"] == "us/statute/pl-9/sec-1"
    assert bundle["probes"]["gate"]["amendment_context"] is True


def test_invalid_yaml_date_does_not_abort_evidence():
    content = (
        "format: rulespec/v1\n"
        "imports:\n  - us:statutes/7/2012/a#household_definition\n"
        "rules:\n- name: x\n  versions:\n  - effective_from: 2026-99-99\n"
        "    formula: gross_income\n"
    )
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    bundle = build_evidence(material, registry=_registry())
    assert bundle["probes"]["registry"]["status"] == "ok"
    assert [f["kind"] for f in bundle["probes"]["registry"]["findings"]] == ["import"]


def test_symlinked_corpus_bucket_is_unavailable(tmp_path):
    root = _corpus(tmp_path, [{"citation_path": "us-ok/statute/1", "body": "x"}])
    real = tmp_path / "elsewhere"
    real.mkdir()
    (root / "data/corpus/provisions/us").mkdir(parents=True, exist_ok=True)
    (root / "data/corpus/provisions/us/statute").symlink_to(
        real, target_is_directory=True
    )
    result = probe_corpus(_material(), corpus=LocalCorpusIndex(root))
    assert _by_subject(result)[CITATION]["status"] == "unavailable"


@pytest.mark.parametrize("kind", ["git", "checkout"])
def test_linked_only_descendant_is_unreadable_not_absent(tmp_path, kind):
    from axiom_encode.triage.probes import cited_module_status

    repo, _sha = _rulespec_repo(tmp_path, RULESPEC_FILES)
    target = repo / "us/statutes/7/2020/a.yaml"
    target.parent.mkdir(parents=True)
    target.symlink_to("../2012/a.yaml")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "link")
    view = (
        GitRefRulespecView(repo, _git(repo, "rev-parse", "HEAD"))
        if kind == "git"
        else CheckoutRulespecView(repo)
    )
    with RulespecSnapshot(view, "us") as snapshot:
        assert cited_module_status(snapshot, "us/statute/7/2020") == "unreadable"
    with RulespecSnapshot(view, "us") as snapshot:
        assert cited_module_status(snapshot, "us/statute/7/2012") == "present"


def test_provenance_uses_the_module_the_resolver_chose(tmp_path):
    files = {
        **RULESPEC_FILES,
        "us/statutes/7/2014/statutes/7/5555.yaml": (
            "format: rulespec/v1\nrules:\n- name: y\n  kind: derived\n"
        ),
    }
    repo, sha = _rulespec_repo(tmp_path, files)
    view = GitRefRulespecView(repo, sha)
    content = "format: rulespec/v1\nimports:\n  - statutes/7/5555#y\nrules: []\n"
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content),
        workspace_context=("us:statutes/7/2014/statutes/7/5555",),
    )
    [finding] = probe_registry(material, snapshot_factory=_factory(view))["findings"]
    assert finding["status"] == "exported"
    assert finding["module"] == "us/statutes/7/2014/statutes/7/5555.yaml"
    assert finding["in_workspace_context"] is True


# --------------------------------------------------------------------------
# Review round 7 regressions
# --------------------------------------------------------------------------


def test_nested_compile_error_is_placed_in_the_inner_module():
    issue = (
        "Axiom rules engine compile failed: failed to load RuleSpec module "
        "`<rulespec-validation-root>/rulespec-us/us/statutes/7/2014/d.yaml`: "
        "failed to load RuleSpec module "
        "`<rulespec-validation-root>/rulespec-us/us/statutes/26/32.yaml`: bad"
    )
    result = probe_import(_material(final_issues=(issue,)))
    assert [(f["subject"], f["status"]) for f in result["findings"]] == [
        ("us:statutes/26/32", "imported_not_authored")
    ]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("got decimal 0.3333333333333333", "got decimal 0.6666666666666666"),
        ("got 123456789012345", "got 123456789012346"),
    ],
)
def test_numeric_results_are_not_normalized_away(first, second):
    assert issue_signature(first) != issue_signature(second)
    assert issue_signature("sha 0123456789abcdef") == issue_signature(
        "sha fedcba9876543210"
    )


@pytest.mark.parametrize("dangling", [True, False])
def test_symlinked_corpus_jurisdiction_is_unavailable(tmp_path, dangling):
    root = _corpus(tmp_path, [{"citation_path": "us-ok/statute/1", "body": "x"}])
    target = tmp_path / ("missing" if dangling else "populated")
    if not dangling:
        (target / "statute").mkdir(parents=True)
    (root / "data/corpus/provisions/us").symlink_to(target, target_is_directory=True)
    result = probe_corpus(_material(), corpus=LocalCorpusIndex(root))
    assert _by_subject(result)[CITATION]["status"] == "unavailable"


# --------------------------------------------------------------------------
# Review round 8 regressions
# --------------------------------------------------------------------------


def test_jurisdiction_prefixed_candidate_keys_match_compile_paths(tmp_path):
    candidate = CandidateMaterial(module_path="us/" + MODULE, rulespec=CANDIDATE)
    material = _material(
        candidate=candidate, final_issues=(_compile_issue("us/" + MODULE),)
    )
    assert material.target_key == "us:statutes/7/2014/d"
    assert [f["status"] for f in probe_import(material)["findings"]] == ["own_module"]
    view = _view(tmp_path)
    statuses = {
        f["subject"]: f["status"]
        for f in probe_registry(material, snapshot_factory=_factory(view))["findings"]
    }
    assert statuses["us:statutes/7/2012/a#household_definition"] == "exported"


def test_citation_normalization_is_not_an_amendment_route(tmp_path):
    output = tmp_path / "out"
    _workspace(output, runner="openai-gpt-6-sol", citation="26 USC 32", source="s")
    workspace = output / "_eval_workspaces/openai-gpt-6-sol/26-USC-32/workspace"
    metadata = json.loads((workspace / "source-metadata.json").read_text())
    metadata["source_attestation"]["requested_corpus_citation_path"] = (
        "us/statute/26/32"
    )
    (workspace / "source-metadata.json").write_text(json.dumps(metadata))
    material = load_local_encode(output, citation="26 USC 32")
    assert material.citation == "26 USC 32"
    assert material.amendment_request is None
    assert material.source_text == "s"


def test_import_without_a_module_path_is_unresolvable(tmp_path):
    view = _view(tmp_path)
    content = 'format: rulespec/v1\nimports:\n  - "#missing"\nrules: []\n'
    material = _material(
        candidate=CandidateMaterial(module_path=MODULE, rulespec=content)
    )
    result = probe_registry(material, snapshot_factory=_factory(view))
    assert [f["status"] for f in result["findings"]] == ["unresolvable_target"]
