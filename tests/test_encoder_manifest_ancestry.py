"""Signed producers are admitted by Git ancestry and main first-parent history."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode.harness.evals import resolve_corpus_source_unit
from tests.environment_assertions import hides_environment
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
    install_test_eval_evidence_keys,
)
from tests.signing_broker_fixtures import SigningBrokerFixture
from tests.test_cli import (
    TEST_APPLY_SIGNING_BROKER,
    TEST_VALIDATION_WAIVER_SHA256,
    _bind_test_corpus_release,
)
from tests.test_cli import (
    TestGuardGenerated as _GuardFixtures,
)


def _git(repo: Path, *args: str, input: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.com",
            "GIT_COMMITTER_NAME": "Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.com",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        },
    ).stdout.strip()


def _write_git_object(repo: Path, kind: str, payload: bytes) -> str:
    """Write Git's actual loose-object format without spawning a writer process."""

    framed = f"{kind} {len(payload)}\0".encode() + payload
    digest = hashlib.sha1(framed).hexdigest()
    object_path = repo / ".git/objects" / digest[:2] / digest[2:]
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(zlib.compress(framed))
    return digest


def _write_git_ref(repo: Path, ref: str, commit: str) -> None:
    path = repo / ".git" / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(commit + "\n")


@dataclass
class _Graph:
    """Construct Git objects while retaining an independent graph oracle."""

    root: Path
    commits: list[str] = field(default_factory=list)
    parents: list[tuple[int, ...]] = field(default_factory=list)
    main: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root.mkdir()
        git_dir = self.root / ".git"
        (git_dir / "refs/heads").mkdir(parents=True)
        (git_dir / "objects").mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        (git_dir / "config").write_text(
            "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
        )
        self.tree = _write_git_object(self.root, "tree", b"")

    def add(self, *parents: int, main: bool = False) -> int:
        index = len(self.commits)
        parent_lines = "".join(f"parent {self.commits[p]}\n" for p in parents)
        commit = _write_git_object(
            self.root,
            "commit",
            (
                f"tree {self.tree}\n{parent_lines}"
                "author Fixture <fixture@example.com> 1767225600 +0000\n"
                "committer Fixture <fixture@example.com> 1767225600 +0000\n"
                f"\nfixture commit {index}\n"
            ).encode(),
        )
        self.commits.append(commit)
        self.parents.append(parents)
        if main:
            self.main.append(index)
            _write_git_ref(self.root, "refs/heads/main", commit)
        return index

    def is_ancestor(self, pin: int, producer: int) -> bool:
        pending = [producer]
        seen = set()
        while pending:
            current = pending.pop()
            if current == pin:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(self.parents[current])
        return False


def _identity(commit: str, version: str = "0.2.2161") -> dict[str, str]:
    return {
        "repository": cli.APPLIED_ENCODING_OFFICIAL_REPOSITORY,
        "commit": commit,
        "version": version,
    }


def _expected(graph: _Graph, pin: int = 1, version: str = "0.2.2161"):
    return cli._GuardEncoderIdentity(
        _identity(graph.commits[pin], version), history_checkout=graph.root
    )


@pytest.fixture
def graph(tmp_path: Path) -> _Graph:
    graph = _Graph(tmp_path / "encoder-history")
    base = graph.add(main=True)
    pin = graph.add(base, main=True)
    second_parent = graph.add(pin)
    merged = graph.add(pin, second_parent, main=True)
    head = graph.add(merged, main=True)
    feature = graph.add(head)
    _git(graph.root, "update-ref", "refs/heads/feature", graph.commits[feature])
    return graph


@pytest.mark.parametrize(
    ("producer", "accepted"),
    [(0, False), (1, True), (2, False), (3, True), (4, True), (5, False)],
    ids=["older-same-version", "pin", "second-parent", "merge", "main-head", "feature"],
)
def test_same_version_is_neither_ordering_nor_main_membership(
    graph, producer, accepted
):
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[producer]), _expected(graph)
    )
    assert (issue is None) is accepted


def test_exact_identity_needs_no_history_and_ignores_resolution_mode():
    producer = {**_identity("a" * 40), "identity_source": "git"}
    expected = {**_identity("a" * 40), "identity_source": "github-sha"}
    assert cli._encoder_producer_admission_issue(producer, expected) is None


def test_exact_commit_cannot_claim_a_different_version(graph):
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[1], "0.2.9999"), _expected(graph)
    )
    assert issue is not None


@pytest.mark.parametrize("version", ["0.2.1", "0.2.2161", "0.2.9999"])
def test_main_descendant_admission_does_not_order_versions(graph, version):
    assert (
        cli._encoder_producer_admission_issue(
            _identity(graph.commits[4], version), _expected(graph)
        )
        is None
    )


def test_first_parent_membership_uses_main_even_when_head_and_default_are_feature(
    graph,
):
    _git(graph.root, "symbolic-ref", "HEAD", "refs/heads/feature")
    _git(graph.root, "update-ref", "refs/remotes/origin/feature", graph.commits[5])
    _git(
        graph.root,
        "symbolic-ref",
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/feature",
    )
    assert cli._encoder_producer_admission_issue(
        _identity(graph.commits[5]), _expected(graph)
    )
    assert (
        cli._encoder_producer_admission_issue(
            _identity(graph.commits[4]), _expected(graph)
        )
        is None
    )


def test_second_parent_reachable_from_main_is_still_rejected(graph):
    assert graph.is_ancestor(1, 2)
    assert (
        graph.commits[2] in _git(graph.root, "rev-list", "refs/heads/main").splitlines()
    )
    assert cli._encoder_producer_admission_issue(
        _identity(graph.commits[2]), _expected(graph)
    )


def test_fetched_main_admits_head_when_local_main_stays_at_pin(graph):
    (graph.root / ".git/FETCH_HEAD").write_text(
        f"{graph.commits[4]}\t\tbranch 'main' of https://github.com/TheAxiomFoundation/axiom-encode\n"
    )
    _git(graph.root, "update-ref", "refs/heads/main", graph.commits[1])
    assert (
        cli._encoder_producer_admission_issue(
            _identity(graph.commits[4]), _expected(graph)
        )
        is None
    )


def test_explicit_main_fetch_is_authoritative_over_ahead_local_main(graph):
    (graph.root / ".git/FETCH_HEAD").write_text(
        f"{graph.commits[1]}\t\tbranch 'main' of https://github.com/TheAxiomFoundation/axiom-encode\n"
    )
    assert cli._encoder_producer_admission_issue(
        _identity(graph.commits[4]), _expected(graph)
    )


def test_origin_main_without_canonical_main_cannot_authorize_producers(graph):
    _write_git_ref(graph.root, "refs/remotes/origin/main", graph.commits[4])
    (graph.root / ".git/refs/heads/main").unlink()
    assert cli._encoder_producer_admission_issue(
        _identity(graph.commits[4]), _expected(graph)
    )


@pytest.mark.parametrize("missing", ["pin", "producer", "main", "checkout"])
def test_unavailable_history_fails_closed(graph, tmp_path, missing):
    expected = _expected(graph)
    producer = _identity(graph.commits[4])
    if missing == "pin":
        expected["commit"] = "f" * 40
    elif missing == "producer":
        producer["commit"] = "f" * 40
    elif missing == "main":
        _git(graph.root, "update-ref", "-d", "refs/heads/main")
    else:
        absent = tmp_path / "unavailable-history"
        absent.mkdir()
        expected = cli._GuardEncoderIdentity(dict(expected), history_checkout=absent)
    issue = cli._encoder_producer_admission_issue(producer, expected)
    assert isinstance(issue, str) and issue


def test_differing_producer_requires_trusted_history_context(graph):
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[4]), _identity(graph.commits[1])
    )
    assert isinstance(issue, str) and issue


def test_shallow_history_cannot_prove_descendant_admission(graph):
    (graph.root / ".git/shallow").write_text(graph.commits[0] + "\n")
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[4]), _expected(graph)
    )
    assert isinstance(issue, str) and "shallow" in issue.lower()


@pytest.mark.parametrize("failure", ["exit-128", "missing-git"])
def test_git_errors_are_rejections(graph, monkeypatch, failure):
    @hides_environment
    def unavailable(command, **kwargs):
        if failure == "missing-git":
            raise OSError("Git executable unavailable")
        return subprocess.CompletedProcess(
            command, 128, "", "fatal: history unavailable"
        )

    monkeypatch.setattr(cli.subprocess, "run", unavailable)
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[4]), _expected(graph)
    )
    assert isinstance(issue, str) and issue


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    steps=st.lists(
        st.tuples(st.booleans(), st.integers(1, 2), st.integers(0, 10)),
        min_size=1,
        max_size=4,
    ),
    pin_selector=st.integers(0, 100),
    producer_selector=st.integers(0, 100),
    pin_version=st.sampled_from(["0.2.1", "0.2.2161", "0.2.9999"]),
    producer_version=st.sampled_from(["0.2.1", "0.2.2161", "0.2.9999"]),
)
def test_admission_exactly_matches_generated_commit_graph_rule(
    tmp_path,
    steps,
    pin_selector,
    producer_selector,
    pin_version,
    producer_version,
):
    # Hypothesis reuses tmp_path; allocate each graph deterministically within it.
    counter = len(list(tmp_path.iterdir()))
    graph = _Graph(tmp_path / f"graph-{counter}")
    graph.add(main=True)
    for merge, branch_length, fork_selector in steps:
        previous_main = graph.main[-1]
        side = graph.main[fork_selector % len(graph.main)]
        for _ in range(branch_length):
            side = graph.add(side)
        graph.add(previous_main, *([side] if merge else []), main=True)
    pin = pin_selector % len(graph.commits)
    producer = producer_selector % len(graph.commits)
    exact = pin == producer and pin_version == producer_version
    forward_main = (
        pin != producer and producer in graph.main and graph.is_ancestor(pin, producer)
    )
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[producer], producer_version),
        _expected(graph, pin, pin_version),
    )
    assert (issue is None) is (exact or forward_main)


@pytest.fixture
def identity_checkout(tmp_path):
    checkout = tmp_path / "provisioned-encoder"
    graph = _Graph(checkout)
    (checkout / "src/axiom_encode").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        f'[project]\nname = "axiom-encode"\nversion = "{cli.__version__}"\n'
    )
    (checkout / "src/axiom_encode/__init__.py").write_text(
        f'__version__ = "{cli.__version__}"\n'
    )
    (checkout / "uv.lock").write_text(
        f'[[package]]\nname = "axiom-encode"\nversion = "{cli.__version__}"\n'
    )
    _git(checkout, "add", ".")
    graph.tree = _git(checkout, "write-tree")
    pin_index = graph.add(main=True)
    producer_index = graph.add(pin_index, main=True)
    pin, producer = graph.commits[pin_index], graph.commits[producer_index]
    (checkout / ".git/HEAD").write_text(pin + "\n")
    _write_git_ref(checkout, "refs/remotes/origin/main", producer)
    return checkout, pin, producer


def test_read_only_identity_carries_history_without_serializing_a_checkout(
    identity_checkout,
):
    checkout, pin, producer = identity_checkout
    identity = cli._read_only_guard_encoder_execution_identity(checkout)
    assert identity.history_checkout == checkout
    assert json.loads(json.dumps(identity)) == _identity(pin, cli.__version__)
    assert (
        cli._encoder_producer_admission_issue(
            _identity(producer, cli.__version__), identity
        )
        is None
    )
    assert _git(checkout, "rev-parse", "HEAD") == pin


@pytest.fixture
def manifest_fixture(tmp_path, monkeypatch):
    install_test_eval_evidence_keys(
        monkeypatch,
        apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
        apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
    )
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    fixtures = _GuardFixtures()
    release = _bind_test_corpus_release(
        repo, tmp_path / "axiom-corpus", body="test source\n"
    )
    fixtures.corpus_release = release
    fixtures.corpus_path = release.root
    fixtures.source_attestation = resolve_corpus_source_unit(
        "us/statute/26/1", release
    ).source_attestation
    fixtures.source_attestation["generation_input_sha256"] = (
        fixtures.source_attestation["resolved_text_sha256"]
    )
    fixtures.source_attestation["rulespec_root"] = "rulespec-us/us"
    monkeypatch.setattr(
        cli,
        "canonical_rulespec_root_identity",
        lambda path: f"rulespec-{Path(path).name.split('-', 1)[0]}/{Path(path).name}",
    )
    fixtures._canonical_guard_repo(repo)
    relative = "us/statutes/26/1.yaml"
    rule = repo / relative
    fixtures._source_backed_rule(rule)
    payload = fixtures._model_manifest(
        [{"path": relative, "sha256": cli._sha256_file(rule)}]
    )
    payload["citation"] = "us/statute/26/1"
    return fixtures, repo, rule, payload


def _set_producer(payload, commit, version="0.2.2161"):
    payload["axiom_encode_version"] = version
    payload["axiom_encode_git"]["commit"] = commit
    payload["axiom_encode_git"]["version"] = version
    execution = payload.get("validation_execution")
    if isinstance(execution, dict):
        execution["axiom_encode"]["commit"] = commit
        execution["axiom_encode"]["version"] = version


def _load(repo, payload, expected):
    manifest = repo / ".axiom/encoding-manifests/us/statutes/26/1.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload) + "\n")
    return cli._load_verified_applied_encoding_manifest_payload(
        repo,
        manifest.relative_to(repo).as_posix(),
        signing_broker=TEST_APPLY_SIGNING_BROKER,
        expected_waiver_set_sha256=TEST_VALIDATION_WAIVER_SHA256,
        expected_encoder_identity=expected,
    )


def _reviewed_promotion(payload):
    payload["tool"] = cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL
    payload["reviewed_rulespec_ref"] = "d" * 40
    return {field: payload[field] for field in cli._REVIEWED_CANDIDATE_MANIFEST_FIELDS}


@pytest.mark.parametrize(
    "tool",
    [cli.APPLIED_ENCODING_MODEL_TOOL, cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL],
)
@pytest.mark.parametrize("producer", [1, 4])
def test_signed_model_and_reviewed_promotion_admit_pin_or_main_head(
    manifest_fixture, graph, tool, producer
):
    _fixtures, repo, _rule, payload = manifest_fixture
    _set_producer(payload, graph.commits[producer])
    if tool == cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL:
        payload = _reviewed_promotion(payload)
    cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    pin_version = "0.2.2092" if producer == 4 else "0.2.2161"
    verified, _root, _digest, issues = _load(
        repo, payload, _expected(graph, version=pin_version)
    )
    assert issues == []
    assert verified == payload


@pytest.mark.parametrize(
    "producer",
    [0, 2, 5, "unknown"],
    ids=["older-same-version", "second-parent", "feature", "not-fetched"],
)
@pytest.mark.parametrize(
    "tool",
    [cli.APPLIED_ENCODING_MODEL_TOOL, cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL],
)
def test_signed_model_and_reviewed_promotion_reject_unadmitted_producers(
    manifest_fixture, graph, tool, producer
):
    _fixtures, repo, _rule, payload = manifest_fixture
    commit = "f" * 40 if producer == "unknown" else graph.commits[producer]
    _set_producer(payload, commit)
    if tool == cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL:
        payload = _reviewed_promotion(payload)
    cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    assert (
        cli._applied_encoding_manifest_signature_issue(
            payload, TEST_APPLY_SIGNING_BROKER
        )
        is None
    )
    verified, _root, _digest, issues = _load(repo, payload, _expected(graph))
    assert verified is None
    assert any("running pinned encoder" in issue for issue in issues)


def test_full_generated_guard_uses_fetched_history_while_encoder_head_stays_at_pin(
    manifest_fixture, identity_checkout, monkeypatch
):
    fixtures, repo, rule, payload = manifest_fixture
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "commit", "--allow-empty", "-m", "guard checkout HEAD")
    checkout, pin, producer = identity_checkout
    _set_producer(payload, producer, cli.__version__)
    cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    manifest = repo / ".axiom/encoding-manifests/us/statutes/26/1.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(
        cli, "_applied_encoding_manifest_verifier", lambda: TEST_APPLY_SIGNING_BROKER
    )
    issues = cli.guard_generated_change_issues(
        repo,
        corpus_path=fixtures.corpus_path,
        changed_files=[
            rule.relative_to(repo).as_posix(),
            manifest.relative_to(repo).as_posix(),
        ],
        expected_encoder_checkout=checkout,
    )
    assert issues == []
    assert _git(checkout, "rev-parse", "HEAD") == pin


@pytest.mark.parametrize(
    "field", ["commit", "version", "identity_source", "provenance-version"]
)
def test_valid_signature_cannot_hide_producer_identity_disagreement(
    manifest_fixture, graph, field
):
    _fixtures, repo, _rule, payload = manifest_fixture
    _set_producer(payload, graph.commits[4])
    execution = payload["validation_execution"]["axiom_encode"]
    if field == "commit":
        # Both identities independently satisfy admission, but must agree.
        execution["commit"] = graph.commits[3]
    elif field == "version":
        execution["version"] = "0.2.9999"
    elif field == "identity_source":
        execution["identity_source"] = cli.APPLY_IDENTITY_SOURCE_GITHUB_SHA
    else:
        payload["axiom_encode_git"]["version"] = "0.2.9999"
    cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    assert (
        cli._applied_encoding_manifest_signature_issue(
            payload, TEST_APPLY_SIGNING_BROKER
        )
        is None
    )
    verified, _root, _digest, issues = _load(repo, payload, _expected(graph))
    assert verified is None
    assert any("disagrees" in issue for issue in issues)


@pytest.mark.parametrize("signature", ["unsigned", "wrong-root"])
def test_ancestry_admission_cannot_bypass_protected_signature(
    manifest_fixture, graph, monkeypatch, signature
):
    _fixtures, repo, _rule, payload = manifest_fixture
    _set_producer(payload, graph.commits[4])
    if signature == "unsigned":
        payload.pop("signature")
    else:
        private = bytes(range(32))
        public = (
            Ed25519PrivateKey.from_private_bytes(private)
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
        wrong_root = SigningBrokerFixture(
            apply_private_key=base64.b64encode(private).decode(),
            apply_public_key=base64.b64encode(public).decode(),
        )
        cli._sign_applied_encoding_manifest(payload, wrong_root)

    def unexpected_admission(*args, **kwargs):
        pytest.fail(
            "Producer admission must run only after protected signature verification"
        )

    monkeypatch.setattr(cli, "_encoder_producer_admission_issue", unexpected_admission)
    verified, _root, _digest, issues = _load(repo, payload, _expected(graph))
    assert verified is None
    assert issues


@pytest.mark.parametrize("mutation", ["bytes", "citation"])
def test_new_main_producer_does_not_relax_final_bytes_or_citation(
    manifest_fixture, graph, mutation
):
    _fixtures, repo, rule, payload = manifest_fixture
    _set_producer(payload, graph.commits[4])
    if mutation == "bytes":
        rule.write_text(rule.read_text() + "# edited after signing\n")
    else:
        attestation = payload["source_attestation"]
        attestation["requested_corpus_citation_path"] = "us/statute/26/2"
        attestation["resolved_corpus_citation_path"] = "us/statute/26/2"
        attestation["row"]["citation_path"] = "us/statute/26/2"
    cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    verified, _root, _digest, issues = _load(repo, payload, _expected(graph))
    assert verified is None
    expected_issue = (
        "stale sha256"
        if mutation == "bytes"
        else "declares corpus path(s) not bound by source_attestation"
    )
    assert any(expected_issue in issue for issue in issues)


@pytest.mark.parametrize("nested_signature", ["valid", "invalid"])
def test_new_retirement_preserves_historical_nested_manifest_rules(
    manifest_fixture, graph, nested_signature
):
    fixtures, repo, rule, nested = manifest_fixture
    _set_producer(nested, "c" * 40, "0.2.1000")
    cli._sign_applied_encoding_manifest(nested, TEST_APPLY_SIGNING_BROKER)
    if nested_signature == "invalid":
        nested["signature"]["value"] = "f" * 64
    rule.unlink()
    retirement = fixtures._retirement_manifest(
        [rule.relative_to(repo).as_posix()], retired_manifest=nested
    )
    _set_producer(retirement, graph.commits[4])
    cli._sign_applied_encoding_manifest(retirement, TEST_APPLY_SIGNING_BROKER)
    verified, _root, _digest, issues = _load(repo, retirement, _expected(graph))
    if nested_signature == "valid":
        assert issues == []
        assert verified == retirement
    else:
        assert verified is None
        assert any("retired_manifest" in issue for issue in issues)


WRAPPER_TOOLS = [
    cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL,
    cli.APPLIED_ENCODING_LEGACY_EXACT_DEPENDENT_TOOL,
    cli.APPLIED_ENCODING_LEGACY_RETAINED_SUCCESSOR_TOOL,
    cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_TOOL,
    cli.APPLIED_ENCODING_PATH_MIGRATION_TOOL,
    cli.APPLIED_ENCODING_RETIRE_TOOL,
]


@pytest.mark.parametrize("tool", WRAPPER_TOOLS)
@pytest.mark.parametrize("producer", [0, 1, 2, 4, 5])
def test_every_wrapper_checks_producer_ancestry(graph, tool, producer):
    model = {
        "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "backend": "codex",
        "tool": cli.APPLIED_ENCODING_MODEL_TOOL,
    }
    payload = {
        "tool": tool,
        "axiom_encode_git": {
            "commit": graph.commits[producer],
            "version": "0.2.2161",
            "dirty_tracked": False,
        },
        "applied_files": [{"path": "us/statutes/26/1.yaml", "sha256": "a" * 64}],
    }
    if tool == cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL:
        payload["reviewed_rulespec_ref"] = "d" * 40
    elif tool == cli.APPLIED_ENCODING_LEGACY_RETAINED_SUCCESSOR_TOOL:
        payload["retained_successor_manifest"] = copy.deepcopy(model)
    elif tool == cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_TOOL:
        payload["replacement_manifest"] = copy.deepcopy(model)
    elif tool == cli.APPLIED_ENCODING_PATH_MIGRATION_TOOL:
        payload["migrated_manifest"] = copy.deepcopy(model)
    elif tool == cli.APPLIED_ENCODING_RETIRE_TOOL:
        payload["retired_manifest"] = copy.deepcopy(model)
        payload["applied_files"] = [{"path": "us/statutes/26/1.yaml", "deleted": True}]
    issues = cli._applied_manifest_tool_execution_issues(
        payload,
        manifest_label="manifest.json",
        expected_encoder_identity=_expected(graph),
    )
    assert (issues == []) is (producer in {1, 4})
