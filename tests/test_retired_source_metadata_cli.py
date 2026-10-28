"""I6: the signed class admits only one whole deterministic migration."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import axiom_encode.cli as cli
from axiom_encode import __version__
from axiom_encode.harness.evals import resolve_corpus_source_unit
from axiom_encode.retired_source_metadata import (
    PLAN_SCHEMA,
    RECEIPT_DIR,
    RetiredSourceMetadataError,
    canonical_json_bytes,
    receipt_identity_sha256,
)
from scripts.provision_verification_supervisor import _install_trusted_git_wrapper
from tests.environment_assertions import Names, hides_environment
from tests.signing_broker_fixtures import SigningBrokerFixture
from tests.test_cli import (
    TEST_VALIDATION_WAIVER_SHA256,
    _bind_test_corpus_release,
)
from tests.test_cli import TestGuardGenerated as _GuardFixtures

PRIMARY = "us/statutes/26/25A.yaml"
COMPANION = "us/statutes/26/25A.test.yaml"
IMPORTER = "us/policies/importer.yaml"
WAIVER = TEST_VALIDATION_WAIVER_SHA256
ENCODER = {
    "repository": cli.APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    "commit": "a" * 40,
    "version": __version__,
}
PROVENANCE = {
    "root": "/repo/axiom-encode",
    "commit": ENCODER["commit"],
    "dirty_tracked": False,
    "version": __version__,
    "version_commit": "b" * 40,
    "identity_source": "git",
}
PRIVATE = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
BROKER = SigningBrokerFixture(
    apply_private_key=b64encode(bytes(range(32))).decode(),
    apply_public_key=b64encode(
        PRIVATE.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode(),
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=Names(cli._rulespec_migration_git_environment()),
    ).stdout.strip()


@pytest.fixture(params=["clean", "foreign", "missing"])
def ambient_git(tmp_path, monkeypatch, request):
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    if request.param == "clean":
        return
    foreign = tmp_path / "foreign"
    if request.param == "foreign":
        foreign.mkdir()
        git(foreign, "init")
        git(
            foreign,
            "-c",
            "user.name=Foreign checkout",
            "-c",
            "user.email=foreign@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "foreign history",
        )
    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))


@pytest.mark.parametrize("all_files", [False, True], ids=["explicit_files", "all"])
def test_guard_gitless_default_refs_run_worktree_verification(
    checkout, monkeypatch, ambient_git, all_files
):
    shutil.rmtree(checkout.repo / ".git")
    calls = []
    real_run = subprocess.run

    @hides_environment
    def capture(command, **kwargs):
        calls.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(cli.subprocess, "run", capture)
    assert (
        cli.guard_generated_change_issues(
            checkout.repo,
            corpus_path=checkout.args.corpus_path,
            changed_files=[],
            all_files=all_files,
        )
        == []
    )
    assert not any(command[-1] == "HEAD^{commit}" for command in calls), calls


@pytest.mark.parametrize("argument", ["base_ref", "head_ref"])
def test_guard_gitless_explicit_refs_are_refused(checkout, ambient_git, argument):
    shutil.rmtree(checkout.repo / ".git")
    ref = "HEAD" if argument == "base_ref" else "reviewed-head"
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        changed_files=[],
        **{argument: ref},
    )
    assert issues == [
        f"Cannot resolve guard {argument} {ref!r} without a Git repository"
    ]


def test_guard_gitless_retrospective_is_refused(checkout, ambient_git):
    shutil.rmtree(checkout.repo / ".git")
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        changed_files=[],
        retrospective=True,
    )
    assert issues == ["Retrospective guard requires a Git repository to freeze history"]


@pytest.mark.parametrize("git_entry", ["malformed", "dangling_symlink"])
def test_guard_damaged_git_metadata_does_not_enable_gitless_fallback(
    tmp_path, git_entry
):
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    if git_entry == "malformed":
        (repo / ".git").write_text("gitdir: missing\n")
    else:
        (repo / ".git").symlink_to(repo / "missing")
    issues = cli.guard_generated_change_issues(
        repo, corpus_path=tmp_path / "corpus", changed_files=[]
    )
    assert len(issues) == 1
    assert issues[0].startswith("Cannot inspect guard Git repository:"), issues


def test_guard_git_probe_refusal_does_not_enable_gitless_fallback(
    tmp_path, monkeypatch
):
    @hides_environment
    def refuse(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 128, b"", b"fatal: detected dubious ownership in repository"
        )

    monkeypatch.setattr(cli.subprocess, "run", refuse)
    issues = cli.guard_generated_change_issues(
        tmp_path, corpus_path=tmp_path / "corpus", changed_files=[]
    )
    assert issues == [
        "Cannot inspect guard Git repository: "
        "fatal: detected dubious ownership in repository"
    ]


@pytest.mark.parametrize("git_checkout", [False, True], ids=["gitless", "git"])
@pytest.mark.parametrize("all_files", [False, True], ids=["diff", "all_with_base"])
@pytest.mark.parametrize("argument", ["base_ref", "head_ref"])
@pytest.mark.parametrize(
    "ref",
    ["--output=guard-output", "-p", "--ext-diff", "--no-index"]
    + [f"HEAD{control}" for control in ("\x00", "\n", "\r", "\t", "\x7f", "\x85")],
)
def test_guard_ref_arguments_cannot_be_git_options(
    checkout, monkeypatch, git_checkout, all_files, argument, ref
):
    if not git_checkout:
        shutil.rmtree(checkout.repo / ".git")
    calls = []
    real_run = subprocess.run

    @hides_environment
    def capture(command, **kwargs):
        calls.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(cli.subprocess, "run", capture)
    options = {"base_ref": checkout.base, "head_ref": "HEAD"}
    options[argument] = ref
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        all_files=all_files,
        **options,
    )
    assert any(f"Invalid guard {argument}" in issue for issue in issues), issues
    assert not (checkout.repo / "guard-output").exists()
    assert all(
        not any(
            value == ref or value.startswith((ref + "^", ref + ":"))
            for value in command
        )
        for command in calls
    ), calls


@pytest.mark.parametrize("all_files", [False, True], ids=["diff", "all_with_base"])
def test_guard_freezes_symbolic_refs_before_transaction_queries(
    checkout, capsys, monkeypatch, ambient_git, all_files
):
    migrate(checkout, capsys)
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "signed migration")
    head = git(checkout.repo, "rev-parse", "HEAD")
    git(checkout.repo, "branch", "guard-base", checkout.base)
    git(checkout.repo, "tag", "-a", "guard-head", "-m", "migration head")
    calls = []
    real_run = subprocess.run

    @hides_environment
    def capture(command, **kwargs):
        calls.append(command)
        result = real_run(command, **kwargs)
        if command[-1] == "guard-head^{commit}":
            # Both names move after resolution; queries must use the snapshot.
            git(checkout.repo, "update-ref", "refs/heads/guard-base", head)
            git(checkout.repo, "update-ref", "refs/tags/guard-head", checkout.base)
        return result

    monkeypatch.setattr(cli.subprocess, "run", capture)
    assert (
        cli.guard_generated_change_issues(
            checkout.repo,
            corpus_path=checkout.args.corpus_path,
            base_ref="guard-base",
            head_ref="guard-head",
            all_files=all_files,
        )
        == []
    )
    resolutions = [command for command in calls if "rev-parse" in command]
    for name in ("guard-base", "guard-head"):
        assert sum(command[-1] == f"{name}^{{commit}}" for command in resolutions) == 1
        assert all(
            command[-4:]
            == ["rev-parse", "--verify", "--end-of-options", f"{name}^{{commit}}"]
            for command in resolutions
            if command[-1] == f"{name}^{{commit}}"
        )
    diffs = [command for command in calls if "diff" in command]
    assert diffs
    for command in diffs:
        boundary = command.index("--end-of-options")
        assert command[-1] == "--"
        assert command[boundary + 1 : -1] in ([checkout.base, head], [head])


def test_guard_resolves_identical_base_and_head_once(checkout, monkeypatch):
    calls = []
    real_run = subprocess.run

    @hides_environment
    def capture(command, **kwargs):
        calls.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(cli.subprocess, "run", capture)
    assert (
        cli.guard_generated_change_issues(
            checkout.repo,
            corpus_path=checkout.args.corpus_path,
            base_ref="HEAD",
            head_ref="HEAD",
        )
        == []
    )
    resolutions = [command for command in calls if command[-1] == "HEAD^{commit}"]
    assert len(resolutions) == 1
    assert resolutions[0][-4:] == [
        "rev-parse",
        "--verify",
        "--end-of-options",
        "HEAD^{commit}",
    ]


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "Migration test")
    git(repo, "config", "user.email", "migration@example.com")
    corpus = tmp_path / "axiom-corpus"
    release = _bind_test_corpus_release(repo, corpus, body="test source\n")
    fixtures = _GuardFixtures()
    fixtures.source_attestation = resolve_corpus_source_unit(
        "us/statute/26/1", release
    ).source_attestation
    fixtures.source_attestation["generation_input_sha256"] = (
        fixtures.source_attestation["resolved_text_sha256"]
    )
    fixtures.source_attestation["rulespec_root"] = "rulespec-us/us"
    source_sha256 = fixtures.source_attestation["source_sha256"]
    source = repo / PRIMARY
    source.parent.mkdir(parents=True)
    source.write_text(f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1
    source_sha256: {source_sha256}
    values:
      rate: 0.2
rules: []
""")
    (repo / COMPANION).write_text("tests: []\n")
    importer = repo / IMPORTER
    importer.parent.mkdir(parents=True)
    importer.write_text(f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1
    source_sha256: {source_sha256}
rules:
  - name: imported
    metadata:
      proof:
        atoms:
          - kind: import
            import:
              target: us:statutes/26/25A#amount
              hash: sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}
""")
    owners = {}
    for relative, files in (
        (PRIMARY, [PRIMARY, COMPANION]),
        (IMPORTER, [IMPORTER]),
    ):
        payload = fixtures._model_manifest(
            [
                {
                    "path": path,
                    "sha256": hashlib.sha256((repo / path).read_bytes()).hexdigest(),
                }
                for path in files
            ]
        )
        cli._sign_applied_encoding_manifest(payload, BROKER)
        owner = repo / cli._applied_encoding_manifest_path(Path(relative))
        owner.parent.mkdir(parents=True, exist_ok=True)
        owner.write_text(json.dumps(payload) + "\n")
        owners[relative] = owner.read_bytes()
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"schema_version": PLAN_SCHEMA, "base_commit": base, "modules": [PRIMARY]}
        )
    )
    monkeypatch.setattr(cli, "load_rulespec_local_corpus_release", lambda *_: release)
    monkeypatch.setattr(cli, "verify_rulespec_validation_waiver_set", lambda *_: WAIVER)
    monkeypatch.setattr(
        cli, "_require_applied_encoding_manifest_signer", lambda: BROKER
    )
    monkeypatch.setattr(
        cli, "_applied_encoding_manifest_verifier", lambda: PRIVATE.public_key()
    )
    monkeypatch.setattr(
        cli, "_current_guard_encoder_execution_identity", lambda: ENCODER
    )
    monkeypatch.setattr(
        cli, "_read_only_guard_encoder_execution_identity", lambda *_: ENCODER
    )
    monkeypatch.setattr(
        cli, "_require_clean_axiom_encode_git_provenance", lambda: PROVENANCE
    )
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()
    return SimpleNamespace(
        repo=repo,
        base=base,
        generator_base=base,
        owners=owners,
        args=SimpleNamespace(
            policy_repo_path=repo, corpus_path=corpus, plan=plan, dry_run=False
        ),
    )


def migrate(checkout, capsys):
    cli.cmd_migrate_retired_source_metadata(checkout.args)
    return json.loads(capsys.readouterr().out)


def changed(report):
    return sorted(
        [
            report["receipt_path"],
            *report["manifest_paths"],
            *(item["path"] for item in report["files"]),
        ]
    )


def guard(checkout, paths):
    return cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        changed_files=paths,
        base_ref=checkout.base,
    )


def repository_bytes(repo):
    """Capture all transaction-visible bytes, including newly created artifacts."""

    return {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(repo).parts
    }


def set_owner_kind(checkout, relative, kind):
    owner_path = cli._applied_encoding_manifest_path(Path(relative))
    owner = checkout.repo / owner_path
    payload = json.loads(checkout.owners[relative])
    if kind == "none":
        owner.unlink()
        return
    if kind == "wrong_location":
        relocated = checkout.repo / cli._applied_encoding_manifest_path(
            Path("us/policies/relocated-owner.yaml")
        )
        relocated.parent.mkdir(parents=True, exist_ok=True)
        owner.rename(relocated)
        return
    if kind == "bad_signature":
        payload["signature"]["value"] = "invalid-signature"
        owner.write_text(json.dumps(payload) + "\n")
        return
    if kind == "v1":
        payload["schema_version"] = "axiom-encode/applied-rulespec/v1"
    elif kind == "manual":
        payload["backend"] = "manual"
    elif kind == "stale_hash":
        payload["applied_files"][0]["sha256"] = "0" * 64
    elif kind in {"duplicate", "duplicate_escaped"}:
        duplicate = checkout.repo / cli._applied_encoding_manifest_path(
            Path("us/policies/duplicate.yaml")
        )
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        raw = owner.read_bytes()
        if kind == "duplicate_escaped":
            raw = raw.replace(PRIMARY.encode(), PRIMARY.replace("/", "\\/").encode())
        duplicate.write_bytes(raw)
    else:
        assert kind == "generator"
    cli._sign_applied_encoding_manifest(payload, BROKER)
    owner.write_text(json.dumps(payload) + "\n")


def commit_owner_changes(checkout):
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "--allow-empty", "-m", "owner assignment")
    checkout.base = git(checkout.repo, "rev-parse", "HEAD")
    plan = json.loads(checkout.args.plan.read_text())
    plan["base_commit"] = checkout.base
    checkout.args.plan.write_text(json.dumps(plan))


@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        (PRIMARY, "none"),
        (PRIMARY, "v1"),
        (PRIMARY, "manual"),
        (IMPORTER, "none"),
        (PRIMARY, "duplicate"),
        (PRIMARY, "duplicate_escaped"),
        (PRIMARY, "wrong_location"),
        (PRIMARY, "bad_signature"),
        (PRIMARY, "stale_hash"),
    ],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_H1_unowned_or_nongenerator_base_is_refused_before_write_or_signing(
    checkout, monkeypatch, relative, kind, dry_run
):
    set_owner_kind(checkout, relative, kind)
    commit_owner_changes(checkout)
    checkout.args.dry_run = dry_run
    before = repository_bytes(checkout.repo)
    signer_requests = []

    def signer():
        signer_requests.append(True)
        return BROKER

    monkeypatch.setattr(cli, "_require_applied_encoding_manifest_signer", signer)
    with pytest.raises(SystemExit, match="manifest ownership"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert signer_requests == []
    assert repository_bytes(checkout.repo) == before


@pytest.mark.parametrize("mutation", ["typed", "content"])
def test_H1_embedded_prior_manifest_cannot_differ_from_base_owner(
    checkout, capsys, mutation
):
    report = migrate(checkout, capsys)
    manifest_path = cli._applied_encoding_manifest_path(Path(PRIMARY))
    payload = json.loads((checkout.repo / manifest_path).read_bytes())
    prior = payload["prior_manifest"]["manifest"]
    if mutation == "typed":
        # Python equality treats False and 0 alike, but signed JSON does not.
        assert prior["axiom_encode_git"]["dirty_tracked"] is False
        prior["axiom_encode_git"]["dirty_tracked"] = 0
    else:
        prior["generation_prompt_sha256"] = "0" * 64
    cli._sign_applied_encoding_manifest(payload, BROKER)
    (checkout.repo / manifest_path).write_text(json.dumps(payload) + "\n")
    assert guard(checkout, changed(report))
    with pytest.raises(RetiredSourceMetadataError, match="prior|replay"):
        cli.retired_source_metadata_change_set(
            checkout.repo, Path(report["receipt_path"])
        )


def test_I4_ambiguous_rewrite_refusal_is_pure_and_never_requests_signer(
    checkout, monkeypatch
):
    source = checkout.repo / PRIMARY
    source.write_text("""format: rulespec/v1
module:
  source_verification:
    corpus_citation_paths:
      - us/statute/26/1
      - us/statute/26/2
    values:
      rate: 0.2
rules: []
""")
    owner_path = cli._applied_encoding_manifest_path(Path(PRIMARY))
    payload = json.loads((checkout.repo / owner_path).read_bytes())
    for entry in payload["applied_files"]:
        if entry["path"] == PRIMARY:
            entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    cli._sign_applied_encoding_manifest(payload, BROKER)
    (checkout.repo / owner_path).write_text(json.dumps(payload) + "\n")
    commit_owner_changes(checkout)
    before = repository_bytes(checkout.repo)
    signer_requests = []

    def signer():
        signer_requests.append(True)
        return BROKER

    monkeypatch.setattr(cli, "_require_applied_encoding_manifest_signer", signer)
    with pytest.raises(SystemExit, match="exactly one ancestor"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert signer_requests == []
    assert repository_bytes(checkout.repo) == before


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    owners=st.tuples(
        st.sampled_from(["none", "v1", "manual", "generator", "duplicate"]),
        st.sampled_from(["none", "v1", "manual", "generator", "duplicate"]),
    )
)
def test_H1_every_planned_file_requires_exactly_one_generator_owner(
    checkout, capsys, owners
):
    initial_base = checkout.generator_base
    # Hypothesis reuses fixture instances; restore the immutable generator base.
    git(checkout.repo, "reset", "--hard", initial_base)
    git(checkout.repo, "clean", "-fd")
    for relative, kind in zip((PRIMARY, IMPORTER), owners, strict=True):
        set_owner_kind(checkout, relative, kind)
    commit_owner_changes(checkout)
    before = repository_bytes(checkout.repo)
    if owners == ("generator", "generator"):
        report = migrate(checkout, capsys)
        assert guard(checkout, changed(report)) == []
    else:
        with pytest.raises(SystemExit, match="manifest ownership"):
            cli.cmd_migrate_retired_source_metadata(checkout.args)
        assert repository_bytes(checkout.repo) == before
    # Keep the next example's plan rooted at the same genuine encoder base.
    plan = json.loads(checkout.args.plan.read_text())
    plan["base_commit"] = initial_base
    checkout.args.plan.write_text(json.dumps(plan))


def test_H1_replay_checks_bound_owner_bytes_at_the_receipt_base(checkout, capsys):
    report = migrate(checkout, capsys)
    receipt = json.loads((checkout.repo / report["receipt_path"]).read_bytes())
    owner_path = cli._applied_encoding_manifest_path(Path(PRIMARY))
    (checkout.repo / owner_path).write_bytes(checkout.owners[PRIMARY] + b"\n")
    git(checkout.repo, "add", owner_path.as_posix())
    git(checkout.repo, "commit", "-m", "different base owner bytes")
    base = git(checkout.repo, "rev-parse", "HEAD")
    receipt["base_commit"] = base
    receipt["base_tree"] = git(checkout.repo, "rev-parse", f"{base}^{{tree}}")
    receipt["plan"]["base_commit"] = base
    receipt["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt["plan"])
    ).hexdigest()
    identity = receipt_identity_sha256(receipt)
    receipt["identity_sha256"] = identity
    relative = RECEIPT_DIR / f"{identity}.json"
    (checkout.repo / relative).write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(RetiredSourceMetadataError, match="owner|replay"):
        cli._retired_source_metadata_replay(checkout.repo, relative)


def test_M1_receipt_without_protected_base_fails_closed(checkout, capsys):
    report = migrate(checkout, capsys)
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        changed_files=changed(report),
        base_ref=None,
    )
    assert any("protected base" in issue for issue in issues)


def test_M1_all_rejects_committed_migration_replayed_onto_unowned_base(
    checkout, capsys
):
    report = migrate(checkout, capsys)
    artifacts = {path: (checkout.repo / path).read_bytes() for path in changed(report)}
    git(checkout.repo, "reset", "--hard", checkout.generator_base)
    git(checkout.repo, "clean", "-fd")
    for relative in (PRIMARY, IMPORTER):
        (checkout.repo / cli._applied_encoding_manifest_path(Path(relative))).unlink()
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "protected base has no encoder owners")
    protected_base = git(checkout.repo, "rev-parse", "HEAD")
    for path, raw in artifacts.items():
        target = checkout.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "replay valid signed migration artifacts")

    explicit_issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        all_files=True,
        base_ref=protected_base,
    )
    assert any(
        "base_commit does not match the protected base" in issue
        for issue in explicit_issues
    )
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        all_files=True,
    )
    assert any(
        report["receipt_path"] in issue and "protected base" in issue
        for issue in issues
    )


def test_M1_all_rejects_committed_migration_with_restored_cascade(checkout, capsys):
    original_importer = (checkout.repo / IMPORTER).read_bytes()
    report = migrate(checkout, capsys)
    (checkout.repo / IMPORTER).write_bytes(original_importer)
    (checkout.repo / cli._applied_encoding_manifest_path(Path(IMPORTER))).write_bytes(
        checkout.owners[IMPORTER]
    )
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "retain migration primary and old cascade")

    explicit_issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        all_files=True,
        base_ref=checkout.base,
    )
    assert any(
        f"migration live postimage differs from replay: {IMPORTER}" in issue
        for issue in explicit_issues
    )
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        all_files=True,
    )
    assert any(
        report["receipt_path"] in issue and "protected base" in issue
        for issue in issues
    )


@pytest.mark.parametrize("mode", ["diff", "all_with_base", "changed_files"])
def test_M1_entry_modes_reject_transaction_with_unrelated_actual_change(
    checkout, capsys, mode
):
    report = migrate(checkout, capsys)
    (checkout.repo / "README.md").write_text("outside the migration transaction\n")
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "migration mixed with unrelated change")
    actual_changed = git(
        checkout.repo, "diff", "--name-only", checkout.base, "HEAD"
    ).splitlines()
    assert set(actual_changed) == {*changed(report), "README.md"}
    options = {"base_ref": checkout.base}
    if mode == "all_with_base":
        options["all_files"] = True
    elif mode == "changed_files":
        # A caller-controlled inventory cannot conceal the committed extra file.
        options["changed_files"] = changed(report)
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        **options,
    )
    assert issues


@pytest.mark.parametrize("mode", ["diff", "all_with_base", "changed_files"])
@pytest.mark.parametrize("alternate_has_migration", [False, True])
def test_M1_actual_transaction_scope_ignores_ambient_git_checkout(
    checkout, capsys, tmp_path, monkeypatch, mode, alternate_has_migration
):
    report = migrate(checkout, capsys)
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "commit genuine signed migration")
    alternate = tmp_path / "alternate-rulespec-us"
    shutil.copytree(checkout.repo, alternate)
    if not alternate_has_migration:
        git(alternate, "reset", "--hard", checkout.base)
    with (alternate / ".git" / "info" / "exclude").open("a") as handle:
        handle.write("\nREADME.md\n")
    (checkout.repo / "README.md").write_text("outside the migration transaction\n")
    git(checkout.repo, "add", "README.md")
    git(checkout.repo, "commit", "-m", "add unrelated change only in real checkout")
    monkeypatch.setenv("GIT_DIR", str(alternate / ".git"))
    # Keep checkout routing valid while foreign Git metadata conceals README.
    monkeypatch.setenv("GIT_WORK_TREE", str(checkout.repo))
    monkeypatch.setenv("GIT_INDEX_FILE", str(alternate / ".git" / "index"))
    redirected = set(
        cli._git_changed_files(checkout.repo, base_ref=checkout.base, head_ref="HEAD")
    ) | set(cli._git_changed_files(checkout.repo, base_ref=None, head_ref="HEAD"))
    assert redirected == set(changed(report))
    options = {"base_ref": checkout.base}
    if mode == "all_with_base":
        options["all_files"] = True
    elif mode == "changed_files":
        options["changed_files"] = changed(report)
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        **options,
    )
    assert any("README.md" in issue for issue in issues), issues


def test_M1_all_rejects_receipt_already_present_at_protected_base(checkout, capsys):
    report = migrate(checkout, capsys)
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "land complete signed migration")
    protected_base = git(checkout.repo, "rev-parse", "HEAD")
    assert git(checkout.repo, "diff", "--name-only", protected_base, "HEAD") == ""
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        all_files=True,
        base_ref=protected_base,
    )
    assert any(report["receipt_path"] in issue for issue in issues)


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    mode=st.sampled_from(
        [
            "diff_with_base",
            "diff_without_base",
            "all_with_base",
            "all_without_base",
            "changed_files_with_base",
            "changed_files_without_base",
        ]
    ),
    receipt_present=st.booleans(),
    base_owners_present=st.booleans(),
    cascade_restored=st.booleans(),
)
def test_M1_every_entry_mode_requires_full_protected_receipt_admission(
    checkout, capsys, mode, receipt_present, base_owners_present, cascade_restored
):
    if not hasattr(checkout, "admission_artifacts"):
        checkout.original_importer = (checkout.repo / IMPORTER).read_bytes()
        checkout.admission_report = migrate(checkout, capsys)
        checkout.admission_artifacts = {
            path: (checkout.repo / path).read_bytes()
            for path in changed(checkout.admission_report)
        }
    report = checkout.admission_report
    # Each example starts from the genuine generator base, even after a failure.
    git(checkout.repo, "reset", "--hard", checkout.generator_base)
    git(checkout.repo, "clean", "-fd")
    if not base_owners_present:
        for relative in (PRIMARY, IMPORTER):
            (
                checkout.repo / cli._applied_encoding_manifest_path(Path(relative))
            ).unlink()
        git(checkout.repo, "add", ".")
        git(checkout.repo, "commit", "-m", "remove protected encoder owners")
    protected_base = git(checkout.repo, "rev-parse", "HEAD")
    for path, raw in checkout.admission_artifacts.items():
        if not receipt_present and path == report["receipt_path"]:
            continue
        target = checkout.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    if cascade_restored:
        (checkout.repo / IMPORTER).write_bytes(checkout.original_importer)
        (
            checkout.repo / cli._applied_encoding_manifest_path(Path(IMPORTER))
        ).write_bytes(checkout.owners[IMPORTER])
    git(checkout.repo, "add", ".")
    if mode != "diff_without_base":
        git(checkout.repo, "commit", "-m", "commit candidate migration tree")
    discovered_paths = git(
        checkout.repo, "diff", "--name-only", protected_base
    ).splitlines()
    if mode == "diff_without_base":
        assert (
            cli._applied_encoding_manifest_path(Path(PRIMARY)).as_posix()
            in discovered_paths
        )
    options = {}
    if mode in {"diff_with_base", "all_with_base", "changed_files_with_base"}:
        options["base_ref"] = protected_base
    if mode.startswith("all_"):
        options["all_files"] = True
    elif mode.startswith("changed_files_"):
        options["changed_files"] = discovered_paths
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        **options,
    )
    admitted = (
        "base_ref" in options
        and receipt_present
        and base_owners_present
        and not cascade_restored
    )
    assert (issues == []) is admitted, (
        mode,
        receipt_present,
        base_owners_present,
        cascade_restored,
        issues,
    )


@pytest.mark.parametrize("operation", ["modify", "delete"])
def test_L4_historical_receipt_edit_has_clear_refusal(checkout, capsys, operation):
    report = migrate(checkout, capsys)
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "land migration")
    checkout.base = git(checkout.repo, "rev-parse", "HEAD")
    receipt = checkout.repo / report["receipt_path"]
    if operation == "modify":
        receipt.write_bytes(receipt.read_bytes() + b"\n")
    else:
        receipt.unlink()
    issues = guard(checkout, [report["receipt_path"]])
    assert any(
        "historical migration receipts cannot be edited or removed" in issue
        for issue in issues
    )


def test_signed_replay_class_and_complete_guard_round_trip(checkout, capsys):
    for relative in (PRIMARY, IMPORTER):
        payload, _root, _digest, issues = (
            cli._load_verified_applied_encoding_manifest_payload(
                checkout.repo,
                cli._applied_encoding_manifest_path(Path(relative)).as_posix(),
                signing_broker=PRIVATE.public_key(),
                expected_waiver_set_sha256=WAIVER,
                expected_encoder_identity=ENCODER,
                local_corpus_release=cli.load_rulespec_local_corpus_release(
                    checkout.repo, checkout.args.corpus_path
                ),
            )
        )
        assert issues == []
        assert payload is not None and payload["backend"] == "codex"
    report = migrate(checkout, capsys)
    assert guard(checkout, changed(report)) == []
    assert len(report["files"]) == 2
    assert len(report["cascade_rewrites"]) == 1
    primary_manifest = json.loads(
        (checkout.repo / cli._applied_encoding_manifest_path(Path(PRIMARY))).read_text()
    )
    assert primary_manifest["axiom_encode_git"] == PROVENANCE
    for item in report["files"]:
        owner_path = cli._applied_encoding_manifest_path(Path(item["path"]))
        owner_raw = checkout.owners[item["path"]]
        binding = {
            "path": owner_path.as_posix(),
            "sha256": hashlib.sha256(owner_raw).hexdigest(),
        }
        receipt = json.loads((checkout.repo / report["receipt_path"]).read_bytes())
        receipt_file = next(
            entry for entry in receipt["files"] if entry["path"] == item["path"]
        )
        assert receipt_file["prior_manifest"] == binding
        manifest = json.loads((checkout.repo / owner_path).read_bytes())
        assert manifest["prior_manifest"] == {
            **binding,
            "manifest": json.loads(owner_raw),
        }
    assert "source_attestation" not in primary_manifest
    assert primary_manifest["applied_files"] == sorted(
        [
            {
                "path": PRIMARY,
                "sha256": hashlib.sha256(
                    (checkout.repo / PRIMARY).read_bytes()
                ).hexdigest(),
            },
            {
                "path": COMPANION,
                "sha256": hashlib.sha256(
                    (checkout.repo / COMPANION).read_bytes()
                ).hexdigest(),
            },
        ],
        key=lambda item: item["path"],
    )
    inventory = cli.retired_source_metadata_change_set(
        checkout.repo, Path(report["receipt_path"])
    )
    assert inventory["changed_paths"] == changed(report)
    assert COMPANION not in inventory["changed_paths"]


def test_signed_cli_runs_through_production_trusted_git_allowlist(
    checkout, capsys, monkeypatch, tmp_path
):
    trusted_git = shutil.which("git")
    assert trusted_git is not None
    tools = tmp_path / "trusted-tools"
    tools.mkdir()
    _install_trusted_git_wrapper(
        tools, Path(sys.executable).resolve(), Path(trusted_git).resolve()
    )
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    report = migrate(checkout, capsys)
    assert guard(checkout, changed(report)) == []


def test_prior_owner_uses_the_callers_verification_context(
    checkout, capsys, monkeypatch
):
    report = migrate(checkout, capsys)
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()

    def forbidden():
        raise AssertionError("explicit verifier must not use ambient signing context")

    monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", forbidden)
    payload, _, _, issues = cli._load_verified_applied_encoding_manifest_payload(
        checkout.repo,
        report["manifest_paths"][0],
        signing_broker=BROKER,
        expected_waiver_set_sha256=WAIVER,
        expected_encoder_identity=ENCODER,
    )
    assert issues == []
    assert payload is not None


def test_dry_run_never_requests_signing_and_replays(checkout, capsys, monkeypatch):
    checkout.args.dry_run = True

    def forbidden():
        raise AssertionError("dry run requested signer")

    monkeypatch.setattr(cli, "_require_applied_encoding_manifest_signer", forbidden)
    report = migrate(checkout, capsys)
    assert report["dry_run"] is True
    assert report["manifest_paths"] == []
    replay, _ = cli._retired_source_metadata_replay(
        checkout.repo, Path(report["receipt_path"])
    )
    assert all(
        (checkout.repo / item.path).read_bytes() == item.after for item in replay.files
    )
    assert all(
        (checkout.repo / cli._applied_encoding_manifest_path(Path(path))).read_bytes()
        == raw
        for path, raw in checkout.owners.items()
    )


@pytest.mark.parametrize("tamper", ["primary", "cascade", "receipt", "companion"])
def test_I6_postimage_and_receipt_tampering_refused(checkout, capsys, tamper):
    report = migrate(checkout, capsys)
    relative = {
        "primary": PRIMARY,
        "cascade": IMPORTER,
        "receipt": report["receipt_path"],
        "companion": COMPANION,
    }[tamper]
    with (checkout.repo / relative).open("ab") as handle:
        handle.write(b"\n# changed\n")
    assert guard(checkout, changed(report))


@pytest.mark.parametrize(
    "split",
    [
        "receipt_only",
        "without_receipt",
        "without_primary",
        "without_cascade",
        "without_manifest",
        "unrelated",
    ],
)
def test_I6_receipt_and_exact_change_set_cannot_be_split(checkout, capsys, split):
    report = migrate(checkout, capsys)
    paths = changed(report)
    if split == "receipt_only":
        paths = [report["receipt_path"]]
    elif split == "without_receipt":
        paths.remove(report["receipt_path"])
    elif split == "without_primary":
        paths.remove(PRIMARY)
    elif split == "without_cascade":
        paths.remove(IMPORTER)
    elif split == "without_manifest":
        paths.remove(report["manifest_paths"][0])
    else:
        paths.append("README.md")
    assert guard(checkout, paths)


def test_I6_class_cannot_be_claimed_without_receipt(checkout, capsys):
    report = migrate(checkout, capsys)
    manifest_path = report["manifest_paths"][0]
    payload = json.loads((checkout.repo / manifest_path).read_text())
    del payload["retired_source_metadata"]
    cli._sign_applied_encoding_manifest(payload, BROKER)
    (checkout.repo / manifest_path).write_text(json.dumps(payload))
    assert guard(checkout, changed(report))
    assert cli._applied_manifest_exact_schema_issues(
        payload, manifest_label=manifest_path
    )


def test_I6_receipt_digest_and_encoder_pin_required(checkout, capsys):
    report = migrate(checkout, capsys)
    path = report["manifest_paths"][0]
    payload = json.loads((checkout.repo / path).read_text())
    payload["retired_source_metadata"]["receipt_sha256"] = "0" * 64
    cli._sign_applied_encoding_manifest(payload, BROKER)
    (checkout.repo / path).write_text(json.dumps(payload))
    assert guard(checkout, changed(report))
    payload = copy.deepcopy(payload)
    payload["axiom_encode_git"]["commit"] = "f" * 40
    assert cli._applied_manifest_tool_execution_issues(
        payload, manifest_label=path, expected_encoder_identity=ENCODER
    )


def test_I6_stale_receipt_cannot_be_transplanted_to_another_protected_base(
    checkout, capsys
):
    report = migrate(checkout, capsys)
    assert (
        cli.guard_generated_change_issues(
            checkout.repo,
            corpus_path=checkout.args.corpus_path,
            changed_files=changed(report),
            base_ref=checkout.base,
        )
        == []
    )
    (checkout.repo / "README.md").write_text("intervening base commit\n")
    git(checkout.repo, "add", "README.md")
    git(checkout.repo, "commit", "-m", "intervening base")
    new_base = git(checkout.repo, "rev-parse", "HEAD")
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        changed_files=changed(report),
        base_ref=new_base,
    )
    assert any(
        "base_commit does not match the protected base" in issue for issue in issues
    )


def test_dry_run_refusal_and_wrong_base_leave_bytes_unchanged(checkout):
    original = (checkout.repo / PRIMARY).read_bytes()
    payload = json.loads(checkout.args.plan.read_text())
    payload["base_commit"] = "f" * 40
    checkout.args.plan.write_text(json.dumps(payload))
    checkout.args.dry_run = True
    with pytest.raises(SystemExit, match="does not match RuleSpec HEAD"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert (checkout.repo / PRIMARY).read_bytes() == original
    assert not (checkout.repo / RECEIPT_DIR).exists()


def test_install_receipt_allowlist_is_one_digest_json(checkout):
    assert cli._is_canonical_apply_transaction_target(
        checkout.repo, RECEIPT_DIR / ("a" * 64 + ".json")
    )
    for tail in ("history.json", "a" * 64 + ".yaml", "nested/" + "a" * 64 + ".json"):
        assert not cli._is_canonical_apply_transaction_target(
            checkout.repo, RECEIPT_DIR / tail
        )


@pytest.mark.parametrize("owner", ["overlapping", "shrinking"])
def test_migration_refuses_stale_or_shrinking_existing_v5_ownership(checkout, owner):
    other = "us/policies/other.yaml"
    owner_primary = PRIMARY if owner == "shrinking" else other
    path = cli._applied_encoding_manifest_path(Path(owner_primary))
    target = checkout.repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    covered = [PRIMARY, COMPANION]
    if owner == "shrinking":
        (checkout.repo / other).write_bytes((checkout.repo / PRIMARY).read_bytes())
        covered.append(other)
    fixtures = _GuardFixtures()
    fixtures.source_attestation = json.loads(checkout.owners[PRIMARY])[
        "source_attestation"
    ]
    payload = fixtures._model_manifest(
        [
            {
                "path": relative,
                "sha256": hashlib.sha256(
                    (checkout.repo / relative).read_bytes()
                ).hexdigest(),
            }
            for relative in covered
        ]
    )
    cli._sign_applied_encoding_manifest(payload, BROKER)
    target.write_text(json.dumps(payload) + "\n")
    commit_owner_changes(checkout)
    verified, _root, _digest, issues = (
        cli._load_verified_applied_encoding_manifest_payload(
            checkout.repo,
            path.as_posix(),
            signing_broker=PRIVATE.public_key(),
            expected_waiver_set_sha256=WAIVER,
            expected_encoder_identity=ENCODER,
            local_corpus_release=cli.load_rulespec_local_corpus_release(
                checkout.repo, checkout.args.corpus_path
            ),
            base_commit=checkout.base,
        )
    )
    assert issues == []
    assert verified is not None and verified["backend"] == "codex"
    before = repository_bytes(checkout.repo)
    original = (checkout.repo / PRIMARY).read_bytes()
    with pytest.raises(SystemExit, match="manifest ownership"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert (checkout.repo / PRIMARY).read_bytes() == original
    assert repository_bytes(checkout.repo) == before
    assert not (checkout.repo / RECEIPT_DIR).exists()


def test_receipt_postcheck_failure_rolls_back_every_file(checkout, monkeypatch):
    before = repository_bytes(checkout.repo)

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected replay refusal")

    monkeypatch.setattr(cli, "_retired_source_metadata_replay", fail)
    with pytest.raises(RuntimeError, match="injected replay refusal"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert repository_bytes(checkout.repo) == before
    assert not (checkout.repo / RECEIPT_DIR).exists()
    assert not (checkout.repo / ".axiom/.apply-transaction").exists()
