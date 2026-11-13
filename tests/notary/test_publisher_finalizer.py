"""Signed enrollment -> real Git publication -> actual merge/finalization.

Only GitHub's HTTP control plane and repository origins are test adapters.
Git object writes, snapshots, signatures, manifests and reconstruction are real.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from axiom_encode.notary.bundle import BUNDLE_ARTIFACT, pack_bundle
from axiom_encode.notary.canonical import sha256_hex, strict_parse
from axiom_encode.notary.finalizer import Finalizer
from axiom_encode.notary.github_inputs import (
    CANDIDATE_ARTIFACT,
    Deployment,
    GitHubSignerInputs,
)
from axiom_encode.notary.github_publication import ChainWriter
from axiom_encode.notary.identity import IdentityRefusal, authenticate_job
from axiom_encode.notary.lineage import STORE_PREFIX
from axiom_encode.notary.producers import ENROLLMENT_PATH, Enrollment
from axiom_encode.notary.provenance import REPORT_ARTIFACT
from axiom_encode.notary.publisher import Publisher
from axiom_encode.notary.remote import RemoteRepository
from axiom_encode.notary.signer import NotarySigner, ReceiptInputs, make_candidate
from axiom_encode.notary.verification import verify_snapshots

from .chain_fixtures import Epoch
from .lineage_fixtures import LANE, generation
from .test_github_publication import CheckAPI
from .test_identity import claims, token
from .test_identity import policy as _policy_fixture
from .test_identity import rsa_key as _rsa_fixture
from .test_producers import enrollment_base
from .test_signer import API as SignerAPI
from .test_signer import archive

policy = _policy_fixture
rsa_key = _rsa_fixture


@pytest.fixture
def deployed(tmp_path, monkeypatch, policy, rsa_key):
    origins = {}
    for repository in (LANE, LANE + "-notary"):
        with RemoteRepository(repository) as setup:
            origin = tmp_path / (repository.split("/")[1] + ".git")
            setup._run("init", "--bare", str(origin), in_repo=False)
            origins[repository] = origin

    def connect(repository, **kwargs):
        result = RemoteRepository(repository)
        result._run("config", "remote.origin.url", str(origins[repository]))
        result._run("config", "protocol.file.allow", "always")
        return result

    # Git tree construction is ordinary 100644 in this fixture; production
    # ChainWriter separately rejects non-chain paths before its push.
    parent = None

    def lane_snapshot(label, blobs):
        nonlocal parent
        with connect(LANE) as repo:
            if parent:
                repo.fetch("refs/heads/main")

            # Use git's index-independent mktree recursively for lane paths.
            def tree(values):
                children = {}
                for name, data in values.items():
                    first, sep, rest = name.partition("/")
                    if sep:
                        children.setdefault(first, {})[rest] = data
                    else:
                        children[first] = data
                rows = []
                for name, data in sorted(children.items()):
                    if isinstance(data, dict):
                        mode, kind, oid = "040000", "tree", tree(data)
                    else:
                        mode, kind = "100644", "blob"
                        oid = (
                            repo._run("hash-object", "-w", "--stdin", input_bytes=data)
                            .decode()
                            .strip()
                        )
                    rows.append(f"{mode} {kind} {oid}\t{name}\0".encode())
                return (
                    repo._run("mktree", "-z", input_bytes=b"".join(rows))
                    .decode()
                    .strip()
                )

            repo._environment.update(
                GIT_AUTHOR_NAME="Fixture",
                GIT_AUTHOR_EMAIL="fixture@example.test",
                GIT_COMMITTER_NAME="Fixture",
                GIT_COMMITTER_EMAIL="fixture@example.test",
            )
            args = ["commit-tree", tree(blobs), "-m", label]
            if parent:
                args += ["-p", parent]
            commit = repo._run(*args).decode().strip()
            repo._run("push", "origin", commit + ":refs/heads/main")
            parent = commit
            return repo.snapshot(commit)

    epoch = Epoch.create(prepare_base=enrollment_base, lane_snapshot=lane_snapshot)
    with connect(LANE + "-notary") as repo:
        last = None
        for index, state in enumerate(epoch.history):
            last = ChainWriter._commit(repo, state.blobs, last)
            epoch.history[index] = replace(state, commit=last)
        repo._run("push", "origin", last + ":refs/heads/chain")
    entry = strict_parse(epoch.active.blobs[ENROLLMENT_PATH])["runtimes"][0]
    from axiom_encode.notary.canonical import jcs_dumps

    enrollment = Enrollment(jcs_dumps(entry))
    record = generation() | {
        "epoch_sha256": epoch.anchor.epoch_sha256,
        "runtime_identity": enrollment.runtime_identity,
        "cli_version": "fixture-cli",
    }
    subject = lane_snapshot(
        "subject",
        epoch.active.blobs
        | {"rules/example.yaml": b"generated\n"}
        | {
            STORE_PREFIX + name: file.raw
            for name, file in epoch.identities.store(record).items()
        },
    )
    with connect(LANE) as repo:
        repo.fetch("refs/heads/main")
        repo._run("push", "origin", subject.commit + ":refs/pull/42/head")
        repo._run("push", "--force", "origin", epoch.active.commit + ":refs/heads/main")
    parent = epoch.active.commit
    report = verify_snapshots(
        epoch.active,
        subject,
        epoch.state().predecessor(),
        epoch.inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )
    policy = replace(policy, workflow_sha_git_oid=epoch.active.commit)

    class API(SignerAPI):
        archives = {}

        def get(self, path):
            if "/git/ref/heads/" in path:
                repository, branch = path.removeprefix("/repos/").split(
                    "/git/ref/heads/"
                )
                with connect(repository) as repo:
                    tip = repo.fetch("refs/heads/" + branch)
                return {
                    "ref": "refs/heads/" + branch,
                    "object": {"type": "commit", "sha": tip},
                }
            return super().get(path)

        def collection(self, path):
            assert "/commits/" in path and path.endswith("/pulls")
            return [deepcopy(self.pr | {"number": 42})]

        def archive(self, repository, artifact_id):
            assert repository == LANE
            return self.archives[artifact_id]

        def put(self, name, raw, member):
            data = archive(raw, member=member)
            number = str(len(self.artifacts) + 1)
            self.artifacts.append(
                {
                    "id": int(number),
                    "name": name,
                    "expired": False,
                    "workflow_run": {"id": 1},
                    "digest": "sha256:" + sha256_hex(data),
                }
            )
            self.archives[number] = data
            return data

    api = API(policy)
    api.artifacts = []
    api.pr["head"]["sha"] = subject.commit
    api.pr["base"].update(sha=epoch.active.commit, ref="main")
    for number, name in ((3, "verify"), (4, "recompute")):
        api.jobs.append(
            {
                "name": name,
                "run_id": 1,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
                "check_run_url": f"https://api.github.com/repos/{LANE}/check-runs/{number}",
            }
        )
    report_archive = api.put(REPORT_ARTIFACT, report, "report.json")
    config = Deployment(
        LANE,
        "101",
        "102",
        epoch.anchor.epoch_sha256,
        "main",
        ".axiom/notary/consumer.json",
        policy.workflow_path,
        policy.audience,
        frozenset({123}),
        1,
        2,
        3,
        43,
        44,
        "Axiom notary admission",
        epoch.inventory,
    )
    inputs = GitHubSignerInputs(config, api)
    audits = []
    inputs._audit = lambda environment="notary-signing": audits.append(environment)
    for module in ("github_inputs", "publisher", "finalizer"):
        monkeypatch.setattr(f"axiom_encode.notary.{module}.RemoteRepository", connect)
    monkeypatch.setattr(
        ChainWriter, "_repository", lambda writer: connect(writer.repository)
    )
    monkeypatch.setattr("axiom_encode.notary.publisher.GitHubReader", lambda _: api)
    monkeypatch.setattr("axiom_encode.notary.finalizer.GitHubReader", lambda _: api)
    monkeypatch.setattr(
        "jwt.PyJWKClient.get_signing_key_from_jwt",
        lambda *_: SimpleNamespace(key=rsa_key.public_key()),
    )
    signed_oidc = token(claims(policy) | {"workflow_sha": epoch.active.commit}, rsa_key)
    identity = authenticate_job(signed_oidc, policy, api)
    material = ReceiptInputs(
        epoch.active, subject, epoch.state(), epoch.inventory, report_archive, b"", "42"
    )
    candidate = make_candidate(api, identity, material, require_recompute=True)
    api.put(CANDIDATE_ARTIFACT, candidate, "candidate.json")
    bundle = NotarySigner(epoch.identities.keys["notary"], policy, api, inputs).receipt(
        signed_oidc,
        sha256_hex(candidate),
        epoch.identities.sidecar(candidate, "approver"),
    )
    receipt = next(
        n[:-5]
        for n, raw in bundle.items()
        if n.endswith(".json")
        and strict_parse(raw).get("schema") == "axiom/notary-receipt/v1"
    )
    api.put(BUNDLE_ARTIFACT, pack_bundle("receipt", receipt, bundle), "bundle.json")
    api.jobs[0].update(status="completed", conclusion="success")
    check_api = CheckAPI()
    publisher_identity = replace(
        identity, environment="notary-publishing", check_run_id="5"
    )
    return SimpleNamespace(
        inputs=inputs,
        api=api,
        epoch=epoch,
        connect=connect,
        publisher=Publisher(inputs, app_api=check_api),
        finalizer=Finalizer(inputs, app_api=check_api),
        checks=check_api,
        identity=publisher_identity,
        receipt=receipt,
        subject=subject,
        lane_snapshot=lane_snapshot,
        audits=audits,
    )


def test_signed_local_emission_publishes_and_finalizes_actual_merge(deployed):
    d = deployed
    result = d.publisher.publish(d.identity, "chain", "fixture-only")
    assert result["artifact_sha256"] == d.receipt
    assert d.checks.rows[0]["conclusion"] == "success"
    merged = d.lane_snapshot("merge", d.subject.blobs)
    d.api.pr.update(state="closed", merged=True, merge_commit_sha=merged.commit)
    identity = replace(d.identity, workflow_sha_git_oid=merged.commit)
    final = d.finalizer.finalize(identity, "chain", "fixture-only")
    assert final["state"] == "finalized"
    assert final["artifact_sha256"] == d.receipt
    assert (
        d.finalizer.finalize(identity, "chain", "fixture-only")["state"]
        == "already-finalized"
    )
    assert "notary-publishing" in d.audits


@pytest.mark.parametrize("change", ["outsider", "read-only", "fork", "moved-head"])
def test_publisher_rereads_live_writer_before_any_write(deployed, change):
    d = deployed
    if change == "outsider":
        d.api.pr["user"]["id"] = 999
    elif change == "read-only":
        d.api.permission["permission"] = "read"
    elif change == "fork":
        d.api.pr["head"]["repo"]["full_name"] = "outsider/fork"
    else:
        d.api.pr["head"]["sha"] = "f" * 40
    with pytest.raises(IdentityRefusal):
        d.publisher.publish(d.identity, "chain", "fixture-only")
    assert d.checks.rows == []
    with d.connect(LANE + "-notary") as repo:
        tip = repo.fetch("refs/heads/chain")
        assert d.receipt + ".json" not in repo.snapshot(tip).blobs


def test_revoked_writer_is_voided_after_merge(deployed):
    d = deployed
    d.publisher.publish(d.identity, "chain", "fixture-only")
    merged = d.lane_snapshot("merge", d.subject.blobs)
    d.api.pr.update(state="closed", merged=True, merge_commit_sha=merged.commit)
    d.api.permission["permission"] = "read"
    final = d.finalizer.finalize(
        replace(d.identity, workflow_sha_git_oid=merged.commit), "chain", "fixture-only"
    )
    assert final["state"] == "void"
    assert d.checks.rows[0]["conclusion"] == "action_required"

    retry = d.finalizer.finalize(
        replace(d.identity, workflow_sha_git_oid=merged.commit), "chain", "fixture-only"
    )
    assert retry["state"] == "already-void"
    assert retry["chain_commit"] == final["chain_commit"]
