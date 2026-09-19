from dataclasses import replace

import pytest

from axiom_encode.notary.canonical import sha256_hex, strict_parse
from axiom_encode.notary.github_inputs import (
    BootstrapCeremony,
    Deployment,
    GitHubSignerInputs,
)
from axiom_encode.notary.identity import IdentityRefusal, JobIdentity

from .chain_fixtures import Epoch
from .lineage_fixtures import LANE
from .test_administration import genesis_args
from .test_protection import environment, ruleset
from .test_signer import archive


class API:
    def __init__(self, epoch, subject, candidate, report):
        self.epoch, self.subject = epoch, subject
        self.base = epoch.active
        self.chain_tip = epoch.history[-1].commit
        self.candidate, self.report = candidate, report
        self.env = environment()
        self.branches = {
            "total_count": 1,
            "branch_policies": [{"name": "main", "type": "branch"}],
        }
        self.prs = [
            {
                "number": 42,
                "state": "open",
                "head": {"sha": subject.commit},
                "base": {"ref": "main"},
            }
        ]
        self.chain_exists = False

    def archives(self):
        return {
            "1": archive(self.candidate, member="candidate.json"),
            "2": archive(self.report),
        }

    def get(self, path):
        if path.endswith("/environments/notary-signing"):
            return self.env
        if "/deployment-branch-policies?" in path:
            return self.branches
        if path.endswith("/rulesets/1"):
            body = ruleset(
                LANE,
                "refs/heads/main",
                ["pull_request", "deletion", "non_fast_forward"],
            )
            body["rules"].append(
                {
                    "type": "required_status_checks",
                    "parameters": {
                        "strict_required_status_checks_policy": True,
                        "required_status_checks": [
                            {"context": "Axiom notary admission", "integration_id": 456}
                        ],
                    },
                }
            )
            return body
        if path.endswith("/rulesets/2"):
            return ruleset(
                LANE + "-notary",
                "refs/heads/chain",
                ["creation", "update"],
                identity=2,
                bypass=[
                    {
                        "actor_id": 123,
                        "actor_type": "Integration",
                        "bypass_mode": "always",
                    }
                ],
            )
        if path.endswith("/rulesets/3"):
            return ruleset(
                LANE + "-notary",
                "refs/heads/chain",
                ["deletion", "non_fast_forward", "required_linear_history"],
                identity=3,
            )
        if path.endswith("/rulesets/4"):
            return ruleset(
                LANE,
                "refs/heads/main",
                ["update"],
                identity=4,
                bypass=[
                    {"actor_id": 9, "actor_type": "Team", "bypass_mode": "pull_request"}
                ],
            )
        if path.endswith("/teams/bootstrap"):
            return {"id": 9}
        if path.endswith("/git/ref/heads/main"):
            return {
                "ref": "refs/heads/main",
                "object": {"type": "commit", "sha": self.base.commit},
            }
        if path.endswith("/git/ref/heads/chain"):
            return {
                "ref": "refs/heads/chain",
                "object": {"type": "commit", "sha": self.chain_tip},
            }
        if "/artifacts?" in path:
            return {
                "total_count": 2,
                "artifacts": [
                    {
                        "id": int(i),
                        "name": "axiom-notary-candidate"
                        if i == "1"
                        else "axiom-notary-report",
                        "expired": False,
                        "workflow_run": {"id": 1},
                        "digest": "sha256:" + sha256_hex(raw),
                    }
                    for i, raw in self.archives().items()
                ],
            }
        raise AssertionError(path)

    def collection(self, path):
        if path.endswith("/teams/bootstrap/members"):
            return [{"id": 123}]
        return self.prs

    def archive(self, repository, artifact_id):
        assert repository == LANE
        return self.archives()[artifact_id]

    def optional_ref(self, repository, branch):
        assert repository == LANE + "-notary" and branch == "chain"
        return {"ref": "refs/heads/chain"} if self.chain_exists else None


@pytest.fixture
def adapter(monkeypatch):
    epoch = Epoch.create()
    _, bundle, subject = epoch.receipt()
    candidate = next(
        raw
        for name, raw in bundle.items()
        if name.endswith(".json")
        and strict_parse(raw)["schema"] == "axiom/notary-receipt-candidate/v1"
    )
    report = next(
        raw
        for name, raw in bundle.items()
        if name.endswith(".json")
        and strict_parse(raw)["schema"] == "axiom/notary-report-pass/v1"
    )
    api = API(epoch, subject, candidate, report)
    config = Deployment(
        LANE,
        "101",
        "102",
        epoch.anchor.epoch_sha256,
        "main",
        ".axiom/notary/consumer.json",
        ".github/workflows/notary.yml",
        "https://notary.example.test/sign",
        frozenset([123]),
        1,
        2,
        3,
        123,
        456,
        "Axiom notary admission",
        epoch.inventory,
    )
    reader = GitHubSignerInputs(config, api)
    identity = JobIdentity(
        LANE,
        LANE + "/.github/workflows/notary.yml@refs/heads/main",
        epoch.active.commit,
        "refs/heads/main",
        "1",
        "1",
        "2",
        "notary-signing",
        "fixture",
    )

    class Repository:
        def __init__(self, repository, **_):
            self.repository = repository

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def fetch(self, ref):
            if ref == "refs/heads/chain":
                return api.chain_tip
            return api.base.commit if ref == "refs/heads/main" else api.subject.commit

        def snapshot(self, commit):
            if commit == api.base.commit:
                return api.base
            assert commit == api.subject.commit
            return api.subject

        def chain_history(self, commit):
            assert commit == api.chain_tip
            return epoch.history

    monkeypatch.setattr(
        "axiom_encode.notary.github_inputs.RemoteRepository", Repository
    )
    return reader, api, identity, epoch


def test_adapter_fetches_exact_remote_evidence(adapter):
    reader, api, identity, epoch = adapter
    inputs = reader.receipt(identity, sha256_hex(api.candidate))
    assert inputs.base == epoch.active and inputs.subject == api.subject
    assert inputs.candidate_raw == api.candidate and inputs.pr_number == "42"
    assert inputs.state.activated
    assert reader.job_policy().workflow_sha_git_oid == epoch.active.commit


@pytest.mark.parametrize(
    "mutation",
    [
        "environment-bypass",
        "base-moved",
        "head-moved",
        "duplicate-pr",
        "candidate-swapped",
        "bad-chain",
    ],
)
def test_adapter_refuses_changed_control_plane_or_evidence(adapter, mutation):
    reader, api, identity, epoch = adapter
    address = sha256_hex(api.candidate)
    if mutation == "environment-bypass":
        api.env["can_admins_bypass"] = True
    elif mutation == "base-moved":
        api.base = replace(api.base, commit="d" * 40)
    elif mutation == "head-moved":
        api.subject = replace(api.subject, commit="d" * 40)
    elif mutation == "duplicate-pr":
        api.prs *= 2
    elif mutation == "candidate-swapped":
        api.candidate = b"{}"
    else:
        epoch.history = epoch.history[:-1]
    with pytest.raises(IdentityRefusal):
        reader.receipt(identity, address)


def test_locked_bootstrap_reads_no_caller_selected_tree(adapter):
    reader, api, identity, epoch = adapter
    reader._ceremony = BootstrapCeremony(genesis_args(epoch), 4, 9, "bootstrap", 123)
    api.base = epoch.base
    api.candidate = epoch.history[0].blobs[epoch.anchor.epoch_sha256 + ".json"]
    identity = replace(identity, workflow_sha_git_oid=epoch.base.commit)
    inputs = reader.genesis(identity, epoch.anchor.epoch_sha256)
    assert inputs.snapshot == epoch.base
    assert inputs.arguments == reader._ceremony.arguments


def test_second_genesis_refuses_even_with_ceremony_and_lock(adapter):
    reader, api, identity, epoch = adapter
    reader._ceremony = BootstrapCeremony(genesis_args(epoch), 4, 9, "bootstrap", 123)
    api.chain_exists = True
    with pytest.raises(IdentityRefusal, match="second_genesis"):
        reader.genesis(identity, epoch.anchor.epoch_sha256)


def test_genesis_cannot_invent_ceremony_configuration(adapter):
    reader, _, identity, epoch = adapter
    with pytest.raises(IdentityRefusal, match="ceremony_required"):
        reader.genesis(identity, epoch.anchor.epoch_sha256)
