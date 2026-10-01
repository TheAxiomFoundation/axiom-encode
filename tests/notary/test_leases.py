import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from axiom_encode.notary.apps import InstallationToken
from axiom_encode.notary.identity import IdentityRefusal, JobIdentity, JobPolicy
from axiom_encode.notary.leases import PublisherLeases, owner

from .lineage_fixtures import LANE


class Apps:
    def __init__(self):
        self.revoked, self.issued = [], 0
        self.chain = SimpleNamespace(revoke=lambda value: self.revoked.append(value))
        self.lane = SimpleNamespace(
            scope=SimpleNamespace(repository=LANE),
            revoke=lambda value: self.revoked.append(value),
        )

    def issue(self):
        self.issued += 1
        return tuple(
            InstallationToken(f"{kind}-{self.issued}", int(time.time()) + 3600)
            for kind in ("chain", "lane")
        )


class API:
    completed = False

    def get(self, path):
        return {
            "total_count": 1,
            "jobs": [
                {
                    "name": "publish",
                    "run_id": 1,
                    "run_attempt": 1,
                    "check_run_url": f"https://api.github.com/repos/{LANE}/check-runs/2",
                    "status": "completed" if self.completed else "in_progress",
                }
            ],
        }


@pytest.fixture
def lease(tmp_path):
    tmp_path.chmod(0o700)
    apps, api = Apps(), API()
    store = PublisherLeases(tmp_path / "state.db", apps, api)
    policy = JobPolicy(
        LANE,
        "101",
        "102",
        ".github/workflows/notary.yml",
        "a" * 40,
        "refs/heads/main",
        "notary-publishing",
        "fixture",
        "publish",
    )
    identity = JobIdentity(
        LANE,
        LANE + "/.github/workflows/notary.yml@refs/heads/main",
        "a" * 40,
        "refs/heads/main",
        "1",
        "1",
        "2",
        "notary-publishing",
        "nonce",
    )
    return store, identity, policy, apps, api


def test_restart_preserves_an_active_grant_and_blocks_concurrent_publisher(lease):
    store, identity, policy, apps, api = lease
    assert store.acquire(identity, policy, kind="publish") == ("chain-1", "lane-1")
    restarted = PublisherLeases(store.path, apps, api)
    assert restarted.acquire(
        replace(identity, jti="fresh-oidc"), policy, kind="publish"
    ) == ("chain-1", "lane-1")
    with pytest.raises(IdentityRefusal, match="in_progress"):
        restarted.acquire(
            replace(identity, run_id="3", check_run_id="4"), policy, kind="publish"
        )
    assert apps.issued == 1 and not apps.revoked


def test_completed_job_tokens_are_revoked_before_new_grant(lease):
    store, identity, policy, apps, api = lease
    store.acquire(identity, policy, kind="publish")
    api.completed = True
    assert store.acquire(
        replace(identity, run_id="3", check_run_id="4"), policy, kind="publish"
    ) == ("chain-2", "lane-2")
    assert apps.revoked == ["chain-1", "lane-1"]


def test_release_after_base_movement_uses_original_policy_and_revokes(lease):
    store, identity, policy, apps, _ = lease
    store.acquire(identity, policy, kind="publish")
    assert store.policy_for_release(owner(identity)) == policy
    store.release(identity)
    assert apps.revoked == ["chain-1", "lane-1"]
    with pytest.raises(IdentityRefusal, match="missing"):
        store.policy_for_release(owner(identity))


def test_wrong_job_cannot_release_a_live_lease(lease):
    store, identity, policy, apps, _ = lease
    store.acquire(identity, policy, kind="publish")
    with pytest.raises(IdentityRefusal, match="missing"):
        store.release(replace(identity, check_run_id="99"))
    assert not apps.revoked


def test_partial_revocation_retains_lease_for_idempotent_recovery(lease):
    store, identity, policy, apps, _ = lease
    store.acquire(identity, policy, kind="publish")
    original = apps.lane.revoke

    def unavailable(value):
        raise IdentityRefusal("network")

    apps.lane.revoke = unavailable
    with pytest.raises(IdentityRefusal, match="network"):
        store.release(identity)
    assert store.policy_for_release(owner(identity)) == policy
    with pytest.raises(IdentityRefusal, match="release_required"):
        store.acquire(identity, policy, kind="publish")
    apps.lane.revoke = original
    store.release(identity)
    assert apps.revoked == ["chain-1", "chain-1", "lane-1"]


def test_broker_state_requires_a_private_directory(tmp_path):
    tmp_path.chmod(0o755)
    with pytest.raises(IdentityRefusal, match="directory"):
        PublisherLeases(tmp_path / "state.db", Apps(), API())


def test_expired_grant_refuses_until_release(lease, monkeypatch):
    store, identity, policy, apps, _ = lease
    store.acquire(identity, policy, kind="publish")
    future = int(time.time()) + 3601
    monkeypatch.setattr("axiom_encode.notary.leases.time.time", lambda: future)
    with pytest.raises(IdentityRefusal, match="release_required"):
        store.acquire(identity, policy, kind="publish")
    store.release(identity)
    assert store.acquire(identity, policy, kind="publish") == ("chain-2", "lane-2")


def test_finalizer_execution_exclusive_across_instances(lease):
    store, _, _, apps, api = lease
    restarted = PublisherLeases(store.path, apps, api)
    with store.execution():
        with pytest.raises(IdentityRefusal, match="finalization_in_progress"):
            with restarted.execution():
                pytest.fail("concurrent finalization")
    with restarted.execution():
        pass


def test_completed_workflow_does_not_allow_takeover_of_running_broker(lease):
    store, identity, policy, apps, api = lease
    with store.finalization(identity, replace(policy, job_name="finalize")):
        api.completed = True
        with pytest.raises(IdentityRefusal, match="finalization_in_progress"):
            store.acquire(
                replace(identity, run_id="3", check_run_id="4"), policy, kind="publish"
            )
        assert apps.revoked == []
    assert apps.revoked == ["chain-1", "lane-1"]


def test_crashed_finalizer_grant_is_revoked_before_same_job_retry(lease):
    store, identity, policy, apps, api = lease
    final = replace(policy, job_name="finalize")
    with store.execution():
        store._acquire(identity, final, kind="finalize")
    with store.finalization(identity, final) as tokens:
        assert tokens == ("chain-2", "lane-2")
        assert apps.revoked == ["chain-1", "lane-1"]
