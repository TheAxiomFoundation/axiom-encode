"""Durable, per-lane publisher serialization across broker restarts.

The database is service-owned credential storage, never a runner artifact.
Time passing does not authorize takeover: a previous publisher's job must
have completed and both outstanding installation tokens must be revoked.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .apps import PublisherApps
from .identity import (
    IdentityRefusal,
    JobIdentity,
    JobPolicy,
    _check_id,
    jobs_for_attempt,
)


def owner(identity: JobIdentity) -> str:
    return ":".join(
        (
            identity.repository,
            identity.run_id,
            identity.run_attempt,
            identity.check_run_id,
        )
    )


class PublisherLeases:
    def __init__(self, database: Path, apps: PublisherApps, api):
        self.path, self.apps, self.api = database, apps, api
        if not database.is_absolute():
            raise IdentityRefusal("broker_state_path")
        directory = database.parent.stat()
        if (
            database.parent.is_symlink()
            or directory.st_uid != os.geteuid()
            or stat.S_IMODE(directory.st_mode) != 0o700
        ):
            raise IdentityRefusal("broker_state_directory")
        fd = os.open(database, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise IdentityRefusal("broker_state_permissions")
        finally:
            os.close(fd)
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS leases (lane TEXT PRIMARY KEY, owner TEXT "
                "NOT NULL, identity TEXT NOT NULL, policy TEXT NOT NULL, kind TEXT NOT "
                "NULL, chain_token TEXT NOT NULL, lane_token TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL, state TEXT NOT NULL)"
            )

    @contextmanager
    def _lock(self, suffix, *, nonblocking=False):
        fd = os.open(
            str(self.path) + suffix, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise IdentityRefusal("broker_state_permissions")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
            except BlockingIOError:
                raise IdentityRefusal("finalization_in_progress") from None
            yield
        finally:
            os.close(fd)

    def execution(self):
        # OS ownership survives other broker workers and releases on crash.
        # It spans the complete finalizer call, including token revocation.
        return self._lock(".execution", nonblocking=True)

    @contextmanager
    def _transaction(self):
        with self._lock(".lock"):
            with self._sql_transaction() as connection:
                yield connection

    @contextmanager
    def _sql_transaction(self):
        connection = sqlite3.connect(self.path, timeout=90)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _finished(self, row):
        identity = JobIdentity(**json.loads(row["identity"]))
        policy = JobPolicy(**json.loads(row["policy"]))
        jobs = jobs_for_attempt(
            self.api, identity.repository, identity.run_id, identity.run_attempt
        )
        matching = [
            job
            for job in jobs
            if isinstance(job, dict)
            and _check_id(job, identity.repository) == identity.check_run_id
            and job.get("name") == policy.job_name
            and str(job.get("run_id")) == identity.run_id
            and str(job.get("run_attempt")) == identity.run_attempt
        ]
        if len(matching) != 1 or matching[0].get("status") != "completed":
            raise IdentityRefusal("publication_in_progress")

    def _revoke(self, connection, row):
        # Persist BEFORE the first external DELETE. A crash/partial failure
        # must never make a partly revoked pair available for reissuance.
        connection.execute(
            "UPDATE leases SET state='revoking' WHERE lane=?", (row["lane"],)
        )
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        # A partial revocation refuses takeover. Idempotent revocation support
        # in the App transport permits recovery if the process crashed after
        # the first successful DELETE and before the transaction committed.
        self.apps.chain.revoke(row["chain_token"])
        self.apps.lane.revoke(row["lane_token"])

    def acquire(self, identity: JobIdentity, policy: JobPolicy, *, kind: str):
        if kind != "publish":
            raise IdentityRefusal("broker_use_finalization_context")
        with self.execution():
            return self._acquire(identity, policy, kind=kind)

    @contextmanager
    def finalization(self, identity: JobIdentity, policy: JobPolicy):
        with self.execution():
            tokens = self._acquire(identity, policy, kind="finalize")
            try:
                yield tokens
            finally:
                self._release(identity)

    def _acquire(self, identity: JobIdentity, policy: JobPolicy, *, kind: str):
        if kind not in {"publish", "finalize"} or policy.job_name != kind:
            raise IdentityRefusal("broker_lease_operation")
        if identity.repository != self.apps.lane.scope.repository:
            raise IdentityRefusal("broker_lease_lane")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE lane=?", (identity.repository,)
            ).fetchone()
            if row is not None:
                if row["owner"] == owner(identity):
                    if row["kind"] != kind or json.loads(row["policy"]) != asdict(
                        policy
                    ):
                        raise IdentityRefusal("broker_lease_identity")
                    if kind == "publish":
                        if (
                            row["state"] != "active"
                            or row["expires_at"] <= int(time.time()) + 60
                        ):
                            raise IdentityRefusal("broker_lease_release_required")
                        return row["chain_token"], row["lane_token"]
                    # Exclusive execution ownership proves no prior broker
                    # invocation remains alive; revoke crashed-job tokens.
                else:
                    self._finished(row)
                self._revoke(connection, row)
                connection.execute(
                    "DELETE FROM leases WHERE lane=?", (identity.repository,)
                )
            chain, lane = self.apps.issue()
            try:
                connection.execute(
                    "INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identity.repository,
                        owner(identity),
                        json.dumps(asdict(identity)),
                        json.dumps(asdict(policy)),
                        kind,
                        chain.value,
                        lane.value,
                        min(chain.expires_at, lane.expires_at),
                        "active",
                    ),
                )
                # Commit before returning credentials to a runner. A restart
                # must retain the lease of every token the runner could hold.
                connection.commit()
            except BaseException:
                self.apps.chain.revoke(chain.value)
                self.apps.lane.revoke(lane.value)
                raise
            return chain.value, lane.value

    def policy_for_release(self, identity_owner: str) -> JobPolicy:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT policy FROM leases WHERE owner=?", (identity_owner,)
            ).fetchone()
            if row is None:
                raise IdentityRefusal("broker_lease_missing")
            return JobPolicy(**json.loads(row["policy"]))

    def release(self, identity: JobIdentity):
        with self.execution():
            self._release(identity)

    def _release(self, identity: JobIdentity):
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE lane=? AND owner=?",
                (identity.repository, owner(identity)),
            ).fetchone()
            if row is None:
                raise IdentityRefusal("broker_lease_missing")
            self._revoke(connection, row)
            connection.execute(
                "DELETE FROM leases WHERE lane=?", (identity.repository,)
            )
