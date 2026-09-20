"""Concrete chain CAS and App-owned checks for the separate publisher.

These adapters consume service-built plans; no route accepts serialized plans
from callers. The chain token cannot write lane content. Git writes only the
literal chain ref in a fresh private bare repository without candidate code.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from urllib.parse import urlencode

from ._schema import digest
from .apps import GitHubAppAPI
from .identity import IdentityRefusal
from .protocol import decimal_id, oid
from .publication import PublicationPlan
from .remote import RemoteRepository
from .verification import Snapshot

_NAME = re.compile(
    r"(?:HEAD\.json|[0-9a-f]{64}\.(?:raw|json(?:\.(?:genesis|transition|admin-approver|approver|notary)\.sig)?))"
)


class ChainWriter:
    def __init__(self, repository: str, token: str):
        if not repository.endswith("-notary"):
            raise IdentityRefusal("publisher_chain_repository")
        self.repository, self._token = repository, token

    def _repository(self):
        return RemoteRepository(self.repository, read_token=self._token)

    @staticmethod
    def _commit(repo: RemoteRepository, files: Mapping[str, bytes], parent: str | None):
        entries = []
        for path, raw in sorted(files.items()):
            if not _NAME.fullmatch(path) or not isinstance(raw, bytes):
                raise IdentityRefusal("publisher_chain_path")
            actual = repo._run("hash-object", "-w", "--stdin", input_bytes=raw).strip()
            expected = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
            if actual.decode() != expected:
                raise IdentityRefusal("publisher_blob_identity")
            entries.append(b"100644 blob " + actual + b"\t" + path.encode() + b"\n")
        tree = repo._run("mktree", input_bytes=b"".join(entries)).decode().strip()
        args = (
            "-c",
            "user.name=Axiom notary publisher",
            "-c",
            "user.email=notary@axiom.foundation",
            "commit-tree",
            tree,
        )
        if parent is not None:
            args += ("-p", parent)
        commit = repo._run(*args, input_bytes=b"Publish verified notary state\n")
        result = commit.decode().strip()
        if not oid(result):
            raise IdentityRefusal("publisher_commit_identity")
        return result

    @staticmethod
    def _push(repo: RemoteRepository, commit: str, expected: str | None):
        # The new commit is always a direct child of expected (or an orphan
        # bootstrap history). The lease supplies an explicit expected-old-OID
        # CAS; it does not create a non-fast-forward update. Server rulesets
        # independently prohibit force updates/deletion, including by the App.
        outcome = repo._run(
            "push",
            "--porcelain",
            "--force-with-lease=refs/heads/chain:" + (expected or ""),
            "origin",
            commit + ":refs/heads/chain",
        )
        # Git reports '=' and exits zero if the desired commit is already
        # current, even when an expected-absent lease did not win creation.
        # Never confuse that no-op with this request's successful CAS.
        updates = [line for line in outcome.splitlines() if b"\t" in line]
        expected_flag = b"*" if expected is None else b" "
        if (
            len(updates) != 1
            or updates[0].split(b"\t", 1)[0] != expected_flag
            or updates[0].split(b"\t")[1] != (commit + ":refs/heads/chain").encode()
        ):
            raise IdentityRefusal("publication_cas_lost")
        observed = repo._run("ls-remote", "--refs", "origin", "refs/heads/chain")
        if observed != (commit + "\trefs/heads/chain\n").encode():
            raise IdentityRefusal("publication_moved_or_uncertain")

    def publish(self, plan: PublicationPlan, previous: Snapshot) -> str:
        if plan.expected_chain_commit != previous.commit:
            raise IdentityRefusal("publication_plan_base")
        with self._repository() as repo:
            if repo.fetch("refs/heads/chain") != previous.commit:
                raise IdentityRefusal("publication_cas_lost")
            remote = repo.snapshot(previous.commit)
            if remote.manifest != previous.manifest or remote.blobs != previous.blobs:
                raise IdentityRefusal("publication_base_bytes")
            if not plan.files:
                return previous.commit
            merged = dict(previous.blobs)
            for path, raw in plan.files.items():
                if path != "HEAD.json" and path in merged and merged[path] != raw:
                    raise IdentityRefusal("publication_rewrite")
                merged[path] = raw
            commit = self._commit(repo, merged, previous.commit)
            self._push(repo, commit, previous.commit)
            return commit

    def bootstrap(self, first: Mapping[str, bytes], final: Mapping[str, bytes]) -> str:
        """Install the validated genesis bundle and finalization as two commits.

        The broker audits the locked lane and fully reconstructs these exact
        two snapshots before calling. Expected-absent CAS rejects every second
        genesis, including concurrent first pushes. No preliminary branch write
        or default-branch initialization is needed.
        """
        if "HEAD.json" in first or "HEAD.json" not in final or set(first) & set(final):
            raise IdentityRefusal("bootstrap_commit_partition")
        with self._repository() as repo:
            first_commit = self._commit(repo, first, None)
            commit = self._commit(repo, dict(first) | dict(final), first_commit)
            self._push(repo, commit, None)
            return commit


class AdmissionChecks:
    """One required check per subject, always from the lane App itself."""

    def __init__(
        self, api: GitHubAppAPI, repository: str, app_id: int, name: str, token
    ):
        self.api, self.repository, self.app_id, self.name, self._token = (
            api,
            repository,
            app_id,
            name,
            token,
        )

    def _checks(self, head: str) -> list[dict]:
        if not oid(head):
            raise IdentityRefusal("check_subject")
        rows = []
        for page in range(1, 101):
            query = urlencode(
                {
                    "check_name": self.name,
                    "filter": "all",
                    "per_page": 100,
                    "page": page,
                }
            )
            result = self.api.request(
                "GET",
                f"/repos/{self.repository}/commits/{head}/check-runs?{query}",
                self._token,
            )
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("check_runs"), list)
                or type(result.get("total_count")) is not int
            ):
                raise IdentityRefusal("check_listing")
            rows.extend(result["check_runs"])
            if len(rows) == result["total_count"]:
                break
            if len(rows) > result["total_count"] or not result["check_runs"]:
                raise IdentityRefusal("check_listing")
        else:
            raise IdentityRefusal("check_listing")
        ours = [
            row
            for row in rows
            if isinstance(row, dict)
            and row.get("name") == self.name
            and isinstance(row.get("app"), dict)
            and row["app"].get("id") == self.app_id
        ]
        if any(
            row.get("head_sha") != head or not decimal_id(str(row.get("id")))
            for row in ours
        ) or len({row["id"] for row in ours}) != len(ours):
            raise IdentityRefusal("check_identity")
        return ours

    def target(self, head: str, epoch: str, *, include_invalidated=False) -> str:
        rows = self._checks(head)
        if not rows:
            raise IdentityRefusal("required_check_missing")
        prefix = "axiom-notary:" + epoch + ":"
        targets = set()
        for row in rows:
            value = row.get("external_id")
            if (
                row.get("status") != "completed"
                or row.get("conclusion")
                not in (
                    {"success", "action_required"}
                    if include_invalidated
                    else {"success"}
                )
                or not isinstance(value, str)
                or not value.startswith(prefix)
                or not digest(value.removeprefix(prefix))
            ):
                raise IdentityRefusal("required_check_not_admitted")
            targets.add(value.removeprefix(prefix))
        if len(targets) != 1:
            raise IdentityRefusal("required_check_target_ambiguous")
        return targets.pop()

    def write(self, head: str, epoch: str, target: str, *, admitted: bool) -> str:
        if not digest(epoch) or not digest(target):
            raise IdentityRefusal("check_artifact_identity")
        body = {
            "name": self.name,
            "external_id": f"axiom-notary:{epoch}:{target}",
            "status": "completed",
            "conclusion": "success" if admitted else "action_required",
            "output": {
                "title": "Notary admission"
                if admitted
                else "Notary admission invalidated",
                "summary": (
                    f"Epoch `{epoch}`; artifact `{target}`. "
                    + (
                        "Pending merge; exact subject and finalized base verified."
                        if admitted
                        else "Reverify from the current finalized base."
                    )
                ),
            },
        }
        prefix = f"/repos/{self.repository}/check-runs"
        # GitHub has no conditional create for checks. Repair same-App
        # duplicates instead of making a duplicate permanently unmodifiable.
        # The pinned admission/finalizer workflows also share a lane-wide
        # concurrency group; differently targeted writes must be serialized.
        for _ in range(3):
            rows = self._checks(head)
            if rows:
                for row in rows:
                    self.api.request(
                        "PATCH", prefix + "/" + str(row["id"]), self._token, body=body
                    )
            else:
                self.api.request(
                    "POST",
                    prefix,
                    self._token,
                    body=body | {"head_sha": head},
                    status=201,
                )
            actual = self._checks(head)
            if actual and all(
                row.get("external_id") == body["external_id"]
                and row.get("status") == "completed"
                and row.get("conclusion") == body["conclusion"]
                for row in actual
            ):
                return str(max(row["id"] for row in actual))
        raise IdentityRefusal("check_write_uncertain")

    def invalidate(self, head: str, epoch: str, target: str):
        """Invalidate only this artifact's checks, preserving a winning sibling."""
        if not digest(epoch) or not digest(target):
            raise IdentityRefusal("check_artifact_identity")
        external = f"axiom-notary:{epoch}:{target}"
        for row in self._checks(head):
            if row.get("external_id") == external:
                self.api.request(
                    "PATCH",
                    f"/repos/{self.repository}/check-runs/{row['id']}",
                    self._token,
                    body={
                        "status": "completed",
                        "conclusion": "action_required",
                        "output": {
                            "title": "Notary artifact void",
                            "summary": f"Artifact `{target}` is permanently void; reverify from the finalized base.",
                        },
                    },
                )
        if any(
            row.get("external_id") == external and row.get("conclusion") == "success"
            for row in self._checks(head)
        ):
            raise IdentityRefusal("check_invalidation_uncertain")
