"""Run-scoped artifact provenance, read from GitHub rather than runner claims."""

from __future__ import annotations

import io
import zipfile

from .canonical import sha256_hex
from .identity import IdentityRefusal, JobIdentity, ReadAPI, _check_id, jobs_for_attempt
from .protocol import decimal_id

REPORT_ARTIFACT = "axiom-notary-report"
REPORT_MEMBER = "report.json"
MAX_ARCHIVE = 8_000_000


def artifact_for_run(api: ReadAPI, repository: str, run_id: str, name: str) -> dict:
    artifacts = []
    for page in range(1, 101):
        result = api.get(
            f"/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"
        )
        if (
            not isinstance(result.get("artifacts"), list)
            or type(result.get("total_count")) is not int
        ):
            raise IdentityRefusal("artifact_metadata")
        artifacts.extend(result["artifacts"])
        if len(artifacts) == result["total_count"]:
            break
        if not result["artifacts"] or len(artifacts) > result["total_count"]:
            raise IdentityRefusal("artifact_listing_incomplete")
    else:
        raise IdentityRefusal("artifact_listing_incomplete")
    matching = [a for a in artifacts if isinstance(a, dict) and a.get("name") == name]
    if len(matching) != 1:
        raise IdentityRefusal("artifact_name_not_unique")
    artifact = matching[0]
    if (
        artifact.get("expired") is not False
        or not decimal_id(str(artifact.get("id")))
        or not isinstance(artifact.get("workflow_run"), dict)
        or str(artifact["workflow_run"].get("id")) != run_id
    ):
        raise IdentityRefusal("artifact_identity")
    return artifact


def read_report_archive(
    archive: bytes, expected_sha256: str, *, member: str = REPORT_MEMBER
) -> bytes:
    if len(archive) > MAX_ARCHIVE or sha256_hex(archive) != expected_sha256:
        raise IdentityRefusal("artifact_digest")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
            entries = zip_file.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != member
                or entries[0].file_size > MAX_ARCHIVE
                or entries[0].flag_bits & 1
            ):
                raise IdentityRefusal("artifact_members")
            return zip_file.read(entries[0])
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as exc:
        raise IdentityRefusal("artifact_archive") from exc


def report_provenance(
    api: ReadAPI,
    identity: JobIdentity,
    archive: bytes,
    *,
    require_recompute: bool,
) -> tuple[dict, bytes]:
    jobs = jobs_for_attempt(
        api, identity.repository, identity.run_id, identity.run_attempt
    )
    required = ("verify", "recompute") if require_recompute else ("verify",)
    verify = None
    for name in required:
        matches = [j for j in jobs if isinstance(j, dict) and j.get("name") == name]
        if (
            len(matches) != 1
            or matches[0].get("status") != "completed"
            or matches[0].get("conclusion") != "success"
            or str(matches[0].get("run_id")) != identity.run_id
            or str(matches[0].get("run_attempt")) != identity.run_attempt
            or _check_id(matches[0], identity.repository) is None
        ):
            raise IdentityRefusal("required_job_not_successful")
        if name == "verify":
            verify = matches[0]
    artifact = artifact_for_run(
        api, identity.repository, identity.run_id, REPORT_ARTIFACT
    )
    archive_digest = sha256_hex(archive)
    if artifact.get("digest") != "sha256:" + archive_digest:
        raise IdentityRefusal("artifact_digest")
    report = read_report_archive(archive, archive_digest)
    return {
        "workflow_ref": identity.workflow_ref,
        "workflow_sha_git_oid": identity.workflow_sha_git_oid,
        "ref": identity.ref,
        "run_id": identity.run_id,
        "run_attempt": identity.run_attempt,
        "check_run_id": _check_id(verify, identity.repository),
        "conclusion": "success",
        "artifact_name": REPORT_ARTIFACT,
        "artifact_id": str(artifact["id"]),
        "artifact_sha256": archive_digest,
    }, report
