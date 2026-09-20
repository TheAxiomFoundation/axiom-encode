"""Custodian-approved enrollment bindings, additional to the v33 key registry.

This deployment policy lives on the authenticated base as a trust file. It
does not change the closed v33 generation record or key-registry schemas.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from ._schema import canonical_object, digest, fields, nonempty, ordered_strings
from .canonical import jcs_dumps, sha256_hex
from .identity import IdentityRefusal, ReadAPI, require_writer_submission
from .lineage import GENERATION, STORE_PREFIX, parse_record
from .protocol import decimal_id, oid, parse_artifact
from .registry import KeyRegistry
from .signatures import verify_detached
from .verification import Snapshot

ENROLLMENT_PATH = ".axiom/notary/producers.json"
ENCODER_REPOSITORY = "TheAxiomFoundation/axiom-encode"


@dataclass(frozen=True)
class Enrollment:
    raw: bytes

    @property
    def body(self) -> dict:
        return canonical_object(self.raw)

    @property
    def runtime_identity(self) -> str:
        return "axiom-runtime:sha256:" + sha256_hex(self.raw)


def parse_enrollments(raw: bytes, registry: KeyRegistry) -> dict[str, Enrollment]:
    body = canonical_object(raw)
    if (
        not fields(body, {"schema", "lane", "runtimes"})
        or body["schema"] != "axiom/notary-producer-enrollments/v1"
        or body["lane"] != registry.lane
        or not isinstance(body["runtimes"], list)
    ):
        raise IdentityRefusal("enrollment_schema")
    enrolled = {}
    producers, actors = [], []
    for entry in body["runtimes"]:
        if not fields(
            entry,
            {
                "producer_spki_sha256",
                "actor_spki_sha256",
                "github_user_ids",
                "encoder",
                "codex_cli",
                "custody_evidence_sha256",
            },
        ):
            raise IdentityRefusal("runtime_enrollment_schema")
        producer, actor = entry["producer_spki_sha256"], entry["actor_spki_sha256"]
        encoder, cli, operators = (
            entry["encoder"],
            entry["codex_cli"],
            entry["github_user_ids"],
        )
        if (
            not digest(producer)
            or producer not in registry.keys["producer"]
            or not digest(actor)
            or actor not in registry.keys["actor"]
            or actor in actors
            or not ordered_strings(operators)
            or not operators
            or not all(decimal_id(value) for value in operators)
            or not fields(
                encoder, {"repository", "git_oid", "version", "package_tree_sha256"}
            )
            or encoder["repository"] != ENCODER_REPOSITORY
            or not oid(encoder["git_oid"])
            or not nonempty(encoder["version"])
            or not digest(encoder["package_tree_sha256"])
            or not fields(cli, {"version", "sha256"})
            or not nonempty(cli["version"])
            or not digest(cli["sha256"])
            or not digest(entry["custody_evidence_sha256"])
        ):
            raise IdentityRefusal("runtime_enrollment_binding")
        enrollment = Enrollment(jcs_dumps(entry))
        enrolled[producer] = enrollment
        producers.append(producer)
        actors.append(actor)
    if not ordered_strings(producers) or set(producers) != set(
        registry.keys["producer"]
    ):
        raise IdentityRefusal("producer_enrollment_partition")
    return enrolled


def require_enrolled_submission(
    api: ReadAPI,
    *,
    pr_number: str,
    base: Snapshot,
    subject: Snapshot,
    registry: KeyRegistry,
    report_raw: bytes,
    merged_commit: str | None = None,
) -> dict:
    """Require the live PR author to operate every consumed lineage signer.

    Current repository write access is checked even for a purely unprotected
    change. Every consumed generation is bound to an enrolled runtime and the
    base's exact encoder pin. Corrections additionally need the ordinary v33
    actor + reviewer signatures, already checked during recomputation.
    """
    report = parse_artifact(report_raw, "report-pass")
    if (
        report is None
        or report["lane"] != registry.lane
        or report["subject_commit_git_oid"] != subject.commit
    ):
        raise IdentityRefusal("submission_report")
    return require_writer_submission(
        api,
        registry.lane,
        pr_number,
        subject.commit,
        contributor_ids=enrolled_contributor_ids(
            base, subject, registry, report["coverage_assignment"]
        ),
        merged_commit=merged_commit,
    )


def enrolled_contributor_ids(base, subject, registry, coverage_assignment):
    """Bind a previously verified coverage assignment to enrolled operators.

    This authenticates enrollment only; it is never a gate or receipt verdict.
    """
    enrolled = parse_enrollments(base.blobs.get(ENROLLMENT_PATH, b""), registry)
    try:
        pins = tomllib.loads(base.blobs[".axiom/workflow-toolchain.toml"].decode())[
            "workflow_toolchain"
        ]
        version, commit = pins["axiom_encode_version"], pins["axiom_encode_ref"]
        if not nonempty(version) or not oid(commit):
            raise ValueError()
    except (KeyError, UnicodeError, ValueError, TypeError) as exc:
        raise IdentityRefusal("encoder_pin_unavailable") from exc
    allowed = set().union(*(set(e.body["github_user_ids"]) for e in enrolled.values()))
    consumed = {d for row in coverage_assignment for d in row["record_sha256s"]}
    for address in sorted(consumed):
        raw = subject.blobs.get(STORE_PREFIX + address + ".json", b"")
        record = parse_record(raw)
        if record is None or sha256_hex(raw) != address:
            raise IdentityRefusal("submission_record")
        role = "producer" if record["schema"] == GENERATION else "actor"
        signature = subject.blobs.get(
            STORE_PREFIX + address + ".json." + role + ".sig", b""
        )
        if not verify_detached(
            signature, body_sha256=address, role=role, registry=registry
        ):
            raise IdentityRefusal("submission_signature")
        signer = canonical_object(signature)["signer_spki_sha256"]
        matches = [
            e
            for p, e in enrolled.items()
            if (p if role == "producer" else e.body["actor_spki_sha256"]) == signer
        ]
        if len(matches) != 1:
            raise IdentityRefusal("signer_not_enrolled")
        enrollment = matches[0]
        entry = enrollment.body
        if (
            entry["encoder"]["git_oid"] != commit
            or entry["encoder"]["version"] != version
        ):
            raise IdentityRefusal("enrolled_encoder_pin_mismatch")
        if role == "producer" and (
            record["runtime_identity"] != enrollment.runtime_identity
            or record["cli_version"] != entry["codex_cli"]["version"]
            or record["cli_sha256"] != entry["codex_cli"]["sha256"]
        ):
            raise IdentityRefusal("enrolled_runtime_mismatch")
        allowed.intersection_update(entry["github_user_ids"])
    return frozenset(allowed)
