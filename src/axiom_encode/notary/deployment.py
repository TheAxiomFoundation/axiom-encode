"""Custodian-owned configuration and separate production service entrypoints."""

from __future__ import annotations

import argparse
import os
import re
import stat
from dataclasses import fields as dataclass_fields
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from ._schema import decode_base64, digest, fields, lane_name
from .apps import AppCredential, AppScope, GitHubAppAPI, PublisherApps
from .broker import PublisherTokenBroker
from .canonical import jcs_dumps, strict_parse
from .chain import BOOTSTRAP_PATHS
from .github_inputs import BootstrapCeremony, Deployment, GitHubSignerInputs
from .identity import IdentityRefusal
from .leases import PublisherLeases
from .protocol import decimal_id, oid, parse_artifact
from .readplane import (
    ProxyReader,
    ReadGateway,
    ReadOperations,
    RPCClient,
    gateway_bearer,
)
from .remote import GitHubReader
from .service import create_app
from .signer import NotarySigner


def custodian_parent(path: Path, *, socket=False):
    if not path.is_absolute() or path != path.resolve():
        raise IdentityRefusal("custodian_path")
    # A private socket directory prevents replacement by another local user.
    # The proxy may have directory execute permission; never write permission.
    if socket:
        info = path.parent.stat()
        if info.st_uid not in {0, os.geteuid()} or stat.S_IMODE(info.st_mode) & 0o027:
            raise IdentityRefusal("custodian_socket_directory")
    for parent in path.parents:
        info = parent.stat()
        sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
        if info.st_uid not in {0, os.geteuid()} or (
            stat.S_IMODE(info.st_mode) & 0o022 and not sticky_root
        ):
            raise IdentityRefusal("custodian_parent")


def custodian_file(path: str, *, private=False, limit=8_000_000) -> bytes:
    value = Path(path)
    custodian_parent(value)
    fd = os.open(value, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid not in {0, os.geteuid()}
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & (0o077 if private else 0o022)
            or info.st_size > limit
        ):
            raise IdentityRefusal("custodian_file_permissions")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise IdentityRefusal("custodian_file_size")
        return raw
    finally:
        os.close(fd)


def parse_deployment(body) -> Deployment:
    if not fields(body, {f.name for f in dataclass_fields(Deployment)}):
        raise IdentityRefusal("deployment_schema")
    if (
        not lane_name(body["repository"])
        or not body["repository"].startswith("TheAxiomFoundation/")
        or not all(
            decimal_id(body[k]) for k in ("repository_id", "repository_owner_id")
        )
        or not digest(body["epoch_sha256"])
    ):
        raise IdentityRefusal("deployment_identity")
    if any(
        type(body[k]) is not int or body[k] <= 0
        for k in (
            "lane_ruleset_id",
            "chain_writer_ruleset_id",
            "chain_integrity_ruleset_id",
            "chain_app_id",
            "lane_app_id",
        )
    ):
        raise IdentityRefusal("deployment_control_ids")
    reviewers = body["reviewer_ids"]
    if (
        not isinstance(reviewers, list)
        or not reviewers
        or any(type(v) is not int or v <= 0 for v in reviewers)
        or reviewers != sorted(set(reviewers))
    ):
        raise IdentityRefusal("deployment_reviewers")
    if (
        not isinstance(body["content_branch"], str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", body["content_branch"]) is None
    ):
        raise IdentityRefusal("deployment_branch")
    for key in ("workflow_path", "finalizer_workflow_path"):
        if (
            not isinstance(body[key], str)
            or re.fullmatch(r"\.github/workflows/[A-Za-z0-9_-]+\.ya?ml", body[key])
            is None
        ):
            raise IdentityRefusal("deployment_workflow")
    if (
        body["workflow_path"] == body["finalizer_workflow_path"]
        or body["consumer_spec_path"] != ".axiom/notary/consumer.json"
    ):
        raise IdentityRefusal("deployment_paths")
    if (
        any(
            not isinstance(body[k], str) or not body[k] or len(body[k]) > 300
            for k in ("signing_audience", "publishing_audience", "check_name")
        )
        or body["signing_audience"] == body["publishing_audience"]
    ):
        raise IdentityRefusal("deployment_audience")
    inventory = jcs_dumps(body["dependency_inventory"])
    if parse_artifact(inventory, "dependency-inventory") is None:
        raise IdentityRefusal("deployment_inventory")
    return Deployment(
        **(
            body
            | {"reviewer_ids": frozenset(reviewers), "dependency_inventory": inventory}
        )
    )


def parse_ceremony(body, deployment):
    if body is None:
        return None
    if not fields(body, {f.name for f in dataclass_fields(BootstrapCeremony)}):
        raise IdentityRefusal("ceremony_schema")
    for key in ("lock_ruleset_id", "bootstrap_team_id", "bootstrap_user_id"):
        if type(body[key]) is not int or body[key] <= 0:
            raise IdentityRefusal("ceremony_identity")
    if (
        not isinstance(body["bootstrap_team_slug"], str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", body["bootstrap_team_slug"]) is None
    ):
        raise IdentityRefusal("ceremony_team")
    args = body["arguments"]
    expected = {
        "lane",
        "notary_repository",
        "prospective",
        "consumer_spec_path",
        "consumer_template",
        "legacy_apply_root",
        "legacy_eval_root",
        "frozen_apply_spki_sha256",
        "frozen_eval_spki_sha256",
        "ceremony_admin_spki_sha256",
        "ceremony_notary_spki_sha256",
        "expected_encoder_identity",
        "local_corpus_release",
    }
    if (
        not fields(args, expected)
        or args["lane"] != deployment.repository
        or args["notary_repository"] != deployment.repository + "-notary"
        or args["consumer_spec_path"] != deployment.consumer_spec_path
    ):
        raise IdentityRefusal("ceremony_arguments")
    if not fields(args["prospective"], set(BOOTSTRAP_PATHS.values())):
        raise IdentityRefusal("ceremony_prospective")
    from .administration import legacy_encoder_identity

    legacy_encoder_identity(args["expected_encoder_identity"])
    prospective = {
        key: decode_base64(value) for key, value in args["prospective"].items()
    }
    template = decode_base64(args["consumer_template"])
    if template is None or any(v is None for v in prospective.values()):
        raise IdentityRefusal("ceremony_bytes")
    corpus = args["local_corpus_release"]
    if corpus is not None:
        # This is a service-owned local release directory, never a request path.
        if (
            not isinstance(corpus, str)
            or not Path(corpus).is_absolute()
            or Path(corpus).resolve() != Path(corpus)
        ):
            raise IdentityRefusal("ceremony_corpus")
        corpus = Path(corpus)
    return BootstrapCeremony(
        **(
            body
            | {
                "arguments": args
                | {
                    "prospective": prospective,
                    "consumer_template": template,
                    "local_corpus_release": corpus,
                }
            }
        )
    )


def require_running_identity(expected, inventory):
    if (
        not fields(
            expected, {"repository", "git_oid", "version", "package_tree_sha256"}
        )
        or expected["repository"] != "TheAxiomFoundation/axiom-encode"
        or not oid(expected["git_oid"])
        or not digest(expected["package_tree_sha256"])
    ):
        raise IdentityRefusal("service_encoder_identity")
    from axiom_encode import __file__ as package_file
    from axiom_encode.cli import (
        APPLIED_ENCODING_OFFICIAL_REPOSITORY,
        _current_guard_encoder_execution_identity,
    )
    from axiom_encode.harness.evals import _deterministic_tree_identity

    actual = _current_guard_encoder_execution_identity()
    package = _deterministic_tree_identity(
        Path(package_file).parent, excluded_directory_names=frozenset({"__pycache__"})
    )
    if (
        actual
        != {
            # The legacy runtime attestation uses a hostname-qualified
            # repository; v33 inventories use GitHub's owner/name form.
            # Both spellings are fixed constants, not a caller normalization.
            "repository": APPLIED_ENCODING_OFFICIAL_REPOSITORY,
            "commit": expected["git_oid"],
            "version": expected["version"],
        }
        or package["tree_sha256"] != expected["package_tree_sha256"]
        or strict_parse(inventory)["verifier"]
        != {"repo": expected["repository"], "git_oid": expected["git_oid"]}
    ):
        raise IdentityRefusal("service_encoder_identity")


def build_app(raw: bytes):
    body = strict_parse(raw)
    common = {
        "schema",
        "role",
        "deployment",
        "ceremony",
        "encoder_identity",
        "credentials",
    }
    if (
        not fields(body, common)
        or body["schema"] != "axiom/notary-service-config/v1"
        or body["role"] not in {"signer", "publisher-broker", "read-plane"}
    ):
        raise IdentityRefusal("service_configuration")
    deployment = parse_deployment(body["deployment"])
    ceremony = parse_ceremony(body["ceremony"], deployment)
    require_running_identity(body["encoder_identity"], deployment.dependency_inventory)
    role, credentials = body["role"], body["credentials"]
    if role in {"signer", "read-plane"}:
        names = (
            {"github_read_token_file", "notary_key_file", "approval_directory"}
            if role == "signer"
            else {
                "github_read_token_file",
                "broker_app_public_key_file",
                "broker_app_id",
                "audience",
            }
        )
        if not fields(credentials, names):
            raise IdentityRefusal("service_credential_separation")
        read_token = (
            custodian_file(
                credentials["github_read_token_file"], private=True, limit=10000
            )
            .decode()
            .strip()
        )
        if not read_token or any(c.isspace() for c in read_token):
            raise IdentityRefusal("service_read_credential")
        api = GitHubReader(read_token)
        if role == "signer":
            key = serialization.load_pem_private_key(
                custodian_file(
                    credentials["notary_key_file"], private=True, limit=10000
                ),
                password=None,
            )
            if not isinstance(key, Ed25519PrivateKey):
                raise IdentityRefusal("notary_key_type")
            inputs = GitHubSignerInputs(
                deployment, api, read_token=read_token, ceremony=ceremony
            )
            from .approval_inbox import ApprovalInbox

            inbox = ApprovalInbox(Path(credentials["approval_directory"]))
            return create_app(
                signer_factory=lambda: NotarySigner(
                    key, inputs.job_policy(), api, inputs
                ),
                approval_loader=inbox.read,
            )
        key = serialization.load_pem_public_key(
            custodian_file(credentials["broker_app_public_key_file"], limit=10000)
        )
        if (
            not isinstance(key, RSAPublicKey)
            or key.key_size < 2048
            or credentials["broker_app_id"] != deployment.chain_app_id
        ):
            raise IdentityRefusal("read_plane_broker_key")
        return create_app(
            read_gateway=ReadGateway(
                ReadOperations(deployment, api, ceremony=ceremony),
                key,
                app_id=credentials["broker_app_id"],
                audience=credentials["audience"],
            )
        )
    if not fields(
        credentials, {"chain_app", "lane_app", "read_plane_endpoint", "state_database"}
    ):
        raise IdentityRefusal("service_credential_separation")
    roots = []
    for name in ("chain_app", "lane_app"):
        spec = credentials[name]
        if not fields(spec, {"scope", "key_file"}) or not fields(
            spec["scope"], {f.name for f in dataclass_fields(AppScope)}
        ):
            raise IdentityRefusal("service_app_configuration")
        key = serialization.load_pem_private_key(
            custodian_file(spec["key_file"], private=True, limit=20000), password=None
        )
        roots.append(AppCredential(AppScope(**spec["scope"]), key, GitHubAppAPI()))
    apps = PublisherApps(*roots)
    endpoint = credentials["read_plane_endpoint"]
    api = ProxyReader(RPCClient(endpoint, lambda: gateway_bearer(apps.chain, endpoint)))
    inputs = GitHubSignerInputs(deployment, api, ceremony=ceremony)
    leases = PublisherLeases(Path(credentials["state_database"]), apps, api)
    return create_app(broker=PublisherTokenBroker(inputs, apps, leases))


def main():
    parser = argparse.ArgumentParser(
        description="Run one custodian-configured notary service role on a private Unix socket"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    target = Path(args.socket)
    custodian_parent(target, socket=True)
    if target.exists():
        raise SystemExit("socket must be new in the private service runtime directory")
    os.umask(0o077)
    app = build_app(custodian_file(args.config))
    import uvicorn

    uvicorn.run(
        app,
        uds=str(target),
        access_log=False,
        log_level="warning",
        proxy_headers=False,
        limit_concurrency=16,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
