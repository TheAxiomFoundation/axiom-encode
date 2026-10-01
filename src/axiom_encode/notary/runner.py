"""Four-job admission runner; no generation, hardware key, or App minting.

The workflow installs this module at a full reviewed encoder commit. Only
verify executes candidate commands; recompute, approve and publish are fresh
jobs using pinned code. Configuration is committed on the protected base.
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from ._schema import fields
from .administration import build_genesis, build_transition
from .canonical import jcs_dumps, sha256_hex, strict_parse
from .chain import Anchor, reconstruct
from .consumer import parse_consumer
from .deployment import parse_ceremony, parse_deployment, require_running_identity
from .github_inputs import GitHubSignerInputs
from .identity import IdentityRefusal, JobIdentity, _check_id, jobs_for_attempt
from .manifest import manifest_sha256
from .protocol import decimal_id, oid, parse_artifact
from .provenance import REPORT_ARTIFACT
from .publisher import Publisher
from .readplane import ProxyReader, RPCClient
from .refusal import Refusal
from .remote import GitHubReader, RemoteRepository, _bounded
from .signer import ReceiptInputs, make_candidate
from .verification import verify_snapshots


class Configuration:
    def __init__(self, raw):
        body = strict_parse(raw)
        if (
            not fields(
                body,
                {
                    "schema",
                    "deployment",
                    "encoder_identity",
                    "signer_endpoint",
                    "publisher_endpoint",
                    "ceremony",
                },
            )
            or body["schema"] != "axiom/notary-runner/v1"
        ):
            raise IdentityRefusal("runner_configuration")
        self.deployment = parse_deployment(body["deployment"])
        if self.deployment.epoch_sha256 != "0" * 64:
            raise IdentityRefusal("runner_epoch_must_be_derived")
        self.encoder = body["encoder_identity"]
        self.ceremony = parse_ceremony(body["ceremony"], self.deployment)
        self.signer_endpoint, self.publisher_endpoint = (
            body["signer_endpoint"],
            body["publisher_endpoint"],
        )
        if self.signer_endpoint == self.publisher_endpoint:
            raise IdentityRefusal("runner_service_separation")
        for endpoint in (self.signer_endpoint, self.publisher_endpoint):
            RPCClient(endpoint, lambda: "")  # Validate exact HTTPS origin.


def require_context(config, job):
    c = config.deployment
    if (
        os.environ.get("GITHUB_REPOSITORY") != c.repository
        or os.environ.get("GITHUB_REPOSITORY_ID") != c.repository_id
        or os.environ.get("GITHUB_REPOSITORY_OWNER_ID") != c.repository_owner_id
        or os.environ.get("GITHUB_JOB") != job
        or os.environ.get("GITHUB_REF") != "refs/heads/" + c.content_branch
        or not oid(os.environ.get("GITHUB_WORKFLOW_SHA"))
        or not decimal_id(os.environ.get("GITHUB_RUN_ID"))
        or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
    ):
        raise IdentityRefusal("runner_context")
    require_running_identity(config.encoder, c.dependency_inventory)


def oidc(audience):
    endpoint = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "")
    token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "")
    parsed = urlparse(endpoint)
    # Runtime-provided GitHub Actions endpoint only, with no redirects/proxies.
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".actions.githubusercontent.com")
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or not token
    ):
        raise IdentityRefusal("runner_oidc_endpoint")
    query = [(k, v) for k, v in parse_qsl(parsed.query) if k != "audience"] + [
        ("audience", audience)
    ]
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            urlunparse(parsed._replace(query=urlencode(query))),
            headers={"Authorization": "Bearer " + token},
            timeout=(5, 30),
            stream=True,
            allow_redirects=False,
        )
        if response.status_code != 200:
            response.close()
            raise IdentityRefusal("runner_oidc_unavailable")
        result = strict_parse(_bounded(response, 65536))
    if not isinstance(result, dict) or not isinstance(result.get("value"), str):
        raise IdentityRefusal("runner_oidc_response")
    return result["value"]


def recompute_identity(config, api):
    c = config.deployment
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    jobs = jobs_for_attempt(api, c.repository, run, attempt)
    rows = [
        row
        for row in jobs
        if row.get("name") == "recompute" and row.get("status") == "in_progress"
    ]
    if len(rows) != 1 or _check_id(rows[0], c.repository) is None:
        raise IdentityRefusal("runner_recompute_job")
    return JobIdentity(
        c.repository,
        c.repository + "/" + c.workflow_path + "@refs/heads/" + c.content_branch,
        os.environ["GITHUB_WORKFLOW_SHA"],
        "refs/heads/" + c.content_branch,
        run,
        attempt,
        _check_id(rows[0], c.repository),
        "",
        "recompute-context",
    )


def lane_inputs(config, pr_number):
    c = config.deployment
    if not decimal_id(pr_number):
        raise IdentityRefusal("runner_pull_request")
    with RemoteRepository(c.repository) as lane:
        tip = lane.fetch("refs/heads/" + c.content_branch)
        if tip != os.environ["GITHUB_WORKFLOW_SHA"]:
            raise IdentityRefusal("runner_base_moved")
        base = lane.snapshot(tip)
        subject = lane.snapshot(lane.fetch(f"refs/pull/{pr_number}/head"))
    bind_epoch(config, base)
    c = config.deployment
    consumer = parse_consumer(base.blobs.get(c.consumer_spec_path, b""))
    if consumer is None:
        if config.ceremony is None:
            raise IdentityRefusal("runner_consumer")
        fingerprint = config.ceremony.arguments["ceremony_notary_spki_sha256"]
    else:
        if (
            consumer["lane"] != c.repository
            or consumer["epoch_sha256"] != c.epoch_sha256
        ):
            raise IdentityRefusal("runner_consumer")
        fingerprint = consumer["notary_spki_sha256"]
    anchor = Anchor(c.repository, c.epoch_sha256, c.repository + "-notary", fingerprint)
    with RemoteRepository(anchor.notary_repository) as chain:
        state = reconstruct(
            chain.chain_history(chain.fetch("refs/heads/chain")), anchor
        )
    if isinstance(state, Refusal) or state.tip.manifest != manifest_sha256(
        base.manifest
    ):
        raise IdentityRefusal("runner_finalized_base")
    return base, subject, state


def bind_epoch(config, base):
    """Avoid placing the genesis digest inside its own committed preimage."""
    consumer = parse_consumer(base.blobs.get(config.deployment.consumer_spec_path, b""))
    if consumer is not None:
        if consumer["lane"] != config.deployment.repository:
            raise IdentityRefusal("runner_consumer")
        epoch = consumer["epoch_sha256"]
    else:
        if config.ceremony is None:
            raise IdentityRefusal("runner_ceremony_required")
        epoch = sha256_hex(genesis_candidate(config, base))
    config.deployment = replace(config.deployment, epoch_sha256=epoch)


def run_gates(base, subject, state, inventory, *, execute=None):
    predecessor = state.predecessor(lane_commit=base.commit)
    cheap = verify_snapshots(base, subject, predecessor, inventory, [])
    parsed = strict_parse(cheap)
    if parsed.get("stage") != "gates":
        # A profile with zero gates is not a deployable validation profile.
        if parsed.get("schema") == "axiom/notary-report-pass/v1":
            raise IdentityRefusal("runner_empty_gate_profile")
        return cheap
    commands = strict_parse(base.blobs.get(".axiom/notary/gate-commands.json", b""))
    profile = parse_artifact(
        base.blobs.get(".axiom/notary/profile.json", b""), "profile"
    )
    if (
        not fields(commands, {"schema", "lane", "commands"})
        or commands["schema"] != "axiom/notary-gate-commands/v1"
        or commands["lane"] != state.anchor.lane
        or not isinstance(commands["commands"], list)
    ):
        raise IdentityRefusal("runner_gate_commands")
    required = [gate["gate_id"] for gate in profile["required_gates"]]
    if [
        row.get("gate_id") for row in commands["commands"] if isinstance(row, dict)
    ] != required:
        raise IdentityRefusal("runner_gate_set")
    for row in commands["commands"]:
        if (
            not fields(row, {"gate_id", "argv", "timeout_seconds"})
            or not isinstance(row["argv"], list)
            or not row["argv"]
            or not all(
                isinstance(arg, str) and arg and "\0" not in arg for arg in row["argv"]
            )
            or type(row["timeout_seconds"]) is not int
            or not 1 <= row["timeout_seconds"] <= 3600
        ):
            raise IdentityRefusal("runner_gate_command")
    with (
        tempfile.TemporaryDirectory(prefix="axiom-notary-gates-") as directory,
        ExitStack() as stack,
    ):
        candidate = Path(directory) / state.anchor.lane.split("/")[1]
        with RemoteRepository(state.anchor.lane) as repository:
            if repository.fetch(subject.commit) != subject.commit:
                raise IdentityRefusal("runner_subject_unavailable")
            repository._run(
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(repository.path),
                str(candidate),
                in_repo=False,
            )
            repository._run(
                "-C",
                str(candidate),
                "-c",
                "core.hooksPath=/dev/null",
                "checkout",
                "--detach",
                subject.commit,
                in_repo=False,
            )
            repository._run(
                "-C",
                str(candidate),
                "config",
                "remote.origin.url",
                "https://github.com/" + state.anchor.lane + ".git",
                in_repo=False,
            )
        public_keys = sorted(state.registry.keys["corpus-release"])
        if len(public_keys) != 1:
            raise IdentityRefusal("runner_corpus_key_selection")
        from cryptography.hazmat.primitives import serialization

        corpus_key = base64.b64encode(
            state.registry.keys["corpus-release"][public_keys[0]].public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        ).decode()
        values = {
            "python": sys.executable,
            "checkout": str(candidate),
            "corpus_public_key": corpus_key,
        }
        from axiom_encode.toolchain import (
            load_rulespec_local_corpus_release,
            local_corpus_release_verification,
        )

        from .assets import provision

        assets = stack.enter_context(
            provision(base, Path(directory), state.anchor.lane)
        )
        values.update(assets)
        with local_corpus_release_verification(corpus_key):
            load_rulespec_local_corpus_release(candidate, Path(assets["corpus_root"]))
        declarations = []
        # No Actions runtime token, OIDC token, GitHub credential, or personal
        # login enters candidate commands. This entire job is still untrusted.
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": directory,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for row in commands["commands"]:
            argv = [arg.format_map(values) for arg in row["argv"]]
            try:
                code = (execute or subprocess.run)(
                    argv,
                    cwd=candidate,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=row["timeout_seconds"],
                    check=False,
                ).returncode
            except (subprocess.TimeoutExpired, OSError):
                code = 1
            declarations.append(
                {"gate_id": row["gate_id"], "outcome": "pass" if code == 0 else "fail"}
            )
    return verify_snapshots(base, subject, predecessor, inventory, declarations)


def administrative_candidate(config, operation, pr_number):
    if operation == "transition":
        base, subject, state = lane_inputs(config, pr_number)
        # The declared reason is fixed in base config's audited workflow, not
        # a caller-provided signing body. Administrative review binds the bytes.
        return build_transition(
            base, subject, state, reason="Reviewed administrative transition"
        )
    if config.ceremony is None:
        raise IdentityRefusal("runner_ceremony_required")
    with RemoteRepository(config.deployment.repository) as repository:
        tip = repository.fetch("refs/heads/" + config.deployment.content_branch)
        if tip != os.environ["GITHUB_WORKFLOW_SHA"]:
            raise IdentityRefusal("runner_base_moved")
        base = repository.snapshot(tip)
    return genesis_candidate(config, base)


def genesis_candidate(config, base):
    if config.ceremony.arguments["local_corpus_release"] is None:
        return build_genesis(base, **config.ceremony.arguments)
    from .assets import provision

    with tempfile.TemporaryDirectory(prefix="axiom-genesis-assets-") as directory:
        with provision(base, Path(directory), config.deployment.repository) as assets:
            return build_genesis(
                base,
                **(
                    config.ceremony.arguments
                    | {"local_corpus_release": Path(assets["corpus_root"])}
                ),
            )


def run(config, job, operation, pr_number, output, candidate_digest):
    require_context(config, job)
    c = config.deployment
    if job == "verify":
        if operation == "receipt":
            base, subject, state = lane_inputs(config, pr_number)
            raw = run_gates(base, subject, state, c.dependency_inventory)
            (output / "report.json").write_bytes(raw)
            if parse_artifact(raw, "report-pass") is None:
                raise IdentityRefusal("runner_verification_refused")
        else:
            administrative_candidate(config, operation, pr_number)
            (output / "report.json").write_bytes(
                jcs_dumps({"administrative_preflight": "pass"})
            )
    elif job == "recompute":
        if operation == "receipt":
            api = GitHubReader(os.environ.get("GITHUB_TOKEN"))
            identity = recompute_identity(config, api)
            base, subject, state = lane_inputs(config, pr_number)
            c = config.deployment
            inputs = GitHubSignerInputs(c, api, ceremony=config.ceremony)
            raw = make_candidate(
                api,
                identity,
                ReceiptInputs(
                    base,
                    subject,
                    state,
                    c.dependency_inventory,
                    inputs._archive(identity, REPORT_ARTIFACT),
                    b"",
                    pr_number,
                ),
                require_recompute=False,
            )
        else:
            raw = administrative_candidate(config, operation, pr_number)
        (output / "candidate.json").write_bytes(raw)
        print("Candidate SHA-256: " + sha256_hex(raw))
    elif job == "approve":
        client = RPCClient(config.signer_endpoint, lambda: oidc(c.signing_audience))
        deadline = time.monotonic() + 3600
        while True:
            result = client.call(
                operation,
                {"candidate_sha256": candidate_digest, "approval_base64": None},
            )
            if result != {"state": "awaiting_hardware_approval"}:
                (output / "bundle.json").write_bytes(jcs_dumps(result))
                break
            if time.monotonic() >= deadline:
                raise IdentityRefusal("runner_hardware_approval_timeout")
            time.sleep(15)
    elif job == "publish":
        if operation != "genesis":
            lane_inputs(config, pr_number)
            c = config.deployment
        client = RPCClient(
            config.publisher_endpoint, lambda: oidc(c.publishing_audience)
        )
        grant = client.call("publish-tokens", {})
        try:
            if not fields(
                grant, {"identity", "lease_owner", "chain_token", "lane_token"}
            ) or not fields(
                grant["identity"],
                {field.name for field in dataclass_fields(JobIdentity)},
            ):
                raise IdentityRefusal("runner_publisher_grant")
            inputs = GitHubSignerInputs(
                c, ProxyReader(client), ceremony=config.ceremony
            )
            with ExitStack() as stack:
                if (
                    operation == "genesis"
                    and config.ceremony.arguments["local_corpus_release"] is not None
                ):
                    from .assets import provision

                    with RemoteRepository(c.repository) as lane:
                        base = lane.snapshot(
                            lane.fetch("refs/heads/" + c.content_branch)
                        )
                    directory = stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="axiom-publisher-assets-")
                    )
                    assets = stack.enter_context(
                        provision(base, Path(directory), c.repository)
                    )
                    inputs._ceremony = replace(
                        config.ceremony,
                        arguments=config.ceremony.arguments
                        | {"local_corpus_release": Path(assets["corpus_root"])},
                    )
                result = Publisher(inputs).publish(
                    JobIdentity(**grant["identity"]),
                    grant["chain_token"],
                    grant["lane_token"],
                )
            (output / "publication.json").write_bytes(jcs_dumps(result))
        finally:
            if isinstance(grant.get("lease_owner"), str):
                client.call("release", {"lease_owner": grant["lease_owner"]})
    else:
        result = RPCClient(
            config.publisher_endpoint, lambda: oidc(c.publishing_audience)
        ).call("finalize", {})
        (output / "finalization.json").write_bytes(jcs_dumps(result))


def main():
    parser = argparse.ArgumentParser(
        description="Run one pinned, non-generating notary workflow job"
    )
    parser.add_argument(
        "job", choices=("verify", "recompute", "approve", "publish", "finalize")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--operation", choices=("receipt", "transition", "genesis"), default="receipt"
    )
    parser.add_argument("--pr-number", default="")
    parser.add_argument("--candidate-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(
            Configuration(args.config.read_bytes()),
            args.job,
            args.operation,
            args.pr_number,
            args.output,
            args.candidate_sha256,
        )
    except Exception:
        raise SystemExit(
            "Notary job refused; no credentials or remote response bodies printed."
        ) from None


if __name__ == "__main__":
    main()
