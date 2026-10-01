"""Deterministic execution, exact-byte lineage and shared notary admission."""

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex, strict_parse
from axiom_encode.notary.deterministic_contract import DETERMINISTIC_GENERATION
from axiom_encode.notary.deterministic_runtime import (
    LinuxDeterministicRuntime,
    captured_files,
    observe_outputs,
    run_adapter,
)
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.lineage import STORE_PREFIX, parse_record
from axiom_encode.notary.producer import (
    correction_export,
    deterministic_export,
    generation_export,
)
from axiom_encode.notary.producer_client import apply_export, packet_files
from axiom_encode.notary.producers import (
    ENROLLMENT_PATH,
    Enrollment,
    parse_enrollments,
    require_enrolled_submission,
)
from axiom_encode.notary.verification import verify_snapshots

from .chain_fixtures import Epoch
from .lineage_fixtures import Identities, public_entry
from .test_identity import policy as _policy_fixture
from .test_identity import rsa_key as _rsa_fixture
from .test_producer import decode
from .test_producer_host import host as _host_fixture
from .test_producer_host import request
from .test_producers import API, enrollment_base
from .test_signer import service as _service_fixture
from .test_verification import snapshot

policy = _policy_fixture
rsa_key = _rsa_fixture
host = _host_fixture
service = _service_fixture


def deterministic_entry(codex):
    entry = deepcopy(codex)
    del entry["codex_cli"]
    return entry | {
        "runtime_kind": "deterministic",
        "generator": {
            "name": "fixture-table",
            "version": "1",
            "entrypoint": "adapter.py",
            "files": [{"path": "adapter.py", "sha256": "a" * 64}],
        },
        "runtime": {"python_path": "bin/python3", "tree_sha256": "b" * 64},
        "inputs": [{"path": "source.txt", "sha256": sha256_hex(b"captured source")}],
        "parameters": {"year": "2026"},
        "outputs": ["rules/example.yaml"],
    }


def deterministic_base(identities):
    files = enrollment_base(identities)
    policy = strict_parse(files[ENROLLMENT_PATH])
    policy["schema"] = "axiom/notary-producer-enrollments/v2"
    policy["runtimes"] = [deterministic_entry(policy["runtimes"][0])]
    files[ENROLLMENT_PATH] = jcs_dumps(policy)
    return files


def export(epoch, entry, *, base=None, key=None, outputs=None):
    return deterministic_export(
        key or epoch.identities.keys["producer"],
        enrollment=Enrollment(jcs_dumps(entry)),
        base=base or epoch.active,
        lane=epoch.anchor.lane,
        epoch=epoch.anchor.epoch_sha256,
        run_id="1" * 32,
        outputs=outputs or {"rules/example.yaml": b"generated\n"},
        **{
            name: entry[name]
            for name in ("generator", "runtime", "inputs", "parameters")
        },
    )


@pytest.fixture
def submission():
    epoch = Epoch.create(prepare_base=deterministic_base)
    policy = strict_parse(epoch.active.blobs[ENROLLMENT_PATH])
    _, files = decode(export(epoch, policy["runtimes"][0]))
    subject = snapshot("c" * 40, epoch.active.blobs | files)
    report = verify_snapshots(
        epoch.active,
        subject,
        epoch.state().predecessor(),
        epoch.inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )
    return (
        epoch,
        policy,
        API(subject.commit),
        dict(
            pr_number="1",
            base=epoch.active,
            subject=subject,
            registry=epoch.state().registry,
            report_raw=report,
        ),
    )


def test_deterministic_bytes_pass_same_admission_with_distinct_signature_scope(
    submission,
):
    epoch, policy, api, args = submission
    assert require_enrolled_submission(api, **args)["user"]["id"] == 123
    packet, files = packet_files(export(epoch, policy["runtimes"][0]))
    path = STORE_PREFIX + packet["record_sha256"] + ".json"
    body = strict_parse(files[path])
    assert body["schema"] == DETERMINISTIC_GENERATION
    assert (
        strict_parse(files[path + ".producer.sig"])["scope"] == DETERMINISTIC_GENERATION
    )
    assert not {"model", "sampling", "cli_version", "prompt_sha256s"} & body.keys()


@pytest.mark.parametrize("field", ["generator", "runtime", "inputs", "parameters"])
def test_signed_record_must_match_protected_enrollment(submission, field):
    epoch, policy, api, args = submission
    entry = deepcopy(policy["runtimes"][0])
    if field == "generator":
        entry[field]["files"][0]["sha256"] = "f" * 64
    elif field == "runtime":
        entry[field]["tree_sha256"] = "f" * 64
    elif field == "inputs":
        entry[field][0]["sha256"] = "f" * 64
    else:
        entry[field]["year"] = "2027"
    _, files = decode(export(epoch, entry))
    args["subject"] = snapshot("c" * 40, epoch.active.blobs | files)
    args["report_raw"] = verify_snapshots(
        epoch.active,
        args["subject"],
        epoch.state().predecessor(),
        epoch.inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )
    with pytest.raises(IdentityRefusal, match="runtime_mismatch"):
        require_enrolled_submission(api, **args)


@pytest.mark.parametrize(
    "mutation",
    [
        "unregistered",
        "wrong-kind",
        "extra-field",
        "v1-det",
        "missing-file",
        "bad-input",
        "parameter-number",
        "duplicate-input",
    ],
)
def test_closed_enrollment_rejects_ambiguous_or_unmeasured_runtime(
    submission, mutation
):
    _, original, _, args = submission
    policy = deepcopy(original)
    entry = policy["runtimes"][0]
    if mutation == "unregistered":
        entry["producer_spki_sha256"] = "f" * 64
    elif mutation == "wrong-kind":
        entry["runtime_kind"] = "codex"
    elif mutation == "extra-field":
        entry["codex_cli"] = {"version": "fake", "sha256": "f" * 64}
    elif mutation == "v1-det":
        policy["schema"] = "axiom/notary-producer-enrollments/v1"
    elif mutation == "missing-file":
        entry["generator"]["entrypoint"] = "missing.py"
    elif mutation == "bad-input":
        entry["inputs"][0]["path"] = "../source"
    elif mutation == "parameter-number":
        entry["parameters"]["year"] = 2026
    else:
        entry["inputs"] *= 2
    with pytest.raises(IdentityRefusal):
        parse_enrollments(jcs_dumps(policy), args["registry"])


@pytest.mark.parametrize(
    "mutation", ["model", "sampling", "missing-inputs", "wrong-scope", "changed-output"]
)
def test_no_fabricated_model_metadata_or_wrong_scope_or_changed_bytes(
    submission, mutation
):
    epoch, policy, _, args = submission
    packet, files = decode(export(epoch, policy["runtimes"][0]))
    path = STORE_PREFIX + packet["record_sha256"] + ".json"
    body = strict_parse(files[path])
    if mutation in {"model", "sampling", "missing-inputs"}:
        if mutation == "missing-inputs":
            del body["inputs"]
        else:
            body[mutation] = "invented"
        assert parse_record(jcs_dumps(body)) is None
        return
    if mutation == "wrong-scope":
        files[path + ".producer.sig"] = epoch.identities.sidecar(
            files[path], "producer"
        )
    else:
        files["rules/example.yaml"] = b"edited after generation"
    report = verify_snapshots(
        args["base"],
        snapshot("c" * 40, args["base"].blobs | files),
        epoch.state().predecessor(),
        epoch.inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )
    assert strict_parse(report)["schema"] == "axiom/notary-report-refusal/v1"


def test_deterministic_export_applies_only_authenticated_exact_bytes(
    submission, tmp_path, monkeypatch
):
    epoch, policy, _, args = submission
    root = tmp_path / "checkout"
    root.mkdir()

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(root), *argv])

    git("init", "-q")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    for path, raw in args["base"].blobs.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    git("add", ".")
    git("commit", "-qm", "fixture")
    packet = strict_parse(export(epoch, policy["runtimes"][0]))
    packet["base_commit_git_oid"] = git("rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(
        "axiom_encode.notary.producer_client.registry_for_base",
        lambda *_: epoch.state().registry,
    )
    apply_export(jcs_dumps(packet), root)
    assert (root / "rules/example.yaml").read_bytes() == b"generated\n"


def test_deterministic_receipt_signing_performs_no_generation(service, monkeypatch):
    signer, oidc, candidate, approval, _, _, _ = service
    monkeypatch.setattr(
        "axiom_encode.notary.producer_worker.main", lambda: pytest.fail("CI model call")
    )
    assert signer.receipt(oidc, sha256_hex(candidate), approval)


@pytest.mark.parametrize(
    "mutation", ["outsider", "revoked", "fork", "missing-approval", "edited"]
)
def test_deterministic_notary_refuses_bad_contribution(service, mutation):
    signer, oidc, candidate, approval, reader, api, _ = service
    if mutation == "outsider":
        api.pr["user"]["id"] = 999
    elif mutation == "revoked":
        api.permission["permission"] = "read"
    elif mutation == "fork":
        api.pr["head"]["repo"]["full_name"] = "stranger/fork"
    elif mutation == "missing-approval":
        approval = b""
    else:
        reader.packet = replace(
            reader.packet,
            subject=snapshot(
                reader.packet.subject.commit,
                reader.packet.subject.blobs | {"rules/example.yaml": b"edited"},
            ),
        )
    with pytest.raises(IdentityRefusal):
        signer.receipt(oidc, sha256_hex(candidate), approval)


def test_host_generate_has_no_auth_or_command_and_preserves_owner_retry(
    host, submission
):
    epoch, policy, _, args = submission
    host.config["runtime_kind"] = "deterministic"
    entry = policy["runtimes"][0]

    @contextmanager
    def base(user):
        yield None, args["base"], Enrollment(jcs_dumps(entry)), []

    host._base = base

    def run(**kwargs):
        assert kwargs["auth"] is None
        host.calls.append(kwargs["request"]["run_id"])
        return {
            "outputs": {"rules/example.yaml": b"generated\n"},
            **{
                name: entry[name]
                for name in ("generator", "runtime", "inputs", "parameters")
            },
        }, None

    host.runtime.run = run
    req = {"operation": "generate", "run_id": "1" * 32}
    result = host.perform(1234, req)
    assert result["refreshed_auth_base64"] is None
    assert result == host.perform(1234, req)
    assert host.calls == ["1" * 32]
    for extra in (
        {"parameters": {}},
        {"outputs": {}},
        {"command": "arbitrary"},
        {"auth_base64": "e30="},
    ):
        with pytest.raises(IdentityRefusal, match="generate_request"):
            host.perform(1234, req | extra)
    with pytest.raises(IdentityRefusal, match="runtime_kind"):
        host.perform(1234, request())


ADAPTER = b"""import json, os, sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_bytes())
assert "OPENAI_API_KEY" not in os.environ
assert "CODEX_HOME" not in os.environ
value=(Path(r["input_root"])/"source.txt").read_bytes()+r["parameters"]["year"].encode()
for relative in r["outputs"]:
    p=Path(r["output_root"])/relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(value)
"""


def test_real_adapter_processes_receive_fixed_inputs_and_parameters(
    tmp_path, monkeypatch
):
    job = tmp_path / "job"
    generator = job / "immutable/generator"
    inputs = job / "immutable/inputs"
    generator.mkdir(parents=True)
    inputs.mkdir()
    (generator / "adapter.py").write_bytes(ADAPTER)
    (inputs / "source.txt").write_bytes(b"captured source")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-adapter")
    monkeypatch.setenv("CODEX_HOME", "must-not-enter-adapter")
    run_adapter(
        job,
        {
            "generator": {"entrypoint": "adapter.py"},
            "parameters": {"year": "2026"},
            "outputs": ["rules/example.yaml"],
        },
    )
    first = observe_outputs(job, "work/output", ["rules/example.yaml"])
    assert first == {"rules/example.yaml": b"captured source2026"}


def test_extra_outputs_and_input_digest_drift_refuse(tmp_path):
    (tmp_path / "source.txt").write_bytes(b"captured source")
    with pytest.raises(IdentityRefusal, match="file_measurement"):
        captured_files(tmp_path, [{"path": "source.txt", "sha256": "a" * 64}])
    out = tmp_path / "work/output"
    out.mkdir(parents=True)
    (out / "extra.yaml").write_bytes(b"extra")
    with pytest.raises(IdentityRefusal, match="output_mapping"):
        observe_outputs(tmp_path, "work/output", ["rule.yaml"])


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "nondeterministic",
        "overwrite-prior",
        "measurement-drift",
        "failed",
        "extra-output",
    ],
)
def test_controller_observes_after_worker_exit_and_refuses_drift(
    tmp_path, monkeypatch, submission, fault
):
    import axiom_encode.notary.deterministic_runtime as runtime_module

    epoch, policy, _, _ = submission
    entry = policy["runtimes"][0]
    runtime = object.__new__(LinuxDeterministicRuntime)
    runtime.config = {
        "runtime_root": "/opt/fixture-runtime",
        "python": sys.executable,
        "encoder_identity": {},
        "dependency_inventory": b"{}",
        "timeout_seconds": 60,
    }
    runtime.worker_uid = runtime.worker_gid = 1234
    monkeypatch.setattr(runtime_module, "require_worker_account", lambda *_: None)
    monkeypatch.setattr(runtime_module, "require_worker_idle", lambda *_: None)
    monkeypatch.setattr(runtime_module, "require_running_identity", lambda *_: None)
    monkeypatch.setattr(os, "chown", lambda *_: None)
    measured = []

    def measure(_):
        measured.append(True)
        if len(measured) == 2 and fault == "measurement-drift":
            raise IdentityRefusal("deterministic_runtime_measurement")
        return {"adapter.py": ADAPTER}, {"source.txt": b"captured source"}

    runtime._measure = measure
    job = tmp_path / "job"
    job.mkdir()
    stopped = []
    attempts = []
    real_run = subprocess.run

    def execute(argv, **kwargs):
        if argv[0] == "/usr/bin/systemctl":
            stopped.append(True)
            return SimpleNamespace(returncode=0)
        if argv[0] != "/usr/bin/systemd-run":
            return real_run(argv, **kwargs)
        assert "--property=PrivateNetwork=yes" in argv
        assert "--property=RestrictAddressFamilies=AF_UNIX" in argv
        attempt = Path(argv[-1])
        attempts.append(attempt)
        assert not (attempt / "auth.json").exists()
        if fault != "failed":
            run_adapter(attempt, strict_parse((attempt / "request.json").read_bytes()))
            if fault in {"nondeterministic", "overwrite-prior"} and len(attempts) == 2:
                (attempt / "work/output/rules/example.yaml").write_bytes(b"different")
                if fault == "overwrite-prior":
                    # Even simulating a filesystem-boundary breach cannot
                    # rewrite the first observation in controller memory.
                    (attempts[0] / "work/output/rules/example.yaml").write_bytes(
                        b"different"
                    )
            if fault == "extra-output":
                (attempt / "work/output/extra.yaml").write_bytes(b"unrequested")
        return SimpleNamespace(returncode=1 if fault == "failed" else 0)

    monkeypatch.setattr(subprocess, "run", execute)
    original_observe = runtime_module.observe_outputs

    def observe(*args):
        assert len(stopped) == len(attempts), (
            "must stop descendants before inspecting outputs"
        )
        return original_observe(*args)

    monkeypatch.setattr(runtime_module, "observe_outputs", observe)
    kwargs = dict(
        job=job,
        request={"operation": "generate", "run_id": "1" * 32},
        base=epoch.active,
        enrollment=Enrollment(jcs_dumps(entry)),
        auth=None,
    )
    try:
        if fault:
            with pytest.raises(IdentityRefusal):
                runtime.run(**kwargs)
        else:
            observed, auth = runtime.run(**kwargs)
            assert observed["outputs"] == {"rules/example.yaml": b"captured source2026"}
            assert auth is None
            assert len(measured) == 3
        assert stopped == [True] * len(attempts)
    finally:
        for directory, _, _ in os.walk(job):
            os.chmod(directory, 0o700)


def test_copied_interpreter_cannot_borrow_unmeasured_stdlib(tmp_path):
    import venv

    from axiom_encode.notary.producer_runtime import DETERMINISTIC_BOOTSTRAP

    root = tmp_path / "copied-runtime"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(root)
    result = subprocess.run(
        [
            str(root / "bin/python"),
            "-I",
            "-B",
            "-c",
            DETERMINISTIC_BOOTSTRAP,
            str(root),
            str(tmp_path / "absent-job"),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"external import roots" in result.stderr


def test_enrollment_rejects_nonadjacent_ancestor_files():
    from axiom_encode.notary.deterministic_contract import file_inventory, output_paths

    assert not file_inventory(
        [{"path": path, "sha256": "a" * 64} for path in ("a", "a-b", "a/b")]
    )
    assert not output_paths(["a.yaml", "a.yaml-b.yaml", "a.yaml/b.yaml"])


def test_mixed_codex_deterministic_and_reviewed_correction_share_coverage():
    identities = Identities.create()
    other = {role: Ed25519PrivateKey.generate() for role in ("producer", "actor")}
    for role, key in other.items():
        identities.body[role].append(public_entry(key))
        identities.body[role].sort(key=lambda row: row["spki_sha256"])

    def prepare(ids):
        files = enrollment_base(ids)
        policy = strict_parse(files[ENROLLMENT_PATH])
        codex = policy["runtimes"][0]
        codex |= {
            "runtime_kind": "codex",
            "producer_spki_sha256": public_entry(ids.keys["producer"])["spki_sha256"],
            "actor_spki_sha256": public_entry(ids.keys["actor"])["spki_sha256"],
        }
        det = deterministic_entry(codex) | {
            "producer_spki_sha256": public_entry(other["producer"])["spki_sha256"],
            "actor_spki_sha256": public_entry(other["actor"])["spki_sha256"],
            "outputs": ["rules/table.yaml"],
        }
        policy["schema"] = "axiom/notary-producer-enrollments/v2"
        policy["runtimes"] = sorted(
            [codex, det], key=lambda row: row["producer_spki_sha256"]
        )
        return files | {ENROLLMENT_PATH: jcs_dumps(policy)}

    epoch = Epoch.create(identities=identities, prepare_base=prepare)
    entries = parse_enrollments(
        epoch.active.blobs[ENROLLMENT_PATH], epoch.state().registry
    )
    codex = entries[public_entry(identities.keys["producer"])["spki_sha256"]]
    det = entries[public_entry(other["producer"])["spki_sha256"]]
    _, codex_files = decode(
        generation_export(
            identities.keys["producer"],
            enrollment=codex,
            base=epoch.active,
            lane=epoch.anchor.lane,
            epoch=epoch.anchor.epoch_sha256,
            run_id="1" * 32,
            outputs={"rules/model.yaml": b"model"},
            model="fixture",
            prompts=["c" * 64],
            source_id="fixture",
            source_bytes=b"source",
            draw_set_id="fixture",
            sampling={"temperature": "0.5", "seed": None},
            independence={
                "sibling_draws_visible": "no",
                "incumbent_encoding_visible": "yes",
            },
            references={"oracles": [], "reference_data": []},
        )
    )
    det_packet, det_files = decode(
        export(
            epoch,
            det.body,
            key=other["producer"],
            outputs={"rules/table.yaml": b"table"},
        )
    )
    intermediate = snapshot(
        epoch.active.commit, epoch.active.blobs | codex_files | det_files
    )
    packet, correction_files = decode(
        correction_export(
            other["actor"],
            enrollment=det,
            base=intermediate,
            lane=epoch.anchor.lane,
            epoch=epoch.anchor.epoch_sha256,
            run_id="3" * 32,
            outputs={"rules/table.yaml": b"corrected table"},
            github_user_id="123",
            reason="reviewed correction",
            predecessor=det_packet["record_sha256"],
        )
    )
    path = STORE_PREFIX + packet["record_sha256"] + ".json"
    subject = snapshot("c" * 40, intermediate.blobs | correction_files)

    def report(subject):
        return verify_snapshots(
            epoch.active,
            subject,
            epoch.state().predecessor(),
            epoch.inventory,
            [{"gate_id": "compile", "outcome": "pass"}],
        )

    assert strict_parse(report(subject))["schema"] == "axiom/notary-report-refusal/v1"
    subject = snapshot(
        "c" * 40,
        subject.blobs
        | {path + ".review.sig": identities.sidecar(correction_files[path], "review")},
    )
    raw = report(subject)
    assert len(strict_parse(raw)["eligible_records"]) == 3
    assert require_enrolled_submission(
        API(subject.commit),
        pr_number="1",
        base=epoch.active,
        subject=subject,
        registry=epoch.state().registry,
        report_raw=raw,
    )


def test_reference_b16_adapter_emits_exact_four_file_set(tmp_path):
    root = Path(__file__).resolve().parents[2]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copy(
        root / "deploy/notary/adapters/b16_note_tables.py", bundle / "adapter.py"
    )
    (bundle / "generate_incidence_tables.py").write_text("""from pathlib import Path
def generate(dest, source, selected):
    assert selected == {"brazil-50", "reciprocal-52"}
    assert source.read_bytes() == b"captured notes"
    dest.mkdir(parents=True, exist_ok=True)
    names=[n+e for n in ("note50-brazil-exemptions","note52-reciprocal-exemptions") for e in (".yaml",".test.yaml")]
    for name in names: (dest/name).write_bytes(b"generated")
    return {name:"unused worker hash" for name in names}
""")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "notes.jsonl").write_bytes(b"captured notes")
    prefix = "us/policies/usitc/us-tariff-incidence/generated/"
    outputs = sorted(
        prefix + n + e
        for n in ("note50-brazil-exemptions", "note52-reciprocal-exemptions")
        for e in (".yaml", ".test.yaml")
    )
    descriptor = tmp_path / "request.json"
    request = {
        "schema": "axiom/deterministic-adapter-request/v1",
        "input_root": str(inputs),
        "output_root": str(tmp_path / "output"),
        "parameters": {"actions": "brazil-50,reciprocal-52"},
        "outputs": outputs,
    }
    descriptor.write_bytes(jcs_dumps(request))
    subprocess.run(
        [sys.executable, "-I", str(bundle / "adapter.py"), str(descriptor)], check=True
    )
    assert observe_outputs(tmp_path, "output", outputs) == dict.fromkeys(
        outputs, b"generated"
    )
    request["parameters"]["actions"] = "301"
    descriptor.write_bytes(jcs_dumps(request))
    assert (
        subprocess.run(
            [sys.executable, "-I", str(bundle / "adapter.py"), str(descriptor)],
            capture_output=True,
        ).returncode
        != 0
    )
