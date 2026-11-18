from dataclasses import asdict
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.notary.canonical import jcs_dumps, strict_parse
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.runner import Configuration, run_gates
from axiom_encode.notary.workflows import render

from .chain_fixtures import Epoch
from .lineage_fixtures import LANE
from .test_broker import Inputs
from .test_producer import decode, emission
from .test_producers import enrollment_base
from .test_verification import snapshot


@pytest.fixture
def config():
    epoch = Epoch.create()
    deployment = asdict(Inputs().config)
    deployment["epoch_sha256"] = "0" * 64
    deployment["reviewer_ids"] = sorted(deployment["reviewer_ids"])
    inventory = strict_parse(epoch.inventory)
    inventory["actions"] = [
        [name + "@" + sha, sha]
        for name, sha in (
            ("actions/upload-artifact", "a" * 40),
            ("astral-sh/setup-uv", "b" * 40),
        )
    ]
    deployment["dependency_inventory"] = inventory
    return Configuration(
        jcs_dumps(
            {
                "schema": "axiom/notary-runner/v1",
                "deployment": deployment,
                "encoder_identity": {"git_oid": "e" * 40},
                "signer_endpoint": "https://sign.example.test",
                "publisher_endpoint": "https://publish.example.test",
                "ceremony": None,
            }
        )
    )


def test_workflow_permissions_job_boundaries_and_pins(config):
    files = render(config)
    admission = yaml.safe_load(files[config.deployment.workflow_path])
    jobs = admission["jobs"]
    assert list(jobs) == ["verify", "recompute", "approve", "publish"]
    assert admission["permissions"] == {}
    assert jobs["verify"]["permissions"] == {"contents": "read"}
    assert "environment" not in jobs["verify"]
    assert jobs["recompute"]["permissions"] == {"contents": "read", "actions": "read"}
    assert (
        jobs["approve"]["permissions"]
        == jobs["publish"]["permissions"]
        == {"id-token": "write"}
    )
    assert jobs["approve"]["environment"] == "notary-signing"
    assert jobs["publish"]["environment"] == "notary-publishing"
    assert "secrets." not in "".join(files.values())
    assert "workflow_call" not in admission["on"]
    finalizer = yaml.safe_load(files[config.deployment.finalizer_workflow_path])[
        "jobs"
    ]["finalize"]
    assert finalizer["permissions"] == {"id-token": "write"}
    assert finalizer["environment"] == "notary-publishing"
    assert "runner finalize" in finalizer["steps"][-1]["run"]


def test_workflow_unknown_or_mutable_pins_refuse(config):
    from dataclasses import replace

    config.deployment = replace(
        config.deployment,
        dependency_inventory=jcs_dumps(
            strict_parse(config.deployment.dependency_inventory)
            | {"actions": [["actions/upload-artifact@main", "a" * 40]]}
        ),
    )
    with pytest.raises(IdentityRefusal, match="action_pins"):
        render(config)


@pytest.fixture
def gate_packet():
    commands = {
        "schema": "axiom/notary-gate-commands/v1",
        "lane": LANE,
        "commands": [
            {
                "gate_id": "compile",
                "argv": [
                    "{python}",
                    "-c",
                    "import os; assert not any(k.endswith('TOKEN') for k in os.environ)",
                ],
                "timeout_seconds": 10,
            }
        ],
    }
    epoch = Epoch.create(
        prepare_base=lambda identities: (
            enrollment_base(identities)
            | {".axiom/notary/gate-commands.json": jcs_dumps(commands)}
        )
    )
    enrollment = strict_parse(epoch.active.blobs[".axiom/notary/producers.json"])
    packet = (epoch, enrollment, None, {"base": epoch.active})
    _, files = decode(emission(packet))
    subject = snapshot("c" * 40, epoch.active.blobs | files)
    return epoch, subject


def test_bad_lineage_refuses_before_candidate_checkout(gate_packet, monkeypatch):
    epoch, subject = gate_packet
    subject = snapshot(
        subject.commit, subject.blobs | {"rules/example.yaml": b"tamper"}
    )
    monkeypatch.setattr(
        "axiom_encode.notary.runner.RemoteRepository",
        lambda *_: pytest.fail("candidate checkout must not run"),
    )
    result = strict_parse(
        run_gates(epoch.active, subject, epoch.state(), epoch.inventory)
    )
    assert result["schema"] == "axiom/notary-report-refusal/v1"
    assert result["stage"] == "assignment"


@pytest.mark.parametrize("exit_code, schema", [(0, "pass"), (1, "refusal")])
def test_gate_outcome_is_actual_process_result_and_has_no_credentials(
    gate_packet, monkeypatch, tmp_path, exit_code, schema
):
    epoch, subject = gate_packet

    class Repository:
        path = tmp_path

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def fetch(self, commit):
            return commit

        def _run(self, *args, **kwargs):
            if args[0] == "clone":
                __import__("pathlib").Path(args[-1]).mkdir()

    monkeypatch.setattr(
        "axiom_encode.notary.runner.RemoteRepository", lambda *_: Repository()
    )
    from contextlib import contextmanager

    @contextmanager
    def assets(*args):
        yield {"corpus_root": str(tmp_path)}

    monkeypatch.setattr("axiom_encode.notary.assets.provision", assets)
    monkeypatch.setattr(
        "axiom_encode.toolchain.load_rulespec_local_corpus_release", lambda *args: None
    )
    monkeypatch.setenv("GITHUB_TOKEN", "fixture-must-not-reach-candidate")
    seen = []

    def execute(argv, **kwargs):
        assert "GITHUB_TOKEN" not in kwargs["env"]
        assert "ACTIONS_ID_TOKEN_REQUEST_TOKEN" not in kwargs["env"]
        seen.append(argv)
        return SimpleNamespace(returncode=exit_code)

    report = strict_parse(
        run_gates(
            epoch.active, subject, epoch.state(), epoch.inventory, execute=execute
        )
    )
    assert report["schema"] == "axiom/notary-report-" + schema + "/v1"
    assert len(seen) == 1
    assert report["gates"] == [
        {"gate_id": "compile", "outcome": "pass" if exit_code == 0 else "fail"}
    ]


def test_epoch_is_derived_without_changing_five_file_activation(config):
    from axiom_encode.notary.github_inputs import BootstrapCeremony
    from axiom_encode.notary.runner import bind_epoch

    from .test_administration import genesis_args

    epoch = Epoch.create(
        prepare_base=lambda _: {
            ".axiom/notary/runner.json": b'{"epoch_sha256":"' + b"0" * 64 + b'"}'
        }
    )
    config.ceremony = BootstrapCeremony(genesis_args(epoch), 10, 11, "fixture", 12)
    bind_epoch(config, epoch.base)
    assert config.deployment.epoch_sha256 == epoch.anchor.epoch_sha256
    bind_epoch(config, epoch.active)
    assert config.deployment.epoch_sha256 == epoch.anchor.epoch_sha256
    changed = set(epoch.active.blobs) - set(epoch.base.blobs)
    assert len(changed) == 5
    assert (
        epoch.active.blobs[".axiom/notary/runner.json"]
        == epoch.base.blobs[".axiom/notary/runner.json"]
    )
