"""Git-backed diagnostics must not trust candidate policy or claim admission."""

import json
import subprocess
import sys

import pytest

from axiom_encode.notary.canonical import jcs_dumps
from axiom_encode.notary.lineage import POLICY_PATH, STORE_PREFIX, LineageClassification
from axiom_encode.notary.preflight import inspect_lineage, main
from axiom_encode.notary.refusal import Refusal
from axiom_encode.notary.registry import REGISTRY_PATH
from tests.notary.lineage_fixtures import EPOCH, LANE, Identities, policy_body


@pytest.fixture
def scenario(git_repo):
    identities = Identities.create()
    base = git_repo.commit(
        {
            REGISTRY_PATH: jcs_dumps(identities.body),
            POLICY_PATH: jcs_dumps(policy_body()),
            "README.md": b"fixture\n",
        },
        message="protected base",
    )
    evidence = {
        STORE_PREFIX + name: file.raw for name, file in identities.store().items()
    }
    subject = git_repo.commit(
        evidence | {"rules/example.yaml": b"generated\n"}, message="candidate"
    )
    return git_repo, identities, base, subject


def inspect(scenario, **overrides):
    repo, identities, base, subject = scenario
    kwargs = dict(
        base=base, subject=subject, lane=LANE, epoch_sha256=EPOCH, **identities.pins
    )
    kwargs.update(overrides)
    return inspect_lineage(repo.path, **kwargs)


def cli_args(scenario):
    repo, identities, base, subject = scenario
    args = [
        "--repository",
        str(repo.path),
        "--base",
        base,
        "--subject",
        subject,
        "--lane",
        LANE,
        "--epoch-sha256",
        EPOCH,
    ]
    for key, value in identities.pins.items():
        args += ["--" + key.replace("_", "-"), value]
    return args


def test_committed_generation_and_dirty_checkout_independence(scenario):
    repo, _, _, _ = scenario
    before = inspect(scenario)
    assert isinstance(before, LineageClassification) and len(before.eligible) == 1
    (repo.path / REGISTRY_PATH).write_text("candidate-controlled garbage")
    (repo.path / POLICY_PATH).write_text("candidate-controlled garbage")
    (repo.path / "rules/example.yaml").write_bytes(b"uncommitted hand edit")
    assert inspect(scenario) == before


def test_candidate_cannot_enroll_itself_by_replacing_registry(scenario):
    repo, _, base, _ = scenario
    attacker = Identities.create()
    # Use an independently signed record, without changing the base's trust roots.
    repo.git("checkout", "--quiet", base)
    evidence = {
        STORE_PREFIX + name: file.raw for name, file in attacker.store().items()
    }
    subject = repo.commit(
        {REGISTRY_PATH: jcs_dumps(attacker.body), "rules/example.yaml": b"generated\n"}
        | evidence,
        message="self enrollment attempt",
    )
    result = inspect(scenario, subject=subject)
    assert result.eligible == ()
    assert result.ineligible[0].reasons == ("invalid-signature",)


def test_candidate_cannot_expand_path_policy(scenario):
    repo, identities, base, _ = scenario
    repo.git("checkout", "--quiet", base)
    from tests.notary.lineage_fixtures import generation

    body = generation()
    body["transitions"][0]["path"] = "outside/example"
    evidence = {
        STORE_PREFIX + name: file.raw for name, file in identities.store(body).items()
    }
    policy = policy_body()
    policy["rules"].append({"action": "include", "prefix": "outside"})
    subject = repo.commit(
        evidence | {POLICY_PATH: jcs_dumps(policy), "outside/example": b"generated\n"},
        message="policy expansion attempt",
    )
    result = inspect(scenario, subject=subject)
    assert result.eligible == ()
    assert result.ineligible[0].reasons == ("unprotected-path-transition",)


@pytest.mark.parametrize("revision", ["HEAD", "--help", "a" * 40, "a" * 64])
def test_requires_resolved_pilot_commit_ids(scenario, revision):
    result = inspect(scenario, subject=revision)
    assert isinstance(result, Refusal) and result.code == "subject-unresolvable"


def test_tree_oid_is_not_a_commit(scenario):
    repo, _, _, subject = scenario
    result = inspect(scenario, subject=repo.rev_parse(subject + "^{tree}"))
    assert isinstance(result, Refusal)


def test_whole_tree_symlink_refuses(scenario):
    repo, _, _, _ = scenario
    (repo.path / "outside-symlink").symlink_to("README.md")
    repo.git("add", "outside-symlink")
    repo.git("commit", "-qm", "symlink")
    result = inspect(scenario, subject=repo.rev_parse("HEAD"))
    assert isinstance(result, Refusal) and result.detail == "symlink"


def test_missing_base_registry_cannot_be_supplied_by_subject(scenario):
    repo, _, _, subject = scenario
    base = repo.commit({REGISTRY_PATH: None}, message="remove base registry")
    result = inspect(scenario, base=base, subject=subject)
    assert isinstance(result, Refusal) and result.code == "policy-invalid"


def test_cli_success_is_not_a_notary_pass_report(scenario, capsys):
    assert main(cli_args(scenario)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["admission"] == "not-evaluated"
    assert output["authority"] == "caller-supplied-base-and-pins"
    assert len(output["eligible_records"]) == 1
    assert "schema" not in output and "diff_coverage" not in output


def test_executable_module_is_read_only(scenario):
    repo, _, _, subject = scenario
    result = subprocess.run(
        [sys.executable, "-m", "axiom_encode.notary.preflight", *cli_args(scenario)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["admission"] == "not-evaluated"
    assert repo.rev_parse("HEAD") == subject
    assert repo.git("status", "--porcelain") == b""


def test_eligible_lineage_does_not_claim_candidate_coverage(scenario):
    repo, _, _, _ = scenario
    changed = repo.commit(
        {"rules/example.yaml": b"different final bytes\n"}, message="uncovered edit"
    )
    result = inspect(scenario, subject=changed)
    assert len(result.eligible) == 1  # coverage is a separate mandatory stage
