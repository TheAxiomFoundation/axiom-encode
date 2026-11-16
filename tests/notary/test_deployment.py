import os
from dataclasses import asdict

import pytest
from cryptography.hazmat.primitives import serialization

from axiom_encode.notary.canonical import jcs_dumps, strict_parse
from axiom_encode.notary.deployment import build_app, custodian_file, parse_deployment
from axiom_encode.notary.identity import IdentityRefusal

from .chain_fixtures import Epoch
from .test_broker import Inputs


@pytest.fixture
def config():
    epoch = Epoch.create()
    deployment = asdict(Inputs().config)
    deployment["reviewer_ids"] = sorted(deployment["reviewer_ids"])
    deployment["dependency_inventory"] = strict_parse(epoch.inventory)
    return {
        "schema": "axiom/notary-service-config/v1",
        "role": "signer",
        "deployment": deployment,
        "ceremony": None,
        "encoder_identity": {
            "repository": "TheAxiomFoundation/axiom-encode",
            "git_oid": "e" * 40,
            "version": "fixture",
            "package_tree_sha256": "f" * 64,
        },
        "credentials": {},
    }


def test_public_configuration_round_trip(config):
    value = parse_deployment(config["deployment"])
    assert value.reviewer_ids == frozenset({123})
    assert strict_parse(value.dependency_inventory)["verifier"]["git_oid"] == "e" * 40


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", "stranger/repo"),
        ("content_branch", "main/../other"),
        ("workflow_path", "../workflow"),
        ("reviewer_ids", [True]),
        ("chain_app_id", False),
        ("epoch_sha256", "missing"),
    ],
)
def test_configuration_refuses_ambiguous_identity(config, field, value):
    config["deployment"][field] = value
    with pytest.raises(IdentityRefusal):
        parse_deployment(config["deployment"])


def test_private_key_file_cannot_be_symlink_or_world_readable(tmp_path):
    path = tmp_path.resolve() / "key"
    path.write_bytes(b"fixture-only")
    path.chmod(0o600)
    assert custodian_file(str(path), private=True) == b"fixture-only"
    link = path.with_name("alias")
    link.symlink_to(path)
    with pytest.raises(IdentityRefusal, match="custodian_path"):
        custodian_file(str(link), private=True)
    path.chmod(0o644)
    with pytest.raises(IdentityRefusal, match="permissions"):
        custodian_file(str(path), private=True)


def test_key_hardlink_and_writable_parent_refuse(tmp_path):
    path = tmp_path.resolve() / "key"
    path.write_bytes(b"fixture-only")
    path.chmod(0o600)
    os.link(path, path.with_name("duplicate"))
    with pytest.raises(IdentityRefusal, match="permissions"):
        custodian_file(str(path), private=True)
    path.with_name("duplicate").unlink()
    tmp_path.chmod(0o777)
    with pytest.raises(IdentityRefusal, match="parent"):
        custodian_file(str(path), private=True)


def test_signer_startup_checks_running_code_and_has_no_publisher_credentials(
    config, tmp_path, monkeypatch
):
    epoch = Epoch.create()
    token, key = tmp_path.resolve() / "read", tmp_path.resolve() / "notary.pem"
    token.write_text("fixture-read-only")
    key.write_bytes(
        epoch.identities.keys["notary"].private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    for path in (token, key):
        path.chmod(0o600)
    config["credentials"] = {
        "github_read_token_file": str(token),
        "notary_key_file": str(key),
        "approval_directory": str(tmp_path.resolve() / "approvals"),
    }
    (tmp_path / "approvals").mkdir(mode=0o700)
    observed = []
    monkeypatch.setattr(
        "axiom_encode.notary.deployment.require_running_identity",
        lambda expected, inventory: observed.append((expected, inventory)),
    )
    assert build_app(jcs_dumps(config))
    assert observed[0][0] == config["encoder_identity"]
    config["credentials"]["lane_app_key"] = "must-refuse"
    with pytest.raises(IdentityRefusal, match="credential_separation"):
        build_app(jcs_dumps(config))


def test_broker_cannot_start_with_read_token_or_notary_key(config, monkeypatch):
    monkeypatch.setattr(
        "axiom_encode.notary.deployment.require_running_identity", lambda *_: None
    )
    config["role"] = "publisher-broker"
    config["credentials"] = {
        "github_read_token_file": "forbidden",
        "notary_key_file": "forbidden",
    }
    with pytest.raises(IdentityRefusal, match="credential_separation"):
        build_app(jcs_dumps(config))


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o755, 0o711])
def test_socket_parent_cannot_be_replaced_or_accessed_by_unrelated_user(tmp_path, mode):
    from axiom_encode.notary.deployment import custodian_parent

    tmp_path.chmod(mode)
    with pytest.raises(IdentityRefusal, match="socket_directory"):
        custodian_parent(tmp_path.resolve() / "service.sock", socket=True)


def test_socket_directory_can_allow_proxy_execute_without_write(tmp_path):
    from axiom_encode.notary.deployment import custodian_parent

    tmp_path.chmod(0o710)
    custodian_parent(tmp_path.resolve() / "service.sock", socket=True)


def test_running_encoder_uses_official_attestation_repository_spelling(monkeypatch):
    from axiom_encode import __version__
    from axiom_encode.notary.deployment import require_running_identity

    # The global fixture uses the real legacy attestation's host-qualified
    # spelling. Public v33 configuration intentionally uses owner/name.
    monkeypatch.setattr(
        "axiom_encode.harness.evals._deterministic_tree_identity",
        lambda *a, **kw: {"tree_sha256": "b" * 64},
    )
    expected = {
        "repository": "TheAxiomFoundation/axiom-encode",
        "git_oid": "a" * 40,
        "version": __version__,
        "package_tree_sha256": "b" * 64,
    }
    inventory = jcs_dumps(
        {"verifier": {"repo": expected["repository"], "git_oid": expected["git_oid"]}}
    )
    require_running_identity(expected, inventory)
    for field, value in [
        ("repository", "outsider/axiom-encode"),
        ("git_oid", "c" * 40),
        ("version", "wrong"),
        ("package_tree_sha256", "d" * 64),
    ]:
        with pytest.raises(IdentityRefusal, match="encoder_identity"):
            require_running_identity(expected | {field: value}, inventory)
