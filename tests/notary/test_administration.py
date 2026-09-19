from dataclasses import replace

import pytest

from axiom_encode.notary.administration import build_genesis, build_transition
from axiom_encode.notary.canonical import jcs_dumps, sha256_hex, strict_parse
from axiom_encode.notary.chain import BOOTSTRAP_PATHS, InvalidChain, reconstruct
from axiom_encode.notary.consumer import epoch_template
from axiom_encode.notary.manifest import manifest_sha256

from .chain_fixtures import Epoch
from .test_verification import snapshot


@pytest.fixture
def epoch():
    return Epoch.create()


def genesis_args(epoch):
    g = strict_parse(epoch.history[0].blobs[epoch.anchor.epoch_sha256 + ".json"])
    return {
        "lane": g["lane"],
        "notary_repository": g["notary_repository"],
        "prospective": {
            path: epoch.history[0].blobs[g["bootstrap_policies"][name] + ".raw"]
            for name, path in BOOTSTRAP_PATHS.items()
        },
        "consumer_spec_path": g["consumer_spec_path"],
        "consumer_template": epoch_template(
            epoch.active.blobs[g["consumer_spec_path"]], epoch.anchor.epoch_sha256
        ),
        "legacy_apply_root": g["legacy_apply_root"],
        "legacy_eval_root": g["legacy_eval_root"],
        "frozen_apply_spki_sha256": epoch.identities.pins["legacy_apply_root"],
        "frozen_eval_spki_sha256": epoch.identities.pins["legacy_eval_root"],
        "ceremony_admin_spki_sha256": epoch.identities.body["admin-approver"][0][
            "spki_sha256"
        ],
        "ceremony_notary_spki_sha256": epoch.anchor.notary_spki_sha256,
        "expected_encoder_identity": {
            "repository": "TheAxiomFoundation/axiom-encode",
            "commit": "a" * 40,
            "version": "fixture",
        },
        "local_corpus_release": None,
    }


def test_genesis_builder_recomputes_fixture(epoch):
    raw = build_genesis(epoch.base, **genesis_args(epoch))
    assert sha256_hex(raw) == epoch.anchor.epoch_sha256


def test_genesis_refuses_unsubstitutable_escaped_epoch_template(epoch):
    args = genesis_args(epoch)
    args["consumer_template"] = args["consumer_template"].replace(
        b"0" * 64, b"\\u0030" * 64
    )
    with pytest.raises(InvalidChain, match="consumer_template"):
        build_genesis(epoch.base, **args)


@pytest.mark.parametrize("suffix", ["", "-other"])
def test_genesis_requires_dedicated_lane_notary_repository(epoch, suffix):
    args = genesis_args(epoch)
    args["notary_repository"] = args["lane"] + suffix
    with pytest.raises(InvalidChain, match="dedicated_notary_repository"):
        build_genesis(epoch.base, **args)


@pytest.mark.parametrize(
    "field",
    [
        "frozen_apply_spki_sha256",
        "frozen_eval_spki_sha256",
        "ceremony_admin_spki_sha256",
        "ceremony_notary_spki_sha256",
    ],
)
def test_genesis_requires_witnessed_roots_and_ceremony_keys(epoch, field):
    args = genesis_args(epoch)
    args[field] = "f" * 64
    with pytest.raises(InvalidChain):
        build_genesis(epoch.base, **args)


def test_genesis_unattested_baseline_is_visible_and_exhaustive(epoch):
    base = snapshot(
        epoch.base.commit, epoch.base.blobs | {"rules/a": b"old", "rules/b": b"other"}
    )
    result = strict_parse(build_genesis(base, **genesis_args(epoch)))
    assert result["v5_attested"] == []
    assert result["baseline_unattested"] == [
        ["rules/a", sha256_hex(b"old")],
        ["rules/b", sha256_hex(b"other")],
    ]


def test_genesis_protected_executable_refuses(epoch):
    base = snapshot(epoch.base.commit, epoch.base.blobs | {"rules/a": b"old"})
    base = replace(
        base,
        manifest=[
            (p, "100755" if p == "rules/a" else m, d) for p, m, d in base.manifest
        ],
    )
    with pytest.raises(InvalidChain, match="protected_mode"):
        build_genesis(base, **genesis_args(epoch))


def test_activation_builder_matches_all_installed_values(epoch):
    state = reconstruct(epoch.history[:2], epoch.anchor)
    raw = build_transition(
        epoch.base, epoch.active, state, reason="activate fixture epoch"
    )
    pending = next(
        raw
        for name, raw in epoch.history[2].blobs.items()
        if name.endswith(".json")
        and name != "HEAD.json"
        and strict_parse(raw)["schema"] == "axiom/notary-transition/v1"
    )
    assert raw == pending


@pytest.mark.parametrize(
    "kind", ["unrelated", "protected", "wrong-policy", "wrong-consumer"]
)
def test_activation_refuses_smuggled_changes(epoch, kind):
    state = reconstruct(epoch.history[:2], epoch.anchor)
    blobs = dict(epoch.active.blobs)
    if kind == "unrelated":
        blobs["README"] = b"smuggled"
    elif kind == "protected":
        blobs["rules/x"] = b"smuggled"
    elif kind == "wrong-policy":
        blobs[".axiom/notary/profile.json"] = b"{}"
    else:
        body = strict_parse(blobs[".axiom/notary/consumer.json"])
        body["notary_repository"] = "Other/chain"
        blobs[".axiom/notary/consumer.json"] = jcs_dumps(body)
    with pytest.raises(InvalidChain):
        build_transition(
            epoch.base, snapshot("f" * 40, blobs), state, reason="activation"
        )


def test_transition_cannot_hide_ordinary_content(epoch):
    subject = snapshot(
        "d" * 40, epoch.active.blobs | {"rules/example.yaml": b"hand edit"}
    )
    with pytest.raises(InvalidChain, match="nontrust_delta"):
        build_transition(
            epoch.active,
            subject,
            epoch.state(),
            reason="admin should not admit content",
        )


def test_noop_transition_refuses(epoch):
    with pytest.raises(InvalidChain, match="state_identical"):
        build_transition(epoch.active, epoch.active, epoch.state(), reason="no-op")


def test_transition_stale_base_refuses(epoch):
    with pytest.raises(InvalidChain, match="stale_base"):
        build_transition(epoch.base, epoch.active, epoch.state(), reason="stale")


def test_successor_cannot_make_a_key_serve_two_roles(epoch):
    registry = strict_parse(epoch.active.blobs[".axiom/notary/keys.json"])
    registry["producer"] = registry["approver"]
    subject = snapshot(
        "d" * 40, epoch.active.blobs | {".axiom/notary/keys.json": jcs_dumps(registry)}
    )
    with pytest.raises(InvalidChain, match="invalid_registry"):
        build_transition(epoch.active, subject, epoch.state(), reason="bad alias")


def test_current_tree_commit_can_differ_after_squash(epoch):
    base = replace(epoch.active, commit="d" * 40)
    registry = strict_parse(base.blobs[".axiom/notary/keys.json"])
    registry["producer"] = []
    subject = snapshot(
        "e" * 40, base.blobs | {".axiom/notary/keys.json": jcs_dumps(registry)}
    )
    raw = build_transition(base, subject, epoch.state(), reason="revoke producer")
    assert strict_parse(raw)["base_tree_manifest_sha256"] == manifest_sha256(
        epoch.active.manifest
    )
