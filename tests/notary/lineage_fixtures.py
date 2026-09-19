"""Ephemeral test identities; no fixture key is a deployment credential."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex
from axiom_encode.notary.lineage import (
    CORRECTION,
    GENERATION,
    StoreFile,
    parse_path_policy,
)
from axiom_encode.notary.registry import ROLES, parse_registry
from axiom_encode.notary.signatures import ROLE_SCOPES

LANE = "TheAxiomFoundation/rulespec-nz"
EPOCH = "e" * 64


def public_entry(key):
    der = key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return {
        "spki_sha256": sha256_hex(der),
        "public_key_spki_der_base64": base64.b64encode(der).decode(),
    }


@dataclass
class Identities:
    keys: dict
    body: dict
    pins: dict

    @classmethod
    def create(cls):
        keys = {
            role: Ed25519PrivateKey.generate()
            for role in (*ROLES, "legacy_apply_root", "legacy_eval_root")
        }
        body = {"schema": "axiom/notary-key-registry/v1", "lane": LANE}
        body.update({role: [public_entry(keys[role])] for role in ROLES})
        pins = {
            "notary_spki_sha256": public_entry(keys["notary"])["spki_sha256"],
            "legacy_apply_root": public_entry(keys["legacy_apply_root"])["spki_sha256"],
            "legacy_eval_root": public_entry(keys["legacy_eval_root"])["spki_sha256"],
        }
        return cls(keys, body, pins)

    def registry(self, body=None):
        return parse_registry(
            jcs_dumps(self.body if body is None else body), lane=LANE, **self.pins
        )

    def sidecar(self, raw, role, *, key=None, scope=None):
        key = key or self.keys["notary" if role in {"genesis", "transition"} else role]
        scope = scope or ROLE_SCOPES[role]
        digest = sha256_hex(raw)
        # Independent encoding of the published envelope, not a production signing helper.
        message = (
            b"axiom-encode/external-signer-sign/v2\0"
            + scope.encode()
            + b"\0"
            + digest.encode()
        )
        return jcs_dumps(
            {
                "schema": "axiom/detached-signature/v1",
                "body_sha256": digest,
                "scope": scope,
                "signer_spki_sha256": public_entry(key)["spki_sha256"],
                "signature_base64": base64.b64encode(key.sign(message)).decode(),
            }
        )

    def store(self, body=None):
        body = generation() if body is None else body
        raw = jcs_dumps(body)
        name = sha256_hex(raw) + ".json"
        roles = ("actor", "review") if body["schema"] == CORRECTION else ("producer",)
        return {name: StoreFile(raw)} | {
            name + f".{role}.sig": StoreFile(self.sidecar(raw, role)) for role in roles
        }


def generation():
    return {
        "schema": GENERATION,
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "runtime_identity": "supervised-host/fixture",
        "model": "fixture",
        "cli_version": "fixture",
        "cli_sha256": "c" * 64,
        "prompt_sha256s": ["d" * 64],
        "emitted_at": "2026-09-18T12:00:00Z",
        "draw_set_id": "fixture-draw-1",
        "sampling": {"temperature": "0.5", "seed": None},
        "independence": {
            "sibling_draws_visible": "unknown",
            "incumbent_encoding_visible": "yes",
        },
        "source_capture": {
            "id": "fixture-source",
            "content_sha256": "f" * 64,
            "oracles": [],
            "reference_data": [],
        },
        "transitions": [
            {
                "path": "rules/example.yaml",
                "before_blob_sha256": None,
                "before_mode": None,
                "after_blob_sha256": sha256_hex(b"generated\n"),
                "after_mode": "100644",
                "patch_note_sha256": None,
            }
        ],
    }


def correction():
    return {
        "schema": CORRECTION,
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "actor": "fixture-actor",
        "reason": "recorded correction",
        "predecessor_record_sha256": None,
        "transitions": deepcopy(generation()["transitions"]),
    }


def policy_body():
    return {
        "schema": "axiom/notary-path-policy/v1",
        "lane": LANE,
        "rules": [{"action": "include", "prefix": "rules"}],
    }


def policy():
    return parse_path_policy(jcs_dumps(policy_body()), lane=LANE)
