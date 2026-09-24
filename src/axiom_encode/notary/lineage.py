"""Closed v33 lineage schemas and deterministic eligibility, without admission."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from ._schema import (
    canonical_object,
    digest,
    fields,
    lane_name,
    nonempty,
    ordered_strings,
    relative_path,
)
from .canonical import sha256_hex
from .deterministic_contract import (
    DETERMINISTIC_GENERATION,
    file_inventory,
    generator_identity,
    parameters,
    runtime_identity,
)
from .refusal import Refusal
from .registry import KeyRegistry
from .signatures import verify_detached

GENERATION = "axiom/lineage-generation/v1"
GENERATION_SCHEMAS = frozenset({GENERATION, DETERMINISTIC_GENERATION})
CORRECTION = "axiom/lineage-correction/v1"
STORE_PREFIX = ".axiom/lineage/"
POLICY_PATH = ".axiom/notary/path-policy.json"
_BODY_NAME = re.compile(r"([0-9a-f]{64})\.json")
_SIDECAR_NAME = re.compile(r"([0-9a-f]{64})\.json\.(producer|actor|review)\.sig")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")


def signature_role(body):
    """Select the cryptographic domain from a parsed, closed record schema."""
    if body["schema"] == DETERMINISTIC_GENERATION:
        return "deterministic-producer"
    return "producer" if body["schema"] == GENERATION else "actor"


@dataclass(frozen=True, slots=True)
class StoreFile:
    """A raw blob plus its Git mode; keys in a store are literal relative names."""

    raw: bytes
    mode: str = "100644"


@dataclass(frozen=True, slots=True)
class IneligibleRecord:
    store_name: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EligibleRecord:
    body_sha256: str
    raw: bytes

    @property
    def body(self) -> dict:
        """Return a fresh view; callers cannot mutate authenticated record bytes."""
        return canonical_object(self.raw)


@dataclass(frozen=True, slots=True)
class LineageClassification:
    eligible: tuple[EligibleRecord, ...]
    ineligible: tuple[IneligibleRecord, ...]


@dataclass(frozen=True, slots=True)
class PathPolicy:
    lane: str
    rules: tuple[tuple[str, str], ...]

    def protects(self, path: str) -> bool:
        protected = False
        for action, prefix in self.rules:
            if path == prefix or path.startswith(prefix + "/"):
                protected = action == "include"
        return protected


def parse_path_policy(raw: bytes, *, lane: str) -> PathPolicy | Refusal:
    refusal = Refusal("policy-invalid", POLICY_PATH, "invalid_path_policy")
    body = canonical_object(raw)
    if not fields(body, {"schema", "lane", "rules"}):
        return refusal
    if (
        body["schema"] != "axiom/notary-path-policy/v1"
        or not lane_name(lane)
        or body["lane"] != lane
        or not isinstance(body["rules"], list)
    ):
        return refusal
    rules = []
    for rule in body["rules"]:
        if not fields(rule, {"action", "prefix"}):
            return refusal
        if rule["action"] not in ("include", "exclude") or not relative_path(
            rule["prefix"]
        ):
            return refusal
        rules.append((rule["action"], rule["prefix"]))
    policy = PathPolicy(lane, tuple(rules))
    # Membership changes only at a rule prefix. Check the reserved roots and
    # every rule boundary inside them, respecting the last-match semantics.
    for reserved in (".axiom/lineage", ".axiom/notary"):
        boundaries = {reserved} | {
            prefix for _, prefix in rules if prefix.startswith(reserved + "/")
        }
        if any(policy.protects(path) for path in boundaries):
            return refusal
    return policy


def _decimal(value: object, *, integer: bool = False) -> bool:
    pattern = _INTEGER if integer else _DECIMAL
    return (
        isinstance(value, str)
        and value != "-0"
        and pattern.fullmatch(value) is not None
    )


def _timestamp(value: object) -> bool:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:[Zz]|\+00:00)",
            value,
        )
        is None
    ):
        return False
    normalized = value.replace("t", "T").replace("z", "Z")
    # RFC 3339 permits a positive leap second. This is syntax/calendar checking,
    # not an assertion that the producer's wall-clock time was witnessed.
    if normalized[17:19] == "60":
        if normalized[11:16] != "23:59":
            return False
        normalized = normalized[:17] + "59" + normalized[19:]
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return True


def _references(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not fields(item, {"name", "version", "content_sha256"}):
            return False
        if not nonempty(item["name"]) or not nonempty(item["version"]):
            return False
        if item["content_sha256"] is not None and not digest(item["content_sha256"]):
            return False
    return ordered_strings([item["name"] for item in value])


def _transitions(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for transition in value:
        if not fields(
            transition,
            {
                "path",
                "before_blob_sha256",
                "before_mode",
                "after_blob_sha256",
                "after_mode",
                "patch_note_sha256",
            },
        ):
            return False
        if not relative_path(transition["path"]):
            return False
        if transition["patch_note_sha256"] is not None and not digest(
            transition["patch_note_sha256"]
        ):
            return False
        for side in ("before", "after"):
            blob = transition[f"{side}_blob_sha256"]
            mode = transition[f"{side}_mode"]
            if blob is None:
                if mode is not None:
                    return False
            elif not digest(blob) or mode not in ("100644", "100755"):
                return False
    # Duplicates have their own parsed-record reason in v33 §2.4.
    return ordered_strings([item["path"] for item in value], unique=False)


def parse_record(raw: bytes) -> dict | None:
    """Parse canonical generation/correction bodies; no signature or coverage claim."""
    body = canonical_object(raw)
    common = {"schema", "lane", "epoch_sha256", "transitions"}
    if body is None:
        return None
    if body.get("schema") == GENERATION:
        if not fields(
            body,
            common
            | {
                "runtime_identity",
                "model",
                "cli_version",
                "cli_sha256",
                "prompt_sha256s",
                "emitted_at",
                "draw_set_id",
                "sampling",
                "independence",
                "source_capture",
            },
        ):
            return None
        if not all(
            nonempty(body[k])
            for k in ("runtime_identity", "model", "cli_version", "draw_set_id")
        ):
            return None
        if not digest(body["cli_sha256"]) or not _timestamp(body["emitted_at"]):
            return None
        prompts = body["prompt_sha256s"]
        if not ordered_strings(prompts) or not all(digest(p) for p in prompts):
            return None
        sampling = body["sampling"]
        if not fields(sampling, {"temperature", "seed"}) or (
            sampling["temperature"] is not None
            and not _decimal(sampling["temperature"])
        ):
            return None
        if sampling["seed"] is not None and not _decimal(
            sampling["seed"], integer=True
        ):
            return None
        independence = body["independence"]
        if not fields(
            independence, {"sibling_draws_visible", "incumbent_encoding_visible"}
        ):
            return None
        if any(v not in ("yes", "no", "unknown") for v in independence.values()):
            return None
        source = body["source_capture"]
        if not fields(source, {"id", "content_sha256", "oracles", "reference_data"}):
            return None
        if not nonempty(source["id"]) or not digest(source["content_sha256"]):
            return None
        if not _references(source["oracles"]) or not _references(
            source["reference_data"]
        ):
            return None
    elif body.get("schema") == DETERMINISTIC_GENERATION:
        if not fields(
            body,
            common
            | {
                "runtime_identity",
                "generator",
                "runtime",
                "inputs",
                "parameters",
                "emitted_at",
            },
        ) or not (
            nonempty(body["runtime_identity"])
            and generator_identity(body["generator"])
            and runtime_identity(body["runtime"])
            and file_inventory(body["inputs"])
            and parameters(body["parameters"])
            and _timestamp(body["emitted_at"])
        ):
            return None
    elif body.get("schema") == CORRECTION:
        if not fields(body, common | {"actor", "reason", "predecessor_record_sha256"}):
            return None
        if not nonempty(body["actor"]) or not nonempty(body["reason"]):
            return None
        predecessor = body["predecessor_record_sha256"]
        if predecessor is not None and not digest(predecessor):
            return None
    else:
        return None
    if not lane_name(body["lane"]) or not digest(body["epoch_sha256"]):
        return None
    return body if _transitions(body["transitions"]) else None


def classify_lineage(
    base: Mapping[str, StoreFile],
    subject: Mapping[str, StoreFile],
    *,
    lane: str,
    epoch_sha256: str,
    registry: KeyRegistry,
    path_policy: PathPolicy,
) -> LineageClassification | Refusal:
    """Implement store immutability and all eligibility reasons, not diff coverage.

    Caller must supply manifest-validated trees and an authenticated base's
    registry/policy/epoch. This pure function does not authenticate those inputs.
    """
    if (
        not lane_name(lane)
        or not digest(epoch_sha256)
        or registry.lane != lane
        or path_policy.lane != lane
    ):
        return Refusal("policy-invalid", None, "lineage_context_mismatch")
    for name in sorted(base, key=lambda name: name.encode("utf-8")):
        if subject.get(name) != base[name]:
            return Refusal("structural", STORE_PREFIX + name, "lineage_history_changed")
    introduced = subject.keys() - base.keys()
    new_bodies = {name for name in introduced if _BODY_NAME.fullmatch(name)}
    eligible = []
    ineligible = []
    for name in sorted(introduced, key=lambda name: name.encode("utf-8")):
        match = _BODY_NAME.fullmatch(name)
        if match is None:
            sidecar = _SIDECAR_NAME.fullmatch(name)
            if sidecar and sidecar[1] + ".json" in new_bodies:
                continue  # evidence for that new body, never a second report entry
            ineligible.append(IneligibleRecord(name, ("unrecognized-store-name",)))
            continue
        raw = subject[name].raw
        actual_digest = sha256_hex(raw)
        reasons = set()
        if match[1] != actual_digest:
            reasons.add("address-mismatch")
        body = parse_record(raw)
        if body is None:
            reasons.add("malformed-record")
        else:
            roles = (
                ("producer",)
                if body["schema"] in GENERATION_SCHEMAS
                else ("actor", "review")
            )
            if not all(
                verify_detached(
                    (
                        subject[name + f".{role}.sig"].raw
                        if name + f".{role}.sig" in introduced
                        else b""
                    ),
                    body_sha256=actual_digest,
                    role=signature_role(body) if role == "producer" else role,
                    registry=registry,
                )
                for role in roles
            ):
                reasons.add("invalid-signature")
            if body["lane"] != lane:
                reasons.add("wrong-lane")
            if body["epoch_sha256"] != epoch_sha256:
                reasons.add("wrong-epoch")
            paths = [transition["path"] for transition in body["transitions"]]
            if len(paths) != len(set(paths)):
                reasons.add("duplicate-transition-paths")
            if any(not path_policy.protects(path) for path in paths):
                reasons.add("unprotected-path-transition")
        if reasons:
            ineligible.append(IneligibleRecord(name, tuple(sorted(reasons))))
        else:
            eligible.append(EligibleRecord(actual_digest, raw))
    return LineageClassification(
        tuple(sorted(eligible, key=lambda record: record.body_sha256)),
        tuple(ineligible),
    )
