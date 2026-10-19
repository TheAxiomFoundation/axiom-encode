"""Closed, reviewed patch packages; never a source or signature authority.

The registry is intentionally empty. Adding an entry requires independent review
of the immutable package and a normal encoder change. Caller hashes do not grant
admission. Command integration must bind these bytes to actual validated state.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

OFFICIAL_REPOSITORY = "https://github.com/TheAxiomFoundation/axiom-encode.git"
SCHEMA = "axiom-encode/reviewed-repair-patch/v1"
MAX_MEMBER_BYTES = 1024 * 1024
MAX_PACKAGE_BYTES = 2 * 1024 * 1024
MEMBERS = ("patch.json", "rules.patch.yaml", "tests.patch.yaml")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True)
class ReviewedPatchRegistration:
    """Code-reviewed immutable Git blobs, never populated from dispatch data."""

    commit: str
    # Entries follow MEMBERS order: Git blob id, SHA256, byte length.
    members: tuple[tuple[str, str, int], ...]


REVIEWED_REPAIR_PATCHES = MappingProxyType({})


@dataclass(frozen=True)
class ReviewedRepairPackage:
    patch_id: str
    commit: str
    manifest_bytes: bytes
    rules_bytes: bytes
    tests_bytes: bytes

    @property
    def manifest(self) -> dict[str, Any]:
        # Return a fresh tree: mutation of a caller's view cannot change authority.
        return strict_json(self.manifest_bytes)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_json(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_MEMBER_BYTES:
        raise ValueError("reviewed patch JSON exceeds bounds or is empty")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("reviewed patch JSON contains duplicate keys")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("reviewed patch JSON contains a nonfinite number")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("reviewed patch JSON is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError("reviewed patch JSON must be an object")
    return value


class _PatchLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result or key == "<<":
                raise ValueError("reviewed patch YAML keys must be unique strings")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def strict_yaml(raw: bytes, *, tests: bool) -> object:
    if not raw or len(raw) > MAX_MEMBER_BYTES:
        raise ValueError("reviewed patch YAML exceeds bounds or is empty")
    try:
        text = raw.decode("utf-8")
        # Aliases/anchors cannot introduce shared mutable or recursively aliased nodes.
        if any(
            isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
            for t in yaml.scan(text)
        ):
            raise ValueError("reviewed patch YAML aliases are not supported")
        value = yaml.load(text, Loader=_PatchLoader)
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise ValueError("reviewed patch YAML is malformed") from exc
    if tests:
        if not isinstance(value, list):
            raise ValueError("reviewed patch tests must be a list of named cases")
        _named_items(value, "cases")
    else:
        allowed = {"format", "module", "using", "units", "imports", "inputs", "rules"}
        if (
            not isinstance(value, dict)
            or set(value) - allowed
            or value.get("format") != "rulespec/v1"
        ):
            raise ValueError("reviewed patch RuleSpec has invalid top-level fields")
        if "rules" not in value:
            raise ValueError("reviewed patch requires a rules list")
        for key in ("rules", "inputs"):
            if key in value:
                _named_items(value[key], key)
    return value


def _named_items(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"reviewed patch {label} must be a list")
    seen = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or item["name"] in seen
        ):
            raise ValueError(f"reviewed patch {label} names must be unique")
        seen.add(item["name"])
        if "repair_remove" in item and (
            set(item) != {"name", "repair_remove"} or item["repair_remove"] is not True
        ):
            raise ValueError("reviewed patch retirement marker is not exact")


def _fields(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"reviewed patch {label} fields are not exact")
    return value


def _digest(value: object, pattern=_SHA) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError("reviewed patch digest or commit is invalid")


def validate_manifest(raw: bytes, patch_id: str) -> dict:
    value = strict_json(raw)
    _fields(
        value,
        {
            "schema",
            "patch_id",
            "citation",
            "rulespec_path",
            "rulespec_base_commit",
            "predecessor",
            "seed",
            "source",
            "contract_sha256",
            "engine_ref",
            "changes",
            "expected_postimage",
        },
        "manifest",
    )
    if value["schema"] != SCHEMA or value["patch_id"] != patch_id:
        raise ValueError("reviewed patch schema or id mismatch")
    from .prepare_signed_backfill import citation_rulespec_path

    if citation_rulespec_path(value["citation"]).as_posix() != value["rulespec_path"]:
        raise ValueError("reviewed patch citation/path mismatch")
    for field in ("rulespec_base_commit", "engine_ref"):
        _digest(value[field], _COMMIT)
    _digest(value["contract_sha256"])
    predecessor = _fields(
        value["predecessor"],
        {
            "path",
            "blob",
            "sha256",
            "schema",
            "rulespec_sha256",
            "tests_sha256",
            "canonical_manifest",
        },
        "predecessor",
    )
    target = Path(value["rulespec_path"])
    canonical = Path(".axiom/encoding-manifests") / target.with_suffix(".json")
    historical = Path(".axiom/encoding-manifests") / Path(
        *target.parts[1:]
    ).with_suffix(".json")
    if predecessor["path"] not in {canonical.as_posix(), historical.as_posix()}:
        raise ValueError("reviewed patch predecessor path is not target-specific")
    if predecessor["canonical_manifest"] != (
        "same" if predecessor["path"] == canonical.as_posix() else "absent"
    ):
        raise ValueError("reviewed patch canonical predecessor state invalid")
    if predecessor["schema"] not in {
        "axiom-encode/applied-rulespec/v1",
        "axiom-encode/applied-rulespec/v5",
    }:
        raise ValueError("reviewed patch predecessor schema unsupported")
    _digest(predecessor["blob"], _COMMIT)
    for field in ("sha256", "rulespec_sha256", "tests_sha256"):
        _digest(predecessor[field])
    seed = _fields(
        value["seed"],
        {
            "run_id",
            "run_attempt",
            "encoder_commit",
            "archive_sha256",
            "rulespec_sha256",
            "tests_sha256",
            "issues_sha256",
        },
        "seed",
    )
    for field in ("run_id", "run_attempt"):
        if not isinstance(seed[field], str) or not re.fullmatch(
            r"[1-9][0-9]*", seed[field]
        ):
            raise ValueError("reviewed patch seed run identity invalid")
    for field in ("archive_sha256", "rulespec_sha256", "tests_sha256", "issues_sha256"):
        _digest(seed[field])
    _digest(seed["encoder_commit"], _COMMIT)
    source = _fields(
        value["source"],
        {
            "corpus_ref",
            "release_name",
            "release_sha256",
            "requested_citation",
            "resolved_citation",
            "body_sha256",
        },
        "source",
    )
    _digest(source["corpus_ref"], _COMMIT)
    for field in ("release_sha256", "body_sha256"):
        _digest(source[field])
    if source["requested_citation"] != value["citation"]:
        raise ValueError("reviewed patch requested source mismatch")
    if any(
        not isinstance(source[k], str) or not source[k]
        for k in ("release_name", "resolved_citation")
    ):
        raise ValueError("reviewed patch source identity invalid")
    changes = _fields(
        value["changes"], {"top_level", "rules", "inputs", "cases"}, "changes"
    )
    if (
        not isinstance(changes["top_level"], list)
        or len(set(changes["top_level"])) != len(changes["top_level"])
        or any(
            k not in {"module", "using", "units", "imports"}
            for k in changes["top_level"]
        )
    ):
        raise ValueError("reviewed patch top-level changes invalid")
    for kind in ("rules", "inputs", "cases"):
        groups = _fields(changes[kind], {"added", "replaced", "removed"}, kind)
        seen = set()
        for names in groups.values():
            if not isinstance(names, list):
                raise ValueError("reviewed patch change inventory must be lists")
            for name in names:
                if not isinstance(name, str) or not name or name in seen:
                    raise ValueError("reviewed patch change inventory overlaps")
                seen.add(name)
    post = _fields(
        value["expected_postimage"],
        {
            "rulespec_sha256",
            "tests_sha256",
            "rulespec_object_sha256",
            "tests_object_sha256",
        },
        "postimage",
    )
    for digest in post.values():
        _digest(digest)
    return value


def _registration(patch_id: object) -> ReviewedPatchRegistration:
    if not isinstance(patch_id, str) or _ID.fullmatch(patch_id) is None:
        raise ValueError("reviewed patch id invalid")
    registration = REVIEWED_REPAIR_PATCHES.get(patch_id)
    if not isinstance(registration, ReviewedPatchRegistration):
        raise ValueError("reviewed patch id is not registered")
    _digest(registration.commit, _COMMIT)
    if not isinstance(registration.members, tuple) or len(registration.members) != len(
        MEMBERS
    ):
        raise ValueError("reviewed patch registry member count invalid")
    for member in registration.members:
        if not isinstance(member, tuple) or len(member) != 3:
            raise ValueError("reviewed patch registry member shape invalid")
        blob, digest, size = member
        _digest(blob, _COMMIT)
        _digest(digest)
        if type(size) is not int or not 0 < size <= MAX_MEMBER_BYTES:
            raise ValueError("reviewed patch registry size invalid")
    if sum(member[2] for member in registration.members) > MAX_PACKAGE_BYTES:
        raise ValueError("reviewed patch registry package exceeds bound")
    return registration


def load_registered_patch(checkout: Path, patch_id: str) -> ReviewedRepairPackage:
    """Read only registry-pinned official Git blobs; no caller path/hash authority."""
    registration = _registration(patch_id)

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(checkout), *args])

    if git("remote", "get-url", "origin").decode().strip() != OFFICIAL_REPOSITORY:
        raise ValueError("reviewed patch checkout must have official origin")
    prefix = f"reviewed-repair-patches/{patch_id}"
    tree = git("ls-tree", "-r", "-z", registration.commit, "--", prefix)
    entries = {}
    for record in tree.split(b"\0"):
        if record:
            header, path = record.split(b"\t", 1)
            mode, kind, blob = header.decode().split()
            entries[path.decode()] = (mode, kind, blob)
    if set(entries) != {f"{prefix}/{name}" for name in MEMBERS}:
        raise ValueError("reviewed patch package has missing or extra files")
    result = []
    total = 0
    for name, (blob, digest, size) in zip(MEMBERS, registration.members, strict=True):
        _digest(blob, _COMMIT)
        _digest(digest)
        if type(size) is not int or not 0 < size <= MAX_MEMBER_BYTES:
            raise ValueError("reviewed patch registry size invalid")
        total += size
        if total > MAX_PACKAGE_BYTES:
            raise ValueError("reviewed patch package exceeds total bound")
        if entries[f"{prefix}/{name}"] != ("100644", "blob", blob):
            raise ValueError("reviewed patch member mode or Git blob mismatch")
        if int(git("cat-file", "-s", blob)) != size:
            raise ValueError("reviewed patch Git blob size mismatch")
        raw = git("cat-file", "blob", blob)
        if len(raw) != size or sha256(raw) != digest:
            raise ValueError("reviewed patch bytes do not match registry")
        result.append(raw)
    validate_manifest(result[0], patch_id)
    strict_yaml(result[1], tests=False)
    strict_yaml(result[2], tests=True)
    return ReviewedRepairPackage(patch_id, registration.commit, *result)


def verify_bindings(package: ReviewedRepairPackage, actual: dict[str, object]) -> None:
    """Compare complete independently measured context, not caller assertions.

    The protected command must assemble actual from the normal authenticated
    seed/source/base/engine/contract validators; this pure comparison grants no
    execution or signing capability on its own.
    """
    fields = {
        "citation",
        "rulespec_path",
        "rulespec_base_commit",
        "predecessor",
        "seed",
        "source",
        "contract_sha256",
        "engine_ref",
    }
    _fields(actual, fields, "measured bindings")
    manifest = package.manifest
    if canonical_json(actual) != canonical_json({k: manifest[k] for k in fields}):
        raise ValueError("reviewed patch differs from authenticated execution bindings")


def reconstruct_postimage(
    package: ReviewedRepairPackage, *, rulespec: bytes, tests: bytes
) -> tuple[bytes, bytes]:
    """Run the existing overlay in scratch space and require reviewed exact bytes."""
    import tempfile

    from .harness.evals import (
        ValidationRetryCandidate,
        _overlay_validation_retry_candidate,
    )

    manifest = package.manifest
    if (
        sha256(rulespec) != manifest["seed"]["rulespec_sha256"]
        or sha256(tests) != manifest["seed"]["tests_sha256"]
    ):
        raise ValueError("reviewed patch seed bytes differ")
    before_rule = strict_yaml(rulespec, tests=False)
    before_test = strict_yaml(tests, tests=True)
    strict_yaml(package.rules_bytes, tests=False)
    strict_yaml(package.tests_bytes, tests=True)
    with tempfile.TemporaryDirectory(prefix="reviewed-repair-overlay-") as temporary:
        root = Path(temporary)
        target = root / "patch.yaml"
        companion = root / "patch.test.yaml"
        target.write_bytes(package.rules_bytes)
        companion.write_bytes(package.tests_bytes)
        _overlay_validation_retry_candidate(
            target,
            artifact_root=root,
            candidate=ValidationRetryCandidate(rulespec.decode(), tests.decode()),
        )
        after_rules, after_tests = target.read_bytes(), companion.read_bytes()
    after_rule = strict_yaml(after_rules, tests=False)
    after_test = strict_yaml(after_tests, tests=True)
    expected = manifest["expected_postimage"]
    measured = {
        "rulespec_sha256": sha256(after_rules),
        "tests_sha256": sha256(after_tests),
        "rulespec_object_sha256": sha256(canonical_json(after_rule)),
        "tests_object_sha256": sha256(canonical_json(after_test)),
    }
    if measured != expected:
        raise ValueError(
            "reviewed repair overlay postimage differs from reviewed bytes"
        )
    changes = {
        "top_level": sorted(
            k
            for k in set(before_rule) | set(after_rule)
            if k not in {"rules", "inputs"} and before_rule.get(k) != after_rule.get(k)
        )
    }
    for kind, left, right in (
        ("rules", before_rule.get("rules", []), after_rule.get("rules", [])),
        ("inputs", before_rule.get("inputs", []), after_rule.get("inputs", [])),
        ("cases", before_test, after_test),
    ):
        a, b = ({x["name"]: x for x in items} for items in (left, right))
        changes[kind] = {
            "added": sorted(b.keys() - a.keys()),
            "removed": sorted(a.keys() - b.keys()),
            "replaced": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
        }
    declared = manifest["changes"]
    if canonical_json(changes) != canonical_json(
        {
            "top_level": sorted(declared["top_level"]),
            **{
                k: {action: sorted(names) for action, names in declared[k].items()}
                for k in ("rules", "inputs", "cases")
            },
        }
    ):
        raise ValueError("reviewed repair semantic inventory differs")
    return after_rules, after_tests


def verify_predecessor(
    checkout: Path, package: ReviewedRepairPackage
) -> dict[str, bytes]:
    """Bind historical lineage to reviewed base bytes; never authenticate HMAC.

    Normal replacement/source/collision guards remain mandatory. In particular,
    this does not relocate, delete, or promote a historical manifest.
    """
    from .corpus_resolver import read_bounded_regular_file

    manifest = package.manifest
    base = manifest["rulespec_base_commit"]
    _digest(base, _COMMIT)
    target = Path(manifest["rulespec_path"])
    previous = manifest["predecessor"]
    canonical = Path(".axiom/encoding-manifests") / target.with_suffix(".json")

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(checkout), *args])

    if git("rev-parse", "HEAD").decode().strip() != base:
        raise ValueError("reviewed repair checkout is not exact reviewed base")
    if git("status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"):
        raise ValueError("reviewed repair requires a strictly clean base")
    if git("remote", "get-url", "origin").decode().strip() != (
        f"https://github.com/TheAxiomFoundation/rulespec-{target.parts[0]}.git"
    ):
        raise ValueError("reviewed repair base origin invalid")
    snapshot = {}
    for path, digest in (
        (target, previous["rulespec_sha256"]),
        (target.with_suffix(".test.yaml"), previous["tests_sha256"]),
        (Path(previous["path"]), previous["sha256"]),
    ):
        entry = git("ls-tree", base, "--", path.as_posix()).decode().strip()
        if not entry:
            raise ValueError("reviewed predecessor file absent from exact base")
        header, name = entry.split("\t", 1)
        mode, kind, blob = header.split()
        if mode != "100644" or kind != "blob" or name != path.as_posix():
            raise ValueError("reviewed predecessor must be tracked regular bytes")
        if path == Path(previous["path"]) and blob != previous["blob"]:
            raise ValueError("reviewed predecessor Git blob differs")
        if int(git("cat-file", "-s", blob)) > MAX_MEMBER_BYTES:
            raise ValueError("reviewed predecessor exceeds size bound")
        raw = git("cat-file", "blob", blob)
        live = read_bounded_regular_file(
            checkout,
            checkout / path,
            label="reviewed predecessor",
            max_bytes=MAX_MEMBER_BYTES,
            required_mode=0o644,
        )
        if raw != live or sha256(raw) != digest:
            raise ValueError("reviewed predecessor bytes differ from bound base")
        snapshot[path.as_posix()] = raw
    old = strict_json(snapshot[previous["path"]])
    if old.get("schema_version") != previous["schema"]:
        raise ValueError("reviewed predecessor schema differs")
    if previous["canonical_manifest"] == "absent":
        if git("ls-tree", base, "--", canonical.as_posix()):
            raise ValueError("canonical manifest unexpectedly present at base")
        path = checkout / canonical
        if path.exists() or path.is_symlink():
            raise ValueError("canonical manifest unexpectedly present live")
    return snapshot


TOOL = "axiom-encode apply-reviewed-repair"
# Existing apply-manifest readers enforce 1 MiB. Reserve envelope space and
# enforce the complete serialized signed manifest before any signer call.
MAX_RECEIPT_BYTES = 768 * 1024
MAX_DECODED_RECEIPT_BYTES = 576 * 1024


def git_blob_id(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def build_repair_binding(
    package: ReviewedRepairPackage, *, rulespec: bytes, tests: bytes
) -> dict:
    """Carry reconstruction evidence, not new trust roots, inside the signature."""
    reconstruct_postimage(package, rulespec=rulespec, tests=tests)
    binding = {
        "schema": "axiom-encode/reviewed-repair-binding/v1",
        "patch_id": package.patch_id,
        "package_commit": package.commit,
        "package": {
            name: base64.b64encode(raw).decode("ascii")
            for name, raw in zip(
                MEMBERS,
                (package.manifest_bytes, package.rules_bytes, package.tests_bytes),
                strict=True,
            )
        },
        "seed": {
            "rulespec": base64.b64encode(rulespec).decode("ascii"),
            "tests": base64.b64encode(tests).decode("ascii"),
        },
    }
    if len(canonical_json(binding)) > MAX_RECEIPT_BYTES:
        raise ValueError("reviewed repair reconstruction receipt exceeds bound")
    verify_repair_binding(binding)
    return binding


def verify_repair_binding(
    binding: object,
) -> tuple[ReviewedRepairPackage, tuple[bytes, bytes]]:
    """Reconstruct signed postimages offline using only code-reviewed registry."""
    item = _fields(
        binding, {"schema", "patch_id", "package_commit", "package", "seed"}, "receipt"
    )
    if len(canonical_json(item)) > MAX_RECEIPT_BYTES:
        raise ValueError("reviewed repair receipt exceeds bound")
    if item["schema"] != "axiom-encode/reviewed-repair-binding/v1":
        raise ValueError("reviewed repair receipt schema invalid")
    registration = _registration(item["patch_id"])
    _digest(item["package_commit"], _COMMIT)
    if item["package_commit"] != registration.commit:
        raise ValueError("reviewed repair receipt is not registered")
    members = _fields(item["package"], set(MEMBERS), "receipt package")
    seed = _fields(item["seed"], {"rulespec", "tests"}, "receipt seed")
    encoded = [*members.values(), *seed.values()]
    if any(not isinstance(v, str) for v in encoded):
        raise ValueError("reviewed repair receipt members must be text")
    # A conservative pre-decode upper bound; never allocate an oversized decoded
    # package and only later discover the aggregate limit.
    if sum(3 * ((len(v) + 3) // 4) for v in encoded) > MAX_DECODED_RECEIPT_BYTES:
        raise ValueError("reviewed repair decoded receipt exceeds bound")
    raws = []
    for name, (blob, digest, size) in zip(MEMBERS, registration.members, strict=True):
        if not isinstance(members[name], str):
            raise ValueError("reviewed repair receipt member is not text")
        raw = _decode_member(members[name])
        if (
            type(size) is not int
            or not 0 < size <= MAX_MEMBER_BYTES
            or len(raw) != size
            or sha256(raw) != digest
            or git_blob_id(raw) != blob
        ):
            raise ValueError("reviewed repair receipt member differs from registry")
        raws.append(raw)
    if sum(map(len, raws)) > MAX_PACKAGE_BYTES:
        raise ValueError("reviewed repair receipt package exceeds bound")
    validate_manifest(raws[0], item["patch_id"])
    package = ReviewedRepairPackage(item["patch_id"], registration.commit, *raws)
    seed = _fields(item["seed"], {"rulespec", "tests"}, "receipt seed")
    if any(not isinstance(v, str) for v in seed.values()):
        raise ValueError("reviewed repair receipt seed must be text")
    return package, reconstruct_postimage(
        package,
        rulespec=_decode_member(seed["rulespec"]),
        tests=_decode_member(seed["tests"]),
    )


@dataclass
class ReviewedRepairResult:
    """Explicit non-model validation context; no promotion/model flags."""

    output_file: str
    citation: str
    source_attestation: dict[str, object]
    context_manifest_file: str
    review_binding: dict[str, object]
    runner: str = "reviewed-repair"
    tool: str = TOOL
    backend: None = None
    model: None = None
    generation_prompt_sha256: None = None
    trace_file: None = None


def _decode_member(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > 4 * ((MAX_MEMBER_BYTES + 2) // 3):
        raise ValueError("reviewed repair encoded member exceeds bound")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("reviewed repair member is not strict base64") from exc
    if len(raw) > MAX_MEMBER_BYTES or base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("reviewed repair base64 is noncanonical or oversized")
    return raw


def verify_manifest_reconstruction(
    payload: dict, checkout: Path, manifest_path: str
) -> None:
    """Verify signed reconstruction and registered base lineage, offline.

    The caller must also run the ordinary v5 signature, execution, waiver,
    source-release and applied-file validators. This adds no signature authority.
    """
    package, postimage = verify_repair_binding(payload.get("reviewed_repair"))
    declared = package.manifest
    target = declared["rulespec_path"]
    companion = target.removesuffix(".yaml") + ".test.yaml"
    canonical = ".axiom/encoding-manifests/" + target.removesuffix(".yaml") + ".json"
    if manifest_path != canonical or payload.get("citation") != declared["citation"]:
        raise ValueError("reviewed repair manifest target differs from registry")
    expected_files = [
        {"path": path, "sha256": sha256(raw)}
        for path, raw in zip((target, companion), postimage, strict=True)
    ]
    if payload.get("applied_files") != expected_files:
        raise ValueError("reviewed repair manifest applied files differ from postimage")
    source = declared["source"]
    attestation = payload.get("source_attestation")
    expected_source = {
        "requested_corpus_citation_path": source["requested_citation"],
        "resolved_corpus_citation_path": source["resolved_citation"],
        "corpus_release": source["release_name"],
        "corpus_release_content_sha256": source["release_sha256"],
        "source_sha256": source["body_sha256"],
        "generation_input_sha256": source["body_sha256"],
    }
    if not isinstance(attestation, dict) or any(
        attestation.get(k) != v for k, v in expected_source.items()
    ):
        raise ValueError("reviewed repair manifest source differs from registry")
    execution = payload.get("validation_execution")
    engine = (
        execution.get("axiom_rules_engine") if isinstance(execution, dict) else None
    )
    if not isinstance(engine, dict) or engine.get("commit") != declared["engine_ref"]:
        raise ValueError("reviewed repair manifest engine differs from registry")
    if execution.get("rulespec_dependencies") != []:
        raise ValueError("reviewed repair does not admit dependency checkouts")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args], stderr=subprocess.PIPE
        )

    country = target.split("/")[0]
    if (
        git("remote", "get-url", "origin").decode().strip()
        != f"https://github.com/TheAxiomFoundation/rulespec-{country}.git"
    ):
        raise ValueError(
            "reviewed repair base must belong to official RuleSpec checkout"
        )
    base = declared["rulespec_base_commit"]
    try:
        git("merge-base", "--is-ancestor", base, "HEAD")
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "reviewed repair base is not an ancestor of current tree"
        ) from exc
    predecessor = declared["predecessor"]
    for path, expected_hash in (
        (target, predecessor["rulespec_sha256"]),
        (companion, predecessor["tests_sha256"]),
        (predecessor["path"], predecessor["sha256"]),
    ):
        tree = git("ls-tree", "-z", base, "--", path).split(b"\0")
        if len(tree) != 2 or not tree[0]:
            raise ValueError("reviewed repair base member absent")
        header, actual_path = tree[0].split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        if mode != "100644" or kind != "blob" or actual_path.decode() != path:
            raise ValueError(
                "reviewed repair base member is not canonical regular bytes"
            )
        if int(git("cat-file", "-s", blob)) > MAX_MEMBER_BYTES:
            raise ValueError("reviewed repair base member exceeds bound")
        raw = git("cat-file", "blob", blob)
        if sha256(raw) != expected_hash:
            raise ValueError("reviewed repair base member differs from registration")
        if path == predecessor["path"]:
            if (
                blob != predecessor["blob"]
                or strict_json(raw).get("schema_version") != predecessor["schema"]
            ):
                raise ValueError("reviewed repair predecessor identity differs")
            if predecessor["canonical_manifest"] == "absent":
                from .corpus_resolver import read_bounded_regular_file

                current = read_bounded_regular_file(
                    checkout,
                    checkout / path,
                    label="reviewed historical predecessor",
                    max_bytes=MAX_MEMBER_BYTES,
                    required_mode=0o644,
                )
                if current != raw:
                    raise ValueError("reviewed historical predecessor changed")
    if predecessor["canonical_manifest"] == "absent" and git(
        "ls-tree", base, "--", canonical
    ):
        raise ValueError("reviewed repair canonical manifest was not absent at base")


def verify_archive_seed(
    raw: bytes, package: ReviewedRepairPackage, *, rulespec: bytes, tests: bytes
) -> None:
    """Bind extracted bytes to the registered immutable target-lane archive."""
    import io
    import tarfile

    declared = package.manifest
    if sha256(raw) != declared["seed"]["archive_sha256"]:
        raise ValueError("reviewed repair archive differs from registry")
    prefix = "generated/target/final-rejected-candidate/"
    target = declared["rulespec_path"].split("/", 1)[1]
    expected = {
        prefix + target: rulespec,
        prefix + target.removesuffix(".yaml") + ".test.yaml": tests,
    }
    issues_path = prefix + "issues.json"
    found = set()
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
        for index, member in enumerate(archive):
            if index >= 10000 or member.name in seen:
                raise ValueError(
                    "reviewed repair archive has duplicate or excessive members"
                )
            seen.add(member.name)
            if member.name not in {*expected, issues_path}:
                continue
            if not member.isfile() or not 0 < member.size <= MAX_MEMBER_BYTES:
                raise ValueError("reviewed repair seed archive member invalid")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("reviewed repair seed archive member absent")
            with stream:
                content = stream.read(MAX_MEMBER_BYTES + 1)
            if len(content) != member.size:
                raise ValueError("reviewed repair seed archive size mismatch")
            if member.name == issues_path:
                if sha256(content) != declared["seed"]["issues_sha256"]:
                    raise ValueError("reviewed repair seed issue digest differs")
            elif content != expected[member.name]:
                raise ValueError("reviewed repair extracted seed differs from archive")
            found.add(member.name)
    if found != {*expected, issues_path}:
        raise ValueError("reviewed repair archive lacks exact retained target seed")


def apply_reviewed_repair(args) -> None:
    """Validate an immutable registered repair and sign through ordinary gates."""
    import argparse
    import tempfile
    from datetime import datetime, timezone

    from . import cli as c
    from .corpus_resolver import read_bounded_regular_file
    from .prepare_signed_backfill import (
        immutable_atomic_source_contract,
        resolve_atomic_repair_mode,
    )

    package = load_registered_patch(Path(args.patch_checkout), args.patch_id)
    declared = package.manifest
    repo = c._resolve_canonical_rulespec_checkout(args.repo, label="RuleSpec checkout")
    engine = c._resolve_explicit_existing_directory(
        args.axiom_rules_path, label="Axiom rules engine"
    )
    corpus = c._resolve_explicit_existing_directory(
        args.corpus_path, label="Axiom Corpus"
    )
    c._recover_apply_transaction(repo)
    before = verify_predecessor(repo, package)
    if (
        not isinstance(args.run_id, str)
        or re.fullmatch(r"[1-9][0-9]*", args.run_id) is None
    ):
        raise ValueError("reviewed repair requires an actual workflow run id")
    mode = resolve_atomic_repair_mode(
        args.source_bundle_json,
        {
            "REPAIR_RUN_ID": declared["seed"]["run_id"],
            "GITHUB_RUN_ID": args.run_id,
            "CITATION": declared["citation"],
            "REPLACE_RULESPEC_PATH": declared["rulespec_path"],
        },
    )
    contract = immutable_atomic_source_contract(args.source_bundle_json)
    if (
        mode != {"mode": "full_artifact", "tests_only": False}
        or contract["source_bundle"]
        or contract["canonical_refresh_bundle"]
        or contract["require_complete_source_unit"] is not True
        or contract.get("manifest_only_refresh", False)
        or contract.get("reviewed_candidate_promotion", False)
    ):
        raise ValueError(
            "reviewed repair requires one full-source ordinary replacement"
        )
    raw_contract = strict_json(args.source_bundle_json.encode())
    if raw_contract.get("schema") == "axiom-encode/reviewed-repair-transaction/v1":
        if raw_contract.get("patch_id") != package.patch_id:
            raise ValueError("reviewed repair wrapper names a different patch")
        raw_contract = raw_contract["transaction"]
    contract_hash = sha256(canonical_json(raw_contract))
    if contract_hash != declared["contract_sha256"]:
        raise ValueError("reviewed repair required contract differs from registration")
    review_contract = c._parse_deferred_output_review_contract_json(
        canonical_json(
            {
                "schema": c._STRUCTURED_REVIEW_CONTRACT_SCHEMA,
                "citation": declared["citation"],
                "rulespec_path": declared["rulespec_path"],
                "required_deferred_outputs": [],
                "required_test_cases": contract["primary_required_test_cases"],
            }
        ).decode()
    )

    def seed_file(raw_path, expected):
        path = Path(raw_path).absolute()
        raw = read_bounded_regular_file(
            path.parent, path, label="reviewed repair seed", max_bytes=MAX_MEMBER_BYTES
        )
        if sha256(raw) != expected:
            raise ValueError("reviewed repair seed bytes differ from registration")
        return raw

    seed = declared["seed"]
    rule_seed = seed_file(args.seed_rulespec, seed["rulespec_sha256"])
    test_seed = seed_file(args.seed_tests, seed["tests_sha256"])
    # Hash a bounded regular archive; its bytes remain immutable evidence and are
    # never executed or unpacked by this command.
    archive_path = Path(args.seed_archive).absolute()
    archive = read_bounded_regular_file(
        archive_path.parent,
        archive_path,
        label="reviewed repair authenticated archive",
        max_bytes=256 * 1024 * 1024,
    )
    if sha256(archive) != seed["archive_sha256"]:
        raise ValueError("reviewed repair archive differs from registration")
    verify_archive_seed(archive, package, rulespec=rule_seed, tests=test_seed)
    binding = build_repair_binding(package, rulespec=rule_seed, tests=test_seed)
    postimage = reconstruct_postimage(package, rulespec=rule_seed, tests=test_seed)
    release = c.load_rulespec_local_corpus_release(repo, corpus)
    source_unit = c.resolve_corpus_source_unit(declared["citation"], release)
    source = {
        "corpus_ref": subprocess.check_output(
            ["git", "-C", str(corpus), "rev-parse", "HEAD"], text=True
        ).strip(),
        "release_name": release.name,
        "release_sha256": release.content_sha256,
        "requested_citation": source_unit.requested,
        "resolved_citation": source_unit.citation_path,
        "body_sha256": sha256(source_unit.body.encode()),
    }
    engine_ref = subprocess.check_output(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True
    ).strip()
    verify_bindings(
        package,
        {
            "citation": source_unit.requested,
            "rulespec_path": declared["rulespec_path"],
            "rulespec_base_commit": declared["rulespec_base_commit"],
            "predecessor": declared["predecessor"],
            "seed": seed,
            "source": source,
            "contract_sha256": contract_hash,
            "engine_ref": engine_ref,
        },
    )
    target = Path(declared["rulespec_path"])
    companion = c._rulespec_test_path(target)
    content_root = repo / target.parts[0]
    relative = Path(*target.parts[1:])
    replacement = c._resolve_encode_replacement_target(
        argparse.Namespace(
            replace_rulespec_path=target.as_posix(),
            apply=True,
            mode="repo-augmented",
            repair_candidate_root=str(Path(args.seed_rulespec).parent),
        ),
        policy_checkout_path=repo,
        policy_repo_path=content_root,
        source_unit=source_unit,
        corpus_release=release,
    )
    if (
        replacement is None
        or replacement.relative_output != relative
        or replacement.legacy_replacement is not None
        or replacement.retired_source_admission is not None
    ):
        raise ValueError(
            "reviewed repair requires the ordinary exact-source replacement route"
        )
    normalized_source = source_unit.body.replace("\r\n", "\n").replace("\r", "\n")
    if normalized_source != source_unit.body:
        raise ValueError("reviewed repair source must already have canonical newlines")
    _verifications, rulespec_root = c._manifest_primary_source_verifications(
        manifest_root=repo, applied_files=[repo / target, repo / companion]
    )
    attestation = {
        **source_unit.source_attestation,
        "generation_input_sha256": sha256(normalized_source.encode()),
        "rulespec_root": rulespec_root,
    }
    with tempfile.TemporaryDirectory() as temporary:
        output_root = Path(temporary)
        context = output_root / "context"
        context.mkdir()
        (context / "source.txt").write_text(normalized_source)
        context_manifest = context / "context-manifest.json"
        context_manifest.write_bytes(
            canonical_json(
                {
                    "source_text_file": "source.txt",
                    "source_metadata": {"source_attestation": attestation},
                }
            )
        )
        generated = output_root / "reviewed-repair" / relative
        generated.parent.mkdir(parents=True)
        generated.write_bytes(postimage[0])
        generated_tests = c._rulespec_test_path(generated)
        generated_tests.write_bytes(postimage[1])
        result = ReviewedRepairResult(
            str(generated),
            declared["citation"],
            attestation,
            str(context_manifest),
            binding,
        )
        # The ordinary resolver above has established exact replacement scope.
        setattr(result, c._REPLACEMENT_OVERLAY_SCOPE_ATTR, True)
        c._enforce_no_apply_collision(source_file=generated, target_file=repo / target)
        valid, issues, supplemental = c._run_generated_encoding_overlay_validation(
            result,
            output_root=output_root,
            policy_repo_path=content_root,
            axiom_rules_path=engine,
            local_corpus_release=release,
            require_complete_source_unit=True,
            deferred_output_review_contract=review_contract,
        )
        if not valid:
            raise RuntimeError(
                "Reviewed repair failed ordinary validation: " + "; ".join(issues)
            )
        exact = {relative: postimage[0], c._rulespec_test_path(relative): postimage[1]}
        if any(exact.get(path) != raw.encode() for path, raw in supplemental.items()):
            raise RuntimeError(
                "Reviewed repair validation produced supplemental mutations"
            )
        if (generated.read_bytes(), generated_tests.read_bytes()) != postimage:
            raise RuntimeError(
                "Reviewed repair validation mutated registered postimages"
            )
        c._require_unchanged_successful_apply_validation(
            result,
            output_root=output_root,
            policy_repo_path=content_root,
            relative_output=relative,
            supplemental_files=supplemental,
            local_corpus_release=release,
        )
        snapshot = getattr(result, c._APPLY_VALIDATION_SNAPSHOT_ATTR)
        execution = snapshot["manifest_validation_execution"]
        manifest_relative = c._applied_encoding_manifest_path(target)
        manifest_path = repo / manifest_relative
        c._require_applied_manifest_not_shrunk(
            manifest_path, new_applied_files={target.as_posix(), companion.as_posix()}
        )
        payload = {
            "schema_version": c.APPLIED_ENCODING_MANIFEST_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tool": TOOL,
            "axiom_encode_version": c.__version__,
            "axiom_encode_git": c._require_clean_axiom_encode_git_provenance(),
            "run_id": args.run_id,
            "citation": declared["citation"],
            "reviewed_repair": binding,
            c.VALIDATION_WAIVER_SET_SHA256_FIELD: c.verify_rulespec_validation_waiver_set(
                repo
            ),
            "applied_files": [
                {"path": path.as_posix(), "sha256": sha256(raw)}
                for path, raw in zip((target, companion), postimage, strict=True)
            ],
            "source_attestation": attestation,
            "validation_execution": execution,
        }
        verify_manifest_reconstruction(payload, repo, manifest_relative.as_posix())
        # The ordinary manifest reader is 1 MiB. Reserve a bounded signature
        # envelope before any signer call and check actual serialization again.
        if len(canonical_json(payload)) + 4096 > 1024 * 1024:
            raise ValueError(
                "Reviewed repair signed manifest would exceed existing reader limit"
            )

        def unchanged():
            c._require_applied_manifest_not_shrunk(
                manifest_path,
                new_applied_files={target.as_posix(), companion.as_posix()},
            )
            if verify_predecessor(repo, package) != before:
                raise RuntimeError(
                    "Reviewed repair predecessor changed after validation"
                )
            c._require_unchanged_successful_apply_validation(
                result,
                output_root=output_root,
                policy_repo_path=content_root,
                relative_output=relative,
                supplemental_files=supplemental,
                local_corpus_release=release,
            )

        unchanged()
        with c._isolated_apply_manifest_signer(preflight=True) as signer:
            c._sign_applied_encoding_manifest(payload, signer)
            manifest_bytes = canonical_json(payload) + b"\n"
            if len(manifest_bytes) > 1024 * 1024:
                raise ValueError(
                    "Reviewed repair signed manifest exceeds existing reader limit"
                )

            def installed():
                c._require_apply_post_install_closure(
                    result,
                    output_root=output_root,
                    policy_repo_path=content_root,
                    relative_output=relative,
                    supplemental_files=supplemental,
                    corpus_path=corpus,
                    planned=exact,
                    manifest_path=manifest_path,
                    manifest_bytes=manifest_bytes,
                    signing_broker=signer,
                )
                verified, _prefix, _digest, failures = (
                    c._load_verified_applied_encoding_manifest_payload(
                        repo,
                        manifest_relative.as_posix(),
                        signing_broker=signer,
                        local_corpus_release=release,
                        expected_encoder_identity=c._current_guard_encoder_execution_identity(),
                    )
                )
                if verified is None or failures:
                    raise RuntimeError(
                        "Reviewed repair installed manifest rejected: "
                        + "; ".join(failures)
                    )
                for path, raw in zip((target, companion), postimage, strict=True):
                    if (repo / path).read_bytes() != raw:
                        raise RuntimeError("Reviewed repair installed bytes differ")

            c._install_apply_transaction(
                [
                    (repo / target, postimage[0]),
                    (repo / companion, postimage[1]),
                    (manifest_path, manifest_bytes),
                ],
                checkout_root=repo,
                expected_originals={
                    repo / target: sha256(before[target.as_posix()]),
                    repo / companion: sha256(before[companion.as_posix()]),
                    manifest_path: (
                        sha256(before[manifest_relative.as_posix()])
                        if manifest_relative.as_posix() in before
                        else None
                    ),
                },
                pre_install_check=unchanged,
                post_install_check=installed,
            )
    print(f"applied reviewed repair {package.patch_id} {manifest_relative.as_posix()}")


def resolve_request(raw: str, dispatch: dict[str, str], checkout: Path) -> dict:
    """Resolve the closed wrapper before any prior-run download or signing."""
    from .prepare_signed_backfill import (
        _load_unambiguous_json,
        resolve_atomic_repair_mode,
        split_atomic_source_input,
    )

    request = _load_unambiguous_json(raw, label="reviewed repair request")
    if (
        not isinstance(request, dict)
        or request.get("schema") != "axiom-encode/reviewed-repair-transaction/v1"
    ):
        return {}
    normalized = split_atomic_source_input(raw)
    patch_id = normalized.get("reviewed_repair_patch_id")
    if patch_id is None:
        return {}
    if dispatch.get("OPEN_PR") != "false":
        raise ValueError("reviewed repair is artifact-only; open_pr must be false")
    if resolve_atomic_repair_mode(raw, dispatch) != {
        "mode": "full_artifact",
        "tests_only": False,
    }:
        raise ValueError("reviewed repair requires an ordinary full artifact repair")
    for field in (
        "QUEUE_DISPATCHER_RUN_ID",
        "QUEUE_ID",
        "QUEUE_ITEM_GENERATION_SHA256",
        "QUEUE_ITEM_ID",
        "QUEUE_MANIFEST_SHA256",
        "DEPENDENT_CITATION",
        "DEPENDENT_REVIEW_FINDING",
        "SECOND_DEPENDENT_CITATION",
        "SECOND_DEPENDENT_REVIEW_FINDING",
        "REPLACE_LEGACY_RULESPEC_PATH",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
    ):
        if dispatch.get(field, ""):
            raise ValueError(f"reviewed repair cannot include {field}")
    for field in (
        "EXISTING_SIGNED_IMPORTS_JSON",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
    ):
        if _load_unambiguous_json(dispatch.get(field) or "[]", label=field) != []:
            raise ValueError(f"reviewed repair cannot include {field}")
    package = load_registered_patch(checkout, patch_id)
    value = package.manifest
    expected = {
        "CITATION": value["citation"],
        "REPLACE_RULESPEC_PATH": value["rulespec_path"],
        "RULESPEC_REF": value["rulespec_base_commit"],
        "CORPUS_REF": value["source"]["corpus_ref"],
        "RULES_ENGINE_REF": value["engine_ref"],
        "REPAIR_RUN_ID": value["seed"]["run_id"],
        "COUNTRY": value["rulespec_path"].split("/")[0],
    }
    if any(dispatch.get(k) != v for k, v in expected.items()):
        raise ValueError("reviewed repair dispatch differs from registered identities")
    wrapper = strict_json(raw.encode())
    if sha256(canonical_json(wrapper["transaction"])) != value["contract_sha256"]:
        raise ValueError(
            "reviewed repair complete v6 transaction differs from registry"
        )
    return {"patch_id": patch_id, "package_commit": package.commit}


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["resolve-request"])
    parser.add_argument("--checkout", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            resolve_request(
                os.environ.get("ATOMIC_SOURCE_JSON", "[]"),
                dict(os.environ),
                arguments.checkout,
            ),
            sort_keys=True,
        )
    )
