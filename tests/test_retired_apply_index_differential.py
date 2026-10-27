"""Compare indexed admission with the complete pre-index admission path.

The implementation through ``retired_apply_payload_admission_issue`` is retained
from c7fe642ee. Keep it independent of the production admission implementation:
only its Git, blob, and live-file reads are replaced by the generated tree below.
This preserves the old validation, lookup, check ordering, and exact diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode import retired_apply as indexed_admission
from tests.environment_assertions import throw

RETIRED_APPLY_INVENTORY_SCHEMA = "axiom-encode/retired-apply-inventory/v1"
_REPOSITORY = re.compile(r"TheAxiomFoundation/rulespec-[a-z0-9]+(?:-[a-z0-9]+)*")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _inventory_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
        or value.startswith(("/", "-", ":"))
        or any(character in value for character in "\\*?[")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("retired apply inventory path must be canonical UTF-8 POSIX")
    return value


def _exact_fields(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"retired apply inventory {label} has unsupported fields")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(
            "retired apply inventory sha256 must be 64 lowercase hex digits"
        )
    return value


def validate_retired_apply_inventory(value: object) -> dict[str, object]:
    """Validate and copy the exact inventory schema shared by trust consumers.

    Digests bind raw bytes, including formatting and signatures. No timestamp,
    branch name, ambient environment variable, or repository file is authority.
    """

    root = _exact_fields(value, {"schema", "repositories"}, "root")
    if root["schema"] != RETIRED_APPLY_INVENTORY_SCHEMA:
        raise ValueError("retired apply inventory schema is unsupported")
    repositories = root["repositories"]
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("retired apply inventory repositories must be nonempty")
    canonical_repositories: list[dict[str, object]] = []
    identities: set[str] = set()
    for raw_repository in repositories:
        repository = _exact_fields(
            raw_repository, {"repository", "cutoff_commit", "artifacts"}, "repository"
        )
        identity = repository["repository"]
        if not isinstance(identity, str) or _REPOSITORY.fullmatch(identity) is None:
            raise ValueError("retired apply inventory repository must be canonical")
        if identity in identities:
            raise ValueError("retired apply inventory has a duplicate repository")
        identities.add(identity)
        cutoff = repository["cutoff_commit"]
        if not isinstance(cutoff, str) or _COMMIT.fullmatch(cutoff) is None:
            raise ValueError("retired apply inventory cutoff must be a full commit SHA")
        artifacts = repository["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("retired apply inventory artifacts must be nonempty")
        canonical_artifacts: list[dict[str, object]] = []
        artifact_paths: set[str] = set()
        for raw_artifact in artifacts:
            artifact = _exact_fields(
                raw_artifact, {"path", "sha256", "covered_files"}, "artifact"
            )
            path = _inventory_path(artifact["path"])
            if path in artifact_paths:
                raise ValueError(
                    "retired apply inventory has a duplicate artifact path"
                )
            artifact_paths.add(path)
            covered_files = artifact["covered_files"]
            if not isinstance(covered_files, list):
                raise ValueError("retired apply inventory covered_files must be a list")
            canonical_covered: list[dict[str, object]] = []
            covered_paths: set[tuple[str, str]] = set()
            for raw_covered in covered_files:
                if not isinstance(raw_covered, dict) or set(raw_covered) not in (
                    {"path", "sha256"},
                    {"path", "sha256", "source_commit"},
                ):
                    raise ValueError(
                        "retired apply inventory covered file has unsupported fields"
                    )
                covered = raw_covered
                covered_path = _inventory_path(covered["path"])
                source = covered.get("source_commit", cutoff)
                if not isinstance(source, str) or _COMMIT.fullmatch(source) is None:
                    raise ValueError(
                        "retired apply covered source must be a full commit SHA"
                    )
                if covered_path == path or (covered_path, source) in covered_paths:
                    raise ValueError(
                        "retired apply inventory has duplicate covered paths"
                    )
                covered_paths.add((covered_path, source))
                copied: dict[str, object] = {
                    "path": covered_path,
                    "sha256": _digest(covered["sha256"]),
                }
                if "source_commit" in covered:
                    copied["source_commit"] = source
                canonical_covered.append(copied)
            canonical_artifacts.append(
                {
                    "path": path,
                    "sha256": _digest(artifact["sha256"]),
                    "covered_files": canonical_covered,
                }
            )
        canonical_repositories.append(
            {
                "repository": identity,
                "cutoff_commit": cutoff,
                "artifacts": canonical_artifacts,
            }
        )
    return {
        "schema": RETIRED_APPLY_INVENTORY_SCHEMA,
        "repositories": canonical_repositories,
    }


@dataclass(frozen=True)
class RetiredApplyContext:
    """Explicit artifact context; the frozen inventory is supplied separately."""

    repo_path: Path
    artifact_path: str | Path
    artifact_bytes: bytes
    head_ref: str = "HEAD"
    retrospective: bool = False


def _git_environment() -> dict[str, str]:
    # Use the broker's immutable subprocess environment when supervised. Avoid
    # importing the large eval module from the trust-file schema validator.
    from .signing_broker import scrub_private_signing_environment

    environment = {
        key: value
        for key, value in scrub_private_signing_environment().items()
        if not key.startswith("GIT_")
    }
    environment.update(
        GIT_GRAFT_FILE=os.devnull,
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="core.commitGraph",
        GIT_CONFIG_VALUE_0="false",
        GIT_TERMINAL_PROMPT="0",
        GIT_LITERAL_PATHSPECS="1",
        GIT_OPTIONAL_LOCKS="0",
    )
    return environment


def _git(repo: Path, *arguments: str, ancestry: bool = False) -> bytes | bool:
    # Keep -C first, matching the provisioned supervisor's trusted Git wrapper.
    environment = _git_environment()
    executable = shutil.which("git", path=environment.get("PATH") or None)
    if executable is None:
        throw(
            ValueError("retired apply verification requires the trusted Git executable")
        )
    result = subprocess.run(
        [executable, "-C", str(repo), *arguments],
        capture_output=True,
        env=environment,
        timeout=30,
        check=False,
    )
    if ancestry and result.returncode in {0, 1}:
        return result.returncode == 0
    if result.returncode != 0:
        throw(ValueError(f"retired apply history query failed: git {arguments[0]}"))
    return result.stdout


def _git_text(repo: Path, *arguments: str) -> str:
    result = _git(repo, *arguments)
    assert isinstance(result, bytes)
    return result.decode("utf-8").strip()


def _regular_path(repo: Path, relative: str) -> Path:
    path = repo
    for component in relative.split("/"):
        path /= component
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("retired apply artifact paths must not contain symlinks")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("retired apply artifact must be a regular file")
    return path


def _live_bytes(repo: Path, relative: str) -> bytes:
    # Keep the existing path diagnostics, then atomically anchor the actual
    # read so a concurrent symlink swap cannot escape the checkout.
    from .corpus_resolver import read_bounded_regular_file

    path = _regular_path(repo, relative)
    return read_bounded_regular_file(
        repo,
        path,
        label="retired apply live artifact",
        max_bytes=16 * 1024 * 1024,
    )


def _blob(repo: Path, commit: str, path: str) -> bytes:
    tree = _git(repo, "ls-tree", "-z", "--full-tree", commit, "--", path)
    assert isinstance(tree, bytes)
    entries = tree.split(b"\0")
    if len(entries) != 2 or entries[1] or b"\t" not in entries[0]:
        raise ValueError("retired apply frozen artifact is absent or ambiguous")
    metadata, listed_path = entries[0].split(b"\t", 1)
    mode, kind, oid = metadata.split()
    if (
        mode != b"100644"
        or kind != b"blob"
        or listed_path.decode("utf-8") != path
        or _COMMIT.fullmatch(oid.decode("ascii")) is None
    ):
        raise ValueError("retired apply frozen artifact must be a regular 0644 blob")
    result = _git(repo, "cat-file", "blob", oid.decode("ascii"))
    assert isinstance(result, bytes)
    return result


def _repository_identity(origin: str) -> str | None:
    patterns = (
        r"https://github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?",
        r"git@github\.com:(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?",
        r"ssh://git@github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, origin)
        if match is not None:
            identity = match.group("slug")
            return identity if _REPOSITORY.fullmatch(identity) else None
    return None


def _repository_cutoff(
    inventory: object,
    repo_path: Path,
    head_ref: str,
    retrospective: bool,
) -> tuple[Path, str, str, dict[str, Any]]:
    trusted = validate_retired_apply_inventory(inventory)
    checkout = Path(os.path.abspath(repo_path))
    if checkout.resolve(strict=True) != checkout:
        raise ValueError(
            "retired apply checkout must be canonical and contain no symlink"
        )
    repo = Path(_git_text(checkout, "rev-parse", "--show-toplevel"))
    if repo.resolve(strict=True) != repo:
        raise ValueError("retired apply Git root must be canonical")
    identity = _repository_identity(_git_text(repo, "remote", "get-url", "origin"))
    if identity is None or repo.name != identity.split("/", 1)[1]:
        raise ValueError("retired apply repository identity is not canonical")
    repositories = trusted["repositories"]
    assert isinstance(repositories, list)
    repository = next(
        (entry for entry in repositories if entry["repository"] == identity), None
    )
    if repository is None:
        raise ValueError("retired apply repository is not frozen in trusted inventory")
    if head_ref != "HEAD" and _COMMIT.fullmatch(head_ref) is None:
        raise ValueError(
            "retired apply historical head must be HEAD or a full commit SHA"
        )
    head = _git_text(repo, "rev-parse", "--verify", "HEAD^{commit}")
    if _COMMIT.fullmatch(head) is None or head_ref not in {"HEAD", head}:
        raise ValueError(
            "retired apply historical head must equal the checked-out HEAD"
        )
    cutoff = repository["cutoff_commit"]
    if _git_text(repo, "rev-parse", "--verify", f"{cutoff}^{{commit}}") != cutoff:
        raise ValueError("retired apply cutoff commit is unavailable")
    if _git_text(repo, "rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("retired apply verification requires complete Git history")
    common = Path(
        _git_text(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    )
    grafts = common / "info/grafts"
    if grafts.exists() or grafts.is_symlink():
        raise ValueError("retired apply history contains forbidden graft metadata")
    if _git_text(repo, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        raise ValueError("retired apply history contains forbidden replacement refs")
    older = bool(_git(repo, "merge-base", "--is-ancestor", head, cutoff, ancestry=True))
    newer = bool(_git(repo, "merge-base", "--is-ancestor", cutoff, head, ancestry=True))
    if retrospective:
        if not older:
            raise ValueError(
                "retired apply retrospective head is newer than the frozen cutoff"
            )
    elif not newer:
        raise ValueError(
            "retired apply historical head must descend from the frozen cutoff"
        )
    return repo, head, cutoff, repository


def retrospective_head_issue(
    inventory: object, repo_path: Path, head_ref: str = "HEAD"
) -> str | None:
    """Preflight an explicit retrospective guard even when no artifact is selected."""

    try:
        _repository_cutoff(inventory, repo_path, head_ref, True)
        return None
    except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
        return str(exc)


def _frozen_artifact(
    inventory: object, context: RetiredApplyContext
) -> tuple[Path, str, str, dict[str, Any], bytes]:
    repo, head, cutoff, repository = _repository_cutoff(
        inventory, context.repo_path, context.head_ref, context.retrospective
    )
    absolute = Path(context.artifact_path)
    if not absolute.is_absolute():
        absolute = Path(os.path.abspath(context.repo_path)) / absolute
    # Preserve the lexical path, so relocated and symlinked files cannot turn
    # into a frozen artifact by resolving filesystem indirection.
    relative = _inventory_path(absolute.relative_to(repo).as_posix())
    artifact = next(
        (entry for entry in repository["artifacts"] if entry["path"] == relative), None
    )
    if artifact is None:
        raise ValueError(
            "retired apply artifact path is not frozen in trusted inventory"
        )
    frozen_bytes = _blob(repo, cutoff, relative)
    if hashlib.sha256(frozen_bytes).hexdigest() != artifact["sha256"]:
        raise ValueError(
            "retired apply frozen artifact bytes disagree with trusted inventory"
        )
    for covered in artifact["covered_files"]:
        source = covered.get("source_commit", cutoff)
        if _git_text(repo, "rev-parse", "--verify", f"{source}^{{commit}}") != source:
            raise ValueError("retired apply covered source commit is unavailable")
        if not _git(repo, "merge-base", "--is-ancestor", source, cutoff, ancestry=True):
            raise ValueError(
                "retired apply covered source must be an ancestor of the cutoff"
            )
        frozen_covered = _blob(repo, source, covered["path"])
        if hashlib.sha256(frozen_covered).hexdigest() != covered["sha256"]:
            raise ValueError(
                "retired apply frozen covered-file bytes disagree with trusted inventory"
            )
    frozen = json.loads(frozen_bytes)
    prefix = relative.split(".axiom/", 1)[0] if ".axiom/" in relative else ""
    frozen_coverage = {
        (entry["path"], entry["sha256"]) for entry in artifact["covered_files"]
    }
    for path, digest in _payload_file_hashes(frozen, prefix):
        if (path, digest) not in frozen_coverage:
            raise ValueError(
                "retired apply inventory does not bind every generated-file byte hash"
            )
    return repo, head, cutoff, artifact, frozen_bytes


def _payload_file_hashes(value: object, prefix: str) -> list[tuple[str, str]]:
    covered: list[tuple[str, str]] = []
    if isinstance(value, dict):
        applied = value.get("applied_files")
        if isinstance(applied, list):
            for entry in applied:
                if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                    covered.append(
                        (
                            _inventory_path(
                                prefix + _inventory_path(entry.get("path"))
                            ),
                            _digest(entry["sha256"]),
                        )
                    )
        for child in value.values():
            covered.extend(_payload_file_hashes(child, prefix))
    elif isinstance(value, list):
        for child in value:
            covered.extend(_payload_file_hashes(child, prefix))
    return covered


def retired_apply_admission_issue(
    inventory: object, context: RetiredApplyContext
) -> str | None:
    """Require an unchanged frozen live artifact, including no restored history."""

    try:
        repo, head, cutoff, artifact, frozen_bytes = _frozen_artifact(
            inventory, context
        )
        if (
            not isinstance(context.artifact_bytes, bytes)
            or context.artifact_bytes != frozen_bytes
        ):
            raise ValueError(
                "retired apply manifest/receipt bytes are not exactly frozen"
            )
        if _live_bytes(repo, artifact["path"]) != frozen_bytes:
            raise ValueError(
                "retired apply live manifest/receipt bytes are not exactly frozen"
            )
        live_covered = [
            entry
            for entry in artifact["covered_files"]
            if entry.get("source_commit", cutoff) == cutoff
        ]
        frozen = json.loads(frozen_bytes)
        prefix = (
            artifact["path"].split(".axiom/", 1)[0]
            if ".axiom/" in artifact["path"]
            else ""
        )
        if isinstance(frozen, dict) and isinstance(frozen.get("applied_files"), list):
            root_live = {
                "applied_files": [
                    entry
                    for entry in frozen["applied_files"]
                    if isinstance(entry, dict) and entry.get("deleted") is not True
                ]
            }
            live_hashes = {(entry["path"], entry["sha256"]) for entry in live_covered}
            if any(
                entry not in live_hashes
                for entry in _payload_file_hashes(root_live, prefix)
            ):
                raise ValueError(
                    "retired apply inventory live coverage must bind cutoff files"
                )
        files = [
            {"path": artifact["path"], "sha256": artifact["sha256"]},
            *live_covered,
        ]
        for covered in files:
            if (
                hashlib.sha256(_blob(repo, head, covered["path"])).hexdigest()
                != covered["sha256"]
            ):
                raise ValueError("retired apply HEAD file bytes are not exactly frozen")
            if (
                hashlib.sha256(_live_bytes(repo, covered["path"])).hexdigest()
                != covered["sha256"]
            ):
                raise ValueError(
                    "retired apply live covered-file bytes are not exactly frozen"
                )
        if head != cutoff and not context.retrospective:
            history = _git_text(
                repo,
                "log",
                "--full-history",
                "-m",
                "--format=%H",
                "--no-renames",
                f"{cutoff}..{head}",
                "--",
                *(entry["path"] for entry in files),
            )
            if history:
                raise ValueError(
                    "retired apply artifact was changed or superseded after the frozen cutoff"
                )
        return None
    except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
        return str(exc)


def _contains_payload(value: object, payload: object) -> bool:
    if isinstance(value, dict):
        return value == payload or any(
            _contains_payload(child, payload) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_payload(child, payload) for child in value)
    return False


def retired_apply_payload_admission_issue(
    inventory: object,
    context: RetiredApplyContext,
    payload: object,
    *,
    nested: bool = False,
) -> str | None:
    """Admit direct signatures or exact nested frozen historical evidence.

    Nested success authenticates evidence only. A new outer artifact still
    requires the active root, and its existing receipt chain must bind the
    source path and original manifest hash before it invokes this function.
    """

    try:
        if not nested:
            if (
                not isinstance(payload, dict)
                or json.loads(context.artifact_bytes) != payload
            ):
                raise ValueError(
                    "retired apply payload does not match its exact manifest/receipt bytes"
                )
            return retired_apply_admission_issue(inventory, context)
        _, _, _, _, frozen_bytes = _frozen_artifact(inventory, context)
        frozen = json.loads(frozen_bytes)
        if not isinstance(payload, dict) or not _contains_payload(frozen, payload):
            raise ValueError(
                "retired apply nested payload is not exact frozen evidence at this path"
            )
        return None
    except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
        return str(exc)


# The code above is the old path. The generated filesystem below models complete
# repository trees while keeping Hypothesis examples independent of Git startup
# cost. Existing real-Git tests cover the low-level read and subprocess boundary.
_MUTATIONS = (
    "caller_bytes",
    "live_manifest",
    "head_manifest",
    "frozen_manifest",
    "frozen_generated",
    "head_generated",
    "live_generated",
    "artifact_digest",
    "covered_digest",
    "coverage_missing",
    "artifact_missing",
    "payload_missing",
    "missing_repository",
    "origin",
    "head_ref",
    "cutoff_unavailable",
    "source_unavailable",
    "shallow",
    "replacement",
    "grafts",
    "superseded",
)


@st.composite
def _generated_scenarios(draw):
    return {
        "repository": draw(st.sampled_from(("us", "uk"))),
        "artifact_count": draw(st.integers(min_value=1, max_value=6)),
        "reverse_repositories": draw(st.booleans()),
        "reverse_artifacts": draw(st.booleans()),
        "filename": draw(
            st.sampled_from(("example.json", "26:1.json", "1437c–1 receipt.json"))
        ),
        "generated_bytes": draw(st.binary(min_size=1, max_size=32)),
        "marker": draw(st.integers(min_value=0, max_value=1000)),
        "head_relation": draw(st.sampled_from(("equal", "newer", "older", "side"))),
        "source": draw(
            st.sampled_from(("default", "cutoff", "prior", "newer", "side"))
        ),
        "deleted": draw(st.booleans()),
        "retrospective": draw(st.booleans()),
        "head_ref": draw(st.sampled_from(("HEAD", "exact"))),
        "mutations": draw(
            st.frozensets(st.sampled_from(_MUTATIONS), min_size=0, max_size=4)
        ),
    }


@dataclass
class _Tree:
    repo: Path
    cutoff: str
    head: str
    prior: str
    newer: str
    side: str
    source: str
    mutations: frozenset[str]
    blobs: dict[tuple[str, str], bytes]
    live: dict[str, bytes]

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return ancestor == descendant or (ancestor, descendant) in {
            (self.prior, self.cutoff),
            (self.prior, self.newer),
            (self.cutoff, self.newer),
            (self.prior, self.side),
        }


def _commit_id(repository: str, label: str) -> str:
    return hashlib.sha1(f"{repository}:{label}".encode()).hexdigest()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _build_generated_trees(tmp_path: Path, scenario: dict):
    repositories = []
    trees: dict[Path, _Tree] = {}
    evidence = []
    for country in ("us", "uk"):
        repo = tmp_path / f"rulespec-{country}"
        (repo / ".git/info").mkdir(parents=True, exist_ok=True)
        grafts = repo / ".git/info/grafts"
        if grafts.exists():
            grafts.unlink()
        selected = country == scenario["repository"]
        mutations = scenario["mutations"] if selected else frozenset()
        cutoff, prior, newer, side = (
            _commit_id(country, label) for label in ("cutoff", "prior", "newer", "side")
        )
        head = {
            "equal": cutoff,
            "newer": newer,
            "older": prior,
            "side": side,
        }[scenario["head_relation"]]
        source = {
            "default": cutoff,
            "cutoff": cutoff,
            "prior": prior,
            "newer": newer,
            "side": side,
        }[scenario["source"]]
        tree = _Tree(repo, cutoff, head, prior, newer, side, source, mutations, {}, {})
        trees[repo] = tree
        if "grafts" in mutations:
            grafts.write_text(cutoff + "\n")
        artifacts = []
        for number in range(scenario["artifact_count"]):
            target = selected and number == 0
            flags = mutations if target else frozenset()
            filename = scenario["filename"] if number == 0 else f"filler-{number}.json"
            artifact_path = f".axiom/encoding-manifests/{filename}"
            generated_path = f"us/26:1–{number} generated.yaml"
            generated_bytes = (
                scenario["generated_bytes"] + country.encode() + bytes([number])
            )
            nested_payload = {
                "marker": scenario["marker"],
                "repository": country,
                "number": number,
            }
            payload = {
                "applied_files": [
                    {
                        "path": generated_path,
                        "sha256": _sha(generated_bytes),
                        "deleted": scenario["deleted"],
                    }
                ],
                "nested": nested_payload,
            }
            manifest_bytes = json.dumps(payload, sort_keys=True).encode() + b"\n"
            covered = {"path": generated_path, "sha256": _sha(generated_bytes)}
            if scenario["source"] != "default":
                covered["source_commit"] = source
            if "covered_digest" in flags:
                covered["sha256"] = "0" * 64
            artifact = {
                "path": artifact_path,
                "sha256": "0" * 64
                if "artifact_digest" in flags
                else _sha(manifest_bytes),
                "covered_files": [] if "coverage_missing" in flags else [covered],
            }
            artifacts.append(artifact)
            for commit in {cutoff, head, prior, newer, side}:
                tree.blobs[commit, artifact_path] = manifest_bytes
                tree.blobs[commit, generated_path] = generated_bytes
            tree.live[artifact_path] = manifest_bytes
            tree.live[generated_path] = generated_bytes
            for flag, commit, path in (
                ("frozen_manifest", cutoff, artifact_path),
                ("head_manifest", head, artifact_path),
                ("frozen_generated", source, generated_path),
                ("head_generated", head, generated_path),
            ):
                if flag in flags:
                    tree.blobs[commit, path] += b"changed"
            if "live_manifest" in flags:
                tree.live[artifact_path] += b"changed"
            if "live_generated" in flags:
                tree.live[generated_path] += b"changed"
            context = indexed_admission.RetiredApplyContext(
                repo,
                artifact_path + ".missing"
                if "artifact_missing" in flags
                else artifact_path,
                manifest_bytes + b" " if "caller_bytes" in flags else manifest_bytes,
                head_ref=(
                    "main"
                    if "head_ref" in mutations
                    else head
                    if scenario["head_ref"] == "exact"
                    else "HEAD"
                ),
                retrospective=scenario["retrospective"],
            )
            supplied_payload = (
                {"different": True} if "payload_missing" in flags else payload
            )
            supplied_nested = (
                {"different": True} if "payload_missing" in flags else nested_payload
            )
            evidence.append((context, supplied_payload, supplied_nested))
        if scenario["reverse_artifacts"]:
            artifacts.reverse()
        if "missing_repository" not in mutations:
            repositories.append(
                {
                    "repository": f"TheAxiomFoundation/rulespec-{country}",
                    "cutoff_commit": cutoff,
                    "artifacts": artifacts,
                }
            )
    if scenario["reverse_repositories"]:
        repositories.reverse()
    return (
        {"schema": RETIRED_APPLY_INVENTORY_SCHEMA, "repositories": repositories},
        trees,
        evidence,
    )


def _install_generated_tree_reads(monkeypatch, trees: dict[Path, _Tree]):
    def git(repo: Path, *arguments: str, ancestry: bool = False):
        tree = trees[repo]
        flags = tree.mutations
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repo).encode() + b"\n"
        if arguments == ("remote", "get-url", "origin"):
            owner = "OtherOwner" if "origin" in flags else "TheAxiomFoundation"
            return f"https://github.com/{owner}/{repo.name}.git\n".encode()
        if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
            return tree.head.encode() + b"\n"
        if arguments[:2] == ("rev-parse", "--verify"):
            commit = arguments[2].removesuffix("^{commit}")
            if (commit == tree.cutoff and "cutoff_unavailable" in flags) or (
                commit == tree.source and "source_unavailable" in flags
            ):
                return b"unavailable\n"
            return commit.encode() + b"\n"
        if arguments == ("rev-parse", "--is-shallow-repository"):
            return b"true\n" if "shallow" in flags else b"false\n"
        if arguments == ("rev-parse", "--path-format=absolute", "--git-common-dir"):
            return str(repo / ".git").encode() + b"\n"
        if arguments == ("for-each-ref", "--format=%(refname)", "refs/replace/"):
            return b"refs/replace/synthetic\n" if "replacement" in flags else b""
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            assert ancestry
            return tree.is_ancestor(arguments[2], arguments[3])
        if arguments[0] == "log":
            return b"changed\n" if "superseded" in flags else b""
        raise AssertionError(f"unexpected synthetic Git query: {arguments}")

    def blob(repo: Path, commit: str, path: str):
        try:
            return trees[repo].blobs[commit, path]
        except KeyError:
            raise ValueError(
                "retired apply frozen artifact is absent or ambiguous"
            ) from None

    def live_bytes(repo: Path, path: str):
        return trees[repo].live[path]

    for module in (sys.modules[__name__], indexed_admission):
        monkeypatch.setattr(module, "_git", git)
        monkeypatch.setattr(module, "_blob", blob)
        monkeypatch.setattr(module, "_live_bytes", live_bytes)


_VALID_SCENARIO = {
    "repository": "us",
    "artifact_count": 3,
    "reverse_repositories": True,
    "reverse_artifacts": True,
    "filename": "1437c–1 receipt.json",
    "generated_bytes": b"synthetic generated tree bytes",
    "marker": 42,
    "head_relation": "newer",
    "source": "default",
    "deleted": False,
    "retrospective": False,
    "head_ref": "HEAD",
    "mutations": frozenset(),
}


@pytest.mark.parametrize(
    "indexed", (False, True), ids=("raw-inventory", "indexed-inventory")
)
@settings(
    max_examples=90,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(scenario=_generated_scenarios())
@example(scenario=_VALID_SCENARIO)
@example(scenario={**_VALID_SCENARIO, "head_relation": "side"})
@example(scenario={**_VALID_SCENARIO, "mutations": frozenset({"superseded"})})
@example(scenario={**_VALID_SCENARIO, "mutations": frozenset({"artifact_missing"})})
@example(scenario={**_VALID_SCENARIO, "head_relation": "older", "retrospective": True})
@example(scenario={**_VALID_SCENARIO, "source": "prior", "deleted": True})
def test_generated_admission_matches_pre_index_path(
    tmp_path, monkeypatch, indexed, scenario
):
    inventory, trees, evidence = _build_generated_trees(tmp_path, scenario)
    _install_generated_tree_reads(monkeypatch, trees)
    trusted = (
        indexed_admission.RetiredApplyInventory(inventory) if indexed else inventory
    )
    expected = [
        (
            retired_apply_admission_issue(inventory, context),
            retired_apply_payload_admission_issue(inventory, context, payload),
            retired_apply_payload_admission_issue(
                inventory, context, nested, nested=True
            ),
            retrospective_head_issue(inventory, context.repo_path, context.head_ref),
        )
        for context, payload, nested in evidence
    ]
    with indexed_admission.retired_apply_operation():
        # Repeat in reverse order to exercise cache hits across different paths
        # and repositories sharing the same artifact path names.
        for order in (range(len(evidence)), reversed(range(len(evidence)))):
            for number in order:
                context, payload, nested = evidence[number]
                actual = (
                    indexed_admission.retired_apply_admission_issue(trusted, context),
                    indexed_admission.retired_apply_payload_admission_issue(
                        trusted, context, payload
                    ),
                    indexed_admission.retired_apply_payload_admission_issue(
                        trusted, context, nested, nested=True
                    ),
                    indexed_admission.retrospective_head_issue(
                        trusted, context.repo_path, context.head_ref
                    ),
                )
                assert actual == expected[number], (scenario, number)
