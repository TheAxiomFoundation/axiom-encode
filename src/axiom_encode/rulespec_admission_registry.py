"""Reviewed admissions and exemptions that live in a RuleSpec repository (#2095).

Repository-specific exemptions used to be encoder code (for example the
coverage exemptions of #1923, #1940 and #1986). A RuleSpec repository can now
keep them under ``.axiom/admissions/`` on its protected default branch, where
its own code owners review them, so that they apply to files from any
admitted producer.

Trust: a pull request must never admit itself. The protected workflow exports
the registry from the protected base commit with ``admission-registry
export`` (Git objects only, regular 0644 blobs, bounded sizes), and passes the
exported directory to the encoder in ``AXIOM_RULESPEC_ADMISSION_REGISTRY``.
Records outside the repository's own country fail the load. The in-encoder
records keep working and are unioned with the registry; a registry record that
reuses an in-encoder key with different content fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REGISTRY_ENV = "AXIOM_RULESPEC_ADMISSION_REGISTRY"
REGISTRY_DIR = PurePosixPath(".axiom/admissions")
REGISTRY_RECORD = "registry.json"
REGISTRY_SCHEMA = "axiom-encode/rulespec-admission-registry/v1"
COVERAGE_EXEMPTIONS_FILE = "coverage-exemptions.jsonl"
# Files a registry may hold. Anything else under .axiom/admissions fails, so
# a new record family needs an encoder change that says how it is consumed.
REGISTRY_FILES = frozenset({COVERAGE_EXEMPTIONS_FILE})
MAX_REGISTRY_FILE_BYTES = 4 * 1024 * 1024
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# rulespec-us, rulespec-us-co, rulespec-ca, ... (the origin repository name).
_REPOSITORY_RE = re.compile(
    r"(?:https://github\.com/|git@github\.com:)TheAxiomFoundation/"
    r"rulespec-([a-z]{2}(?:-[a-z0-9]+)*)(?:\.git)?/?"
)
_JURISDICTION_RE = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)*")


class AdmissionRegistryError(ValueError):
    """The registry is malformed, untrusted, or outside its repository."""


@dataclass(frozen=True)
class AdmissionRegistry:
    country: str
    commit: str
    coverage_exemptions: tuple[object, ...]
    # sha256 over the registry record and file bytes, for check output.
    digest: str


def _git_environment() -> dict[str, str]:
    # GIT_DIR, GIT_INDEX_FILE, GIT_CONFIG_* and friends would override -C.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_GRAFT_FILE="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
    )
    return environment


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        timeout=60,
        env=_git_environment(),
    )
    if result.returncode != 0:
        raise AdmissionRegistryError(
            f"git {args[0]} failed: {result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout


def _jurisdiction_of_checkout(repo: Path) -> str:
    """Return the jurisdiction of the checkout's official origin repository."""

    url = _git(repo, "remote", "get-url", "origin").decode("utf-8").strip()
    match = _REPOSITORY_RE.fullmatch(url)
    if match is None:
        raise AdmissionRegistryError(
            "the RuleSpec checkout origin must be "
            "TheAxiomFoundation/rulespec-<jurisdiction>"
        )
    return match.group(1)


def export_registry(
    repo: Path, commit: str, output: Path, *, require_protected: bool = True
) -> AdmissionRegistry:
    """Export the registry of one exact commit from Git objects, not the tree.

    The working tree can hold pull-request bytes; Git objects at the protected
    base commit cannot. Only regular 0644 blobs with a known name are copied.
    An absent registry exports as an empty registry.
    """

    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise AdmissionRegistryError("the registry commit must be a full commit")
    repo = Path(repo)
    country = _jurisdiction_of_checkout(repo)
    resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode()
    if resolved.strip() != commit:
        raise AdmissionRegistryError("the registry commit is not a direct commit")
    if require_protected:
        # Only reviewed records on the protected default branch count. The
        # protected workflow fetches origin/main before it exports.
        main = _git(repo, "rev-parse", "--verify", "refs/remotes/origin/main^{commit}")
        history = _git(repo, "rev-list", "--first-parent", main.decode().strip())
        if commit not in history.decode().split():
            raise AdmissionRegistryError(
                "the registry commit is not on the first-parent history of origin/main"
            )
    listing = _git(
        repo, "ls-tree", "-r", "-z", "--full-tree", commit, "--", str(REGISTRY_DIR)
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for item in filter(None, listing.split(b"\0")):
        metadata, raw_path = item.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        path = PurePosixPath(raw_path.decode("utf-8"))
        if path.parent != REGISTRY_DIR or path.name not in REGISTRY_FILES:
            raise AdmissionRegistryError(f"unknown registry file: {path}")
        if kind != "blob" or mode != "100644":
            raise AdmissionRegistryError(f"registry file is not a regular file: {path}")
        size = int(_git(repo, "cat-file", "-s", oid).decode().strip())
        if size > MAX_REGISTRY_FILE_BYTES:
            raise AdmissionRegistryError(f"registry file is too large: {path}")
        (output / path.name).write_bytes(_git(repo, "cat-file", "blob", oid))
    (output / REGISTRY_RECORD).write_text(
        json.dumps(
            {"schema": REGISTRY_SCHEMA, "country": country, "commit": commit},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return load_registry(output)


def _in_country(citation_path: object, country: str) -> bool:
    """A country repository (rulespec-us) also holds its states (us-co); a
    state repository (rulespec-us-co) holds only its own citations."""

    if not isinstance(citation_path, str):
        return False
    jurisdiction = citation_path.split("/", 1)[0]
    if jurisdiction == country:
        return True
    return re.fullmatch(r"[a-z]{2}", country) is not None and jurisdiction.startswith(
        country + "-"
    )


def load_registry(root: Path) -> AdmissionRegistry:
    """Load an exported registry. Raise on any malformed or foreign record."""

    from axiom_encode.harness.coverage_exemptions import CoverageExemption

    root = Path(root)
    names = {entry.name for entry in root.iterdir()}
    unknown = names - REGISTRY_FILES - {REGISTRY_RECORD}
    if unknown:
        raise AdmissionRegistryError(
            "unknown registry files: " + ", ".join(sorted(unknown))
        )
    try:
        record = json.loads((root / REGISTRY_RECORD).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionRegistryError(f"registry record is unreadable: {exc}") from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "country", "commit"}
        or record["schema"] != REGISTRY_SCHEMA
        or not isinstance(record["country"], str)
        or _JURISDICTION_RE.fullmatch(record["country"]) is None
        or not isinstance(record["commit"], str)
        or _COMMIT_RE.fullmatch(record["commit"]) is None
    ):
        raise AdmissionRegistryError("registry record is malformed")
    country = record["country"]
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    exemptions: list[object] = []
    path = root / COVERAGE_EXEMPTIONS_FILE
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise AdmissionRegistryError("coverage exemptions must be a regular file")
        if path.stat().st_size > MAX_REGISTRY_FILE_BYTES:
            raise AdmissionRegistryError("coverage exemptions file is too large")
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                continue
            label = f"{COVERAGE_EXEMPTIONS_FILE}:{number}"
            try:
                entry = CoverageExemption.from_mapping(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise AdmissionRegistryError(f"{label}: {exc}") from exc
            if not _in_country(entry.citation_path, country) or not _in_country(
                entry.owner_evidence.get("citation_path"), country
            ):
                raise AdmissionRegistryError(
                    f"{label}: citation is outside the {country} repository"
                )
            exemptions.append(entry)
        keys = [entry.key for entry in exemptions]
        if len(set(keys)) != len(keys):
            raise AdmissionRegistryError("duplicate coverage exemption key")
    return AdmissionRegistry(
        country=country,
        commit=record["commit"],
        coverage_exemptions=tuple(exemptions),
        digest=digest.hexdigest(),
    )


def check_working_tree_registry(repo: Path) -> AdmissionRegistry:
    """Check the registry at a checkout's HEAD commit, for its own PR CI.

    This validates a proposed registry change. It never activates records:
    only an export from the protected base does.
    """

    import tempfile

    repo = Path(repo)
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    status = _git(repo, "status", "--porcelain", "--", str(REGISTRY_DIR))
    if status.strip():
        raise AdmissionRegistryError("the registry has uncommitted changes")
    with tempfile.TemporaryDirectory(prefix="axiom-admission-registry-") as tmp:
        return export_registry(
            repo, head, Path(tmp) / "registry", require_protected=False
        )


def active_registry_root() -> Path | None:
    value = os.environ.get(REGISTRY_ENV)
    return Path(value) if value else None
