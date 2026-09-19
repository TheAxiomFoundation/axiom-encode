"""Service-owned, read-only GitHub and raw Git inputs; never checks out code."""

from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from ._schema import lane_name
from .canonical import strict_parse
from .identity import IdentityRefusal
from .manifest import fsck_clean
from .protocol import decimal_id, oid
from .refusal import Refusal
from .verification import Snapshot

MAX_RESPONSE = 16_000_000


def _bounded(response, limit: int = MAX_RESPONSE) -> bytes:
    result = bytearray()
    try:
        for chunk in response.iter_content(65536):
            result.extend(chunk)
            if len(result) > limit:
                raise IdentityRefusal("remote_response_too_large")
        return bytes(result)
    finally:
        response.close()


class GitHubReader:
    """No mutation method; the optional credential must itself be read-only."""

    def __init__(self, token: str | None = None):
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if token:
            self._session.headers["Authorization"] = "Bearer " + token

    def _response(self, path: str):
        if not path.startswith(("/repos/", "/orgs/")) or any(
            c in path for c in ("..", "#", "\\", "\r", "\n")
        ):
            raise IdentityRefusal("remote_api_path")
        try:
            return self._session.get(
                "https://api.github.com" + path,
                timeout=(5, 30),
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException:
            raise IdentityRefusal("remote_api_unavailable") from None

    def _json(self, path: str):
        response = self._response(path)
        if response.status_code != 200:
            response.close()
            raise IdentityRefusal("remote_api_unavailable")
        body = strict_parse(_bounded(response))
        if not isinstance(body, (dict, list)):
            raise IdentityRefusal("remote_api_json")
        return body

    def get(self, path: str) -> dict:
        body = self._json(path)
        if not isinstance(body, dict):
            raise IdentityRefusal("remote_api_object")
        return body

    def optional_ref(self, repository: str, branch: str) -> dict | None:
        """Prove absence using a successful, complete Contents-authorized list.

        GitHub can mask missing Contents access as 404 even when repository
        metadata is visible. No unsuccessful response establishes absence.
        """
        if not lane_name(repository) or re.fullmatch(r"[A-Za-z0-9_-]+", branch) is None:
            raise IdentityRefusal("remote_ref")
        branches = self.collection(f"/repos/{repository}/branches")
        if not all(
            isinstance(row, dict) and isinstance(row.get("name"), str)
            for row in branches
        ):
            raise IdentityRefusal("remote_branch_list")
        matching = [row for row in branches if row["name"] == branch]
        if len(matching) > 1:
            raise IdentityRefusal("remote_branch_list")
        return matching[0] if matching else None

    def collection(self, path: str) -> list:
        if "?" in path:
            raise IdentityRefusal("remote_collection_query")
        rows = []
        for page in range(1, 101):
            result = self._json(path + f"?per_page=100&page={page}")
            if not isinstance(result, list):
                raise IdentityRefusal("remote_api_array")
            rows.extend(result)
            if len(result) < 100:
                return rows
        raise IdentityRefusal("remote_collection_limit")

    def archive(self, repository: str, artifact_id: str) -> bytes:
        if not lane_name(repository) or not decimal_id(artifact_id):
            raise IdentityRefusal("remote_artifact_identity")
        response = self._response(
            f"/repos/{repository}/actions/artifacts/{artifact_id}/zip"
        )
        location = response.headers.get("Location", "")
        status = response.status_code
        response.close()
        target = urlparse(location)
        if (
            status != 302
            or target.scheme != "https"
            or target.username
            or target.password
            or target.port not in (None, 443)
            or target.fragment
            or not target.hostname
            or not target.hostname.endswith(
                (".blob.core.windows.net", ".actions.githubusercontent.com")
            )
        ):
            raise IdentityRefusal("artifact_redirect")
        # Never forward GitHub credentials to the signed blob URL or proxies.
        with requests.Session() as download:
            download.trust_env = False
            try:
                result = download.get(
                    location, timeout=(5, 30), allow_redirects=False, stream=True
                )
                if result.status_code != 200:
                    result.close()
                    raise IdentityRefusal("artifact_download")
                return _bounded(result, 8_000_000)
            except requests.RequestException:
                raise IdentityRefusal("artifact_download") from None


class RemoteRepository:
    """Fresh private bare repository with a fixed GitHub origin and clean config.

    Every fetched object is fsck'ed. Only raw trees and blobs are read. No
    candidate checkout, hooks, attributes, local actions, Python or shell runs.
    """

    def __init__(
        self,
        repository: str,
        *,
        read_token: str | None = None,
        git: str = "/usr/bin/git",
    ):
        if not lane_name(repository) or not repository.startswith(
            "TheAxiomFoundation/"
        ):
            raise IdentityRefusal("remote_repository")
        if not Path(git).is_absolute():
            raise IdentityRefusal("remote_git_executable")
        self.repository, self._git = repository, git
        self._temporary = tempfile.TemporaryDirectory(prefix="axiom-notary-remote-")
        self.path = Path(self._temporary.name) / "objects.git"
        self._environment = {
            "PATH": os.defpath,
            "HOME": self._temporary.name,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_COUNT": "0",
        }
        if read_token:
            encoded = base64.b64encode(
                ("x-access-token:" + read_token).encode()
            ).decode()
            self._environment.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                    "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + encoded,
                }
            )
        try:
            self._run("init", "--bare", str(self.path), in_repo=False)
            self._run(
                "config",
                "remote.origin.url",
                "https://github.com/" + repository + ".git",
            )
            for key, value in (
                ("gc.auto", "0"),
                ("core.hooksPath", "/dev/null"),
                ("http.followRedirects", "false"),
                ("http.sslVerify", "true"),
                ("protocol.file.allow", "never"),
                ("fetch.fsckObjects", "true"),
                ("transfer.fsckObjects", "true"),
            ):
                self._run("config", key, value)
        except BaseException:
            self.close()
            raise

    def _run(self, *args: str, in_repo: bool = True) -> bytes:
        command = [self._git, "--no-replace-objects"]
        if in_repo:
            command.extend(["-C", str(self.path)])
        try:
            result = subprocess.run(
                [*command, *args],
                env=self._environment,
                capture_output=True,
                check=False,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise IdentityRefusal("remote_git_unavailable") from None
        if result.returncode:
            # Git diagnostics may contain a credential-bearing URL/header.
            raise IdentityRefusal("remote_git_failure")
        return result.stdout

    def fetch(self, ref: str) -> str:
        if (
            not oid(ref)
            and re.fullmatch(
                r"refs/(?:heads/[A-Za-z0-9_./-]+|pull/[1-9][0-9]*/head)", ref
            )
            is None
        ):
            raise IdentityRefusal("remote_ref")
        if ".." in ref or ref.endswith("/"):
            raise IdentityRefusal("remote_ref")
        self._run("fetch", "--no-tags", "--no-recurse-submodules", "origin", ref)
        commit = (
            self._run("rev-parse", "--verify", "FETCH_HEAD^{commit}").decode().strip()
        )
        if not oid(commit) or (oid(ref) and commit != ref):
            raise IdentityRefusal("remote_commit")
        self._run("fsck", "--full", "--strict", "--no-reflogs")
        return commit

    def snapshot(self, commit: str) -> Snapshot:
        # The process must have a deployment-owned environment; this repository
        # was created here and never admits candidate local Git configuration.
        if fsck_clean(self.path) is not True:
            raise IdentityRefusal("remote_fsck")
        result = Snapshot.read(self.path, commit)
        if isinstance(result, Refusal):
            raise IdentityRefusal("remote_tree_structural")
        return result

    def chain_history(self, commit: str) -> list[Snapshot]:
        if not oid(commit):
            raise IdentityRefusal("remote_chain_commit")
        lines = (
            self._run("rev-list", "--reverse", "--parents", commit)
            .decode()
            .splitlines()
        )
        previous, history = None, []
        for line in lines:
            values = line.split()
            if len(values) != (1 if previous is None else 2) or (
                previous is not None and values[1] != previous
            ):
                raise IdentityRefusal("remote_chain_not_linear")
            history.append(self.snapshot(values[0]))
            previous = values[0]
        return history

    def close(self):
        self._environment.clear()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
