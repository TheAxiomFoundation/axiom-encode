"""Fresh public verification assets pinned in the finalized base.

No token or model credential is used. Corpus signatures resolve exclusively
through the authenticated registry. Download failure is never a gate pass.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import requests

from ._schema import fields
from .canonical import strict_parse
from .identity import IdentityRefusal
from .protocol import oid
from .remote import RemoteRepository, _bounded

PATH = ".axiom/notary/assets.json"


@contextmanager
def provision(base, directory: Path, lane: str):
    body = strict_parse(base.blobs.get(PATH, b""))
    if (
        not fields(body, {"schema", "lane", "repositories", "corpus_release_base_url"})
        or body["schema"] != "axiom/notary-assets/v1"
        or body["lane"] != lane
        or not isinstance(body["repositories"], list)
    ):
        raise IdentityRefusal("runner_assets_configuration")
    repositories, previous = {}, ""
    for row in body["repositories"]:
        if (
            not fields(row, {"name", "repository", "git_oid"})
            or not isinstance(row["name"], str)
            or not re.fullmatch(r"(?:corpus|engine|rulespec_[a-z]{2})", row["name"])
            or row["name"] <= previous
            or not oid(row["git_oid"])
        ):
            raise IdentityRefusal("runner_assets_repository")
        name = row["name"]
        expected = (
            "axiom-corpus"
            if name == "corpus"
            else "axiom-rules-engine"
            if name == "engine"
            else name.replace("_", "-")
        )
        if row["repository"] != "TheAxiomFoundation/" + expected:
            raise IdentityRefusal("runner_assets_repository")
        previous = name
        target = directory / expected
        with RemoteRepository(row["repository"]) as remote:
            remote.fetch(row["git_oid"])
            remote._run(
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(remote.path),
                str(target),
                in_repo=False,
            )
            remote._run(
                "-C",
                str(target),
                "-c",
                "core.hooksPath=/dev/null",
                "checkout",
                "--detach",
                row["git_oid"],
                in_repo=False,
            )
            remote._run(
                "-C",
                str(target),
                "config",
                "remote.origin.url",
                "https://github.com/" + row["repository"] + ".git",
                in_repo=False,
            )
        repositories[name] = target
    if not {"corpus", "engine"} <= set(repositories):
        raise IdentityRefusal("runner_assets_required")
    url = body["corpus_release_base_url"]
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or ".." in parsed.path
    ):
        raise IdentityRefusal("runner_assets_origin")
    from axiom_encode.toolchain import parse_rulespec_toolchain_bytes

    # Parse exact committed bytes without borrowing organization variables.
    toolchain = parse_rulespec_toolchain_bytes(
        base.blobs[".axiom/toolchain.toml"], root=directory
    )
    relative = (
        "releases/"
        + toolchain.corpus_release
        + "/"
        + toolchain.corpus_release_content_sha256
        + ".json"
    )
    destination = repositories["corpus"] / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(
                url.rstrip("/") + "/" + relative,
                timeout=(5, 120),
                stream=True,
                allow_redirects=False,
                headers={"User-Agent": "axiom-notary-verifier"},
            )
            if response.status_code != 200:
                response.close()
                raise IdentityRefusal("runner_corpus_download")
            raw = _bounded(response, 64_000_000)
        with destination.open("xb") as stream:
            stream.write(raw)
    # Consumers validate named release address, signature, and source bytes
    # through load_rulespec_local_corpus_release before using any provision.
    yield {name + "_root": str(path) for name, path in repositories.items()}
