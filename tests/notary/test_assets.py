"""Real local Git transport regression: checkouts retain canonical identities."""

import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from axiom_encode.notary.assets import provision
from axiom_encode.notary.canonical import jcs_dumps
from axiom_encode.notary.remote import RemoteRepository


def test_asset_clones_keep_canonical_origins_and_exact_commits(tmp_path, monkeypatch):
    def git(*args):
        return (
            subprocess.check_output(["git", *map(str, args)], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

    repos, specs = {}, []
    for name, repo in [("corpus", "axiom-corpus"), ("engine", "axiom-rules-engine")]:
        source = tmp_path / name
        source.mkdir()
        git("init", source)
        (source / "fixture").write_text(name)
        if name == "corpus":
            release = source / "releases/test" / ("a" * 64 + ".json")
            release.parent.mkdir(parents=True)
            release.write_text("fixture release; authenticity checked by consumers")
        git("-C", source, "add", ".")
        git(
            "-C",
            source,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-m",
            "fixture",
        )
        oid = git("-C", source, "rev-parse", "HEAD")
        repos["TheAxiomFoundation/" + repo] = source, oid
        specs.append(
            {"name": name, "repository": "TheAxiomFoundation/" + repo, "git_oid": oid}
        )

    @contextmanager
    def local_remote(repository):
        with RemoteRepository(repository) as remote:
            source, oid = repos[repository]
            remote._run("-c", "protocol.file.allow=always", "fetch", str(source), oid)
            remote.fetch = lambda ref: oid if ref == oid else None
            yield remote

    monkeypatch.setattr("axiom_encode.notary.assets.RemoteRepository", local_remote)
    monkeypatch.setattr(
        "axiom_encode.toolchain.parse_rulespec_toolchain_bytes",
        lambda *a, **kw: SimpleNamespace(
            corpus_release="test", corpus_release_content_sha256="a" * 64
        ),
    )
    base = SimpleNamespace(
        blobs={
            ".axiom/notary/assets.json": jcs_dumps(
                {
                    "schema": "axiom/notary-assets/v1",
                    "lane": "TheAxiomFoundation/rulespec-nz",
                    "repositories": specs,
                    "corpus_release_base_url": "https://corpus.example.test",
                }
            ),
            ".axiom/toolchain.toml": b"fixture",
        }
    )
    work = tmp_path / "assets"
    work.mkdir()
    with provision(base, work, "TheAxiomFoundation/rulespec-nz") as roots:
        for spec in specs:
            checkout = Path(roots[spec["name"] + "_root"])
            assert git("-C", checkout, "rev-parse", "HEAD") == spec["git_oid"]
            assert (
                git("-C", checkout, "remote", "get-url", "origin")
                == "https://github.com/" + spec["repository"] + ".git"
            )
            assert (checkout / "fixture").read_text() == spec["name"]
