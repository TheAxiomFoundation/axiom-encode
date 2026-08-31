"""Tests for the RuleSpec-owned corpus release selector contract."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from axiom_encode import signing_broker
from axiom_encode.corpus_release import RELEASE_OBJECT_PUBLIC_KEY_ENV
from axiom_encode.corpus_resolver import InvalidCorpusReleaseError
from axiom_encode.toolchain import (
    RuleSpecToolchainError,
    load_rulespec_corpus_release_pin,
    load_rulespec_local_corpus_release,
    validation_waiver_digest_transition_issues,
    verify_rulespec_validation_waiver_set,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    bind_test_corpus_release,
)

RELEASE_NAME = "test-rulespec-release"
WAIVER_TEXT = "validate_failures: {}\n"
WAIVER_SHA256 = hashlib.sha256(WAIVER_TEXT.encode()).hexdigest()


def _toolchain_bytes(
    waiver_bytes: bytes,
    *,
    release_name: str = RELEASE_NAME,
    content_sha256: str = "a" * 64,
) -> bytes:
    return (
        "[toolchain]\n"
        f'axiom_corpus_release = "{release_name}"\n'
        f'axiom_corpus_release_content_sha256 = "{content_sha256}"\n'
        "validation_waiver_set_sha256 = "
        f'"{hashlib.sha256(waiver_bytes).hexdigest()}"\n'
    ).encode()


def test_pending_waiver_digest_transition_binds_exact_base_and_head_bytes():
    base_waivers = b"validate_failures: {}\n"
    head_waivers = b"validate_failures:\n  us/statutes/26/1.yaml:\n    pending: {}\n"

    assert (
        validation_waiver_digest_transition_issues(
            base_toolchain=_toolchain_bytes(base_waivers),
            head_toolchain=_toolchain_bytes(head_waivers),
            base_waivers=base_waivers,
            head_waivers=head_waivers,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("corpus-release", "may not change the corpus release"),
        ("corpus-digest", "may not change the corpus release"),
        ("formatting", "bytes may change only"),
        ("extra-key", "must contain exactly"),
        ("stale-head-digest", "does not bind the exact head"),
        ("stale-base-digest", "does not bind the exact protected-base"),
    ],
)
def test_pending_waiver_digest_transition_rejects_every_other_toolchain_edit(
    mutation: str,
    expected: str,
):
    base_waivers = b"validate_failures: {}\n"
    head_waivers = b"validate_failures:\n  us/statutes/26/1.yaml:\n    pending: {}\n"
    base_toolchain = _toolchain_bytes(base_waivers)
    head_toolchain = _toolchain_bytes(head_waivers)

    if mutation == "corpus-release":
        head_toolchain = _toolchain_bytes(head_waivers, release_name="other-release")
    elif mutation == "corpus-digest":
        head_toolchain = _toolchain_bytes(head_waivers, content_sha256="b" * 64)
    elif mutation == "formatting":
        head_toolchain += b"# unrelated formatting\n"
    elif mutation == "extra-key":
        head_toolchain += b'unrelated = "value"\n'
    elif mutation == "stale-head-digest":
        head_toolchain = base_toolchain
    elif mutation == "stale-base-digest":
        base_toolchain = _toolchain_bytes(head_waivers)

    issues = validation_waiver_digest_transition_issues(
        base_toolchain=base_toolchain,
        head_toolchain=head_toolchain,
        base_waivers=base_waivers,
        head_waivers=head_waivers,
    )

    assert any(expected in issue for issue in issues)


@pytest.mark.parametrize(
    "mutation",
    [
        "whitespace",
        "key-order",
        "quote-style",
        "leading-comment",
        "inline-comment",
        "crlf",
        "final-newline",
    ],
)
def test_pending_waiver_digest_transition_rejects_semantic_toml_reformatting(
    mutation: str,
):
    base_waivers = b"validate_failures: {}\n"
    head_waivers = b"validate_failures:\n  us/statutes/26/1.yaml:\n    pending: {}\n"
    base_toolchain = _toolchain_bytes(base_waivers)
    head_toolchain = _toolchain_bytes(head_waivers)

    if mutation == "whitespace":
        head_toolchain = head_toolchain.replace(
            b'axiom_corpus_release = "',
            b'axiom_corpus_release    =    "',
        )
    elif mutation == "key-order":
        table, release, content, waiver = head_toolchain.splitlines(keepends=True)
        head_toolchain = b"".join((table, content, release, waiver))
    elif mutation == "quote-style":
        head_toolchain = head_toolchain.replace(
            f'"{RELEASE_NAME}"'.encode(),
            f"'{RELEASE_NAME}'".encode(),
        )
    elif mutation == "leading-comment":
        head_toolchain = b"# semantically inert\n" + head_toolchain
    elif mutation == "inline-comment":
        head_toolchain = head_toolchain.replace(
            f'axiom_corpus_release = "{RELEASE_NAME}"\n'.encode(),
            f'axiom_corpus_release = "{RELEASE_NAME}" # same value\n'.encode(),
        )
    elif mutation == "crlf":
        head_toolchain = head_toolchain.replace(b"\n", b"\r\n")
    elif mutation == "final-newline":
        head_toolchain = head_toolchain.removesuffix(b"\n")

    assert validation_waiver_digest_transition_issues(
        base_toolchain=base_toolchain,
        head_toolchain=head_toolchain,
        base_waivers=base_waivers,
        head_waivers=head_waivers,
    ) == (
        "head toolchain bytes may change only the canonical "
        "validation_waiver_set_sha256 value",
    )


def test_pending_waiver_digest_transition_preserves_unusual_base_formatting():
    base_waivers = b"validate_failures: {}\n"
    head_waivers = b"validate_failures:\n  us/statutes/26/1.yaml:\n    pending: {}\n"
    base_digest = hashlib.sha256(base_waivers).hexdigest()
    head_digest = hashlib.sha256(head_waivers).hexdigest()
    base_toolchain = (
        "# existing toolchain formatting\r\n"
        "[toolchain] # retained table comment\r\n"
        "validation_waiver_set_sha256\t = \t"
        f'"{base_digest}" # exact waiver bytes\r\n'
        f"axiom_corpus_release_content_sha256 = '{'a' * 64}'\r\n"
        f"axiom_corpus_release = '{RELEASE_NAME}'"
    ).encode()
    head_toolchain = base_toolchain.replace(
        base_digest.encode(),
        head_digest.encode(),
        1,
    )

    assert (
        validation_waiver_digest_transition_issues(
            base_toolchain=base_toolchain,
            head_toolchain=head_toolchain,
            base_waivers=base_waivers,
            head_waivers=head_waivers,
        )
        == ()
    )


@pytest.mark.parametrize("side", ["protected-base", "head"])
@pytest.mark.parametrize(
    ("digest_kind", "bad_digest"),
    [
        ("wrong", "0" * 64),
        ("uppercase", "A" * 64),
        ("nonhex", "g" * 64),
        ("short", "a" * 63),
    ],
)
def test_pending_waiver_digest_transition_rejects_wrong_or_malformed_digest(
    side: str,
    digest_kind: str,
    bad_digest: str,
):
    base_waivers = b"validate_failures: {}\n"
    head_waivers = b"validate_failures:\n  us/statutes/26/1.yaml:\n    pending: {}\n"
    base_toolchain = _toolchain_bytes(base_waivers)
    head_toolchain = _toolchain_bytes(head_waivers)
    target_waivers = base_waivers if side == "protected-base" else head_waivers
    actual_digest = hashlib.sha256(target_waivers).hexdigest().encode()
    if side == "protected-base":
        base_toolchain = base_toolchain.replace(
            actual_digest,
            bad_digest.encode(),
            1,
        )
    else:
        head_toolchain = head_toolchain.replace(
            actual_digest,
            bad_digest.encode(),
            1,
        )

    issues = validation_waiver_digest_transition_issues(
        base_toolchain=base_toolchain,
        head_toolchain=head_toolchain,
        base_waivers=base_waivers,
        head_waivers=head_waivers,
    )

    expected = (
        f"{side} toolchain does not bind the exact {side} waiver bytes"
        if digest_kind == "wrong"
        else f"{side} toolchain is invalid"
    )
    assert any(expected in issue for issue in issues)


def test_pending_waiver_digest_transition_rejects_semantic_noop():
    waivers = b"validate_failures: {}\n"
    toolchain = _toolchain_bytes(waivers)

    issues = validation_waiver_digest_transition_issues(
        base_toolchain=toolchain,
        head_toolchain=toolchain,
        base_waivers=waivers,
        head_waivers=waivers,
    )

    assert any("did not change" in issue for issue in issues)


def test_pending_waiver_digest_transition_requires_canonical_base_assignment():
    base_waivers = b"validate_failures: {}\n"
    head_waivers = b"validate_failures:\n  us/statutes/26/1.yaml:\n    pending: {}\n"
    base_toolchain = _toolchain_bytes(base_waivers).replace(
        f'"{hashlib.sha256(base_waivers).hexdigest()}"'.encode(),
        f"'{hashlib.sha256(base_waivers).hexdigest()}'".encode(),
    )

    issues = validation_waiver_digest_transition_issues(
        base_toolchain=base_toolchain,
        head_toolchain=_toolchain_bytes(head_waivers),
        base_waivers=base_waivers,
        head_waivers=head_waivers,
    )

    assert any("exactly one canonical" in issue for issue in issues)


def _write_toolchain(
    root: Path,
    release_name: str = RELEASE_NAME,
    *,
    content_sha256: str = "a" * 64,
) -> Path:
    (root / "known-validation-gaps.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "known-validation-gaps.yaml").write_text(WAIVER_TEXT, encoding="utf-8")
    path = root / ".axiom" / "toolchain.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release_name}"\n'
        f'axiom_corpus_release_content_sha256 = "{content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{WAIVER_SHA256}"\n',
        encoding="utf-8",
    )
    return path


def _write_corpus_release(root: Path, name: str = RELEASE_NAME):
    version = "2026-01-01"
    provision = root / f"data/corpus/provisions/us/statute/{version}.jsonl"
    provision.parent.mkdir(parents=True)
    provision.write_text(
        json.dumps(
            {
                "id": "test:us/statute/1",
                "citation_path": "us/statute/1",
                "body": "authoritative source",
                "jurisdiction": "us",
                "document_class": "statute",
                "version": version,
                "source_path": "sources/us/statute/test",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return bind_test_corpus_release(
        root,
        name,
        [("us", "statute", version)],
    )


def test_load_rulespec_local_corpus_release_binds_exact_named_selector(tmp_path):
    rulespec = tmp_path / "rulespec-us"
    corpus = tmp_path / "corpus"
    module = rulespec / "rules" / "program.rulespec.md"
    module.parent.mkdir(parents=True)
    module.write_text("# Program\n", encoding="utf-8")
    release = _write_corpus_release(corpus)
    _write_toolchain(rulespec, content_sha256=release.content_sha256)

    release = load_rulespec_local_corpus_release(module, corpus)

    assert release.root == corpus.resolve()
    assert release.name == RELEASE_NAME
    assert release.content_sha256
    assert release.selector_sha256
    assert (
        release.provisions_root == (corpus / "data" / "corpus" / "provisions").resolve()
    )


def test_toolchain_accepts_checkout_with_canonical_program_specs(tmp_path):
    rulespec = tmp_path / "rulespec-us"
    (rulespec / "programs/us/snap").mkdir(parents=True)
    _write_toolchain(rulespec)

    assert load_rulespec_corpus_release_pin(rulespec) == (RELEASE_NAME, "a" * 64)


def test_environment_corpus_key_cannot_replace_protected_broker(
    tmp_path,
    monkeypatch,
):
    rulespec = tmp_path / "rulespec-us"
    corpus = tmp_path / "corpus"
    release = _write_corpus_release(corpus)
    _write_toolchain(rulespec, content_sha256=release.content_sha256)
    monkeypatch.setattr(signing_broker, "_active_broker", None)
    monkeypatch.setenv(RELEASE_OBJECT_PUBLIC_KEY_ENV, TEST_RELEASE_PUBLIC_KEY)

    with pytest.raises(RuleSpecToolchainError, match="protected signing broker"):
        load_rulespec_local_corpus_release(rulespec, corpus)


def test_nested_rulespec_path_rejects_multiple_toolchains(tmp_path):
    monorepo = tmp_path / "rulespec-us"
    child = monorepo / "jurisdictions" / "us"
    module = child / "program.rulespec.md"
    module.parent.mkdir(parents=True)
    module.write_text("# Program\n", encoding="utf-8")
    _write_toolchain(monorepo, "parent-release")
    _write_toolchain(child, "child-release")

    with pytest.raises(RuleSpecToolchainError, match="multiple ancestor"):
        load_rulespec_corpus_release_pin(module)


def test_git_checkout_requires_toolchain_at_repository_root(tmp_path):
    checkout = tmp_path / "rulespec-us"
    module = checkout / "jurisdictions" / "us" / "program.rulespec.md"
    module.parent.mkdir(parents=True)
    module.write_text("# Program\n", encoding="utf-8")
    (checkout / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    _write_toolchain(module.parent, "nested-release")

    with pytest.raises(RuleSpecToolchainError, match="exactly one.*at its root"):
        load_rulespec_corpus_release_pin(module)


@pytest.mark.parametrize(
    "contents",
    [
        "[toolchain]\n",
        "[toolchain]\naxiom_corpus_release = ''\n",
        "[toolchain]\naxiom_corpus_release = ' padded '\n",
        "[toolchain]\naxiom_corpus_release = 'bad release'\n",
        "axiom_corpus_release = 'top-level'\n",
        (
            "[toolchain]\n"
            f"axiom_corpus_release = '{RELEASE_NAME}'\n"
            f"axiom_corpus_release_content_sha256 = '{'a' * 64}'\n"
            f"validation_waiver_set_sha256 = '{WAIVER_SHA256}'\n"
            "[legacy]\nenabled = true\n"
        ),
    ],
)
def test_toolchain_rejects_missing_or_noncanonical_selector(tmp_path, contents):
    rulespec = tmp_path / "rulespec-us"
    path = _write_toolchain(rulespec)
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(RuleSpecToolchainError):
        load_rulespec_corpus_release_pin(rulespec)


def test_named_selector_does_not_fall_back_to_current(tmp_path):
    rulespec = tmp_path / "rulespec-us"
    corpus = tmp_path / "corpus"
    _write_toolchain(rulespec, "missing-release")
    (corpus / "data/corpus/provisions").mkdir(parents=True)

    with pytest.raises(InvalidCorpusReleaseError, match="missing-release"):
        load_rulespec_local_corpus_release(rulespec, corpus)


def test_local_release_rejects_waiver_bytes_not_bound_by_toolchain(tmp_path):
    rulespec = tmp_path / "rulespec-us"
    corpus = tmp_path / "corpus"
    _write_toolchain(rulespec)
    release = _write_corpus_release(corpus)
    _write_toolchain(rulespec, content_sha256=release.content_sha256)
    (rulespec / "known-validation-gaps.yaml").write_text(
        "validate_failures:\n  changed: true\n",
        encoding="utf-8",
    )

    with pytest.raises(RuleSpecToolchainError, match="sha256 does not match"):
        load_rulespec_local_corpus_release(rulespec, corpus)


def test_waiver_set_symlink_is_rejected(tmp_path):
    rulespec = tmp_path / "rulespec-us"
    _write_toolchain(rulespec)
    waiver = rulespec / "known-validation-gaps.yaml"
    outside = tmp_path / "outside-waivers.yaml"
    outside.write_text(WAIVER_TEXT, encoding="utf-8")
    waiver.unlink()
    waiver.symlink_to(outside)

    with pytest.raises(RuleSpecToolchainError, match="safely open"):
        verify_rulespec_validation_waiver_set(rulespec)


def test_mutable_current_release_name_is_rejected(tmp_path):
    rulespec = tmp_path / "rulespec-us"
    corpus = tmp_path / "corpus"
    _write_toolchain(rulespec, "current")
    (corpus / "data/corpus/provisions").mkdir(parents=True)

    with pytest.raises(RuleSpecToolchainError, match="current.*reserved"):
        load_rulespec_local_corpus_release(rulespec, corpus)


def test_toolchain_symlink_is_rejected(tmp_path):
    rulespec = tmp_path / "rulespec-us"
    actual = tmp_path / "actual-toolchain.toml"
    actual.write_text(
        f'[toolchain]\naxiom_corpus_release = "{RELEASE_NAME}"\n',
        encoding="utf-8",
    )
    path = rulespec / ".axiom" / "toolchain.toml"
    path.parent.mkdir(parents=True)
    path.symlink_to(actual)

    with pytest.raises(RuleSpecToolchainError, match="must not be a symlink"):
        load_rulespec_corpus_release_pin(rulespec)


@pytest.mark.parametrize("name", ["rulespec-us-co", "rulespec", "workspace"])
def test_toolchain_rejects_noncanonical_checkout_roots(tmp_path, name):
    rulespec = tmp_path / name
    _write_toolchain(rulespec)

    with pytest.raises(
        RuleSpecToolchainError,
        match="exact canonical rulespec-<country> checkout",
    ):
        load_rulespec_corpus_release_pin(rulespec)


def test_production_code_contains_no_legacy_corpus_release_escape_hatches():
    source_root = Path(__file__).parents[1] / "src" / "axiom_encode"
    forbidden_tokens = {
        "require_release": "release validation opt-out",
        "local_corpus_only": "obsolete local-only mode flag",
        "_candidate_local_axiom_corpus_paths": "ambient corpus discovery",
        "AXIOM_CORPUS_REPO": "ambient corpus environment override",
        "AXIOM_CORPUS_ARTIFACT_ROOT": "legacy corpus artifact override",
        "AXIOM_CORPUS_ROOT": "ambient corpus automation override",
        "--corpus-root": "legacy corpus CLI flag",
        "resolve_supabase_corpus_source": "remote corpus resolver",
        "CorpusRemoteError": "remote corpus resolver error surface",
        "_MAX_CORPUS_CLAIM_BYTES": "retired corpus claim artifact scanner",
        "data/corpus/claims": "unversioned corpus claim artifact lookup",
        "_fetch_local_source_claim_record": "release-agnostic claim lookup",
        "_uk_source_path_aliases": "hard-coded corpus alias bridge",
        "_canonical_corpus_citation_path_alias": "hard-coded corpus alias bridge",
        "exact-alias-dependency": "unattested corpus alias classification",
        "def source_verification_block": "raw source-text pinning helper",
        "def corpus_provisions_root": "legacy corpus layout discovery",
        "current.json": "mutable release selector path",
    }

    for path in source_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token, label in forbidden_tokens.items():
            assert token not in source, f"{path}: retained {label}: {token}"

        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "claims":
                raise AssertionError(
                    f"{path}:{getattr(node, 'lineno', '?')}: retained corpus "
                    "claim artifact path segment"
                )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = [*node.args.posonlyargs, *node.args.args]
            positional_defaults = [
                *(None for _ in range(len(positional) - len(node.args.defaults))),
                *node.args.defaults,
            ]
            defaulted_arguments = [
                *zip(positional, positional_defaults, strict=True),
                *zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True),
            ]
            for argument, default in defaulted_arguments:
                if argument.arg not in {"release_name", "release_selector"}:
                    continue
                assert not (
                    isinstance(default, ast.Constant) and default.value == "current"
                ), f"{path}:{node.lineno}: {argument.arg} defaults to current"


def test_production_validator_constructors_choose_release_binding_explicitly():
    repository_root = Path(__file__).parents[1]
    source_paths = [
        *(repository_root / "src" / "axiom_encode").rglob("*.py"),
        *(repository_root / "scripts").glob("*.py"),
    ]
    explicitly_unbound_private_callers = {
        ("src/axiom_encode/cli.py", "cmd_compile"),
        ("src/axiom_encode/cli.py", "cmd_test"),
        ("src/axiom_encode/cli.py", "_rulespec_companion_test_failures"),
    }

    for path in source_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        relative_path = path.relative_to(repository_root).as_posix()
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ValidatorPipeline"
            ):
                continue
            parent = parents.get(node)
            while parent is not None and not isinstance(
                parent, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                parent = parents.get(parent)
            caller = parent.name if parent is not None else "<module>"
            release_keywords = [
                keyword
                for keyword in node.keywords
                if keyword.arg == "local_corpus_release"
            ]
            assert len(release_keywords) == 1, (
                f"{relative_path}:{node.lineno}: ValidatorPipeline in {caller} is "
                "not explicit about its LocalCorpusRelease binding"
            )
            release_value = release_keywords[0].value
            if isinstance(release_value, ast.Constant) and release_value.value is None:
                assert (relative_path, caller) in explicitly_unbound_private_callers, (
                    f"{relative_path}:{node.lineno}: ValidatorPipeline in {caller} "
                    "uses an unbound corpus outside a private compiler helper"
                )

        for node in ast.walk(tree):
            if relative_path != "src/axiom_encode/cli.py" or not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "args"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "corpus_path"
                and isinstance(node.args[2], ast.Constant)
                and node.args[2].value is None
            ):
                continue
            raise AssertionError(
                f"{relative_path}:{node.lineno}: optional corpus_path compatibility "
                "lookup is forbidden"
            )
