"""Change 5b of the merge guard design (#2172), workflow half: switched off.

The evidence step of the hosted signed workflow ("Package exact generated
changes") accepts the ``applied-rulespec/v6`` promotion manifest only while
every lock of the v6 writer is open. With the committed locks it refuses every
v6 manifest, and its v5 behaviour is the same as before this change.

On-state behaviour is reached in tests only: in this process by patching the
constants, and in the step's own subprocess by a test prelude that patches
them before it runs the step. Nothing here reads git history or the network.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from axiom_encode import applied_manifest_v6 as v6
from axiom_encode import candidate_promotion_evidence as evidence
from axiom_encode import cli
from axiom_encode import reviewed_pr_admission as admission
from tests.test_cli import TEST_APPLY_SIGNING_BROKER

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/targeted-signed-reencode.yml"
STEP = "Package exact generated changes"
APP = 987654
PR_NUMBER = 2044
REPOSITORY = "TheAxiomFoundation/rulespec-us"
BRANCH_PATH = f"/repos/{REPOSITORY}/branches/main"
CITATION = "us/statute/7/2015/f"
PRIMARY = "us/statutes/7/2015/f.yaml"
COMPANION = "us/statutes/7/2015/f.test.yaml"
MANIFEST = ".axiom/encoding-manifests/statutes/7/2015/f.yaml.json"
PAVEL = {"id": 110687043, "login": "PavelMakarchuk"}
TRANSACTION = {
    "schema": "axiom-encode/atomic-source-transaction/v5",
    "source_bundle": [],
    "canonical_refresh_bundle": [],
    "primary_required_test_cases": [],
    "require_complete_source_unit": True,
    "manifest_only_refresh": False,
    "reviewed_candidate_promotion": True,
}
LIVE_PROMOTION = json.dumps(
    {
        "schema": "axiom-encode/reviewed-pr-promotion/v1",
        "rulespec_pr": PR_NUMBER,
        "transaction": TRANSACTION,
    }
)
RECORDED_BRANCH = json.loads(
    (
        ROOT / "tests/fixtures/merge_guard_requirement/rulespec_us_branch_main.json"
    ).read_text("utf-8")
)["response"]
REVIEWED_EVIDENCE = "axiom-encode/reviewed-candidate-promotion-evidence/v1"
CANDIDATE_EVIDENCE = "axiom-encode/candidate-promotion-evidence/v1"


def branch_record(*, guard_app_id=APP):
    """The public branch record of main, with the guard check required."""

    record = copy.deepcopy(RECORDED_BRANCH)
    checks = [{"context": "validate / validate", "app_id": 15368}]
    if guard_app_id is not None:
        checks.append({"context": admission.GUARD_CHECK_NAME, "app_id": guard_app_id})
    record["protection"]["required_status_checks"] = {
        "enforcement_level": "everyone",
        "contexts": [check["context"] for check in checks],
        "checks": checks,
    }
    return record


def v5_promotion(ref: str) -> dict:
    """An unsigned v5 promotion payload with the exact live field set."""

    payload = {
        "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "generated_at": "2026-11-15T00:00:00Z",
        "tool": v6.PROMOTION_TOOL,
        "axiom_encode_version": "0.2.0",
        "axiom_encode_git": {"commit": "c" * 40, "dirty": False},
        "run_id": "12345",
        "citation": CITATION,
        "reviewed_rulespec_ref": ref,
        "validation_waiver_set_sha256": "0" * 64,
        "applied_files": [
            {"path": PRIMARY, "sha256": "1" * 64},
            {"path": COMPANION, "sha256": "2" * 64},
        ],
        "source_attestation": {"rulespec_root": "rulespec-us/us"},
        "validation_execution": {"policy_pre_apply": {}},
    }
    assert set(payload) | {"signature"} == cli._REVIEWED_CANDIDATE_MANIFEST_FIELDS
    return payload


def sign(payload: dict) -> dict:
    cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    return payload


def written_v6(ref: str, **scope) -> dict:
    """What the switched-off writer of change 5 emits, signed."""

    arguments = {
        "repository": REPOSITORY,
        "pull_request": PR_NUMBER,
        "base_sha": "a" * 40,
        "author_id": PAVEL["id"],
        "dispatcher": dict(PAVEL),
        **scope,
    }
    with (
        patch.object(admission, "V6_WRITER_ENABLED", True),
        patch.object(v6, "V6_READER_ENABLED", True),
    ):
        payload = v6.as_v6_candidate_payload(
            v5_promotion(ref), v5_ref_field="reviewed_rulespec_ref", **arguments
        )
    return sign(payload)


REF = "b" * 40


class Branch:
    """A stand-in for the anonymous GitHub read of lock 3."""

    def __init__(self, value):
        self.value, self.calls = value, []

    def __call__(self, path):
        self.calls.append(path)
        if isinstance(self.value, Exception):
            raise self.value
        if path != BRANCH_PATH:
            raise AssertionError(f"unexpected read {path}")
        return copy.deepcopy(self.value)


@pytest.fixture
def on(monkeypatch):
    """Every lock of the writer open. Tests only."""

    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)
    monkeypatch.setattr(v6, "V6_READER_ENABLED", True)


def admit(payload, *, fetch=None, **overrides):
    arguments = {
        "atomic_source_json": LIVE_PROMOTION,
        "country": "us",
        "citation": CITATION,
        "rulespec_ref": REF,
        "fetch": Branch(branch_record()) if fetch is None else fetch,
        **overrides,
    }
    return evidence.require_candidate_promotion_evidence(payload, **arguments)


def outcome(payload, **kwargs):
    try:
        return ("ok", admit(payload, **kwargs))
    except ValueError as exc:
        return ("refused", str(exc))


# --- the locks are the writer's, and they are shut -----------------------------


def test_locks_are_committed_shut_and_have_no_runtime_override():
    assert admission.V6_WRITER_ENABLED is False
    assert admission.GUARD_APP_ID is None
    assert v6.V6_READER_ENABLED is False
    source = Path(evidence.__file__).read_text()
    for forbidden in ("os.environ", "getenv", "import os", "argparse", "sys.argv"):
        assert forbidden not in source
    # No switch of its own: it reads no constant of the locks itself.
    for name in ("V6_WRITER_ENABLED", "V6_READER_ENABLED", "GUARD_APP_ID"):
        assert name not in _code_without_docstrings(source)


def test_off_the_written_v6_manifest_is_refused_before_anything_is_read():
    branch = Branch(branch_record())
    with pytest.raises(ValueError, match="writer is switched off"):
        admit(written_v6(REF), fetch=branch)
    assert branch.calls == []


def _lock_states():
    return itertools.product(
        (False, True),  # lock 1
        (False, True),  # the reader switch
        (None, 15368, APP),  # lock 2 (15368 is GitHub Actions)
        ("required", "recorded", "unreadable", "rate-limited"),  # lock 3
    )


@pytest.mark.parametrize(("writer", "reader", "app_id", "branch"), _lock_states())
def test_admitted_only_with_every_lock_of_the_writer_open(
    monkeypatch, writer, reader, app_id, branch
):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", writer)
    monkeypatch.setattr(v6, "V6_READER_ENABLED", reader)
    monkeypatch.setattr(admission, "GUARD_APP_ID", app_id)
    fetch = Branch(
        {
            "required": branch_record(),
            "recorded": RECORDED_BRANCH,
            "unreadable": OSError("connection reset"),
            "rate-limited": admission.AdmissionError("rate limited"),
        }[branch]
    )
    result = outcome(written_v6(REF), fetch=fetch)
    if writer and reader and app_id == APP and branch == "required":
        assert result == ("ok", REF)
        assert fetch.calls == [BRANCH_PATH]
    else:
        assert result[0] == "refused"
        # Never "nothing there": the read happens only behind locks 1 and 2.
        assert fetch.calls == (
            [BRANCH_PATH] if writer and reader and app_id == APP else []
        )


@pytest.mark.parametrize("value", [1, "True", "yes", object()])
def test_only_an_exact_true_switch_opens_a_lock(monkeypatch, value):
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)
    for name, module in (("V6_WRITER_ENABLED", admission), ("V6_READER_ENABLED", v6)):
        monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
        monkeypatch.setattr(v6, "V6_READER_ENABLED", True)
        monkeypatch.setattr(module, name, value)
        assert outcome(written_v6(REF))[0] == "refused", name


def test_the_locks_are_checked_with_the_writers_own_functions(on, monkeypatch):
    payload = written_v6(REF)
    calls = []
    real_writer = v6.require_candidate_writer
    real_guard = admission.require_guard_enforced
    monkeypatch.setattr(
        v6,
        "require_candidate_writer",
        lambda: calls.append("writer") or real_writer(),
    )
    monkeypatch.setattr(
        admission,
        "require_guard_enforced",
        lambda repository, *, fetch: (
            calls.append(("guard", repository)) or real_guard(repository, fetch=fetch)
        ),
    )
    assert admit(payload) == REF
    assert calls == ["writer", ("guard", REPOSITORY)]


# --- on (tests only): the written manifest is evidence of exactly this run -----


def test_on_the_written_v6_manifest_is_admitted(on):
    fetch = Branch(branch_record())
    assert admit(written_v6(REF), fetch=fetch) == REF
    assert fetch.calls == [BRANCH_PATH]


def _drop(field):
    return lambda payload: payload.pop(field)


def _scope_set(path, value):
    def mutate(payload):
        target = payload[v6.ATTESTATION_SCOPE_FIELD]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _scope_drop(path):
    def mutate(payload):
        target = payload[v6.ATTESTATION_SCOPE_FIELD]
        for key in path[:-1]:
            target = target[key]
        target.pop(path[-1])

    return mutate


def _set(field, value):
    return lambda payload: payload.__setitem__(field, value)


# Mutations that keep this run's bindings (citation, ref, repository, PR).
CONTRACT_MUTATIONS = {
    **{
        f"drop {field}": _drop(field)
        for field in sorted(cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS[0])
    },
    "extra field": _set("note", "x"),
    "the v5 ref beside the v6 ref": _set("reviewed_rulespec_ref", REF),
    "deterministic_execution": _set("deterministic_execution", {}),
    "per-file waiver bindings beside the digest": _set(
        "validation_waiver_bindings", []
    ),
    "backend": _set("backend", "codex"),
    "model tool": _set("tool", "axiom-encode encode --apply"),
    "scope schema": _scope_set(("schema",), "axiom-encode/attestation-scope/v2"),
    "scope owner review": _scope_set(("owner_review",), "before-signing"),
    "scope validation": _scope_set(("validation",), "none"),
    "scope origin": _scope_set(("origin",), "hosted"),
    "scope merge requirement": _scope_set(("merge_requirement",), "none"),
    "scope extra key": _scope_set(("run_attempt",), 1),
    "scope drops dispatcher": _scope_drop(("dispatcher",)),
    "candidate extra key": _scope_set(("candidate", "title"), "x"),
    "candidate drops base": _scope_drop(("candidate", "base_sha")),
    "candidate base not a sha": _scope_set(("candidate", "base_sha"), "main"),
    "candidate author bool": _scope_set(("candidate", "author_id"), True),
    "candidate author zero": _scope_set(("candidate", "author_id"), 0),
    "candidate head other": _scope_set(("candidate", "head_sha"), "c" * 40),
    "dispatcher id float": _scope_set(("dispatcher", "id"), 110687043.0),
    "dispatcher login invalid": _scope_set(("dispatcher", "login"), "a b"),
    "dispatcher extra key": _scope_set(("dispatcher", "type"), "User"),
    "scope not an object": _set(v6.ATTESTATION_SCOPE_FIELD, []),
    "rulespec root of another repository": _set(
        "source_attestation", {"rulespec_root": "rulespec-uk/uk"}
    ),
    "unchanged": lambda payload: None,
    "per-file waiver variant": lambda payload: (
        payload.pop("validation_waiver_set_sha256"),
        payload.__setitem__("validation_waiver_bindings", []),
    ),
}


def _reader_admits(payload) -> bool:
    return v6.v6_admissible(payload) and not v6.contract_issues(
        payload,
        manifest_label="m",
        field_variants=cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS,
    )


@pytest.mark.parametrize("name", sorted(CONTRACT_MUTATIONS))
def test_evidence_and_reader_agree_both_ways(on, name):
    """Two-way pin: the step admits exactly what the reader admits.

    Each mutation keeps this run's bindings, so any difference between the two
    decisions would be a definition the step does not share with the reader.
    """

    payload = written_v6(REF)
    CONTRACT_MUTATIONS[name](payload)
    assert (outcome(payload)[0] == "ok") is _reader_admits(payload), name


def test_the_pin_corpus_has_both_decisions(on):
    decisions = set()
    for mutate in CONTRACT_MUTATIONS.values():
        payload = written_v6(REF)
        mutate(payload)
        decisions.add(_reader_admits(payload))
    assert decisions == {True, False}


def test_evidence_uses_the_readers_definitions():
    assert evidence.V6_FIELD_VARIANTS is cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS
    assert set(evidence.V6_FIELD_VARIANTS) == set(
        v6.v6_field_variants(
            cli._APPLIED_MANIFEST_SCHEMA_VARIANTS["reviewed candidate promotion"],
            v5_ref_field="reviewed_rulespec_ref",
        )
    )
    tree = ast.parse(Path(evidence.__file__).read_text())
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # Field and scope names come from the reader, never a second spelling.
    for name in (
        "candidate_rulespec_ref",
        "attestation_scope",
        "reviewed_rulespec_ref",
        v6.MANIFEST_SCHEMA_V6,
    ):
        assert name not in literals, name
    # The scope members it reads are members of the reader's field sets.
    assert "candidate" in v6.ATTESTATION_SCOPE_FIELDS
    assert {"repository", "pull_request"} <= v6.ATTESTATION_CANDIDATE_FIELDS
    assert literals & (
        v6.ATTESTATION_SCOPE_FIELDS | v6.ATTESTATION_CANDIDATE_FIELDS
    ) == {"candidate", "repository", "pull_request"}
    attributes = {
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    assert {
        "_manifest_v6.is_v6_labelled",
        "_manifest_v6.require_candidate_writer",
        "_manifest_v6.is_promotion_repository",
        "_manifest_v6.v6_admissible",
        "_manifest_v6.contract_issues",
        "_manifest_v6.CANDIDATE_REF_FIELD",
        "_manifest_v6.ATTESTATION_SCOPE_FIELD",
        "_admission.require_guard_enforced",
    } <= attributes


def _uk_manifest(payload):
    """Consistent in itself, so the reader admits it, but for rulespec-uk."""

    payload[v6.ATTESTATION_SCOPE_FIELD]["candidate"]["repository"] = (
        "TheAxiomFoundation/rulespec-uk"
    )
    payload["source_attestation"] = {"rulespec_root": "rulespec-uk/uk"}


RUN_MISMATCHES = {
    "another citation": (None, {"citation": "us/statute/7/2015/g"}),
    "another requested ref": (None, {"rulespec_ref": "c" * 40}),
    "another country": (None, {"country": "uk"}),
    "another pull request": (
        _scope_set(("candidate", "pull_request"), PR_NUMBER + 1),
        {},
    ),
    "scope names rulespec-uk": (_uk_manifest, {}),
}


@pytest.mark.parametrize("name", sorted(RUN_MISMATCHES))
def test_a_scope_that_names_another_run_is_refused(on, name):
    mutate, overrides = RUN_MISMATCHES[name]
    payload = written_v6(REF)
    if mutate is not None:
        mutate(payload)
    reads = []

    def fetch(path):
        reads.append(path)
        return branch_record()

    # The reader alone cannot see these: only this run's inputs can.
    assert _reader_admits(payload)
    assert outcome(payload, fetch=fetch, **overrides)[0] == "refused"
    # Lock 3 is read for this run's repository, never the manifest's.
    country = overrides.get("country", "us")
    assert reads == [f"/repos/TheAxiomFoundation/rulespec-{country}/branches/main"]


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "",
        "{",
        "null",
        json.dumps(TRANSACTION),
        json.dumps(
            {
                "schema": "axiom-encode/reviewed-pr-promotion/v1",
                "rulespec_pr": PR_NUMBER,
                "transaction": {**TRANSACTION, "reviewed_candidate_promotion": False},
            }
        ),
        json.dumps(
            {
                "schema": "axiom-encode/reviewed-pr-promotion/v1",
                "rulespec_pr": str(PR_NUMBER),
                "transaction": TRANSACTION,
            }
        ),
        '{"schema":"axiom-encode/reviewed-pr-promotion/v1","rulespec_pr":2044,'
        '"rulespec_pr":2045,"transaction":{}}',
        json.dumps({**TRANSACTION, "manifest_only_refresh": True}),
    ],
)
def test_an_unreadable_or_other_lane_is_refused(on, raw):
    assert outcome(written_v6(REF), atomic_source_json=raw)[0] == "refused"


@pytest.mark.parametrize(
    "parsed",
    [
        {"reviewed_candidate_promotion": False, "rulespec_pr": PR_NUMBER},
        {"reviewed_candidate_promotion": None, "rulespec_pr": PR_NUMBER},
        {"reviewed_candidate_promotion": True},
        {"reviewed_candidate_promotion": True, "rulespec_pr": 0},
        {"reviewed_candidate_promotion": True, "rulespec_pr": True},
        {"reviewed_candidate_promotion": True, "rulespec_pr": str(PR_NUMBER)},
    ],
)
def test_the_lane_check_does_not_rely_on_the_parser_alone(on, monkeypatch, parsed):
    """Today only the promotion wrapper yields ``rulespec_pr``; check it anyway."""

    monkeypatch.setattr(evidence, "split_atomic_source_input", lambda raw: parsed)
    with pytest.raises(ValueError, match="live candidate pull request promotion"):
        admit(written_v6(REF))


@pytest.mark.parametrize("value", [None, 1, ["us"]])
def test_an_unreadable_run_input_is_refused(on, value):
    for name in ("atomic_source_json", "country", "citation", "rulespec_ref"):
        assert outcome(written_v6(REF), **{name: value})[0] == "refused", name


@pytest.mark.parametrize("payload", [None, [], "x", "v5", "V6"])
def test_only_a_v6_label_is_judged(on, payload):
    if payload == "v5":
        payload = v5_promotion(REF)
    elif payload == "V6":
        payload = {
            **written_v6(REF),
            "schema_version": "axiom-encode/applied-rulespec/V6",
        }
    with pytest.raises(ValueError, match="not labelled"):
        admit(payload)


# --- the hosted workflow step -------------------------------------------------


def _steps():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return workflow["jobs"]["encode"]["steps"]


def _package_step():
    return next(step for step in _steps() if step.get("name") == STEP)


def _package_script() -> str:
    marker = (
        '"${workflow_python[@]}" - \\\n'
        '  "$artifact/context-manifest.json" \\\n'
        "  \"$artifact/apply-manifests.json\" <<'PY'\n"
    )
    return (
        _package_step()["run"]
        .split(marker, 1)[1]
        .split('\nPY\n"${workflow_python[@]}" - "$artifact/metadata.json"', 1)[0]
    )


# This change, as (now, before). The block that handles a v6 label is removed
# whole; the other edits are restored to their earlier text.
_IMPORT = (
    "from axiom_encode.applied_manifest_v6 import (\n"
    "    CANDIDATE_REF_FIELD,\n"
    "    is_v6_labelled,\n"
    ")\n"
    "# Changed manifests admitted below as v6 promotion evidence.\n"
    "v6_evidence_paths = set()\n"
)
_REF_NOW = (
    "            or evidence_manifest.get(\n"
    "                CANDIDATE_REF_FIELD\n"
    "                if relative_path in v6_evidence_paths\n"
    '                else "reviewed_rulespec_ref"\n'
    "            )\n"
)
_REF_BEFORE = '            or evidence_manifest.get("reviewed_rulespec_ref")\n'
_SCHEMA_NOW = (
    "                        CANDIDATE_PROMOTION_EVIDENCE_SCHEMA\n"
    "                        if relative_path in v6_evidence_paths\n"
    "                        else "
    '"axiom-encode/reviewed-candidate-promotion-evidence/v1"\n'
)
_SCHEMA_BEFORE = (
    '                        "axiom-encode/reviewed-candidate-promotion-evidence/v1"\n'
)
_V6_BLOCK = re.compile(
    r"^    if is_v6_labelled\(payload\):\n(?:^        .*\n|^\n)*?^        continue\n",
    re.MULTILINE,
)
_NESTED_BLOCK = re.compile(
    r"^    if effective is not payload and is_v6_labelled\(effective\):\n"
    r"(?:^        .*\n)*?^        \)\n",
    re.MULTILINE,
)


def _step_before_this_change(script: str) -> str:
    assert script.count(_IMPORT) == 1
    script = script.replace(_IMPORT, "")
    for block in (_NESTED_BLOCK, _V6_BLOCK):
        found = block.findall(script)
        assert len(found) == 1
        script = script.replace(found[0], "")
    for now, before in ((_REF_NOW, _REF_BEFORE), (_SCHEMA_NOW, _SCHEMA_BEFORE)):
        assert script.count(now) == 1
        script = script.replace(now, before)
    for name in (
        "is_v6_labelled",
        "CANDIDATE_REF_FIELD",
        "candidate_promotion_evidence",
        "CANDIDATE_PROMOTION_EVIDENCE_SCHEMA",
        "applied_manifest_v6",
        "COUNTRY",
        "v6_evidence_paths",
    ):
        assert name not in script, name
    return script


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, env=_clean_env()
    ).strip()


def _clean_env() -> dict[str, str]:
    env = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "TMPDIR", "LANG", "SYSTEMROOT")
        if name in os.environ
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    return env


def make_run(tmp_path: Path) -> tuple[Path, str]:
    """A RuleSpec checkout at an exact candidate head, and the runner files."""

    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README.md").write_text("fixture\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "fixture")
    (tmp_path / "source-bundle.json").write_text("[]\n")
    (tmp_path / "canonical-refresh-bundle.json").write_text("[]\n")
    return repo, _git(repo, "rev-parse", "HEAD")


def write_manifest(repo: Path, payload, relative: str = MANIFEST) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text + "\n")
    path.chmod(0o644)
    return path


_PRELUDE = """
import json, sys
from axiom_encode import applied_manifest_v6, reviewed_pr_admission
locks = json.loads({locks!r})
reviewed_pr_admission.V6_WRITER_ENABLED = locks["writer"]
applied_manifest_v6.V6_READER_ENABLED = locks["reader"]
reviewed_pr_admission.GUARD_APP_ID = locks["app_id"]
def fetch(path):
    with open({reads!r}, "a") as stream:
        stream.write(path + "\\n")
    if locks["branch"] is None:
        raise OSError("no network in tests")
    if path != {branch_path!r}:
        raise AssertionError(path)
    return json.loads(json.dumps(locks["branch"]))
reviewed_pr_admission.github_json = fetch
with open({script!r}) as stream:
    code = compile(stream.read(), "<stdin>", "exec")
exec(code, {{"__name__": "__main__"}})
"""


def run_step(
    tmp_path: Path,
    ref: str,
    *,
    script: str | None = None,
    locks: dict | None = None,
    artifact: str = "artifact",
    **env_overrides: str,
) -> dict:
    """Run the step's Python as the workflow does, and capture what it did.

    Without ``locks`` the script runs on stdin with the committed constants,
    as in the workflow. With ``locks`` a test prelude patches them first.
    """

    script = _package_script() if script is None else script
    out = tmp_path / artifact
    out.mkdir()
    env = {
        **_clean_env(),
        "PYTHONPATH": str(ROOT / "src"),
        "RUNNER_TEMP": str(tmp_path),
        "RULESPEC_CHECKOUT": "rulespec-us",
        "RULESPEC_REF": ref,
        "CITATION": CITATION,
        "COUNTRY": "us",
        "REPLACE_RULESPEC_PATH": PRIMARY,
        "ATOMIC_SOURCE_JSON": LIVE_PROMOTION,
        "REVIEW_FINDING": "",
        "REVIEW_FINDING_PRESENT": "false",
        **env_overrides,
    }
    env = {name: value for name, value in env.items() if value is not None}
    # The workflow event and commit a real runner provides (current main reads
    # them in this step before the manifest loop).
    event = tmp_path / f"{artifact}-event.json"
    event.write_text(
        json.dumps(
            {
                "inputs": {
                    "citation": env.get("CITATION", ""),
                    "review_finding": env.get("REVIEW_FINDING", ""),
                }
            }
        )
    )
    env.setdefault("GITHUB_EVENT_PATH", str(event))
    env.setdefault("GITHUB_SHA", "a" * 40)
    arguments = [str(out / "context-manifest.json"), str(out / "apply-manifests.json")]
    reads = tmp_path / f"{artifact}-reads.txt"
    if locks is None:
        command, stdin = [sys.executable, "-", *arguments], script
    else:
        script_path = tmp_path / f"{artifact}-step.py"
        script_path.write_text(script)
        prelude = _PRELUDE.format(
            locks=json.dumps(locks),
            reads=str(reads),
            branch_path=BRANCH_PATH,
            script=str(script_path),
        )
        command, stdin = [sys.executable, "-c", prelude, *arguments], None
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        # Line numbers of a traceback name the script's own lines; this change
        # adds lines above them.
        "stderr": re.sub(r"line \d+", "line N", completed.stderr),
        "files": {path.name: path.read_bytes() for path in sorted(out.iterdir())},
        "reads": reads.read_text().splitlines() if reads.exists() else [],
    }


OPEN = {"writer": True, "reader": True, "app_id": APP, "branch": branch_record()}


def _evidence_file(schema: str) -> bytes:
    return (
        json.dumps(
            {"citation": CITATION, "rulespec_path": PRIMARY, "schema": schema},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _inventory(manifest: Path) -> bytes:
    return (
        json.dumps(
            {
                "schema": "axiom-encode/applied-manifest-inventory/v1",
                "items": [
                    {
                        "citation": CITATION,
                        "path": MANIFEST,
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _legacy_wrapper(nested: dict, *, label=None) -> dict:
    """A legacy-replacement manifest that nests ``nested``."""

    return {
        "schema_version": label or cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply --replace-legacy-rulespec-path",
        "replacement_manifest": nested,
        "replacement": {"receipt_path": "x", "receipt_sha256": "0" * 64},
    }


def _v5_run(name: str, repo: Path, ref: str) -> dict[str, str]:
    """Write one v5 case into the checkout; return its environment overrides."""

    payload = v5_promotion(ref)
    if name == "promotion":
        write_manifest(repo, sign(payload))
    elif name == "promotion, another ref":
        payload["reviewed_rulespec_ref"] = "c" * 40
        write_manifest(repo, sign(payload))
    elif name == "promotion, model tool":
        payload["tool"] = "axiom-encode encode --apply"
        write_manifest(repo, sign(payload))
    elif name == "promotion, no ref":
        payload.pop("reviewed_rulespec_ref")
        write_manifest(repo, sign(payload))
    elif name == "promotion, v5 label carrying the v6 fields":
        v6_shape = written_v6(ref)
        v6_shape["schema_version"] = cli.APPLIED_ENCODING_MANIFEST_SCHEMA
        write_manifest(repo, v6_shape)
    elif name == "promotion, v5 label carrying both refs":
        v6_shape = written_v6(ref)
        v6_shape["schema_version"] = cli.APPLIED_ENCODING_MANIFEST_SCHEMA
        v6_shape["reviewed_rulespec_ref"] = ref
        write_manifest(repo, v6_shape)
    elif name == "promotion, label spelled V6":
        v6_shape = written_v6(ref)
        v6_shape["schema_version"] = "axiom-encode/applied-rulespec/V6"
        write_manifest(repo, v6_shape)
    elif name == "promotion, another citation":
        payload["citation"] = "us/statute/7/2015/g"
        write_manifest(repo, sign(payload))
    elif name == "promotion, two manifests":
        write_manifest(repo, sign(copy.deepcopy(payload)))
        write_manifest(repo, sign(payload), MANIFEST.replace("f.yaml", "g.yaml"))
    elif name == "promotion, no manifest":
        pass
    elif name == "promotion, manifest not an object":
        write_manifest(repo, "[]")
    elif name == "promotion, manifest not json":
        write_manifest(repo, "{")
    elif name == "promotion, legacy replacement nesting a v5 manifest":
        write_manifest(repo, _legacy_wrapper(sign(payload)))
    elif name == "promotion, static lane":
        write_manifest(repo, sign(payload))
        return {"ATOMIC_SOURCE_JSON": json.dumps(TRANSACTION)}
    elif name == "manifest-only refresh":
        write_manifest(repo, sign(payload))
        return {
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    **TRANSACTION,
                    "manifest_only_refresh": True,
                    "reviewed_candidate_promotion": False,
                }
            )
        }
    else:  # pragma: no cover
        raise AssertionError(name)
    return {}


V5_RUNS = (
    "promotion",
    "promotion, another ref",
    "promotion, model tool",
    "promotion, no ref",
    "promotion, v5 label carrying the v6 fields",
    "promotion, v5 label carrying both refs",
    "promotion, label spelled V6",
    "promotion, another citation",
    "promotion, two manifests",
    "promotion, no manifest",
    "promotion, manifest not an object",
    "promotion, manifest not json",
    "promotion, legacy replacement nesting a v5 manifest",
    "promotion, static lane",
    "manifest-only refresh",
)


@pytest.mark.parametrize("name", V5_RUNS)
def test_off_every_v5_run_is_the_same_as_before_this_change(tmp_path, name):
    """Differential: the step against itself with this change reverted.

    The step's own outcome is checked first, so a change of behaviour fails
    here even when the revert below no longer applies.
    """

    repo, ref = make_run(tmp_path)
    overrides = _v5_run(name, repo, ref)
    script = _package_script()
    after = run_step(tmp_path, ref, script=script, artifact="after", **overrides)
    accepted = {
        "promotion": REVIEWED_EVIDENCE,
        "promotion, v5 label carrying both refs": REVIEWED_EVIDENCE,
        "promotion, static lane": REVIEWED_EVIDENCE,
        "manifest-only refresh": "axiom-encode/manifest-only-refresh-evidence/v1",
    }
    if name in accepted:
        assert after["returncode"] == 0, after["stderr"]
        assert after["files"]["context-manifest.json"] == _evidence_file(accepted[name])
    else:
        assert after["returncode"] != 0
        assert after["files"] == {}
    before = run_step(
        tmp_path,
        ref,
        script=_step_before_this_change(script),
        artifact="before",
        **overrides,
    )
    assert after == before


def test_the_reverted_step_is_the_step_without_its_v6_names():
    script = _package_script()
    before = _step_before_this_change(script)
    # The import and the flag (6 lines), the nested refusal (8), the v6 block
    # (30), the ref (+4), the schema (+2).
    assert len(script.splitlines()) - len(before.splitlines()) == 6 + 8 + 30 + 4 + 2
    assert 'or evidence_manifest.get("reviewed_rulespec_ref")' in before


def test_off_the_hosted_step_refuses_the_written_v6_manifest(tmp_path):
    repo, ref = make_run(tmp_path)
    write_manifest(repo, written_v6(ref))
    result = run_step(tmp_path, ref)
    assert result["returncode"] != 0
    assert (
        f"v6 manifest {MANIFEST} is not promotion evidence: "
        "the applied-rulespec/v6 writer is switched off"
    ) in result["stderr"]
    assert result["files"] == {} and result["reads"] == []


@pytest.mark.parametrize(
    ("locks", "error"),
    [
        ({**OPEN, "writer": False}, "writer is switched off"),
        ({**OPEN, "reader": False}, "reader is switched off"),
        ({**OPEN, "app_id": None}, "GUARD_APP_ID"),
        ({**OPEN, "branch": RECORDED_BRANCH}, "not a required check of main"),
        ({**OPEN, "branch": None}, "cannot be read"),
    ],
)
def test_each_shut_lock_alone_keeps_the_hosted_step_refusing(tmp_path, locks, error):
    repo, ref = make_run(tmp_path)
    write_manifest(repo, written_v6(ref))
    result = run_step(tmp_path, ref, locks=locks)
    assert result["returncode"] != 0
    assert error in result["stderr"]
    assert result["files"] == {}


def test_on_the_hosted_step_accepts_the_written_v6_manifest(tmp_path):
    repo, ref = make_run(tmp_path)
    manifest = write_manifest(repo, written_v6(ref))
    result = run_step(tmp_path, ref, locks=OPEN)
    assert result["returncode"] == 0, result["stderr"]
    assert result["files"] == {
        "apply-manifests.json": _inventory(manifest),
        "context-manifest.json": _evidence_file(CANDIDATE_EVIDENCE),
    }
    # One read of lock 3, for this run's repository.
    assert result["reads"] == [BRANCH_PATH]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("drop a field", "does not match the exact unreviewed candidate"),
        ("another citation", "is not for the requested citation"),
        ("extra field", "does not match the exact unreviewed candidate"),
        ("scope owner review", "attestation scope is invalid: owner_review"),
        ("another pull request", "another pull request than this run"),
        ("another ref", "is not the requested RuleSpec ref"),
        ("not a live promotion", "live candidate pull request promotion"),
        ("no country", "RuleSpec repository of this run is invalid"),
        ("a second v6 manifest", "found 2"),
        ("a v5 manifest beside it", "found 2"),
    ],
)
def test_on_the_hosted_step_refuses(tmp_path, change, error):
    repo, ref = make_run(tmp_path)
    payload = written_v6(ref)
    overrides: dict[str, str | None] = {}
    if change == "drop a field":
        payload.pop("run_id")
    elif change == "another citation":
        payload.pop("signature")
        payload["citation"] = "us/statute/7/2015/g"
        payload = sign(payload)
    elif change == "extra field":
        payload["reviewed_rulespec_ref"] = ref
    elif change == "scope owner review":
        payload[v6.ATTESTATION_SCOPE_FIELD]["owner_review"] = "before-signing"
    elif change == "another pull request":
        payload = written_v6(ref, pull_request=PR_NUMBER + 1)
    elif change == "another ref":
        payload = written_v6("c" * 40)
    elif change == "not a live promotion":
        overrides["ATOMIC_SOURCE_JSON"] = json.dumps(TRANSACTION)
    elif change == "no country":
        overrides["COUNTRY"] = None
    elif change == "a second v6 manifest":
        write_manifest(repo, written_v6(ref), MANIFEST.replace("f.yaml", "g.yaml"))
    elif change == "a v5 manifest beside it":
        write_manifest(
            repo, sign(v5_promotion(ref)), MANIFEST.replace("f.yaml", "g.yaml")
        )
    write_manifest(repo, payload)
    result = run_step(tmp_path, ref, locks=OPEN, **overrides)
    assert result["returncode"] != 0
    assert error in result["stderr"], result["stderr"]
    assert "context-manifest.json" not in result["files"]


NESTED = "is not promotion evidence: only a top-level manifest may carry the v6 label"


@pytest.mark.parametrize("locks", [None, OPEN], ids=["locks shut", "locks open"])
@pytest.mark.parametrize(
    "outer", [None, "axiom-encode/applied-rulespec/v6"], ids=["v5 outer", "v6 outer"]
)
def test_a_v6_label_nested_in_a_legacy_replacement_is_refused(tmp_path, locks, outer):
    """The decision is made once, on the top-level manifest; a nested v6 label
    is refused by name, before any lock is read, as the reader's nested
    readers refuse it."""

    repo, ref = make_run(tmp_path)
    write_manifest(repo, _legacy_wrapper(written_v6(ref), label=outer))
    result = run_step(tmp_path, ref, locks=locks)
    assert result["returncode"] != 0
    assert f"v6 manifest nested in {MANIFEST} {NESTED}" in result["stderr"]
    assert result["files"] == {} and result["reads"] == []


@pytest.mark.parametrize("locks", [None, OPEN], ids=["locks shut", "locks open"])
def test_a_v6_legacy_replacement_nesting_v5_is_not_evidence(tmp_path, locks):
    repo, ref = make_run(tmp_path)
    write_manifest(
        repo,
        _legacy_wrapper(
            sign(v5_promotion(ref)), label="axiom-encode/applied-rulespec/v6"
        ),
    )
    result = run_step(tmp_path, ref, locks=locks)
    assert result["returncode"] != 0
    assert f"v6 manifest {MANIFEST} is not promotion evidence" in result["stderr"]
    assert result["files"] == {}


def test_the_step_decides_v6_once_on_the_top_level_manifest():
    script = _package_script()
    # The ref field and the evidence record follow the gate's decision, not a
    # second label test on another object.
    assert script.count("is_v6_labelled(") == 2
    assert "is_v6_labelled(evidence_manifest)" not in script
    assert script.count("relative_path in v6_evidence_paths") == 2
    assert script.count("v6_evidence_paths.add(relative_path)") == 1


def test_the_step_hands_the_reader_its_inputs_and_no_seam():
    script = _package_script()
    calls = [
        node
        for node in ast.walk(ast.parse(script))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_candidate_promotion_evidence"
    ]
    assert len(calls) == 1
    (call,) = calls
    assert [ast.unparse(argument) for argument in call.args] == ["payload"]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords} == {
        "atomic_source_json": "os.environ.get('ATOMIC_SOURCE_JSON') or '[]'",
        "country": "os.environ.get('COUNTRY', '')",
        "citation": "os.environ.get('CITATION', '')",
        "rulespec_ref": "os.environ.get('RULESPEC_REF', '')",
    }
    assert _package_step()["env"]["COUNTRY"] == "${{ inputs.country }}"
    # The v6 names come from the reader; the step spells none of them.
    for literal in (
        '"candidate_rulespec_ref"',
        '"attestation_scope"',
        "applied-rulespec/v6",
        '"axiom-encode/candidate-promotion-evidence/v1"',
    ):
        assert literal not in script


def test_only_the_evidence_step_of_the_workflow_knows_v6():
    """The rest of the workflow is still a v5-only reader (design, section 5)."""

    text = WORKFLOW.read_text()
    assert "applied-rulespec/v6" not in text
    for step in _steps():
        run = step.get("run", "")
        knows_v6 = any(
            name in run
            for name in (
                "applied_manifest_v6",
                "candidate_promotion_evidence",
                "is_v6_labelled",
                "CANDIDATE_REF_FIELD",
            )
        )
        assert knows_v6 is (step.get("name") == STEP), step.get("name")


def test_after_the_evidence_step_nothing_reads_the_manifest_label_or_ref():
    """So a v6 manifest that the step admits reaches publication."""

    steps = _steps()
    names = [step.get("name") for step in steps]
    start = names.index(STEP) + 1
    end = names.index("Push lane branch and open draft pull request") + 1
    for step in steps[start:end]:
        run = step.get("run", "")
        for name in ("applied-rulespec/v5", "reviewed_rulespec_ref", "schema_version"):
            assert name not in run, (step.get("name"), name)


def _code_without_docstrings(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)
    return ast.unparse(tree)
