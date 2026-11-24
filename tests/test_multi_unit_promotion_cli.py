"""Several new units in one candidate: one signing run, one manifest per unit.

The admission half is in ``test_multi_unit_promotion_admission.py``. Here the
promotion command signs every unit of a multi-unit admission or none, and the
dispatch preflight builds the batch selector from the operator's list.
"""

import contextlib
import copy
import datetime as datetime_module
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import axiom_encode.cli as cli
from axiom_encode import prepare_signed_backfill as helper
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.cli import (
    _APPLY_VALIDATION_SNAPSHOT_ATTR,
    _TRUSTED_RUNTIME_ENV,
    APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL,
    _applied_encoding_manifest_path,
    _applied_encoding_manifest_signature_issue,
    cmd_promote_reviewed_candidate,
)
from axiom_encode.reviewed_pr_admission import AdmissionError
from tests.test_cli import (
    AXIOM_ENCODE_TEST_VERSION,
    TEST_APPLY_SIGNING_BROKER,
    _git,
    _sha256_file,
)

NAMES = ("a", "b", "c")
FIXED_NOW = datetime_module.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime_module.UTC)
GIT_DATE = {
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}


def citation(name: str) -> str:
    return f"us/statute/7/2015/{name}"


def primary(name: str) -> str:
    return f"us/statutes/7/2015/{name}.yaml"


def manifest_of(repo: Path, name: str) -> Path:
    return repo / _applied_encoding_manifest_path(Path(primary(name)))


def rule_text(name: str, imports: tuple[str, ...] = ()) -> str:
    return (
        "format: rulespec/v1\n"
        + (
            "imports:\n" + "".join(f"  - {item}\n" for item in imports)
            if imports
            else ""
        )
        + "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {citation(name)}\n"
        f"    source_sha256: {'a' * 64}\n"
        "rules: []\n"
    )


def candidate(tmp_path: Path, imports: dict | None = None) -> tuple[Path, str]:
    """A candidate checkout that adds every unit; the same commit every time."""
    repo = tmp_path / "rulespec-us"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    for name in NAMES:
        rule = repo / primary(name)
        rule.parent.mkdir(parents=True, exist_ok=True)
        rule.write_text(rule_text(name, (imports or {}).get(name, ())))
        rule.with_name(f"{name}.test.yaml").write_text(f"[]  # {name}\n")
    _git(repo, "add", ".")
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "candidate"],
        check=True,
        capture_output=True,
        env={**os.environ, **GIT_DATE},
    )
    return repo, _git(repo, "rev-parse", "HEAD").stdout.strip()


def record(repo: Path, head: str, name: str) -> dict:
    rule = repo / primary(name)
    return {
        "schema": "axiom-encode/reviewed-rulespec-admission/v1",
        "country": "us",
        "rulespec_ref": head,
        "rulespec_pr": 42,
        "pr_base_branch": "codex/candidate",
        "base_sha": "b" * 40,
        "binding": {
            "citation": citation(name),
            "rulespec_path": primary(name),
            "rulespec_sha256": _sha256_file(rule),
            "companion_sha256": _sha256_file(rule.with_name(f"{name}.test.yaml")),
        },
        "declared": None,
        "purpose": "Test admission.",
    }


SHARED = {
    "country": "us",
    "pr": 42,
    "branch": "codex/candidate",
    "actor": "PavelMakarchuk",
    "run_id": "12345",
}


def batch_admission(repo: Path, head: str, names=NAMES) -> dict:
    return {
        "workflow_commit": "e" * 40,
        "selector": {
            **SHARED,
            "head": head,
            "units": [
                {"citation": citation(name), "rulespec_path": primary(name)}
                for name in names
            ],
        },
        "records": [record(repo, head, name) for name in names],
    }


def single_admission(repo: Path, head: str, name: str) -> dict:
    return {
        "workflow_commit": "e" * 40,
        "selector": {
            **SHARED,
            "head": head,
            "citation": citation(name),
            "rulespec_path": primary(name),
        },
        "record": record(repo, head, name),
    }


class FixedDatetime(datetime_module.datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


class Run:
    """One promotion run with everything outside the command stubbed."""

    def __init__(self, tmp_path, repo, head, live, *, lead="a", fixed_time=False):
        self.repo = repo
        self.tmp_path = tmp_path
        self.live = live
        self.fixed_time = fixed_time
        self.validated: list[str] = []
        self.signed: list[str] = []
        self.revalidations = 0
        self.fail_validation: str | None = None
        self.fail_signature: str | None = None
        self.fail_revalidation_at: int | None = None
        self.fail_verification: str | None = None
        self.installs: list[list[str]] = []
        for directory in ("axiom-rules-engine", "axiom-corpus"):
            (tmp_path / directory).mkdir(exist_ok=True)
        self.args = SimpleNamespace(
            repo=repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            corpus_path=tmp_path / "axiom-corpus",
            rulespec_dependency_root=[],
            rulespec_path=primary(lead),
            citation=citation(lead),
            reviewed_rulespec_ref=head,
            run_id="12345",
        )

    def _validate(self, result, **_kwargs):
        name = result.citation.rsplit("/", 1)[1]
        self.validated.append(name)
        if name == self.fail_validation:
            return False, [f"companion test of {name} failed"], {}
        setattr(
            result,
            _APPLY_VALIDATION_SNAPSHOT_ATTR,
            {
                "manifest_validation_execution": {
                    "schema": "axiom-encode/apply-validation-execution/v2",
                    "unit": name,
                }
            },
        )
        return True, [], {}

    def _sign(self, payload, broker):
        name = payload["citation"].rsplit("/", 1)[1]
        if name == self.fail_signature:
            raise RuntimeError("signing broker refused the request")
        self.signed.append(name)
        return self._real_sign(payload, broker)

    def _revalidate(self, value):
        self.revalidations += 1
        if self.revalidations == self.fail_revalidation_at:
            raise AdmissionError("candidate has unresolved changes requested")
        return value

    def _install(self, files, **kwargs):
        self.installs.append(sorted(Path(target).name for target, _raw in files))
        return self._real_install(files, **kwargs)

    def _verify(self, repo, relative, **_kwargs):
        name = Path(relative).stem
        issues = ["signature mismatch"] if name == self.fail_verification else []
        return ({} if not issues else None), "", "d" * 64, issues

    def __call__(self, monkeypatch):
        runtime = self.tmp_path / "runtime-attestation.json"
        runtime.write_text(json.dumps({"reviewed_pr_admission": self.live}))
        records = self.tmp_path / "reviewed-rulespec-refs"
        records.mkdir(exist_ok=True)
        monkeypatch.setenv(_TRUSTED_RUNTIME_ENV, "1")
        self._real_sign = cli._sign_applied_encoding_manifest
        self._real_install = cli._install_apply_transaction
        provenance = {
            "root": "/opt/axiom-verification",
            "commit": "b" * 40,
            "dirty_tracked": False,
            "version": AXIOM_ENCODE_TEST_VERSION,
            "version_commit": "b" * 40,
            "identity_source": "trusted-runtime-attestation",
        }
        stubs = {
            "_load_trusted_runtime_attestation": {
                "return_value": {"path": str(runtime), "commit": "e" * 40}
            },
            "_promotion_reviewed_rulespec_record_root": {"return_value": records},
            "_recover_apply_transaction": {},
            "load_rulespec_local_corpus_release": {"return_value": object()},
            "_manifest_primary_source_verifications": {
                "return_value": ([], "rulespec-us/us")
            },
            "resolve_corpus_source_unit": {
                "side_effect": lambda cited, _release: SimpleNamespace(
                    body=f"source text of {cited}\n",
                    amendment_documents=(),
                    source_attestation={"requested_corpus_citation_path": cited},
                )
            },
            "_run_generated_encoding_overlay_validation": {
                "side_effect": self._validate
            },
            "_require_clean_axiom_encode_git_provenance": {"return_value": provenance},
            "verify_rulespec_validation_waiver_set": {"return_value": "c" * 64},
            "_target_uses_per_file_waiver_bindings": {"return_value": False},
            "_current_guard_encoder_execution_identity": {
                "return_value": {
                    "repository": APPLIED_ENCODING_OFFICIAL_REPOSITORY,
                    "commit": "b" * 40,
                    "version": AXIOM_ENCODE_TEST_VERSION,
                    "identity_source": "trusted-runtime-attestation",
                }
            },
            "_load_verified_applied_encoding_manifest_payload": {
                "side_effect": self._verify
            },
            "_sign_applied_encoding_manifest": {"side_effect": self._sign},
            "_install_apply_transaction": {"side_effect": self._install},
        }
        with contextlib.ExitStack() as stack:
            for name, kwargs in stubs.items():
                stack.enter_context(patch(f"axiom_encode.cli.{name}", **kwargs))
            stack.enter_context(
                patch(
                    "axiom_encode.reviewed_pr_admission.revalidate_admission",
                    side_effect=self._revalidate,
                )
            )
            self.signer = stack.enter_context(
                patch("axiom_encode.cli._isolated_apply_manifest_signer")
            )
            self.signer.return_value.__enter__.return_value = TEST_APPLY_SIGNING_BROKER
            if self.fixed_time:
                stack.enter_context(patch("axiom_encode.cli.datetime", FixedDatetime))
            cmd_promote_reviewed_candidate(self.args)

    def untracked(self) -> list[str]:
        return _git(
            self.repo, "ls-files", "--others", "--exclude-standard"
        ).stdout.splitlines()

    def assert_nothing_written(self):
        assert self.untracked() == []
        assert _git(self.repo, "status", "--porcelain").stdout == ""
        for name in NAMES:
            assert not manifest_of(self.repo, name).exists()


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)


@pytest.fixture
def run(tmp_path, on):
    repo, head = candidate(tmp_path)
    return Run(tmp_path, repo, head, batch_admission(repo, head))


# --- One run signs every unit -------------------------------------------------


def test_one_run_writes_one_manifest_per_unit(run, monkeypatch, capsys):
    run(monkeypatch)
    assert run.validated == list(NAMES)
    assert run.signed == list(NAMES)
    # Before validation and before signing, once each for the whole batch.
    assert run.revalidations == 2
    assert run.signer.call_count == 1
    assert run.untracked() == [
        f".axiom/encoding-manifests/us/statutes/7/2015/{name}.json" for name in NAMES
    ]
    # One transaction holds every manifest.
    assert run.installs == [["a.json", "b.json", "c.json"]]
    for name in NAMES:
        payload = json.loads(manifest_of(run.repo, name).read_text())
        assert payload["tool"] == APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL
        assert payload["citation"] == citation(name)
        assert payload["reviewed_rulespec_ref"] == run.args.reviewed_rulespec_ref
        assert payload["run_id"] == "12345"
        assert payload["validation_execution"]["unit"] == name
        assert [item["path"] for item in payload["applied_files"]] == [
            primary(name),
            primary(name).removesuffix(".yaml") + ".test.yaml",
        ]
        assert payload["applied_files"][0]["sha256"] == _sha256_file(
            run.repo / primary(name)
        )
        assert (
            _applied_encoding_manifest_signature_issue(
                payload, TEST_APPLY_SIGNING_BROKER
            )
            is None
        )
    assert capsys.readouterr().out.splitlines() == [
        "promoted reviewed candidate "
        f".axiom/encoding-manifests/us/statutes/7/2015/{name}.json"
        for name in NAMES
    ]


@pytest.mark.parametrize("lead", NAMES)
def test_the_arguments_may_name_any_unit_of_the_batch(tmp_path, on, monkeypatch, lead):
    repo, head = candidate(tmp_path)
    run = Run(tmp_path, repo, head, batch_admission(repo, head), lead=lead)
    run(monkeypatch)
    assert run.signed == list(NAMES)


def test_each_manifest_equals_a_single_unit_promotion_of_that_unit(
    tmp_path, on, monkeypatch
):
    """Byte for byte, signature included, with the clock held."""
    repo, head = candidate(tmp_path / "batch")
    batch = Run(
        tmp_path / "batch", repo, head, batch_admission(repo, head), fixed_time=True
    )
    batch(monkeypatch)
    for name in NAMES:
        alone_root = tmp_path / f"alone-{name}"
        alone_repo, alone_head = candidate(alone_root)
        assert alone_head == head
        alone = Run(
            alone_root,
            alone_repo,
            alone_head,
            single_admission(alone_repo, alone_head, name),
            lead=name,
            fixed_time=True,
        )
        alone(monkeypatch)
        assert alone.untracked() == [
            f".axiom/encoding-manifests/us/statutes/7/2015/{name}.json"
        ]
        assert (
            manifest_of(repo, name).read_bytes()
            == manifest_of(alone_repo, name).read_bytes()
        )


def test_a_batch_of_one_equals_the_single_unit_promotion(tmp_path, on, monkeypatch):
    repo, head = candidate(tmp_path / "batch")
    for name in ("b", "c"):
        _git(repo, "rm", "-q", primary(name), primary(name)[:-5] + ".test.yaml")
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "one unit"],
        check=True,
        capture_output=True,
        env={**os.environ, **GIT_DATE},
    )
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    as_batch = Run(
        tmp_path / "batch",
        repo,
        head,
        batch_admission(repo, head, ("a",)),
        fixed_time=True,
    )
    as_batch(monkeypatch)
    batch_bytes = manifest_of(repo, "a").read_bytes()
    manifest_of(repo, "a").unlink()
    alone = Run(
        tmp_path / "batch",
        repo,
        head,
        single_admission(repo, head, "a"),
        fixed_time=True,
    )
    alone(monkeypatch)
    assert manifest_of(repo, "a").read_bytes() == batch_bytes
    assert as_batch.installs == alone.installs == [["a.json"]]


# --- All or nothing -----------------------------------------------------------


@pytest.mark.parametrize("failing", NAMES)
def test_one_unit_that_fails_validation_leaves_nothing_signed(
    run, monkeypatch, failing
):
    run.fail_validation = failing
    with pytest.raises(RuntimeError) as refusal:
        run(monkeypatch)
    message = str(refusal.value)
    assert message.startswith(f"unit {citation(failing)} ({primary(failing)}): ")
    assert f"companion test of {failing} failed" in message
    # No signature is requested for any unit, also not for one validated before.
    assert run.signed == []
    assert run.installs == []
    assert run.validated == list(NAMES[: NAMES.index(failing) + 1])
    run.assert_nothing_written()


@pytest.mark.parametrize("failing", NAMES)
def test_one_signature_that_fails_leaves_no_manifest(run, monkeypatch, failing):
    run.fail_signature = failing
    with pytest.raises(RuntimeError) as refusal:
        run(monkeypatch)
    assert str(refusal.value) == (
        f"unit {citation(failing)} ({primary(failing)}): "
        "signing broker refused the request"
    )
    # Every unit was validated first; no manifest of any unit is written.
    assert run.validated == list(NAMES)
    assert run.installs == []
    run.assert_nothing_written()


@pytest.mark.parametrize("call", [1, 2])
def test_an_admission_that_is_withdrawn_leaves_nothing_signed(run, monkeypatch, call):
    """Before validation (1) or after it and before the first signature (2)."""
    run.fail_revalidation_at = call
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        run(monkeypatch)
    assert run.signed == []
    assert run.validated == ([] if call == 1 else list(NAMES))
    run.assert_nothing_written()


@pytest.mark.parametrize("failing", NAMES)
def test_a_manifest_that_fails_verification_removes_every_manifest(
    run, monkeypatch, failing
):
    run.fail_verification = failing
    with pytest.raises(RuntimeError) as refusal:
        run(monkeypatch)
    assert f"unit {citation(failing)} ({primary(failing)}): " in str(refusal.value)
    assert run.installs == [["a.json", "b.json", "c.json"]]
    run.assert_nothing_written()


@pytest.mark.parametrize("changed", NAMES)
@pytest.mark.parametrize("member", ["primary", "companion"])
def test_a_unit_file_that_changes_after_validation_stops_the_install(
    run, monkeypatch, changed, member
):
    """Every unit's bytes are read again before the install, not the first's."""
    target = run.repo / primary(changed)
    if member == "companion":
        target = target.with_name(f"{changed}.test.yaml")
    real_sign = run._sign

    def sign_then_change(payload, broker):
        result = real_sign(payload, broker)
        if payload["citation"] == citation(NAMES[-1]):
            target.write_text("changed after validation\n")
        return result

    run._sign = sign_then_change
    verified = []
    real_verify = run._verify
    run._verify = lambda *args, **kwargs: (
        verified.append(args[1]),
        real_verify(*args, **kwargs),
    )[1]
    with pytest.raises((ValueError, RuntimeError)):
        run(monkeypatch)
    assert run.signed == list(NAMES)
    # Stopped by the check before the install: no manifest was ever written,
    # so none was verified.
    assert verified == []
    for name in NAMES:
        assert not manifest_of(run.repo, name).exists()
    assert run.untracked() == []


@pytest.mark.parametrize("field", ["rulespec_sha256", "companion_sha256"])
def test_a_unit_whose_bytes_differ_from_its_record_is_refused_and_named(
    run, monkeypatch, field
):
    run.live["records"][1]["binding"][field] = "0" * 64
    with pytest.raises(ValueError) as refusal:
        run(monkeypatch)
    assert str(refusal.value) == (
        f"unit {citation('b')} ({primary('b')}): "
        "reviewed candidate bytes differ from its admission record binding"
    )
    assert run.validated == [] and run.signed == []
    run.signer.assert_not_called()
    run.assert_nothing_written()


def test_a_unit_that_already_has_a_manifest_is_refused_and_named(
    tmp_path, on, monkeypatch
):
    repo, _head = candidate(tmp_path)
    existing = manifest_of(repo, "c")
    existing.parent.mkdir(parents=True)
    existing.write_text("{}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "already signed")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    run = Run(tmp_path, repo, head, batch_admission(repo, head))
    with pytest.raises(ValueError) as refusal:
        run(monkeypatch)
    assert str(refusal.value) == (
        f"unit {citation('c')} ({primary('c')}): "
        "reviewed candidate already has an apply manifest"
    )
    assert run.validated == [] and run.signed == []
    assert run.untracked() == []


def test_a_run_that_leaves_any_other_path_changed_fails(run, monkeypatch):
    """The checkout may differ from the candidate by the manifests only."""
    real_validate = run._validate

    def validate_and_litter(result, **kwargs):
        (run.repo / "us/statutes/7/2015/stray.txt").write_text("left behind\n")
        return real_validate(result, **kwargs)

    run._validate = validate_and_litter
    with pytest.raises(RuntimeError, match="changed paths other than its manifest"):
        run(monkeypatch)


# --- The admission must be this operation's -----------------------------------


def test_a_record_of_another_unit_is_refused_and_named(run, monkeypatch):
    run.live["records"][0], run.live["records"][1] = (
        run.live["records"][1],
        run.live["records"][0],
    )
    with pytest.raises(ValueError) as refusal:
        run(monkeypatch)
    assert str(refusal.value) == (
        f"unit {citation('a')} ({primary('a')}): reviewed candidate citation or "
        "path differs from its admission record"
    )
    assert run.validated == [] and run.signed == []
    run.assert_nothing_written()


@pytest.mark.parametrize("change", ["fewer", "more"])
def test_records_must_be_one_per_unit(run, monkeypatch, change):
    if change == "fewer":
        run.live["records"].pop()
    else:
        run.live["records"].append(copy.deepcopy(run.live["records"][0]))
    with pytest.raises(ValueError, match="does not match this operation"):
        run(monkeypatch)
    assert run.validated == [] and run.signed == []
    run.assert_nothing_written()


def test_a_unit_without_a_binding_is_refused(run, monkeypatch):
    run.live["records"][2]["binding"] = None
    with pytest.raises(ValueError, match="unit .*c.*has no promotion binding"):
        run(monkeypatch)
    assert run.validated == [] and run.signed == []


def test_a_batch_never_promotes_unbound_also_at_a_frozen_legacy_ref(run, monkeypatch):
    """The frozen pre-record refs may promote unbound only as a single unit."""
    run.live["records"][2]["binding"] = None
    monkeypatch.setattr(
        helper,
        "UNBOUND_PROMOTION_REVIEWED_RULESPEC_REFS",
        frozenset({("us", run.args.reviewed_rulespec_ref)}),
    )
    with pytest.raises(ValueError, match="unit .*c.*has no promotion binding"):
        run(monkeypatch)
    assert run.validated == [] and run.signed == []


def test_a_single_unit_error_does_not_name_a_unit(tmp_path, on, monkeypatch):
    """The unit prefix belongs to the batch form; single-unit text is unchanged."""
    repo, head = candidate(tmp_path)
    run = Run(tmp_path, repo, head, single_admission(repo, head, "b"), lead="b")
    run.fail_validation = "b"
    with pytest.raises(RuntimeError) as refusal:
        run(monkeypatch)
    assert str(refusal.value) == (
        "reviewed candidate failed current overlay validation: "
        "companion test of b failed"
    )
    assert run.signed == []
    run.live["record"]["binding"]["rulespec_sha256"] = "0" * 64
    with pytest.raises(ValueError) as refusal:
        run(monkeypatch)
    assert str(refusal.value) == (
        "reviewed candidate bytes differ from its admission record binding"
    )


def test_arguments_that_name_no_unit_of_the_batch_are_refused(
    tmp_path, on, monkeypatch
):
    repo, head = candidate(tmp_path)
    run = Run(tmp_path, repo, head, batch_admission(repo, head, ("a", "b")), lead="c")
    with pytest.raises(ValueError, match="does not match this operation"):
        run(monkeypatch)
    assert run.revalidations == 0 and run.validated == [] and run.signed == []
    run.assert_nothing_written()


@pytest.mark.parametrize(
    ("field", "value"),
    [("head", "f" * 40), ("run_id", "54321"), ("country", "nz")],
)
def test_a_batch_admission_of_another_run_or_head_is_refused(
    run, monkeypatch, field, value
):
    run.live["selector"][field] = value
    if field == "country":
        for unit in run.live["selector"]["units"]:
            unit["citation"] = "nz" + unit["citation"][2:]
            unit["rulespec_path"] = "nz" + unit["rulespec_path"][2:]
    with pytest.raises(ValueError, match="does not match this operation"):
        run(monkeypatch)
    assert run.revalidations == 0 and run.validated == [] and run.signed == []


def test_a_batch_admission_of_another_encoder_commit_is_refused(run, monkeypatch):
    run.live["workflow_commit"] = "f" * 40
    with pytest.raises(ValueError, match="does not match this operation"):
        run(monkeypatch)
    assert run.revalidations == 0 and run.signed == []


def test_switch_off_refuses_a_batch_admission_before_any_work(tmp_path, monkeypatch):
    """With the switch off the command does not know the batch form."""
    repo, head = candidate(tmp_path)
    run = Run(tmp_path, repo, head, batch_admission(repo, head))
    assert admission.MULTI_UNIT_PROMOTION_ENABLED is False
    with pytest.raises(AdmissionError, match="promotion selector has unexpected"):
        run(monkeypatch)
    assert run.revalidations == 0 and run.validated == [] and run.signed == []
    run.signer.assert_not_called()
    run.assert_nothing_written()


def test_a_batch_over_the_limit_is_refused_before_any_work(run, monkeypatch):
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS", 2)
    with pytest.raises(AdmissionError, match="names 3 units; the limit is 2"):
        run(monkeypatch)
    assert run.revalidations == 0 and run.validated == [] and run.signed == []


# --- Units of one batch do not import each other ------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "us:statutes/7/2015/a#x",
        "statutes/7/2015/a#x",
        "us:statutes/7/2015/a",
        # Spellings the validator's resolver follows to the same file.
        "US:statutes/7/2015/a",
        "Us:statutes/7/2015/a#x",
        "':statutes/7/2015/a'",
        "'x y:statutes/7/2015/a'",
        "\"'statutes/7/2015/a'\"",
        "'\"us:statutes/7/2015/a\"'",
        "' /statutes/7/2015/a/ '",
    ],
)
def test_a_unit_that_imports_another_unit_of_the_batch_is_refused(
    tmp_path, on, monkeypatch, target
):
    repo, head = candidate(tmp_path, imports={"b": (target,)})
    run = Run(tmp_path, repo, head, batch_admission(repo, head))
    with pytest.raises(ValueError) as refusal:
        run(monkeypatch)
    assert str(refusal.value) == (
        f"{primary('b')} imports {primary('a')}, another unit of the same batch; "
        "promote the imported unit first"
    )
    # Refused before the signer is prepared and before any validation.
    run.signer.assert_not_called()
    assert run.validated == [] and run.signed == []
    run.assert_nothing_written()


def test_a_unit_may_import_a_module_outside_the_batch(tmp_path, on, monkeypatch):
    repo, head = candidate(
        tmp_path, imports={"b": ("us:statutes/7/2014/a#x", "us:statutes/7/2015/ab#x")}
    )
    run = Run(tmp_path, repo, head, batch_admission(repo, head))
    run(monkeypatch)
    assert run.signed == list(NAMES)


def test_a_single_unit_promotion_is_not_checked_for_batch_imports(
    tmp_path, on, monkeypatch
):
    """The import rule belongs to the batch form only."""
    repo, head = candidate(tmp_path, imports={"b": ("us:statutes/7/2015/a#x",)})
    run = Run(tmp_path, repo, head, single_admission(repo, head, "b"), lead="b")
    run(monkeypatch)
    assert run.signed == ["b"]


def test_the_import_check_follows_the_validators_resolver(tmp_path, on):
    """Every spelling the resolver follows to a batch unit is refused."""
    from axiom_encode.harness import validator_pipeline

    repo, _head = candidate(tmp_path)
    a, b = Path(primary("a")), Path(primary("b"))
    for spelling in (
        "US:statutes/7/2015/a",
        ":statutes/7/2015/a",
        "x y:statutes/7/2015/a",
        "'statutes/7/2015/a'",
        '"us:statutes/7/2015/a"',
        "Us:statutes/7/2015/a#x",
        "statutes/7/2015/a",
        "us:statutes/7/2015/c#x",
        "us:statutes/7/2014/a#x",
        "statutes/7/2015/zz",
    ):
        raw = yaml.safe_dump(
            {"format": "rulespec/v1", "imports": [spelling], "rules": []}
        ).encode()
        (repo / b).write_bytes(raw)
        resolved = validator_pipeline._resolve_rulespec_import_file_static(
            spelling, rules_file=repo / b, policy_repo_path=repo / "us"
        )
        issue = cli._promotion_batch_import_issue(
            repo, {a: (repo / a).read_bytes(), b: raw}
        )
        in_batch = resolved is not None and resolved.resolve() in {
            (repo / a).resolve(),
            (repo / primary("c")).resolve(),
        }
        # ``c`` is not passed as a unit here, so only ``a`` counts.
        expected = resolved is not None and resolved.resolve() == (repo / a).resolve()
        assert (issue is not None) is expected, (spelling, resolved, issue, in_batch)
        # The reviewed spellings of a batch unit are refused outright.
        if spelling.replace("'", "").replace('"', "").split("#")[0].endswith("2015/a"):
            assert issue == (
                f"{primary('b')} imports {primary('a')}, another unit of the "
                "same batch; promote the imported unit first"
            ), spelling
        else:
            assert issue is None, spelling
    assert (
        cli._promotion_batch_import_issue(
            repo, {a: (repo / a).read_bytes(), b: (repo / a).read_bytes()}
        )
        is None
    )


def test_an_import_hidden_by_a_second_imports_key_is_refused(tmp_path, on):
    """The text is read as the validator reads it, not only the parsed YAML.

    A YAML parser keeps the last of two ``imports`` keys; the validator's text
    reading sees both blocks.
    """
    repo, _head = candidate(tmp_path)
    a, b = Path(primary("a")), Path(primary("b"))
    raw = (
        b"imports:\n  - us:statutes/7/2015/a#x\n"
        b"rules: []\n"
        b"imports:\n  - us:statutes/7/2014/other#x\n"
    )
    assert yaml.safe_load(raw)["imports"] == ["us:statutes/7/2014/other#x"]
    assert cli._promotion_batch_import_issue(
        repo, {a: (repo / a).read_bytes(), b: raw}
    ) == (
        f"{primary('b')} imports {primary('a')}, another unit of the same batch; "
        "promote the imported unit first"
    )


def test_a_unit_that_names_itself_in_its_imports_is_not_a_batch_import(tmp_path, on):
    """The rule is about another unit; the validator judges a self-import."""
    repo, _head = candidate(tmp_path)
    a, b = Path(primary("a")), Path(primary("b"))
    raw = yaml.safe_dump({"imports": ["us:statutes/7/2015/b#x"], "rules": []}).encode()
    assert (
        cli._resolve_rulespec_import_file_static(
            "us:statutes/7/2015/b#x", rules_file=repo / b, policy_repo_path=repo / "us"
        )
        is not None
    )
    assert (
        cli._promotion_batch_import_issue(repo, {a: (repo / a).read_bytes(), b: raw})
        is None
    )


def test_an_import_reached_through_an_alias_of_the_checkout_is_refused(tmp_path, on):
    """Through a symlinked checkout path the batch import is still refused."""
    repo, _head = candidate(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)
    a, b = Path(primary("a")), Path(primary("b"))
    raw = yaml.safe_dump({"imports": ["us:statutes/7/2015/a#x"], "rules": []}).encode()
    (repo / b).write_bytes(raw)
    units = {a: (repo / a).read_bytes(), b: raw}
    assert cli._promotion_batch_import_issue(repo, units) == (
        f"{primary('b')} imports {primary('a')}, another unit of the same "
        "batch; promote the imported unit first"
    )
    # The resolver refuses the aliased root as unsafe: refused, fail closed.
    assert cli._promotion_batch_import_issue(alias / "rulespec-us", units) in {
        f"{primary('b')} has an import that cannot be resolved safely",
        f"{primary('b')} imports {primary('a')}, another unit of the same "
        "batch; promote the imported unit first",
    }


def test_an_import_in_another_letter_case_is_refused_when_it_reaches_a_unit(
    tmp_path, on, monkeypatch
):
    """A resolver result that differs from a unit file in case only is the unit."""
    repo, _head = candidate(tmp_path)
    a, b = Path(primary("a")), Path(primary("b"))
    raw = yaml.safe_dump({"imports": ["us:statutes/7/2015/A#x"], "rules": []}).encode()
    monkeypatch.setattr(
        cli,
        "_resolve_rulespec_import_file_static",
        lambda item, **kwargs: repo / "us/statutes/7/2015/A.yaml",
    )
    assert cli._promotion_batch_import_issue(
        repo, {a: (repo / a).read_bytes(), b: raw}
    ) == (
        f"{primary('b')} imports {primary('a')}, another unit of the same batch; "
        "promote the imported unit first"
    )


def test_the_import_check_resolves_with_the_runs_dependency_roots(
    tmp_path, on, monkeypatch
):
    """The resolver gets the dependency roots of the run, as validation does."""
    repo, head = candidate(tmp_path, imports={"b": ("us:statutes/7/2014/x#y",)})
    roots = (tmp_path / "rulespec-ca",)
    seen = []
    real = cli._resolve_rulespec_import_file_static

    def spy(item, **kwargs):
        seen.append(kwargs["rulespec_dependency_roots"])
        return real(item, **{**kwargs, "rulespec_dependency_roots": ()})

    monkeypatch.setattr(cli, "_resolve_rulespec_import_file_static", spy)
    monkeypatch.setattr(cli, "_normalize_rulespec_dependency_roots", lambda _r: roots)
    monkeypatch.setattr(
        cli,
        "_authoritative_rulespec_dependency_scope",
        lambda _roots: contextlib.nullcontext(),
    )
    run = Run(tmp_path, repo, head, batch_admission(repo, head))
    run(monkeypatch)
    assert seen and all(value == roots for value in seen)


def test_an_import_of_a_companion_of_another_unit_is_refused(tmp_path, on):
    repo, _head = candidate(tmp_path)
    a, b = Path(primary("a")), Path(primary("b"))
    raw = yaml.safe_dump({"imports": ["us:statutes/7/2015/a.test#x"]}).encode()
    resolved = cli._resolve_rulespec_import_file_static(
        "us:statutes/7/2015/a.test#x", rules_file=repo / b, policy_repo_path=repo / "us"
    )
    issue = cli._promotion_batch_import_issue(
        repo, {a: (repo / a).read_bytes(), b: raw}
    )
    assert (issue is not None) is (resolved is not None)


def test_an_import_the_resolver_refuses_as_unsafe_stops_the_batch(tmp_path, on):
    repo, _head = candidate(tmp_path)
    a, b = Path(primary("a")), Path(primary("b"))
    raw = yaml.safe_dump({"imports": ["us:statutes/../../outside#x"]}).encode()
    assert (
        cli._promotion_batch_import_issue(repo, {a: (repo / a).read_bytes(), b: raw})
        == f"{primary('b')} has an import that cannot be resolved safely"
    )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            b"imports:\n  - {alias: 'statutes/7/2015/a#x'}\n",
            "has a malformed imports list",
        ),
        (b"imports:\n  - 7\n", "has a malformed imports list"),
        (b"imports:\n  - [statutes/7/2015/a]\n", "has a malformed imports list"),
        (b"- a\n- b\n", "cannot be read for its imports"),
        (b"just text\n", "cannot be read for its imports"),
        (b"", "cannot be read for its imports"),
        (b"a: [\n", "cannot be read for its imports"),
        (b"\xff\xfe", "cannot be read for its imports"),
    ],
)
def test_a_primary_that_cannot_be_read_for_imports_stops_the_batch(
    tmp_path, on, raw, message
):
    """Fail closed: no unit is skipped because its imports cannot be read."""
    repo, _head = candidate(tmp_path)
    a, b = Path(primary("a")), Path(primary("b"))
    assert (
        cli._promotion_batch_import_issue(repo, {a: (repo / a).read_bytes(), b: raw})
        == f"{primary('b')} {message}"
    )


@pytest.mark.parametrize(
    "text", ["imports: us:statutes/7/2015/a#x\n", "imports: {a: 1}\n"]
)
def test_a_unit_whose_imports_cannot_be_read_is_refused(
    tmp_path, on, monkeypatch, text
):
    repo, _head = candidate(tmp_path)
    (repo / primary("b")).write_text(text + rule_text("b"))
    _git(repo, "commit", "-am", "malformed imports")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    run = Run(tmp_path, repo, head, batch_admission(repo, head))
    with pytest.raises(ValueError, match="b.yaml has a malformed imports list"):
        run(monkeypatch)
    assert run.validated == [] and run.signed == []


# --- The dispatch wrapper and the preflight -----------------------------------


def wrapper(units=None) -> str:
    payload = {
        "schema": "axiom-encode/reviewed-pr-promotion/v1",
        "rulespec_pr": 42,
        "transaction": {
            "schema": "axiom-encode/atomic-source-transaction/v5",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
            "require_complete_source_unit": True,
            "manifest_only_refresh": False,
            "reviewed_candidate_promotion": True,
        },
    }
    if units is not None:
        payload["units"] = units
    return json.dumps(payload)


def test_the_wrapper_without_units_is_unchanged():
    normalized = helper.split_atomic_source_input(wrapper())
    assert "promotion_units" not in normalized
    assert normalized["rulespec_pr"] == 42


def test_switch_off_refuses_the_units_key_as_before():
    """With the switch off the wrapper is read exactly as on main."""
    assert admission.MULTI_UNIT_PROMOTION_ENABLED is False
    with pytest.raises(
        ValueError,
        match="^reviewed PR promotion requires exactly transaction and rulespec_pr$",
    ):
        helper.split_atomic_source_input(wrapper([citation("a")]))
    # A malformed list gets the same error: the key itself is unknown.
    with pytest.raises(
        ValueError, match="requires exactly transaction and rulespec_pr"
    ):
        helper.split_atomic_source_input(wrapper([]))


@pytest.mark.parametrize("value", [None, 0, 1, "True"])
def test_only_exactly_true_admits_the_units_key(monkeypatch, value):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", value)
    with pytest.raises(
        ValueError, match="requires exactly transaction and rulespec_pr"
    ):
        helper.split_atomic_source_input(wrapper([citation("a")]))


def test_the_units_key_stays_unknown_when_the_switch_cannot_be_read(monkeypatch):
    """Outside the package the wrapper cannot read the switch: the key is refused."""
    import axiom_encode

    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)
    assert helper.split_atomic_source_input(wrapper([citation("a")]))[
        "promotion_units"
    ] == [citation("a")]
    monkeypatch.delattr(axiom_encode, "reviewed_pr_admission")
    monkeypatch.setitem(sys.modules, "axiom_encode.reviewed_pr_admission", None)
    with pytest.raises(ImportError):
        from axiom_encode import reviewed_pr_admission  # noqa: F401
    with pytest.raises(
        ValueError,
        match="^reviewed PR promotion requires exactly transaction and rulespec_pr$",
    ):
        helper.split_atomic_source_input(wrapper([citation("a")]))
    # Without the key the wrapper still reads.
    assert helper.split_atomic_source_input(wrapper())["rulespec_pr"] == 42


def test_the_wrapper_carries_the_unit_list(on):
    units = [citation(name) for name in NAMES]
    normalized = helper.split_atomic_source_input(wrapper(units))
    assert normalized["promotion_units"] == units
    assert {
        key: value for key, value in normalized.items() if key != "promotion_units"
    } == (helper.split_atomic_source_input(wrapper()))
    # The signed transaction contract does not see the list.
    assert helper.immutable_atomic_source_contract(wrapper(units)) == (
        helper.immutable_atomic_source_contract(wrapper())
    )


@pytest.mark.parametrize(
    "units",
    [
        [],
        "us/statute/7/2015/a",
        [citation("a"), citation("a")],
        [citation("a"), 7],
        [""],
        ["x" * 513],
        [f"us/statute/7/2015/{index}" for index in range(65)],
        {"a": 1},
    ],
)
def test_a_malformed_unit_list_is_refused(on, units):
    with pytest.raises(ValueError, match="units must be a list of distinct citations"):
        helper.split_atomic_source_input(wrapper(units))


def test_the_wrapper_still_refuses_an_unknown_key(on):
    payload = json.loads(wrapper([citation("a")]))
    payload["other"] = 1
    with pytest.raises(
        ValueError, match="requires exactly transaction and rulespec_pr"
    ):
        helper.split_atomic_source_input(json.dumps(payload))


@pytest.fixture
def script(monkeypatch, tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "scripts/prepare_reviewed_pr_promotion.py"
    )
    spec = importlib.util.spec_from_file_location("batch_promotion_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "TheAxiomFoundation/axiom-encode")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("DISPATCH_ACTOR", "PavelMakarchuk")
    monkeypatch.setattr(
        "sys.argv",
        [
            str(path),
            "prepare",
            "--repo",
            str(tmp_path),
            "--selector",
            str(tmp_path / "selector.json"),
        ],
    )
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    module.verified = []
    monkeypatch.setattr(
        admission,
        "verify_reviewed_pr",
        lambda selector: module.verified.append(selector),
    )
    monkeypatch.setattr(
        module,
        "load",
        lambda name: admission if name == "reviewed_pr_admission" else helper,
    )
    return module


def dispatch(units, lead="a") -> str:
    return json.dumps(
        {
            "country": "us",
            "citation": citation(lead),
            "rulespec_ref": "a" * 40,
            "replace_rulespec_path": primary(lead),
            "pr_base_branch": "codex/candidate",
            "open_pr": True,
            "source_bundle_json": wrapper(units),
        }
    )


def test_the_preflight_builds_the_batch_selector_from_the_list(
    script, on, monkeypatch, tmp_path
):
    # The operator's order does not matter; the selector is in path order.
    monkeypatch.setenv(
        "DISPATCH_JSON", dispatch([citation("c"), citation("a"), citation("b")], "b")
    )
    script.main()
    written = json.loads((tmp_path / "selector.json").read_text())
    assert written == {
        **SHARED,
        "head": "a" * 40,
        "units": [
            {"citation": citation(name), "rulespec_path": primary(name)}
            for name in NAMES
        ],
    }
    # The provisioner reads the file back with the same function.
    assert script.verified == [admission.selector_from_payload(written)]
    assert len((tmp_path / "selector.json").read_bytes()) <= 4096
    assert (tmp_path / "output").read_text() == "enabled=true\n"


def test_the_preflight_refuses_a_unit_list_while_the_switch_is_off(
    script, monkeypatch, tmp_path
):
    monkeypatch.setenv("DISPATCH_JSON", dispatch([citation(name) for name in NAMES]))
    with pytest.raises(
        ValueError, match="requires exactly transaction and rulespec_pr"
    ):
        script.main()
    assert script.verified == []
    assert not (tmp_path / "selector.json").exists()
    # The preflight's own check, should a unit list ever reach it with the
    # switch off.
    monkeypatch.setattr(
        helper,
        "split_atomic_source_input",
        lambda raw: {"rulespec_pr": 42, "promotion_units": [citation("a")]},
    )
    with pytest.raises(ValueError, match="^multi-unit promotion is switched off$"):
        script.main()
    assert script.verified == []
    assert not (tmp_path / "selector.json").exists()
    assert not (tmp_path / "output").exists()


def test_the_preflight_without_units_writes_the_single_selector(
    script, on, monkeypatch, tmp_path
):
    monkeypatch.setenv("DISPATCH_JSON", dispatch(None))
    script.main()
    written = json.loads((tmp_path / "selector.json").read_text())
    assert set(written) == set(admission.PromotionSelector.__dataclass_fields__)
    assert script.verified == [admission.PromotionSelector(**written)]


def test_the_citation_input_must_be_a_unit_of_the_list(
    script, on, monkeypatch, tmp_path
):
    monkeypatch.setenv("DISPATCH_JSON", dispatch([citation("a"), citation("b")], "c"))
    with pytest.raises(ValueError, match="citation input to be one of its units"):
        script.main()
    assert script.verified == []
    assert not (tmp_path / "selector.json").exists()


def test_the_path_input_must_be_the_canonical_path_of_the_citation(
    script, on, monkeypatch, tmp_path
):
    payload = json.loads(dispatch([citation("a"), citation("b")]))
    payload["replace_rulespec_path"] = primary("b")
    monkeypatch.setenv("DISPATCH_JSON", json.dumps(payload))
    with pytest.raises(ValueError, match="canonical source path"):
        script.main()
    assert script.verified == []


def test_the_preflight_refuses_a_list_over_the_limit(script, on, monkeypatch, tmp_path):
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS", 2)
    monkeypatch.setenv("DISPATCH_JSON", dispatch([citation(name) for name in NAMES]))
    with pytest.raises(AdmissionError, match="names 3 units; the limit is 2"):
        script.main()
    assert script.verified == []
    assert not (tmp_path / "selector.json").exists()
