"""Operation lifetime, inventory work and subprocess bounds for frozen trust."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from base64 import b64encode
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import pytest

from axiom_encode import cli, retired_apply
from axiom_encode.retired_apply import (
    RetiredApplyInventory,
    retired_apply_admission_issue,
    retired_apply_operation,
)
from tests import test_retired_apply as _frozen_fixtures
from tests import test_retired_apply_paths as _reader_fixtures
from tests.test_retired_apply import _git
from tests.test_signing_broker import _base_protocol_status, _client_for_status

frozen = _frozen_fixtures.frozen
frozen_repo = _reader_fixtures.frozen_repo


@pytest.mark.parametrize(
    "mutation",
    [
        "__init__",
        "__setstate__",
        "__setattr__",
        "__delattr__",
        "object.__setattr__",
        "object.__delattr__",
        "__dict__",
        "__reduce__",
        "__reduce_ex__",
        "pickle",
        "copy",
        "deepcopy",
        "repository-comparison",
        "artifacts-comparison",
        "artifact-comparison",
        "covered-comparison",
    ],
)
def test_attached_inventory_cannot_change_broker_admission(frozen, mutation):
    replayed = frozen.manifest.with_name("not-frozen.json")
    replayed.write_bytes(frozen.manifest_bytes)
    new_head = frozen.commit("replay outside protected inventory")
    context = replace(frozen.context, artifact_path=replayed)
    replacement_wire = copy.deepcopy(frozen.inventory)
    repository = replacement_wire["repositories"][0]
    repository["cutoff_commit"] = new_head
    repository["artifacts"][0]["path"] = replayed.relative_to(frozen.repo).as_posix()
    replacement = RetiredApplyInventory(replacement_wire)
    status = _base_protocol_status()
    status["retired_apply_public_keys"] = [b64encode(b"d" * 32).decode()]
    status["retired_apply_inventory"] = frozen.inventory
    client = _client_for_status(status)
    try:
        inventory = client.retired_apply_inventory

        def assert_authority_unchanged():
            assert client.retired_apply_inventory is inventory
            with retired_apply_operation():
                assert retired_apply_admission_issue(inventory, frozen.context) is None
                assert "not frozen" in retired_apply_admission_issue(inventory, context)

        assert_authority_unchanged()
        if mutation == "__init__":
            # Immutable tuple storage makes repeated initialization a no-op.
            inventory.__init__(replacement_wire)
        elif mutation.endswith("-comparison"):
            identity = "TheAxiomFoundation/rulespec-us"
            original_repository = inventory[identity]
            replacement_repository = replacement[identity]
            original_artifacts = original_repository["artifacts"]
            replacement_artifacts = replacement_repository["artifacts"]
            original_artifact = next(iter(original_artifacts.values()))
            replacement_artifact = next(iter(replacement_artifacts.values()))
            layers = {
                "repository-comparison": (
                    original_repository,
                    replacement_repository,
                ),
                "artifacts-comparison": (original_artifacts, replacement_artifacts),
                "artifact-comparison": (original_artifact, replacement_artifact),
                "covered-comparison": (
                    original_artifact["covered_files"][0],
                    {"path": "changed.yaml", "sha256": "d" * 64},
                ),
            }
            original, changed = layers[mutation]

            class ReflectedMutation:
                def __eq__(self, other):
                    if isinstance(other, dict):
                        other.clear()
                        other.update(changed)
                    return False

            assert (original == ReflectedMutation()) is False
        else:
            target = inventory
            if mutation in {"copy", "deepcopy"}:
                target = getattr(copy, mutation)(inventory)
                assert target is inventory
            with pytest.raises((AttributeError, TypeError)):
                if mutation in {"__setstate__", "copy", "deepcopy"}:
                    target.__setstate__([dict(replacement)])
                elif mutation == "__setattr__":
                    target.__setattr__("_repositories", dict(replacement))
                elif mutation == "__delattr__":
                    target.__delattr__("_repositories")
                elif mutation == "object.__setattr__":
                    object.__setattr__(target, "_repositories", dict(replacement))
                elif mutation == "object.__delattr__":
                    object.__delattr__(target, "_repositories")
                elif mutation == "__dict__":
                    target.__dict__["_repositories"] = dict(replacement)
                elif mutation == "__reduce__":
                    target.__reduce__()
                elif mutation == "__reduce_ex__":
                    target.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.loads(pickle.dumps(target))
        assert_authority_unchanged()
    finally:
        client.close()


def test_guard_validates_and_copies_inventory_only_at_attach(frozen_repo):
    fixture = frozen_repo
    status = _base_protocol_status()
    for label in ("apply", "eval", "corpus_release"):
        public_key = getattr(fixture.broker, f"{label}_public_key_raw")
        if public_key is not None:
            status[f"{label}_public_key"] = b64encode(public_key).decode()
    status["corpus_release_public_keys"] = [status["corpus_release_public_key"]]
    status["retired_apply_public_keys"] = [
        b64encode(key).decode() for key in fixture.broker.retired_apply_public_keys_raw
    ]
    inventory = fixture.broker.retired_apply_inventory
    status["retired_apply_inventory"] = inventory
    real_deepcopy = copy.deepcopy
    real_cutoff = retired_apply._repository_cutoff
    scopes = []
    inventory_copies = 0

    def record_scope(*args, **kwargs):
        scopes.append(retired_apply._operation.get())
        return real_cutoff(*args, **kwargs)

    def counted_copy(value, *args, **kwargs):
        nonlocal inventory_copies
        if value is inventory or isinstance(value, RetiredApplyInventory):
            inventory_copies += 1
        return real_deepcopy(value, *args, **kwargs)

    with (
        patch.object(copy, "deepcopy", side_effect=counted_copy),
        patch.object(
            retired_apply,
            "validate_retired_apply_inventory",
            wraps=retired_apply.validate_retired_apply_inventory,
        ) as validate,
        patch.object(retired_apply, "_git", wraps=retired_apply._git) as git,
        patch.object(retired_apply, "_repository_cutoff", side_effect=record_scope),
    ):
        client = _client_for_status(status)
        try:
            assert validate.call_count == 1  # Includes the validator's detached copy.
            with (
                patch.object(cli, "get_signing_broker", return_value=client),
                patch.object(
                    cli, "_applied_encoding_manifest_verifier", return_value=client
                ),
            ):
                previous_scope = None
                for _ in range(2):
                    git.reset_mock()
                    scopes.clear()
                    assert (
                        cli.guard_generated_change_issues(
                            fixture.repo, corpus_path=fixture.corpus, all_files=True
                        )
                        == []
                    )
                    assert retired_apply._operation.get() is None
                    assert scopes and scopes[0] is not None
                    assert all(scope is scopes[0] for scope in scopes)
                    assert scopes[0] is not previous_scope
                    previous_scope = scopes[0]
                    assert (
                        sum(
                            call.args[1:] == ("rev-parse", "--is-shallow-repository")
                            for call in git.call_args_list
                        )
                        == 1
                    )
            assert validate.call_count == 1
            assert inventory_copies == 0
        finally:
            client.close()


def test_raw_inventory_is_validated_once_per_operation(frozen):
    with patch.object(
        retired_apply,
        "validate_retired_apply_inventory",
        wraps=retired_apply.validate_retired_apply_inventory,
    ) as validate:
        for number in range(2):
            with retired_apply_operation():
                for _ in range(3):
                    assert (
                        retired_apply_admission_issue(frozen.inventory, frozen.context)
                        is None
                    )
            assert validate.call_count == number + 1


@pytest.mark.parametrize("count", [1, 24])
def test_git_subprocesses_are_constant_per_admission(frozen, monkeypatch, count):
    original = frozen.inventory["repositories"][0]["artifacts"][0]
    artifacts = []
    for number in range(count):
        path = f".axiom/encoding-manifests/{number}.json"
        (frozen.repo / path).write_bytes(frozen.manifest_bytes)
        artifacts.append({**original, "path": path})
    cutoff = frozen.commit()
    repository = frozen.inventory["repositories"][0]
    repository.update(cutoff_commit=cutoff, artifacts=artifacts)
    # Exercise the per-artifact full-history query as well as cutoff ancestry.
    (frozen.repo / "README.txt").write_text("Unrelated post-cutoff commit\n")
    frozen.commit()
    index = RetiredApplyInventory(frozen.inventory)
    calls = Counter()
    actual_git = retired_apply._git

    def counted_git(repo, *arguments, **kwargs):
        calls[arguments] += 1
        return actual_git(repo, *arguments, **kwargs)

    monkeypatch.setattr(retired_apply, "_git", counted_git)
    with retired_apply_operation():
        for artifact in artifacts:
            context = replace(frozen.context, artifact_path=artifact["path"])
            before = calls.total()
            assert retired_apply_admission_issue(index, context) is None
            assert calls.total() - before <= (20 if before == 0 else 10)
    # One repository preflight and one covered-source validation. Each later
    # artifact needs one HEAD query, eight blob subprocesses and its history log.
    assert calls.total() <= 10 + 10 * count
    assert calls[("for-each-ref", "--format=%(refname)", "refs/replace/")] == 1
    assert calls[("rev-parse", "--is-shallow-repository")] == 1
    assert calls[("merge-base", "--is-ancestor", cutoff, cutoff)] == 1


@pytest.mark.parametrize("mutation", ["origin", "grafts", "replace", "shallow"])
def test_repository_preflight_is_fresh_for_each_operation(frozen, mutation):
    index = RetiredApplyInventory(frozen.inventory)
    with retired_apply_operation():
        assert retired_apply_admission_issue(index, frozen.context) is None
    if mutation == "origin":
        _git(
            frozen.repo,
            "remote",
            "set-url",
            "origin",
            "https://github.com/Other/rulespec-us",
        )
        expected = "identity"
    elif mutation == "grafts":
        (frozen.repo / ".git/info/grafts").touch()
        expected = "graft"
    elif mutation == "replace":
        _git(frozen.repo, "update-ref", "refs/replace/" + "a" * 40, frozen.cutoff)
        expected = "replacement"
    else:
        (frozen.repo / ".git/shallow").write_text(frozen.cutoff + "\n")
        expected = "complete Git history"
    with retired_apply_operation():
        issue = retired_apply_admission_issue(index, frozen.context)
    assert issue is not None and expected in issue


def test_operation_tracks_changed_head_and_retrospective_mode(frozen):
    index = RetiredApplyInventory(frozen.inventory)
    with retired_apply_operation():
        assert retired_apply_admission_issue(index, frozen.context) is None
        (frozen.repo / "README.txt").write_text("New HEAD\n")
        frozen.commit()
        # Warm the new HEAD before asking for the stale explicit SHA. Otherwise
        # a key missing head_ref still reruns preflight on the changed HEAD.
        assert retired_apply_admission_issue(index, frozen.context) is None
        # A prior HEAD-keyed snapshot cannot bypass either the head_ref binding
        # or the retrospective cutoff restriction on this new HEAD.
        pinned = replace(frozen.context, head_ref=frozen.cutoff)
        assert "checked-out HEAD" in retired_apply_admission_issue(index, pinned)
        assert retired_apply_admission_issue(index, frozen.context) is None
        retrospective = replace(frozen.context, retrospective=True)
        assert "newer than" in retired_apply_admission_issue(index, retrospective)
        assert retired_apply_admission_issue(index, frozen.context) is None
        frozen.generated.write_bytes(frozen.generated_bytes + b"changed")
        frozen.commit("supersede covered bytes")
        frozen.generated.write_bytes(frozen.generated_bytes)
        frozen.commit("restore covered bytes")
        assert "superseded" in retired_apply_admission_issue(index, frozen.context)


def test_operation_never_reuses_another_repository_or_inventory(frozen, tmp_path):
    index = RetiredApplyInventory(frozen.inventory)
    other = tmp_path / "other" / frozen.repo.name
    other.parent.mkdir()
    # This checkout has the same identity, object IDs and paths, but forbidden
    # metadata. The repository path must participate in every cache key.
    _git(tmp_path, "clone", "--no-hardlinks", str(frozen.repo), str(other))
    _git(
        other,
        "remote",
        "set-url",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-us",
    )
    (other / ".git/info/grafts").touch()
    untrusted = copy.deepcopy(frozen.inventory)
    untrusted["repositories"][0]["repository"] = "TheAxiomFoundation/rulespec-ca"
    with retired_apply_operation():
        assert retired_apply_admission_issue(index, frozen.context) is None
        other_context = replace(
            frozen.context,
            repo_path=other,
            artifact_path=frozen.manifest.relative_to(frozen.repo),
        )
        assert "graft" in retired_apply_admission_issue(index, other_context)
        assert "not frozen" in retired_apply_admission_issue(
            RetiredApplyInventory(untrusted), frozen.context
        )


def test_nested_operations_and_exceptions_discard_their_caches(frozen):
    index = RetiredApplyInventory(frozen.inventory)
    with retired_apply_operation():
        assert retired_apply_admission_issue(index, frozen.context) is None
        outer = retired_apply._operation.get()
        with pytest.raises(RuntimeError, match="abort"):
            with retired_apply_operation():
                assert retired_apply._operation.get() is not outer
                assert retired_apply_admission_issue(index, frozen.context) is None
                raise RuntimeError("abort")
        assert retired_apply._operation.get() is outer
    assert retired_apply._operation.get() is None


def _synthetic_inventory(count, cutoff):
    payload = b'{"applied_files": []}'
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "schema": retired_apply.RETIRED_APPLY_INVENTORY_SCHEMA,
        "repositories": [
            {
                "repository": "TheAxiomFoundation/rulespec-us",
                "cutoff_commit": cutoff,
                "artifacts": [
                    {
                        "path": f".axiom/encoding-manifests/{number}.json",
                        "sha256": digest,
                        "covered_files": [],
                    }
                    for number in range(count)
                ],
            }
        ],
    }, payload


def test_inventory_mapping_handles_collisions_without_exposing_mutable_storage(
    monkeypatch,
):
    monkeypatch.setattr(retired_apply, "hash", lambda key: 0, raising=False)
    inventory, _ = _synthetic_inventory(32, "a" * 40)
    index = RetiredApplyInventory(inventory)
    identity = "TheAxiomFoundation/rulespec-us"
    assert list(index) == [identity]
    assert identity in index and "missing" not in index
    artifacts = index[identity]["artifacts"]
    paths = [artifact["path"] for artifact in inventory["repositories"][0]["artifacts"]]
    assert list(artifacts) == paths
    assert len(artifacts) == len(paths)
    assert artifacts == dict(artifacts)
    assert not artifacts != dict(artifacts)
    for path in paths:
        assert artifacts[path]["path"] == path
    with pytest.raises(KeyError):
        artifacts["missing"]

    def assert_native_storage(value):
        if type(value) is tuple:
            for child in value:
                assert_native_storage(child)
        else:
            assert type(value) in {str, int, type(None)}

    # Even explicit base-class tuple access exposes only immutable native data.
    for value in tuple.__iter__(index):
        assert_native_storage(value)


@pytest.mark.parametrize("warm", [False, True])
def test_inventory_class_spoof_does_not_bypass_validation(warm):
    class Counterfeit(tuple):
        __slots__ = ()

        @property
        def __class__(self):
            return RetiredApplyInventory

    # isinstance honors a spoofed __class__; trusted instances need exact types.
    inventory, _ = _synthetic_inventory(1, "a" * 40)
    index = RetiredApplyInventory(inventory)
    with retired_apply_operation():
        if warm:
            assert retired_apply._inventory_index(index) is index
        object.__setattr__(index, "__class__", Counterfeit)
        assert isinstance(index, RetiredApplyInventory)
        with pytest.raises(ValueError, match="retired apply inventory"):
            retired_apply._inventory_index(index)


def _synthetic_admission_times(frozen, monkeypatch, counts=(1000, 5000)):
    """Measure the hot admission path, with deterministic constant-cost Git IO.

    Keep production inventory indexing, path binding and admission logic. Setup
    is O(N) once at attach; the last artifact exercises accidental linear scans.
    """
    monkeypatch.setattr(retired_apply, "_blob", lambda *args: b'{"applied_files": []}')
    monkeypatch.setattr(
        retired_apply, "_live_bytes", lambda *args: b'{"applied_files": []}'
    )
    actual_git = retired_apply._git

    def synthetic_git(repo, *arguments, **kwargs):
        if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
            return frozen.cutoff.encode()
        return actual_git(repo, *arguments, **kwargs)

    monkeypatch.setattr(retired_apply, "_git", synthetic_git)
    measurements = {}
    for count in counts:
        inventory, payload = _synthetic_inventory(count, frozen.cutoff)
        index = RetiredApplyInventory(inventory)
        context = replace(
            frozen.context,
            artifact_path=f".axiom/encoding-manifests/{count - 1}.json",
            artifact_bytes=payload,
        )
        with retired_apply_operation():
            assert retired_apply_admission_issue(index, context) is None
            samples = []
            # Remove filesystem scheduling noise after real canonical preflight,
            # just as blob/live/Git IO above is synthetic. Keep inventory and
            # admission work intact so linear scans cannot hide behind IO.
            with monkeypatch.context() as hot_io:
                hot_io.setattr(Path, "resolve", lambda self, **kwargs: self)
                for _ in range(5):
                    start = perf_counter()
                    for _ in range(200):
                        assert retired_apply_admission_issue(index, context) is None
                    samples.append((perf_counter() - start) / 200)
        measurements[count] = median(samples)
    return measurements


def test_per_admission_timing_does_not_scale_with_inventory(frozen, monkeypatch):
    times = _synthetic_admission_times(frozen, monkeypatch)
    # Median batches tolerate scheduler noise. A 5x scan/validation/copy is
    # rejected; the small absolute allowance covers microsecond timer noise.
    assert times[5000] <= times[1000] * 2 + 0.00002, times
    print("indexed hot admission seconds:", json.dumps(times, sort_keys=True))
