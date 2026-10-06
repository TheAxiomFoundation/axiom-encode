"""Git probe budgets and UK root admission under transient host load."""

import math
import random
import struct
import subprocess
from pathlib import Path

import pytest

from axiom_encode import repo_routing
from axiom_encode.harness.dependency_stubs import UnsafeRulespecContextPath
from axiom_encode.harness.validator_pipeline import _candidate_rulespec_repo_roots
from axiom_encode.repo_routing import inspect_canonical_rulespec_checkout


def _init_checkout(path: Path, origin: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", origin],
        cwd=path,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("probe", ["rev-parse", "remote"])
def test_uk_compile_roots_recover_transient_git_timeout(monkeypatch, tmp_path, probe):
    """A slow identity probe must not make a valid UK root appear unsafe."""
    checkout = tmp_path / "rulespec-uk"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-uk.git")
    policy_root = checkout / "uk"
    policy_root.mkdir()
    original_run = subprocess.run
    timed_out = False
    attempts = []

    def overloaded_git(command, *args, **kwargs):
        nonlocal timed_out
        if command[3] == probe:
            attempts.append(kwargs["timeout"])
            if not timed_out:
                timed_out = True
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(repo_routing.subprocess, "run", overloaded_git)

    assert _candidate_rulespec_repo_roots("rulespec-uk", policy_root) == [policy_root]
    assert timed_out
    assert len(attempts) >= 2


@pytest.mark.parametrize(
    "probe",
    [
        "_git_top_level",
        "_git_origin_repo_name",
        "_git_config_input_paths",
        "_git_system_config_input_paths",
    ],
)
def test_git_probes_retry_timeout_with_configured_budget(monkeypatch, tmp_path, probe):
    checkout = tmp_path / "rulespec-uk"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-uk.git")
    monkeypatch.setenv("AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.delenv("GIT_CONFIG_SYSTEM", raising=False)
    original_run = subprocess.run
    attempts = []
    target_command = {
        "_git_top_level": "rev-parse",
        "_git_origin_repo_name": "remote",
        "_git_config_input_paths": "config",
        "_git_system_config_input_paths": "var",
    }[probe]
    timed_out = False

    def overloaded_git(command, *args, **kwargs):
        nonlocal timed_out
        attempts.append(kwargs["timeout"])
        if command[3] == target_command and not timed_out:
            timed_out = True
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(repo_routing.subprocess, "run", overloaded_git)

    assert getattr(repo_routing, probe)(str(checkout)) is not None
    assert attempts == ([12.5] * (3 if probe == "_git_config_input_paths" else 2))


@pytest.mark.parametrize("error", ["timeout", "unavailable"])
def test_uk_compile_roots_reject_failed_git_probe_with_bounded_attempts(
    monkeypatch, tmp_path, error
):
    checkout = tmp_path / "rulespec-uk"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-uk.git")
    policy_root = checkout / "uk"
    policy_root.mkdir()
    monkeypatch.delenv("AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS", raising=False)
    attempts = []

    def failed_git(command, **kwargs):
        attempts.append(kwargs["timeout"])
        if error == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise FileNotFoundError(2, "git unavailable")

    monkeypatch.setattr(repo_routing.subprocess, "run", failed_git)

    with pytest.raises(
        UnsafeRulespecContextPath, match="exact direct jurisdiction child"
    ):
        _candidate_rulespec_repo_roots("rulespec-uk", policy_root)
    assert attempts == ([10.0, 10.0] if error == "timeout" else [10.0])


@pytest.mark.parametrize(
    "value",
    ["", "invalid", "0", "-1", "nan", "inf", "300.00000000000006", "301", "1e308"],
)
def test_git_probe_timeout_rejects_invalid_configuration(monkeypatch, tmp_path, value):
    checkout = tmp_path / "rulespec-uk"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-uk.git")
    monkeypatch.setenv("AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS", value)

    def unexpected_git(*args, **kwargs):
        pytest.fail("Invalid timeout must be rejected before launching Git")

    monkeypatch.setattr(repo_routing.subprocess, "run", unexpected_git)

    assert inspect_canonical_rulespec_checkout(checkout) == (
        None,
        "git-top-level-probe-invalid-timeout",
    )


def test_git_probe_timeout_categorizes_extreme_budget(monkeypatch, tmp_path):
    """A real identity probe must categorize a budget subprocess cannot represent."""
    checkout = tmp_path / "rulespec-uk"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-uk.git")
    monkeypatch.setenv("AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS", "1e308")

    assert inspect_canonical_rulespec_checkout(checkout) == (
        None,
        "git-top-level-probe-invalid-timeout",
    )


def test_accepted_git_probe_budgets_reach_a_real_probe_without_overflow(
    monkeypatch, tmp_path
):
    """Generated positive float budgets through 300 seconds launch safely.

    Tiny budgets may expire, but must reach Git and yield a categorized timeout
    rather than an uncaught numeric error. Use the standard library because this
    repository has no property-testing dependency.
    """
    checkout = tmp_path / "rulespec-uk"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-uk.git")
    budgets = [math.ulp(0.0), math.ulp(1.0), 0.5, 10.0, 12.5, 300.0]
    budgets.extend([math.nextafter(300.0, 0.0), math.ldexp(1.0, -1022)])
    # Positive IEEE 754 bit patterns preserve numerical order. Sample throughout
    # the domain, including subnormal and very small normal numbers.
    randomizer = random.Random(1777)
    maximum_bits = struct.unpack("!Q", struct.pack("!d", 300.0))[0]
    budgets.extend(
        struct.unpack("!d", struct.pack("!Q", randomizer.randint(1, maximum_bits)))[0]
        for _ in range(64)
    )
    original_run = subprocess.run
    attempts = []

    def observed_git(command, **kwargs):
        attempts.append(kwargs["timeout"])
        return original_run(command, **kwargs)

    monkeypatch.setattr(repo_routing.subprocess, "run", observed_git)
    for budget in budgets:
        attempts.clear()
        monkeypatch.setenv("AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS", repr(budget))
        assert repo_routing._git_probe_timeout_seconds() == budget
        try:
            assert repo_routing._git_top_level(str(checkout)) == checkout.resolve()
        except repo_routing._GitProbeError as exc:
            assert exc.category == "timeout"
        assert attempts in ([budget], [budget, budget])
