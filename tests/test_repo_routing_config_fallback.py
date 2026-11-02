"""Configuration discovery failures must preserve fresh checkout admission."""

import subprocess
from pathlib import Path

import pytest

from axiom_encode import repo_routing


def _init_checkout(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/TheAxiomFoundation/rulespec-uk.git",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("config_probe", ["var", "config"])
@pytest.mark.parametrize("failure", ["timeout", "unavailable"])
def test_configuration_failure_disables_cache_and_keeps_fresh_identity_checks(
    monkeypatch, tmp_path, config_probe, failure
):
    checkout = tmp_path / "rulespec-uk"
    _init_checkout(checkout)
    for name in (
        "AXIOM_ENCODE_GIT_PROBE_TIMEOUT_SECONDS",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
    ):
        monkeypatch.delenv(name, raising=False)
    original_run = subprocess.run
    config_attempts = []
    identity_probes = []

    def failed_configuration(command, *args, **kwargs):
        probe = command[3]
        if probe == config_probe:
            config_attempts.append(kwargs["timeout"])
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            raise FileNotFoundError(2, "git unavailable")
        if probe in {"rev-parse", "remote"}:
            identity_probes.append(probe)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(repo_routing.subprocess, "run", failed_configuration)

    with repo_routing._rulespec_routing_cache_scope() as cache:
        for _ in range(2):
            assert repo_routing.inspect_canonical_rulespec_checkout(checkout) == (
                "rulespec-uk",
                None,
            )
            assert cache.checkout_inspections == {}
        assert identity_probes == ["rev-parse", "remote"] * 2
        assert config_attempts == [10.0] * (4 if failure == "timeout" else 2)

        original_run(
            [
                "git",
                "-C",
                str(checkout),
                "remote",
                "set-url",
                "origin",
                "https://github.com/TheAxiomFoundation/rulespec-us.git",
            ],
            check=True,
            capture_output=True,
        )
        assert repo_routing.inspect_canonical_rulespec_checkout(checkout) == (
            None,
            "git-origin-name-mismatch",
        )
        assert identity_probes == ["rev-parse", "remote"] * 3
        assert config_attempts == [10.0] * (6 if failure == "timeout" else 3)
        assert cache.checkout_inspections == {}
