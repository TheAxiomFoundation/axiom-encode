"""Select the real engine for lifetime fixtures, or skip on a missing capability.

These tests take the ``real_engine`` fixture, so CI's real-engine job runs them
against its pinned engine. An engine without the ``run-lifetime`` subcommand
skips them with a capability reason rather than a missing-engine one, so the
skip stands under ``AXIOM_REQUIRE_REAL_ENGINE=1`` and stays visible in the
report until the pin gains the subcommand.
"""

import subprocess
from pathlib import Path

import pytest

LIFETIME_SUBCOMMAND = "run-lifetime"


def lifetime_capable_binary(real_engine) -> Path:
    """Return the bound binary when its CLI lists the lifetime subcommand."""

    probe = subprocess.run(
        [str(real_engine.binary), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if LIFETIME_SUBCOMMAND not in probe.stdout + probe.stderr:
        pytest.skip(
            f"pinned engine CLI lacks the {LIFETIME_SUBCOMMAND} subcommand; "
            "lifetime fixtures need an engine pin that implements it"
        )
    return real_engine.binary
