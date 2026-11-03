"""Select the real engine for lifetime fixtures, or skip on a missing capability.

These tests take the ``real_engine`` fixture, so CI's real-engine job runs them
against its pinned engine. An engine without the ``run-lifetime`` subcommand
skips them with a capability reason rather than a missing-engine one, so the
skip stands under ``AXIOM_REQUIRE_REAL_ENGINE=1`` and stays visible in the
report until the pin gains the subcommand. The check is the same probe that
decides whether generation is offered lifetime guidance.
"""

from pathlib import Path

import pytest

from axiom_encode.harness.lifetime_fixtures import (
    LIFETIME_SUBCOMMAND,
    engine_supports_lifetime,
)


def lifetime_capable_binary(real_engine) -> Path:
    """Return the bound binary when its CLI lists the lifetime subcommand."""

    if not engine_supports_lifetime(real_engine.binary):
        pytest.skip(
            f"pinned engine CLI lacks the {LIFETIME_SUBCOMMAND} subcommand; "
            "lifetime fixtures need an engine pin that implements it"
        )
    return real_engine.binary
