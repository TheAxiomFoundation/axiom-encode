"""Keep the supervisor's bounded worker contract compatible with Python.

The Go tests consume the same fixture and enforce its accepted/refused cases.
These checks exercise the real Python parser without requiring Go, a broker,
signing keys, or network access in the Python-only CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_encode import cli

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "cmd/axiom-encode-signing-supervisor/testdata/waiver_audit_workers.json"
)
_CONTRACT = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_supervisor_worker_contract_covers_its_complete_accepted_set() -> None:
    assert _CONTRACT["env"] == cli._WAIVER_AUDIT_WORKERS_ENV
    assert _CONTRACT["max_digits"] == 2
    assert _CONTRACT["max_value"] == cli._WAIVER_AUDIT_WORKERS_MAX == 32
    accepted = _CONTRACT["accepted"]
    assert len(accepted) == len(set(accepted)) == 43
    assert set(accepted) == {str(value) for value in range(10)} | {
        f"{value:02d}" for value in range(33)
    }
    assert {"-1", "33", "000"} <= set(_CONTRACT["refused"])


@pytest.mark.parametrize("raw", _CONTRACT["accepted"])
@pytest.mark.parametrize("module_count", [0, 1, 2, 32, 100])
def test_python_accepts_every_supervisor_worker_value_with_the_same_meaning(
    monkeypatch: pytest.MonkeyPatch, raw: str, module_count: int
) -> None:
    monkeypatch.setenv(_CONTRACT["env"], raw)

    assert cli._waiver_audit_worker_count(module_count) == max(
        1, min(int(raw) or 1, module_count)
    )


@pytest.mark.parametrize(("raw", "expected"), [("33", 32), ("000", 1)])
def test_supervisor_worker_contract_is_stricter_than_python(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    # Python accepts values above the supervisor's maximum and longer digit
    # strings: it caps 33 to 32 and treats zero as one worker. Go intentionally
    # rejects both so only a short, bounded value crosses the trust boundary.
    assert raw in _CONTRACT["refused"]
    monkeypatch.setenv(_CONTRACT["env"], raw)

    assert cli._waiver_audit_worker_count(100) == expected


def test_python_also_refuses_the_supervisor_negative_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "-1" in _CONTRACT["refused"]
    monkeypatch.setenv(_CONTRACT["env"], "-1")

    with pytest.raises(ValueError, match="must not be negative"):
        cli._waiver_audit_worker_count(100)
