"""Partial repair documents retain the authenticated candidate's unit registry."""

from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import evals

CAD = {"name": "CAD", "kind": "currency", "minor_units": 2}
USD = {"name": "USD", "kind": "currency", "minor_units": 2}
MISSING = object()


def _documents(tmp_path: Path, preserved_units=MISSING, generated_units=MISSING):
    rule = {
        "name": "amount",
        "kind": "parameter",
        "dtype": "Money",
        "unit": "CAD",
        "versions": [{"effective_from": "2025-01-01", "formula": 100}],
    }
    preserved = {"format": "rulespec/v1", "module": {}, "rules": [rule]}
    generated = {"format": "rulespec/v1", "module": {}, "rules": []}
    if preserved_units is not MISSING:
        preserved["units"] = preserved_units
    if generated_units is not MISSING:
        generated["units"] = generated_units
    path = tmp_path / "repair.yaml"
    path.write_text(yaml.safe_dump(generated), encoding="utf-8")
    candidate = evals.ValidationRetryCandidate(rulespec=yaml.safe_dump(preserved))
    return path, candidate, rule


@pytest.mark.parametrize("generated_units", [MISSING, []])
def test_partial_repair_restores_currency_declaration(tmp_path, generated_units):
    path, candidate, rule = _documents(tmp_path, [CAD], generated_units)
    repairs = evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    payload = yaml.safe_load(path.read_text())
    assert payload["units"] == [CAD]
    assert payload["rules"] == [rule]
    assert "unit:CAD" in repairs


def test_partial_registry_preserves_new_and_omitted_units(tmp_path):
    path, candidate, _ = _documents(tmp_path, [CAD], [USD])
    repairs = evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    assert yaml.safe_load(path.read_text())["units"] == [USD, CAD]
    assert "unit:CAD" in repairs


def test_explicit_unit_replacement_is_not_silently_reverted(tmp_path):
    replacement = {**CAD, "minor_units": 3}
    path, candidate, _ = _documents(tmp_path, [CAD], [replacement])
    repairs = evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    assert yaml.safe_load(path.read_text())["units"] == [replacement]
    assert "unit:CAD" not in repairs


def test_explicit_unit_removal_is_consumed_without_restoration(tmp_path):
    path, candidate, _ = _documents(
        tmp_path, [CAD, USD], [{"name": "USD", "repair_remove": True}]
    )
    repairs = evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    assert yaml.safe_load(path.read_text())["units"] == [CAD]
    assert "removed_unit:USD" in repairs
    assert "repair_remove" not in path.read_text()


def test_no_registry_does_not_invent_a_currency(tmp_path):
    path, candidate, _ = _documents(tmp_path)
    evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    assert "units" not in yaml.safe_load(path.read_text())


@pytest.mark.parametrize("side", ["preserved", "generated"])
@pytest.mark.parametrize("invalid", [None, {}, "CAD", [CAD, CAD], [3], [{"name": 3}]])
def test_invalid_registry_refuses_before_writing(tmp_path, side, invalid):
    path, candidate, _ = _documents(
        tmp_path,
        invalid if side == "preserved" else [CAD],
        invalid if side == "generated" else [],
    )
    before = path.read_bytes()
    error = evals._PreservedRepairOverlayError if side == "preserved" else ValueError
    with pytest.raises(error, match="units"):
        evals._overlay_validation_retry_candidate(
            path, artifact_root=tmp_path, candidate=candidate
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "marker",
    [
        {"name": "USD", "repair_remove": True},
        {"name": "CAD", "repair_remove": False},
        {"name": "CAD", "repair_remove": True, "minor_units": 2},
    ],
)
def test_unknown_or_malformed_unit_removal_refuses(tmp_path, marker):
    path, candidate, _ = _documents(tmp_path, [CAD], [marker])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="units"):
        evals._overlay_validation_retry_candidate(
            path, artifact_root=tmp_path, candidate=candidate
        )
    assert path.read_bytes() == before
