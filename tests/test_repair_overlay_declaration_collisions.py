"""A repair overlay never declares one symbol as both an input and a rule (#1837).

The overlay restores every named item a repair omits. Restoring a retained
derived rule whose name the repair now declares as an input left both
declarations in the merged artifact; validation then read the name as derived
and the companion's filer facts disappeared. A collision is now rejected with
feedback that names both declarations. Converting a retained rule into an
input, or the reverse, stays possible through the exact removal marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import evals as ev
from axiom_encode.harness import validator_pipeline as vp

NAME = "taxpayer_is_section_152_dependent_of_another_individual"
PREREQUISITE = "claimable_as_dependent_on_another_return"


def _input(name: str, entity: str = "TaxUnit") -> dict[str, str]:
    return {"name": name, "entity": entity, "dtype": "Boolean", "period": "Year"}


def _rule(name: str, formula: str, entity: str = "Person") -> dict[str, object]:
    return {
        "name": name,
        "kind": "derived",
        "entity": entity,
        "dtype": "Judgment",
        "period": "Year",
        "versions": [{"effective_from": "0001-01-01", "formula": formula}],
    }


def _artifact(inputs, rules) -> dict[str, object]:
    payload: dict[str, object] = {"format": "rulespec/v1", "rules": rules}
    if inputs:
        payload["inputs"] = inputs
    return payload


def _overlay(tmp_path: Path, generated, preserved):
    file = tmp_path / "candidate.yaml"
    file.write_text(yaml.safe_dump(generated, sort_keys=False))
    repairs = ev._overlay_validation_retry_candidate(
        file,
        artifact_root=tmp_path,
        candidate=ev.ValidationRetryCandidate(
            rulespec=yaml.safe_dump(preserved, sort_keys=False)
        ),
    )
    return repairs, yaml.safe_load(file.read_text()), file


# The reduced New Mexico 7-2-39 case from #1837: the retained candidate derives
# the dependency fact on Person; the repair supplies it as a TaxUnit filer fact.
RETAINED = _artifact([_input(PREREQUISITE, "Person")], [_rule(NAME, PREREQUISITE)])
ADMISSION = _rule("admission", f"not {NAME}", entity="TaxUnit")


def test_restoring_a_rule_that_shadows_a_replacement_input_is_rejected(tmp_path):
    generated = _artifact([_input(NAME)], [ADMISSION])
    original = yaml.safe_dump(generated, sort_keys=False)
    with pytest.raises(ValueError) as raised:
        _overlay(tmp_path, generated, RETAINED)
    assert not isinstance(raised.value, ev._PreservedRepairOverlayError)
    message = str(raised.value)
    assert f"`{NAME}`" in message
    assert "retained candidate's derived rule on Person" in message
    assert "this repair's input on TaxUnit" in message
    assert f"{{name: {NAME}, repair_remove: true}}" in message
    assert "`rules`" in message
    # Nothing is written: the caller restores the retained candidate.
    assert (tmp_path / "candidate.yaml").read_text() == original


def test_removal_marker_converts_the_retained_rule_into_the_filer_fact(tmp_path):
    generated = _artifact(
        [_input(NAME)], [{"name": NAME, "repair_remove": True}, ADMISSION]
    )
    repairs, merged, file = _overlay(tmp_path, generated, RETAINED)
    assert f"removed_rule:{NAME}" in repairs
    assert {item["name"] for item in merged["inputs"]} == {NAME, PREREQUISITE}
    assert {item["name"] for item in merged["rules"]} == {"admission"}
    stat = file.stat()
    summary = vp._rulespec_reference_summary_cached(
        file, stat.st_mtime_ns, stat.st_size
    )
    # The supplied filer fact is an input slot again, not a derived name.
    assert NAME in summary.input_slots


def test_restoring_an_input_that_shadows_a_replacement_rule_is_rejected(tmp_path):
    retained = _artifact([_input(NAME)], [ADMISSION])
    generated = _artifact([], [_rule(NAME, PREREQUISITE), ADMISSION])
    with pytest.raises(ValueError) as raised:
        _overlay(tmp_path, generated, retained)
    message = str(raised.value)
    assert "retained candidate's input on TaxUnit" in message
    assert "this repair's derived rule on Person" in message
    assert "`inputs`" in message


def test_removal_marker_converts_the_retained_input_into_a_rule(tmp_path):
    retained = _artifact([_input(NAME), _input(PREREQUISITE, "Person")], [ADMISSION])
    generated = _artifact(
        [{"name": NAME, "repair_remove": True}],
        [_rule(NAME, PREREQUISITE), ADMISSION],
    )
    repairs, merged, _ = _overlay(tmp_path, generated, retained)
    assert f"removed_input:{NAME}" in repairs
    assert {item["name"] for item in merged["inputs"]} == {PREREQUISITE}
    assert {item["name"] for item in merged["rules"]} == {NAME, "admission"}


def test_a_repair_that_declares_both_itself_is_rejected(tmp_path):
    generated = _artifact([_input(NAME)], [_rule(NAME, PREREQUISITE), ADMISSION])
    with pytest.raises(ValueError) as raised:
        _overlay(tmp_path, generated, RETAINED)
    assert not isinstance(raised.value, ev._PreservedRepairOverlayError)
    message = str(raised.value)
    assert "this repair's input on TaxUnit" in message
    assert "this repair's derived rule on Person" in message


def test_a_retained_candidate_that_declares_both_is_an_unusable_base(tmp_path):
    retained = _artifact([_input(NAME)], [_rule(NAME, PREREQUISITE)])
    generated = _artifact([], [ADMISSION])
    with pytest.raises(ev._PreservedRepairOverlayError) as raised:
        _overlay(tmp_path, generated, retained)
    assert "retained candidate's input on TaxUnit" in str(raised.value)


def test_restoring_omitted_items_without_a_collision_is_unchanged(tmp_path):
    retained = _artifact(
        [_input(PREREQUISITE, "Person"), _input("filing_status")],
        [_rule(NAME, PREREQUISITE), ADMISSION],
    )
    generated = _artifact([_input("filing_status")], [ADMISSION])
    repairs, merged, _ = _overlay(tmp_path, generated, retained)
    assert f"input:{PREREQUISITE}" in repairs
    assert f"rule:{NAME}" in repairs
    assert {item["name"] for item in merged["inputs"]} == {
        "filing_status",
        PREREQUISITE,
    }
    assert {item["name"] for item in merged["rules"]} == {"admission", NAME}


NAMES = ("alpha", "beta", "gamma", "delta")
ROLE = st.sampled_from(["absent", "input", "rule"])
REPAIR_ROLE = st.sampled_from(
    ["absent", "input", "rule", "remove_input", "remove_rule", "input_remove_rule"]
)


@settings(max_examples=200, deadline=None)
@given(
    retained_roles=st.tuples(*(ROLE for _ in NAMES)),
    repair_roles=st.tuples(*(REPAIR_ROLE for _ in NAMES)),
)
def test_overlay_never_yields_a_declaration_collision_or_drops_an_item(
    tmp_path_factory, retained_roles, repair_roles
):
    retained_inputs, retained_rules, repair_inputs, repair_rules = [], [], [], []
    for name, role in zip(NAMES, retained_roles, strict=True):
        if role == "input":
            retained_inputs.append(_input(name))
        elif role == "rule":
            retained_rules.append(_rule(name, "true"))
    for name, role in zip(NAMES, repair_roles, strict=True):
        if role in {"input", "input_remove_rule"}:
            repair_inputs.append(_input(name))
        if role == "rule":
            repair_rules.append(_rule(name, "true"))
        if role == "remove_input":
            repair_inputs.append({"name": name, "repair_remove": True})
        if role in {"remove_rule", "input_remove_rule"}:
            repair_rules.append({"name": name, "repair_remove": True})
    retained = _artifact(retained_inputs, retained_rules)
    generated = _artifact(repair_inputs, repair_rules)
    tmp_path = tmp_path_factory.mktemp("overlay")
    try:
        _, merged, _ = _overlay(tmp_path, generated, retained)
    except ValueError:
        return
    inputs = {item["name"] for item in merged.get("inputs", [])}
    rules = {item["name"] for item in merged["rules"]}
    assert not inputs & rules
    for name, before, after in zip(NAMES, retained_roles, repair_roles, strict=True):
        # The repair's own declaration always survives.
        if after in {"input", "input_remove_rule"}:
            assert name in inputs
        if after == "rule":
            assert name in rules
        # A retained declaration survives unless the repair removed it.
        if before == "input" and after not in {"remove_input"}:
            assert name in inputs
        if before == "rule" and after not in {"remove_rule", "input_remove_rule"}:
            assert name in rules
