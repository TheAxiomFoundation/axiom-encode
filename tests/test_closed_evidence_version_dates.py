"""Keep synthetic closed evidence tied to actual version dates and proof paths."""

import copy
from datetime import date

import pytest
import yaml

from axiom_encode.harness.measurement_role_evidence import bind_measurement_roles
from axiom_encode.legacy_exact_dependent_concepts import (
    LegacyExactDependentConceptError,
    derive_exact_dependent_parameter_replacements,
)
from tests import test_legacy_exact_dependent_concepts as legacy
from tests import test_measurement_role_evidence as measurement
from tests import test_partnership_reporting_evidence as partnership
from tests import test_repayment_alternative_evidence as repayment_module


def aliases(payload, selected=("effective_from", "effective_to"), *, proofs=True):
    """Rewrite a synthetic input without changing a date or proof authority."""
    for rule in payload["rules"]:
        for version in rule.get("versions", []):
            for canonical, alias in (
                ("effective_from", "from"),
                ("effective_to", "to"),
            ):
                if canonical in selected and canonical in version:
                    version[alias] = version.pop(canonical)
        if proofs:
            for atom in rule.get("metadata", {}).get("proof", {}).get("atoms", []):
                for canonical, alias in (
                    ("effective_from", "from"),
                    ("effective_to", "to"),
                ):
                    if canonical in selected:
                        atom["path"] = atom["path"].replace(
                            "." + canonical, "." + alias
                        )


@pytest.mark.parametrize(
    "selected",
    [("effective_from",), ("effective_to",), ("effective_from", "effective_to")],
)
def test_schedule_m_date_aliases_preserve_measurement_and_reporting_evidence(selected):
    payload, cases, source = measurement.current_material()
    aliases(payload, selected)
    bound = bind_measurement_roles(payload, source, measurement.CITATION)
    assert bound is not None
    assert len(bound.current) == 2
    assert measurement.issues(payload, cases, source) == ()


@pytest.mark.parametrize("rule_index", [0, 1, 2, 3])
@pytest.mark.parametrize("damage", ["proof_path", "duplicate_date", "date_object"])
def test_schedule_m_aliases_preserve_proof_and_date_refusals(rule_index, damage):
    payload, cases, source = measurement.current_material()
    aliases(payload)
    rule = payload["rules"][rule_index]
    if damage == "proof_path":
        for atom in rule["metadata"]["proof"]["atoms"]:
            atom["path"] = atom["path"].replace(".from", ".effective_from")
    elif damage == "duplicate_date":
        rule["versions"][0]["effective_from"] = rule["versions"][0]["from"]
    else:
        rule["versions"][0]["from"] = date(2025, 1, 1)
    assert measurement.issues(payload, cases, source)


@pytest.mark.parametrize(
    "selected",
    [("effective_from",), ("effective_to",), ("effective_from", "effective_to")],
)
def test_partnership_date_aliases_keep_exact_reporting_certificate(selected):
    payload, cases, source = partnership.material()
    expected = partnership.certify(payload, cases, source)
    assert expected is not None
    aliases(payload, selected)
    actual = partnership.certify(payload, cases, source)
    assert actual is not None
    assert actual.witnesses == expected.witnesses


@pytest.mark.parametrize(
    "damage", ["proof_path", "duplicate_date", "date_object", "unknown_key"]
)
def test_partnership_aliases_keep_closed_refusal_boundaries(damage):
    payload, cases, source = partnership.material()
    aliases(payload, proofs=damage != "proof_path")
    version = payload["rules"][0]["versions"][0]
    if damage == "duplicate_date":
        version["effective_to"] = version["to"]
    elif damage == "date_object":
        version["from"] = date(2025, 1, 1)
    elif damage == "unknown_key":
        version["unsupported"] = True
    assert partnership.certify(payload, cases, source) is None


@pytest.fixture
def repayment_world(tmp_path):
    return repayment_module.repayment.__wrapped__(tmp_path)


@pytest.mark.parametrize(
    "selected",
    [("effective_from",), ("effective_to",), ("effective_from", "effective_to")],
)
def test_repayment_date_aliases_keep_owned_dates_and_alternative_tree(
    repayment_world, selected
):
    expected = repayment_module.coverage(repayment_world)
    assert len(expected.records) == 1
    for payload in repayment_world[:2]:
        aliases(payload, selected)
    actual = repayment_module.coverage(repayment_world)
    assert len(actual.records) == 1
    before, after = expected.records[0], actual.records[0]
    assert (after.start, after.end, after.tree, after.raw_span) == (
        before.start,
        before.end,
        before.tree,
        before.raw_span,
    )


@pytest.mark.parametrize(
    "damage", ["proof_path", "duplicate_date", "date_object", "unknown_key"]
)
def test_repayment_aliases_keep_closed_refusal_boundaries(repayment_world, damage):
    for payload in repayment_world[:2]:
        aliases(payload, proofs=damage != "proof_path")
    version = repayment_world[0]["rules"][0]["versions"][0]
    if damage == "duplicate_date":
        version["effective_from"] = version["from"]
    elif damage == "date_object":
        version["from"] = date(2031, 1, 1)
    elif damage == "unknown_key":
        version["unsupported"] = True
    assert repayment_module.coverage(repayment_world).records == ()


def legacy_alias_bytes(raw):
    payload = yaml.safe_load(raw)
    aliases(payload)
    return yaml.safe_dump(payload).encode()


def test_legacy_dependent_use_and_parameter_aliases_preserve_exact_replacement():
    kwargs = dict(
        dependent_primary_raw=legacy._dependent(),
        retained_modules=legacy._retained_modules(),
    )
    expected = derive_exact_dependent_parameter_replacements(**kwargs)
    actual = derive_exact_dependent_parameter_replacements(
        dependent_primary_raw=legacy_alias_bytes(kwargs["dependent_primary_raw"]),
        retained_modules=tuple(
            (old, new, legacy_alias_bytes(before), legacy_alias_bytes(after))
            for old, new, before, after in kwargs["retained_modules"]
        ),
    )
    assert actual == expected


@pytest.mark.parametrize("damage", ["duplicate_date", "date_object"])
def test_legacy_dependent_aliases_keep_date_refusals(damage):
    payload = yaml.safe_load(legacy._dependent())
    aliases(payload)
    version = payload["rules"][0]["versions"][0]
    if damage == "duplicate_date":
        version["effective_from"] = version["from"]
    else:
        version["from"] = date(2026, 1, 1)
    with pytest.raises(LegacyExactDependentConceptError):
        derive_exact_dependent_parameter_replacements(
            dependent_primary_raw=yaml.safe_dump(payload).encode(),
            retained_modules=legacy._retained_modules(),
        )


def test_legacy_parameter_to_alias_preserves_runtime_contract_refusal():
    old, new, before, after = copy.deepcopy(legacy._retained_modules()[0])
    payload = yaml.safe_load(after)
    aliases(payload)
    payload["rules"][0]["versions"][-1]["to"] = "2030-12-31"
    with pytest.raises(LegacyExactDependentConceptError, match="cannot be proved"):
        derive_exact_dependent_parameter_replacements(
            dependent_primary_raw=legacy._dependent(),
            retained_modules=((old, new, before, yaml.safe_dump(payload).encode()),),
        )
