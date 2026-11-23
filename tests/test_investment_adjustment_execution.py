import copy
import json
from pathlib import Path

import pytest

from axiom_encode.harness.bound_investment_adjustment_operands import (
    bind_investment_adjustment,
)
from axiom_encode.harness.investment_adjustment_execution import (
    ENGINE_PIN,
    verify_investment_adjustment_compiled,
)

F = Path(__file__).parent / "fixtures"
CONTENT = (F / "schedule-n-static.yaml").read_text()
PARENT = (F / "schedule-n-authenticated-body.txt").read_text()
GUIDE = (F / "schedule-n-guide.txt").read_text()
ARTIFACT = json.loads((F / "schedule-n-compiled.json").read_text())
BOUND = bind_investment_adjustment(CONTENT, PARENT, GUIDE)


def verify(artifact):
    return verify_investment_adjustment_compiled(
        BOUND, CONTENT, PARENT, GUIDE, artifact, ENGINE_PIN
    )


def test_actual_compiled_program_matches_independent_all_rule_lowering():
    assert len(verify(ARTIFACT)) == 18


@pytest.mark.parametrize("index", range(18))
def test_coherent_top_and_version_expression_substitution_rejected(index):
    artifact = copy.deepcopy(ARTIFACT)
    rule = artifact["program"]["derived"][index]
    replacement = {"kind": "literal", "value": {"kind": "integer", "value": 0}}
    rule["expr"] = replacement
    rule["versions"][0]["expr"] = replacement
    with pytest.raises(ValueError, match="compiled program"):
        verify(artifact)


@pytest.mark.parametrize("index", range(21))
def test_each_input_catalog_identity_required(index):
    artifact = copy.deepcopy(ARTIFACT)
    artifact["metadata"]["input_catalog"][index]["request_names"].append(
        "foreign.input"
    )
    with pytest.raises(ValueError, match="metadata"):
        verify(artifact)


@pytest.mark.parametrize(
    "field", ["entity", "dtype", "unit", "period", "corpus_citation_path", "id"]
)
def test_derived_identity_type_or_source_mutation_rejected(field):
    artifact = copy.deepcopy(ARTIFACT)
    artifact["program"]["derived"][0][field] = "wrong"
    with pytest.raises(ValueError, match="compiled program"):
        verify(artifact)


def test_rate_and_unit_precision_and_extra_relation_rejected():
    for mutation in ("rate", "unit", "relation"):
        artifact = copy.deepcopy(ARTIFACT)
        if mutation == "rate":
            artifact["program"]["parameters"][0]["versions"][0]["values"]["0"][
                "value"
            ] = "0.4"
        elif mutation == "unit":
            artifact["program"]["units"][-1]["minor_units"] = 2.0
        else:
            artifact["program"]["relations"] = [{"name": "foreign"}]
        with pytest.raises(ValueError, match="compiled program"):
            verify(artifact)


def test_evaluation_order_cannot_omit_or_precede_dependencies():
    artifact = copy.deepcopy(ARTIFACT)
    artifact["metadata"]["evaluation_order"].pop()
    with pytest.raises(ValueError, match="evaluation order"):
        verify(artifact)
    artifact = copy.deepcopy(ARTIFACT)
    artifact["metadata"]["evaluation_order"].reverse()
    with pytest.raises(ValueError, match="dependency order"):
        verify(artifact)


def test_wrong_engine_and_forged_binding_refuse():
    with pytest.raises(ValueError, match="unbound"):
        verify_investment_adjustment_compiled(
            BOUND, CONTENT, PARENT, GUIDE, ARTIFACT, "other"
        )
    with pytest.raises(ValueError, match="unbound"):
        verify_investment_adjustment_compiled(
            BOUND, CONTENT + "\n", PARENT, GUIDE, ARTIFACT, ENGINE_PIN
        )
