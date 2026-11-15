"""A deferral in a manual unit names the module's own `policies/` target.

Arizona DES FAA5 medical blocks 6 and 10 (local runs a1ac394d and 76e1c8f5)
each deferred one branch whose amount comes from another manual page. Both
candidates wrote `us-az:manuals/...#<output>`, copying the corpus class of
`us-az/manual/...`. `manuals/` is not a RuleSpec module root, so the target
check refused them, and the completeness check then read the invented
`/assistant-meals` and `/private-vehicles` labels as source branches with no
text, which no reason can satisfy. The modules live at `us-az:policies/...`;
the prompt never said so and the refusal named only a statute example.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.checks import get_check
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE,
    _numeric_profile_for_citation_path,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_deferred_output_issues,
    numeric_value_is_grounded,
)
from axiom_encode.prompts.encoder import ENCODER_PROMPT, get_encoder_prompt

FIXTURES = Path(__file__).parent / "fixtures"
MODULE = "us-az:policies/des/faa5/na-medical-expenses-and-deduction"
UNITS = {
    "6": f"{MODULE}/block-6/assistant-meals#assistant_meals_allowable_amount",
    "10": (
        f"{MODULE}/block-10/private-vehicles"
        "#snap_medical_private_vehicle_transportation_amount"
    ),
}


def _fixture(unit: str) -> tuple[dict, str, list, str]:
    directory = FIXTURES / f"az_batch3_medical_block_{unit}"
    provenance = json.loads((directory / "provenance.json").read_text())
    texts = {}
    for name, key in (
        ("candidate.yaml", "candidate"),
        ("candidate.test.yaml", "companion"),
        ("source.txt", "source"),
    ):
        text = (directory / name).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == provenance[key]["sha256"]
        texts[key] = text
    cases = yaml.safe_load(texts["companion"])
    return provenance, texts["candidate"], cases, texts["source"]


def _deferred_output(candidate: str) -> str:
    return yaml.safe_load(candidate)["module"]["deferred_outputs"][0]["output"]


def _complete_source_issues(candidate: str, cases: list, source: str, citation: str):
    profile = _numeric_profile_for_citation_path(citation)
    bindings = sc.collect_artifact_numeric_bindings(
        candidate,
        extract_named_scalars=extract_named_scalar_occurrences,
    )
    return list(
        sc.analyze_complete_source_unit(
            candidate,
            source,
            corpus_citation_path=citation,
            test_cases=cases,
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text,
                profile=profile,
            ),
            extract_numeric_grounding_occurrences=functools.partial(
                extract_typed_numeric_occurrences_from_text, profile=profile
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            artifact_numeric_values=tuple(value for _name, value in bindings),
            artifact_numeric_bindings=bindings,
        ).issues
    )


def _deferral_issues(issues) -> list[str]:
    return [issue for issue in issues if "[complete-source-unit:deferral]" in issue]


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_recorded_refusal_names_the_module_root(unit):
    provenance, candidate, _cases, _source = _fixture(unit)
    output = _deferred_output(candidate)
    assert output.startswith("us-az:manuals/")

    issues = find_deferred_output_issues(candidate)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.startswith(provenance["recorded_issues"][0])
    assert issue.check == CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE.id
    assert issue.found == output
    assert issue.expected == UNITS[unit]
    assert "`manuals/` is a corpus document class" in issue
    assert "under `policies/`" in issue
    assert UNITS[unit] in issue


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_the_suggested_target_passes_both_deferral_checks(unit):
    provenance, candidate, cases, source = _fixture(unit)
    citation = provenance["citation"]
    before = _deferral_issues(
        _complete_source_issues(candidate, cases, source, citation)
    )
    assert [issue.split("\n", 1)[0] for issue in before] == provenance[
        "recorded_issues"
    ][1:]

    suggested = find_deferred_output_issues(candidate)[0].expected
    repaired = candidate.replace(_deferred_output(candidate), suggested)

    assert find_deferred_output_issues(repaired) == []
    assert (
        _deferral_issues(_complete_source_issues(repaired, cases, source, citation))
        == []
    )


def test_block_10_has_no_other_complete_source_finding_once_the_target_is_fixed():
    provenance, candidate, cases, source = _fixture("10")
    repaired = candidate.replace(_deferred_output(candidate), UNITS["10"])
    assert (
        _complete_source_issues(repaired, cases, source, provenance["citation"]) == []
    )


@pytest.mark.parametrize(
    "output",
    [
        "us-az:manuals/des/faa5/block-10",
        "us-az:manuals",
        "block-10#amount",
        "us-az:handbooks/des/faa5/block-10#amount",
    ],
)
def test_no_root_hint_without_a_corrected_target(output):
    content = yaml.safe_dump(
        {"module": {"deferred_outputs": [{"output": output, "reason": "x"}]}}
    )
    (issue,) = find_deferred_output_issues(content)
    assert issue.check == CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE.id
    assert "corpus document class" not in issue
    assert issue.expected == "<jurisdiction>:<root>/<path>#<output>"


def test_check_is_declared():
    check = get_check(CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE.id)
    assert check is CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE
    assert "`policies/`" in check.fix


def test_prompt_states_the_policies_root_for_manual_units():
    assert "never `manuals/` or `forms/`" in ENCODER_PROMPT
    assert (
        "`us-az:policies/des/faa5/na-medical-expenses-and-deduction/block-10"
        "#private_vehicle_mileage_amount`" in ENCODER_PROMPT
    )
    complete = get_encoder_prompt(
        "us-az/manual/des/faa5/na-medical-expenses-and-deduction/block-10",
        "policies/des/faa5/na-medical-expenses-and-deduction/block-10.yaml",
        corpus_citation_path=(
            "us-az/manual/des/faa5/na-medical-expenses-and-deduction/block-10"
        ),
        require_complete_source_unit=True,
    )
    assert "Never invent a branch label" in complete
