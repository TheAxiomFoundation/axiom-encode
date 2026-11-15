"""A deferral in a manual unit names the module's own `policies/` target.

Arizona DES FAA5 medical blocks 6 and 10 (local runs a1ac394d and 76e1c8f5)
each deferred one branch whose amount comes from another manual page. The
complete-source prompt told the model to write each deferred output as
`us-az:manuals/des/faa5/.../block-10/<marker>/<marker>#<name>`: the
completeness validator's corpus-shaped anchor. `manuals/` is not a RuleSpec
module root, so the target check refused both candidates, and the invented
`/assistant-meals` and `/private-vehicles` markers were read as source branches
with no text, which no reason can satisfy. The modules live at
`us-az:policies/...`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.checks import get_check
from axiom_encode.harness import evals
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.evals import EvalWorkspace
from axiom_encode.harness.validator_pipeline import (
    CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE,
    _numeric_profile_for_citation_path,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_deferred_output_issues,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures"
MODULE = "us-az:policies/des/faa5/na-medical-expenses-and-deduction"
CORPUS = "us-az/manual/des/faa5/na-medical-expenses-and-deduction"
UNITS = {
    "6": f"{MODULE}/block-6#assistant_meals_allowable_amount",
    "10": f"{MODULE}/block-10#snap_medical_private_vehicle_transportation_amount",
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


def _deferral(output: str, corpus_citation_path: str | None) -> str:
    module: dict = {"deferred_outputs": [{"output": output, "reason": "x"}]}
    if corpus_citation_path:
        module["source_verification"] = {"corpus_citation_path": corpus_citation_path}
    return yaml.safe_dump({"module": module})


def _eval_prompt(tmp_path, corpus_citation_path: str, *, complete: bool = True):
    source_file = tmp_path / "source.txt"
    source_file.write_text("The cost is allowable.")
    workspace = EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=tmp_path / "context-manifest.json",
        source_metadata={
            "source_attestation": {
                "requested_corpus_citation_path": corpus_citation_path
            }
        },
    )
    return evals._build_eval_prompt(
        citation=corpus_citation_path,
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="a.yaml",
        include_tests=True,
        runner_backend="codex",
        require_complete_source_unit=complete,
    )


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_recorded_refusal_names_the_module_target(unit):
    provenance, candidate, _cases, _source = _fixture(unit)
    output = _deferred_output(candidate)
    assert output.startswith("us-az:manuals/")

    issues = find_deferred_output_issues(candidate)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.startswith(provenance["recorded_issues"][0])
    assert issue.check == CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE.id
    assert issue.found == output
    # The source prints no marker, so the expected target has no branch.
    assert issue.expected == UNITS[unit]
    assert "`manuals/` is a corpus document class" in issue
    assert f"own target `{UNITS[unit]}`" in issue


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
    ("output", "corpus_citation_path", "expected"),
    [
        ("us:statute/26/21/b#credit", "us/statute/26/21", "us:statutes/26/21#credit"),
        (
            "us:regulation/7/273/9/d#deduction",
            "us/regulation/7/273/9",
            "us:regulations/7-cfr/273/9#deduction",
        ),
        (
            "us-ks:forms/department-of-revenue/k-40es#tax",
            "us-ks/form/department-of-revenue/k-40es",
            "us-ks:policies/department-of-revenue/k-40es#tax",
        ),
    ],
)
def test_singular_and_form_roots_name_the_module_target(
    output, corpus_citation_path, expected
):
    (issue,) = find_deferred_output_issues(_deferral(output, corpus_citation_path))
    assert issue.check == CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE.id
    assert issue.expected == expected
    assert "is a corpus document class" in issue


@pytest.mark.parametrize(
    ("output", "corpus_citation_path"),
    [
        ("us-az:manuals/des/faa5/block-10#amount", None),
        ("us-az:manuals/des/faa5/block-10", f"{CORPUS}/block-10"),
        ("us-az:manuals", f"{CORPUS}/block-10"),
        ("block-10#amount", f"{CORPUS}/block-10"),
        ("us-az:handbooks/des/faa5/block-10#amount", f"{CORPUS}/block-10"),
    ],
)
def test_no_root_hint_without_a_module_target(output, corpus_citation_path):
    (issue,) = find_deferred_output_issues(_deferral(output, corpus_citation_path))
    assert issue.check == CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE.id
    assert "corpus document class" not in issue
    assert issue.expected == "<jurisdiction>:<root>/<path>#<output>"


def test_check_is_declared():
    check = get_check(CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE.id)
    assert check is CHECK_DEFERRED_OUTPUT_TARGET_NOT_ABSOLUTE
    assert "`policies/`" in check.fix


@pytest.mark.parametrize(
    ("citation", "anchor"),
    [
        (f"{CORPUS}/block-10", f"{MODULE}/block-10"),
        (
            "us-wa/manual/dshs/eaz/esa-applications-assistance-expedited-service"
            "-basic-food/block-5",
            "us-wa:policies/dshs/eaz/esa-applications-assistance-expedited-service"
            "-basic-food/block-5",
        ),
        (
            "us-ks/form/department-of-revenue/k-40es",
            "us-ks:policies/department-of-revenue/k-40es",
        ),
        # Valid anchors are unchanged.
        ("us/statute/26/2/a", "us:statutes/26/2/a"),
        ("us/regulation/42/457/622", "us:regulations/42-cfr/457/622"),
        (
            "us/guidance/irs/rev-proc-2025-32/page-15",
            "us:policies/irs/rev-proc-2025-32/page-15",
        ),
    ],
)
def test_prompt_anchor_is_a_module_target(citation, anchor):
    assert evals._complete_source_deferral_anchor(citation) == anchor


def test_manual_prompt_names_the_policies_anchor_and_no_invented_markers(tmp_path):
    prompt = _eval_prompt(tmp_path, f"{CORPUS}/block-10")
    assert "us-az:manuals/" not in prompt
    assert f"`{MODULE}/block-10/<marker>/<marker>#<name>`" in prompt
    assert f"own target `{MODULE}/block-10#<name>`" in prompt
    assert "never invent a marker" in prompt


@pytest.mark.parametrize(
    ("citation", "complete"),
    [
        ("us/statute/26/2/a", True),
        ("us/guidance/irs/rev-proc-2025-32/page-15", True),
        (f"{CORPUS}/block-10", False),
    ],
)
def test_unmarked_prose_note_is_only_in_complete_manual_and_form_prompts(
    tmp_path, citation, complete
):
    prompt = _eval_prompt(tmp_path, citation, complete=complete)
    assert "never invent a marker" not in prompt
