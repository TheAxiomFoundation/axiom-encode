from __future__ import annotations

import copy
import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as s
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness.measurement_role_evidence import bind_measurement_roles

FIXTURE = Path(__file__).parent / "fixtures" / "measurement_roles"
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-m-student-loan-interest"


def material():
    return (
        yaml.safe_load((FIXTURE / "policy.yaml").read_text()),
        yaml.safe_load((FIXTURE / "cases.yaml").read_text()),
        (FIXTURE / "source.txt").read_text(),
    )


def issues(doc, cases, source):
    content = yaml.safe_dump(doc, sort_keys=False)
    profile = v._numeric_profile_for_citation_path(CITATION)
    bindings = s.collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    return s.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        source_context={CITATION: source},
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        artifact_numeric_values=tuple(x[1] for x in bindings),
        artifact_numeric_bindings=bindings,
    ).issues


def test_actual_proof_copy_ordinary_witnesses_preserve_remaining_obligations():
    doc, cases, source = material()
    found = issues(doc, cases, source)
    assert len(found) == 2
    assert "numeric value 2025" in found[0]
    assert "paired positive/blocking" in found[1]
    assert not any("formula clause" in issue for issue in found)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_input",
        "rule_collision",
        "imports",
        "default",
        "wrong_unit",
        "wrong_entity",
        "missing_unit",
        "missing_proof",
        "wrong_proof_path",
        "wrong_citation",
        "wrong_dates",
        "wrong_formula",
        "wrong_history",
    ],
)
def test_context_refuses_unbound_or_ambiguous_roles(mutation):
    doc, _, source = material()
    before = copy.deepcopy(doc)
    owner = doc["rules"][0]
    if mutation == "duplicate_input":
        doc["inputs"].append(copy.deepcopy(doc["inputs"][0]))
    elif mutation == "rule_collision":
        owner["name"] = doc["inputs"][0]["name"]
    elif mutation == "imports":
        doc["imports"] = [{"path": "other"}]
    elif mutation == "default":
        doc["inputs"][0]["default"] = False
    elif mutation == "wrong_unit":
        doc["inputs"][1]["unit"] = "USD"
    elif mutation == "wrong_entity":
        doc["inputs"][1]["entity"] = "Family"
    elif mutation == "missing_unit":
        doc["units"] = []
    elif mutation == "missing_proof":
        owner["metadata"]["proof"]["atoms"] = []
    elif mutation == "wrong_proof_path":
        for atom in owner["metadata"]["proof"]["atoms"]:
            atom["path"] = "versions[1].formula"
    elif mutation == "wrong_citation":
        for atom in owner["metadata"]["proof"]["atoms"]:
            atom["source"]["corpus_citation_path"] = "other"
    elif mutation == "wrong_dates":
        owner["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "wrong_formula":
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            " - ", " + "
        )
    elif mutation == "wrong_history":
        source = source.replace("1998 through", "1999 through")
    assert doc != before or mutation == "wrong_history"
    assert bind_measurement_roles(doc, source, CITATION) is None


@pytest.mark.parametrize("mutation", ["assertion", "period", "selector", "unreached"])
def test_no_full_witness_means_no_measurement_consumption(mutation):
    doc, cases, source = material()
    if mutation == "assertion":
        for case in cases:
            key = next(
                k for k in case["output"] if k.endswith("#" + doc["rules"][0]["name"])
            )
            case["output"][key] += 1
    elif mutation == "period":
        for case in cases:
            case["period"]["end"] = "2024-12-31"
    elif mutation == "selector":
        for case in cases:
            del case["input"][
                next(k for k in case["input"] if "carryforward_calculated" in k)
            ]
    else:
        cases = cases[:5]  # Only prior-carryforward branch; no historical witness.
    found = issues(doc, cases, source)
    assert any("numeric value 1997" in issue for issue in found)
    assert any("formula" in issue for issue in found)


def test_unknown_source_year_remains_an_obligation():
    doc, cases, source = material()
    source += "\nThe additional adjustment is 2031.\n"
    assert any("numeric value 2031" in issue for issue in issues(doc, cases, source))


@pytest.mark.parametrize(
    "mutation",
    [
        "quoted_header",
        "quoted_roles",
        "wrong_code",
        "wrong_year",
        "prefixed_header",
        "duplicate_roles",
        "unbalanced_quote",
    ],
)
def test_source_context_cannot_turn_examples_into_measurement_authority(mutation):
    doc, cases, source = material()
    before = source
    if mutation == "quoted_header":
        first, rest = source.split("\n", 1)
        source = '"' + first + '"\n' + rest
    elif mutation == "quoted_roles":
        source = source.replace("Also complete", '"Also complete').replace(
            "to a future year.", 'to a future year."'
        )
    elif mutation == "wrong_code":
        source = source.replace("D. M -V", "D. N -V")
    elif mutation == "wrong_year":
        source = source.replace("(2025-12)", "(2024-12)")
    elif mutation == "prefixed_header":
        source = "Example: " + source
    elif mutation == "duplicate_roles":
        source += (
            "\n" + source[source.index("Also complete") : source.index("If interest")]
        )
    else:
        source += '\n"'
    assert source != before
    assert bind_measurement_roles(doc, source, CITATION) is None
    assert any("numeric value 1997" in issue for issue in issues(doc, cases, source))


def current_material():
    doc, cases, source = material()
    for index, quotes in (
        (
            1,
            [
                "Complete this schedule if interest was paid in 2025 on a student loan granted to you.",
                "Interest paid in 2025 on a student loan",
            ],
        ),
        (2, ["Amount carried to line 385 of your 2025 return"]),
    ):
        for quote in quotes:
            assert source.count(quote) == 1
            doc["rules"][index]["metadata"]["proof"]["atoms"].append(
                {
                    "path": "versions[0].formula",
                    "kind": "formula",
                    "source": {"corpus_citation_path": CITATION, "excerpt": quote},
                }
            )
    return doc, cases, source


def test_current_roles_need_their_own_ordinary_principal_execution():
    doc, cases, source = current_material()
    found = issues(doc, cases, source)
    assert len(found) == 1
    assert "paired positive/blocking" in found[0]
    assert not any("numeric-recall" in issue for issue in found)


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_operation",
        "wrong_operation",
        "wrong_assertion",
        "wrong_ast",
        "duplicate_operation",
        "missing_role",
        "wrong_dates",
        "missing_dependency_assertion",
    ],
)
def test_current_role_refusal_restores_its_year_obligation(index, mutation):
    doc, cases, source = current_material()
    before = copy.deepcopy((doc, cases, source))
    rule = doc["rules"][index]
    operation = (
        "Add lines 46 and 48."
        if index == 1
        else "Carry all or part of this amount to line 385 of your return."
    )
    atoms = rule["metadata"]["proof"]["atoms"]
    if mutation == "missing_operation":
        atoms[:] = [a for a in atoms if a["source"]["excerpt"] != operation]
    elif mutation == "wrong_operation":
        for atom in atoms:
            if atom["source"]["excerpt"] == operation:
                atom["source"]["excerpt"] = "Subtract line 60 from line 52."
    elif mutation == "wrong_assertion":
        for case in cases:
            key = next(k for k in case["output"] if k.endswith("#" + rule["name"]))
            case["output"][key] += 1
    elif mutation == "wrong_ast":
        rule["versions"][0]["formula"] += " + 0"
    elif mutation == "duplicate_operation":
        source += "\n" + operation + "\n"
    elif mutation == "missing_role":
        atoms.pop()
    elif mutation == "wrong_dates":
        rule["versions"][0]["effective_to"] = "2026-12-31"
    else:
        # For line52, an independently computed but unasserted dependency cannot
        # supply the ordinary corroborated witness. For direct line385, remove
        # its own assertion instead.
        name = doc["rules"][0]["name"] if index == 1 else rule["name"]
        for case in cases:
            del case["output"][
                next(k for k in case["output"] if k.endswith("#" + name))
            ]
    assert (doc, cases, source) != before
    assert any("numeric value 2025" in issue for issue in issues(doc, cases, source))


def test_source_offsets_and_unknown_year_are_preserved_with_current_roles():
    doc, cases, source = current_material()
    source += "\nAn additional calculation uses 2026.\n"
    bound = bind_measurement_roles(doc, source, CITATION)
    assert bound is not None
    for role in (*bound.roles, *(r for b in bound.current for r in b.roles)):
        for start, end, year in role.spans:
            assert source[start:end] == str(year)
    assert any("numeric value 2026" in issue for issue in issues(doc, cases, source))


@pytest.mark.parametrize("rule_index", [0, 1, 2])
def test_measurement_consumers_refuse_listed_assertions_from_another_row(rule_index):
    doc, cases, source = current_material()
    name = doc["rules"][rule_index]["name"]
    for case in cases:
        key = next(k for k in case["output"] if k.endswith("#" + name))
        case["output"][key] = [case["output"][key]]
        row = copy.deepcopy(case["input"])
        paid = next(k for k in row if "paid_2025_line_48" in k)
        row[paid] += 1
        case["tables"] = {"Person": [row]}
    found = issues(doc, cases, source)
    assert any("formula" in issue for issue in found)
    assert any("numeric value" in issue for issue in found)
