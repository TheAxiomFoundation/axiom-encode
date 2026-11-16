"""Document ownership is source-only; it never certifies caller facts or runtime."""

import copy
from pathlib import Path

import pytest

from axiom_encode.bound_form_structure import recognize_bound_form_structure
from axiom_encode.harness.bound_form_obligations import bind_document_requirements

SOURCE = (
    Path(__file__).parent / "fixtures/informational_referral/source.txt"
).read_text()
CITATION = "ca/policy/cra/t1-2025/nl479-credits/primary-document"
CLAUSE = "If you need more space, attach an additional sheet of paper."


def fixture():
    name = "additional_sheet_attachment_required"
    fact = "needs_additional_space"
    atoms = [
        {
            "path": "versions[0].formula",
            "kind": "formula",
            "source": {"corpus_citation_path": CITATION, "excerpt": CLAUSE},
        }
    ]
    atoms += [
        {
            "path": "versions[0]." + k,
            "kind": "effective_period",
            "source": {"corpus_citation_path": CITATION, "excerpt": "2025"},
        }
        for k in ("effective_from", "effective_to")
    ]
    rule = {
        "name": name,
        "kind": "derived",
        "dtype": "Judgment",
        "entity": "Person",
        "period": "Year",
        "metadata": {"proof": {"atoms": atoms}},
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": fact,
            }
        ],
    }
    inp = {
        "name": fact,
        "dtype": "Boolean",
        "entity": "Person",
        "period": "Year",
        "description": "The individual completing this form needs more space for its entries.",
    }
    return {name: rule}, {fact: inp}


def bind(rules, inputs, source=SOURCE):
    form = recognize_bound_form_structure(source, CITATION)
    assert form is not None
    return bind_document_requirements(form, rules, inputs)


def test_source_owned_requirement_exact_frame_and_no_unused_inputs():
    r, i = fixture()
    result = bind(r, i)
    assert result is not None and len(result) == 1
    record = result[0]
    assert SOURCE[slice(*record.source_span)] == CLAUSE
    assert record.entity == "Person" and record.frame == ("2025-01-01", "2025-12-31")
    assert record.input_name == "needs_additional_space"
    assert bind({}, {}) == ()
    assert bind({}, i) is None


@pytest.mark.parametrize(
    "name,accepted",
    [
        ("filer_needs_additional_space", True),
        ("reported_space_requirement", False),
        ("is_eligible_for_credit", False),
    ],
)
def test_name_correlation_not_description_only(name, accepted):
    r, i = fixture()
    fact = i.pop("needs_additional_space")
    fact["name"] = name
    i[name] = fact
    next(iter(r.values()))["versions"][0]["formula"] = name
    # Eligibility reuse keeps the copied needs-space description: still refused.
    assert (bind(r, i) is not None) is accepted


@pytest.mark.parametrize(
    "field,value",
    [
        ("entity", "Family"),
        ("dtype", "Money"),
        ("period", "Month"),
        ("description", "The individual is eligible for the credit."),
        ("unit", "CAD"),
        ("default", False),
    ],
)
def test_fact_contract_refuses_unrelated_or_defaulted_facts(field, value):
    r, i = fixture()
    next(iter(i.values()))[field] = value
    assert bind(r, i) is None


@pytest.mark.parametrize(
    "formula",
    [
        "not needs_additional_space",
        "needs_additional_space and is_eligible",
        "amount > 0",
        "true",
        "other_rule",
        "input.needs_additional_space",
        "needs_additional_space + 1",
    ],
)
def test_no_inverted_arithmetic_cross_rule_or_extra_dependencies(formula):
    r, i = fixture()
    next(iter(r.values()))["versions"][0]["formula"] = formula
    assert bind(r, i) is None


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "nonexistent",
        "clipped",
        "foreign",
        "duplicate",
        "extra",
        "wrong_year",
        "wrong_kind",
        "null",
    ],
)
def test_exact_full_clause_and_date_proofs_required(change):
    r, i = fixture()
    atoms = next(iter(r.values()))["metadata"]["proof"]["atoms"]
    if change == "missing":
        atoms.pop()
    elif change == "nonexistent":
        atoms[0]["source"]["excerpt"] = "If eligible, attach a sheet."
    elif change == "clipped":
        atoms[0]["source"]["excerpt"] = "attach an additional sheet of paper."
    elif change == "foreign":
        atoms[0]["source"]["corpus_citation_path"] = "xx/policy/unrelated"
    elif change == "duplicate":
        atoms[2] = copy.deepcopy(atoms[1])
    elif change == "extra":
        atoms[0]["source"]["unknown"] = "claim"
    elif change == "wrong_year":
        atoms[1]["source"]["excerpt"] = "2024"
    elif change == "wrong_kind":
        atoms[0]["kind"] = "definition"
    else:
        atoms[0] = None
    assert bind(r, i) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("entity", "Family"),
        ("kind", "parameter"),
        ("dtype", "Money"),
        ("unit", "CAD"),
        ("period", "Month"),
    ],
)
def test_only_typed_person_judgment_owner(field, value):
    r, i = fixture()
    next(iter(r.values()))[field] = value
    assert bind(r, i) is None


def test_extra_version_unknown_input_and_duplicate_owner_refused():
    r, i = fixture()
    rule = next(iter(r.values()))
    rule["versions"].append(copy.deepcopy(rule["versions"][0]))
    assert bind(r, i) is None
    rule["versions"].pop()
    i["unused_fact"] = dict(next(iter(i.values())), name="unused_fact")
    assert bind(r, i) is None
    del i["unused_fact"]
    r["duplicate_requirement"] = dict(rule, name="duplicate_requirement")
    assert bind(r, i) is None


@pytest.mark.parametrize(
    "tail",
    [
        " If eligible.",
        " less assistance",
        " plus a recovery amount",
        " or submit a receipt.",
    ],
)
def test_operative_clause_tails_not_ignored(tail):
    r, i = fixture()
    changed = CLAUSE + tail
    next(iter(r.values()))["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        changed
    )
    assert bind(r, i, SOURCE.replace(CLAUSE, changed)) is None


def test_duplicate_source_clause_and_wrong_frame_refused():
    r, i = fixture()
    assert bind(r, i, SOURCE + "\n" + CLAUSE + "\n") is None
    next(iter(r.values()))["versions"][0]["effective_to"] = "2026-12-31"
    assert bind(r, i) is None


@pytest.mark.parametrize(
    "name",
    [
        "does_not_need_space",
        "never_needs_space",
        "needs_no_space",
        "eligible_filer_needs_additional_space",
    ],
)
def test_contradictory_or_arbitrary_prefix_names_refused(name):
    r, i = fixture()
    fact = i.pop("needs_additional_space")
    fact["name"] = name
    i[name] = fact
    next(iter(r.values()))["versions"][0]["formula"] = name
    assert bind(r, i) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("condition", "is_eligible"),
        ("selector", "other"),
        ("rounding", "floor"),
        ("unknown", True),
    ],
)
def test_rule_semantic_extras_refused(field, value):
    r, i = fixture()
    next(iter(r.values()))[field] = value
    assert bind(r, i) is None


def test_metadata_extras_not_silently_dropped():
    r, i = fixture()
    next(iter(r.values()))["metadata"]["unknown"] = True
    assert bind(r, i) is None


def table_fixture():
    rules, inputs = fixture()
    rule = next(iter(rules.values()))
    fact = next(iter(inputs.values()))
    name = "is_claiming_physical_activity_tax_credit"
    fact.update(
        name=name,
        description=(
            "The filer is using the physical activity tax credit section of this form; "
            "this does not assert eligibility or that the table is completed."
        ),
    )
    rule["name"] = "table_requirement"
    rule["versions"][0]["formula"] = name
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        "If applicable, complete the following table and include all of your family's\n"
        "eligible physical activity expenses."
    )
    return {rule["name"]: rule}, {name: fact}


def test_sheet_and_table_have_disjoint_complete_ownership():
    sheet_rules, sheet_inputs = fixture()
    table_rules, table_inputs = table_fixture()
    result = bind(sheet_rules | table_rules, sheet_inputs | table_inputs)
    assert result is not None and len(result) == 2
    assert len({item.source_span for item in result}) == 2
    assert len({item.input_name for item in result}) == 2


@pytest.mark.parametrize(
    "defect", ["eligibility", "negated", "sheet_fact", "proof", "default"]
)
def test_table_premise_cannot_be_relabelled_by_description(defect):
    rules, inputs = table_fixture()
    fact = next(iter(inputs.values()))
    rule = next(iter(rules.values()))
    if defect in {"eligibility", "negated", "sheet_fact"}:
        name = {
            "eligibility": "is_eligible_for_credit",
            "negated": "is_not_claiming_physical_activity_tax_credit",
            "sheet_fact": "needs_additional_space",
        }[defect]
        fact["name"] = name
        inputs = {name: fact}
        rule["versions"][0]["formula"] = name
    elif defect == "proof":
        rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "Complete a table."
    else:
        fact["default"] = False
    assert bind(rules, inputs) is None


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("Do not: ", ""),
        ("Do not:\n", ""),
        ("Do not:\n\n", ""),
        ("", " unless another instruction applies."),
        ("", "\nunless another instruction applies."),
        ("", "\n\nunless another instruction applies."),
        ("", "\n\nExcept where otherwise stated."),
        ("", "\n\nProvided that the family is eligible."),
        ("", "\n\nSubject to other instructions."),
        ("", "\n\nBut only for eligible claimants."),
        ("", "\n\nHowever, eligibility is required."),
    ],
)
def test_table_enclosing_negation_or_qualification_refuses_ownership(
    newline, prefix, suffix
):
    rules, inputs = table_fixture()
    atom = next(iter(rules.values()))["metadata"]["proof"]["atoms"][0]
    clause = atom["source"]["excerpt"]
    source = SOURCE.replace(clause, prefix + clause + suffix).replace("\n", newline)
    atom["source"]["excerpt"] = clause.replace("\n", newline)
    assert bind(rules, inputs, source) is None


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_table_whole_paragraph_preserves_exact_raw_span(newline):
    rules, inputs = table_fixture()
    atom = next(iter(rules.values()))["metadata"]["proof"]["atoms"][0]
    atom["source"]["excerpt"] = atom["source"]["excerpt"].replace("\n", newline)
    source = SOURCE.replace("\n", newline)
    result = bind(rules, inputs, source)
    assert result is not None and len(result) == 1
    assert source[slice(*result[0].source_span)] == atom["source"]["excerpt"]
