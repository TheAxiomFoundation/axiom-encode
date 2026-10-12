"""Bare outlines must keep branch and condition evidence separate."""

import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)


FIXTURES = Path(__file__).parent / "fixtures" / "nz_bare_labels"
CITATION = "nz/statute/act/public/2018/0032/section/71"


SOURCE = """In this subpart, unless the context otherwise requires,—
qualifying benefit means—
a
a main benefit under this Act;
b
a benefit specified in this section.
"""


def test_bare_paragraphs_do_not_join_context_exception_to_first_branch():
    branches = sc.recognize_source_structure(SOURCE)
    paragraphs = [branch for branch in branches if branch.path]
    assert [branch.path for branch in paragraphs] == [("a",), ("b",)]
    assert paragraphs[0].text.strip() == "a main benefit under this Act;"
    conditions = sc._source_exception_branches(
        SOURCE,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert len(conditions) == 1
    assert "unless the context otherwise requires" in conditions[0].text
    assert "main benefit" not in conditions[0].text
    assert "a a main" not in sc._collapse_text(conditions[0].text)


def test_ambiguous_bare_sequence_keeps_legacy_source_obligation():
    ambiguous = SOURCE.replace("\nb\n", "\nd\n")
    branches = sc.recognize_source_structure(ambiguous)
    conditions = sc._source_exception_branches(
        ambiguous,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert conditions
    assert "unless the context otherwise requires" in conditions[0].text


def test_verbatim_s71_retains_three_definitions_and_three_paragraphs():
    source = (FIXTURES / "s71.txt").read_text()
    branches = sc.recognize_source_structure(source)
    assert [branch.label for branch in branches if branch.kind == "definition"] == [
        "qualifying benefit",
        "relevant payment date",
        "winter period",
    ]
    paragraphs = [branch for branch in branches if branch.path]
    assert [branch.path for branch in paragraphs] == [("a",), ("b",), ("c",)]
    assert paragraphs[-1].text.strip() == "veteran’s pension"
    clauses = tuple(sc._source_clause_spans(source, branches=branches))
    assert any(text.startswith("relevant payment date") for _, _, text in clauses)
    assert any(text.startswith("winter period") for _, _, text in clauses)
    conditions = sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    context = next(branch for branch in conditions if "unless" in branch.text)
    assert "qualifying benefit" not in context.text
    assert all("\na\n" not in branch.text for branch in branches)
    assert "a a main benefit" not in sc._collapse_text(context.text)


def _analyze(source, payload, cases):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _source_rule(name, formula, excerpt, source_path=CITATION, *, dtype="Bool"):
    return {
        "name": name,
        "kind": "derived",
        "dtype": dtype,
        "source": source_path,
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
        "versions": [{"formula": formula}],
    }


def _qualifying_benefit_candidate(source):
    """An executable candidate for the verbatim qualifying-benefit definition."""

    inputs = [
        "context_otherwise_requires",
        "context_qualifying_benefit",
        "main_benefit_under_this_act",
        "receives_new_zealand_superannuation",
        "receives_veterans_pension",
    ]
    formula = (
        "if context_otherwise_requires: context_qualifying_benefit else: "
        "main_benefit_under_this_act or receives_new_zealand_superannuation "
        "or receives_veterans_pension"
    )
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [{"name": name, "dtype": "Bool"} for name in inputs],
        "rules": [
            _source_rule("qualifying_benefit", formula, source),
            _source_rule(
                "qualifying_main_benefit",
                "main_benefit_under_this_act",
                "a main benefit under this Act; or",
                f"{CITATION}(a)",
            ),
            _source_rule(
                "qualifying_superannuation",
                "receives_new_zealand_superannuation",
                "New Zealand superannuation; or",
                f"{CITATION}(b)",
            ),
            _source_rule(
                "qualifying_veterans_pension",
                "receives_veterans_pension",
                "veteran’s pension",
                f"{CITATION}(c)",
            ),
        ],
    }
    cases = []
    for overrides in (
        {},
        {"main_benefit_under_this_act": True},
        {"receives_new_zealand_superannuation": True},
        {"receives_veterans_pension": True},
        {"main_benefit_under_this_act": True, "context_otherwise_requires": True},
        {"context_qualifying_benefit": True, "context_otherwise_requires": True},
    ):
        values = dict.fromkeys(inputs, False) | overrides
        expected = (
            values["context_qualifying_benefit"]
            if values["context_otherwise_requires"]
            else any(values[name] for name in inputs[2:])
        )
        cases.append(
            {
                "period": "2026-07-01",
                "input": values,
                "output": {
                    "qualifying_benefit": expected,
                    "qualifying_main_benefit": values[inputs[2]],
                    "qualifying_superannuation": values[inputs[3]],
                    "qualifying_veterans_pension": values[inputs[4]],
                },
            }
        )
    return payload, cases


def _s71_candidate(source):
    """Represent every definition, with source-stated dates as local facts."""

    qualifying, remainder = source.split("relevant payment date", 1)
    relevant, winter = ("relevant payment date" + remainder).split("winter period", 1)
    winter = "winter period" + winter
    payload, cases = _qualifying_benefit_candidate(qualifying)
    payload["inputs"].extend(
        [
            {"name": "instalment_payment_date", "dtype": "Date"},
            {"name": "calendar_year_may_first", "dtype": "Date"},
            {"name": "context_relevant_payment_date", "dtype": "Date"},
            {"name": "context_winter_period_start_date", "dtype": "Date"},
            {"name": "context_winter_period_duration_weeks", "dtype": "Int32"},
        ]
    )
    payload["rules"].extend(
        [
            _source_rule(
                "relevant_payment_date",
                "if context_otherwise_requires: context_relevant_payment_date "
                "else: instalment_payment_date",
                relevant,
                dtype="Date",
            ),
            _source_rule(
                "winter_period_start_date",
                "if context_otherwise_requires: context_winter_period_start_date "
                "else: calendar_year_may_first",
                winter,
                dtype="Date",
            ),
            _source_rule(
                "winter_period_weeks",
                "if context_otherwise_requires: context_winter_period_duration_weeks "
                "else: winter_period_duration_weeks",
                winter,
                dtype="Int32",
            ),
        ]
    )
    for name, value, excerpt in (
        ("minimum_received_winter_days", 1, "for 1 or more days during the winter period"),
        ("winter_period_duration_weeks", 22, "the 22-week period starting on 1 May"),
        ("winter_period_start_day", 1, "starting on 1 May"),
    ):
        payload["rules"].append(
            {
                "name": name,
                "kind": "parameter",
                "dtype": "Int32",
                "source": CITATION,
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].value",
                                "kind": "value",
                                "source": {
                                    "corpus_citation_path": CITATION,
                                    "excerpt": excerpt,
                                },
                            }
                        ]
                    }
                },
                "versions": [{"value": value}],
            }
        )
    for case in cases:
        case["input"].update(
            instalment_payment_date="2026-07-03",
            calendar_year_may_first="2026-05-01",
            context_relevant_payment_date="2026-07-04",
            context_winter_period_start_date="2026-06-01",
            context_winter_period_duration_weeks=20,
        )
        context = case["input"]["context_otherwise_requires"]
        case["output"].update(
            relevant_payment_date="2026-07-04" if context else "2026-07-03",
            winter_period_start_date="2026-06-01" if context else "2026-05-01",
            winter_period_weeks=20 if context else 22,
        )
    return payload, cases


def test_verbatim_s71_honest_context_candidate_passes_complete_source_gates():
    source = (FIXTURES / "s71.txt").read_text()
    payload, cases = _s71_candidate(source)
    result = _analyze(source, payload, cases)
    assert not result.issues


def test_verbatim_s71_dropped_context_exception_keeps_tests_obligation():
    source = (FIXTURES / "s71.txt").read_text()
    payload, cases = _s71_candidate(source)
    payload["rules"][0]["versions"][0]["formula"] = (
        "main_benefit_under_this_act or receives_new_zealand_superannuation "
        "or receives_veterans_pension"
    )
    for rule in payload["rules"]:
        formula = rule["versions"][0].get("formula", "")
        if formula.startswith("if context_otherwise_requires:"):
            rule["versions"][0]["formula"] = formula.split(" else: ", 1)[1]
    for case in cases:
        case["output"]["qualifying_benefit"] = any(
            case["input"][name]
            for name in (
                "main_benefit_under_this_act",
                "receives_new_zealand_superannuation",
                "receives_veterans_pension",
            )
        )
        case["output"].update(
            relevant_payment_date="2026-07-03",
            winter_period_start_date="2026-05-01",
            winter_period_weeks=22,
        )
    result = _analyze(source, payload, cases)
    assert any("[complete-source-unit:tests]" in issue for issue in result.issues)
    assert any("context otherwise requires" in issue for issue in result.issues)


def test_exact_bare_branch_accepts_existing_runtime_gap_wording():
    source = (
        "The secretary shall perform the following actions:—\n"
        "a\nThe secretary shall file an annual administration report;\n"
        "b\nThe secretary may require a copy of the federal return to be filed.\n"
    )
    branches = sc.recognize_source_structure(source)
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/b#return_copy",
                    "reason": (
                        f"{CITATION}(b) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("b",)}
    assert not issues


@pytest.mark.parametrize(
    "reason",
    [
        "The runtime has no document-filing capability for the federal return copy.",
        f"{CITATION}(b) cannot be encoded until the runtime has more capabilities.",
        f"{CITATION}(a) cannot be encoded until the runtime has a "
        "document-filing capability for the required federal return copy.",
    ],
)
def test_bare_branch_keeps_imprecise_runtime_gap_deferrals_rejected(reason):
    source = (
        "The secretary shall perform the following actions:—\n"
        "a\nThe secretary shall file an annual administration report;\n"
        "b\nThe secretary may require a copy of the federal return to be filed.\n"
    )
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/b#return_copy",
                    "reason": reason,
                }
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload,
        corpus_citation_path=CITATION,
        source_text=source,
        branches=sc.recognize_source_structure(source),
    )
    assert not covered
    assert any("[complete-source-unit:deferral]" in issue for issue in issues)


def test_inserted_bare_subsection_uses_normalized_gate_branch_paths():
    source = "".join(
        f"{label}\nThe secretary shall perform the following actions:—\n"
        "a\nThe secretary shall file an annual administration report;\n"
        "b\nThe secretary may require a copy of the federal return to be filed.\n"
        for label in ("1", "2", "2A", "3")
    )
    branches = sc.recognize_source_structure(source)
    assert ("2a", "b") in {branch.path for branch in branches}
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/2A/b#return_copy",
                    "reason": (
                        f"{CITATION}(2A)(b) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("2a", "b")}
    assert not issues


def test_labelled_definition_text_does_not_invent_source_unit_definition_branch():
    source = (
        "The following definitions apply:—\n"
        "a\nqualifying benefit means a main benefit under this Act;\n"
        "b\nwinter period means the 22-week period starting on 1 May.\n"
    )
    branches = sc.recognize_source_structure(source)
    assert [branch.path for branch in branches] == [("a",), ("b",)]
    assert all(branch.kind != "definition" for branch in branches)


@pytest.mark.parametrize("path", [(), ("a",), ("b",), ("c",)])
def test_s71_calendar_runtime_gap_is_not_authorized_by_structure(path):
    source = (FIXTURES / "s71.txt").read_text()
    suffix = "".join(f"({part})" for part in path)
    assert not sc._reason_names_source_bound_runtime_gap(
        f"{CITATION}{suffix} cannot be encoded until the runtime has a "
        "date-arithmetic capability for the winter period and relevant payment date.",
        source,
        path=path,
        corpus_citation_path=CITATION,
    )
