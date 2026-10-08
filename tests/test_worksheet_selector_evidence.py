"""Worksheet output references use source proof rather than generated names."""

import copy

import pytest

from tests.test_worksheet_source_clauses import ROW, analyze, fixture

SELECTOR = "deferred_security_options_closing_balance_line_65220"


def worksheet():
    payload, cases = fixture()
    payload["rules"][0]["name"] = SELECTOR
    payload["rules"][1]["versions"][0]["formula"] = SELECTOR + " > 0"
    for case in cases:
        case["output"][SELECTOR] = case["output"].pop("amount_on_line_3")
    return payload, cases


@pytest.mark.parametrize("excerpt", [ROW, "Closing balance: line 1 minus line 2"])
def test_source_row_connects_field_code_name_to_printed_output_line(excerpt):
    payload, cases = worksheet()
    payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = excerpt
    assert not analyze(payload, cases).issues


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_citation",
        "short_excerpt",
        "fabricated_excerpt",
        "wrong_output_line",
        "nonformula_proof",
        "missing_proof",
        "wrong_notice_proof",
        "multiple_versions",
        "missing_assertion",
        "wrong_computation",
        "reversed_comparison",
        "wrong_threshold",
        "inclusive_threshold",
    ],
)
def test_unbound_or_incorrect_notice_is_rejected(mutation):
    payload, cases = worksheet()
    selector, notice = payload["rules"]
    atom = selector["metadata"]["proof"]["atoms"][0]
    if mutation == "wrong_citation":
        atom["source"]["corpus_citation_path"] = "ca/policy/other"
    elif mutation == "short_excerpt":
        atom["source"]["excerpt"] = "Closing balance"
    elif mutation == "fabricated_excerpt":
        atom["source"]["excerpt"] = "Unrelated balance: line 1 minus line 2"
    elif mutation == "wrong_output_line":
        atom["source"]["excerpt"] = ROW.replace("= 3", "= 4")
    elif mutation == "nonformula_proof":
        atom["path"] = "description"
    elif mutation == "missing_proof":
        selector["metadata"] = {}
    elif mutation == "wrong_notice_proof":
        notice["metadata"]["proof"]["atoms"][0]["source"]["corpus_citation_path"] = (
            "ca/policy/other"
        )
    elif mutation == "multiple_versions":
        selector["versions"].append(copy.deepcopy(selector["versions"][0]))
    elif mutation == "missing_assertion":
        for case in cases:
            case["output"].pop(SELECTOR)
    elif mutation == "wrong_computation":
        selector["versions"][0]["formula"] = "opening_balance_line_1 + benefits_line_2"
    else:
        operator, cutoff = {
            "reversed_comparison": ("<", 0),
            "wrong_threshold": (">", 1),
            "inclusive_threshold": (">=", 0),
        }[mutation]
        notice["versions"][0]["formula"] = f"{SELECTOR} {operator} {cutoff}"
        for case in cases:
            value = case["output"][SELECTOR]
            holds = (
                value < cutoff
                if operator == "<"
                else value >= cutoff
                if operator == ">="
                else value > cutoff
            )
            case["output"]["show_on_notice"] = "holds" if holds else "not_holds"
    assert any(
        "complete-source-unit:tests" in issue
        for issue in analyze(payload, cases).issues
    )


@pytest.mark.parametrize("change", ["duplicate", "separated", "wrong_reference"])
def test_source_mapping_must_be_unique_and_adjacent(monkeypatch, change):
    from tests import test_worksheet_source_clauses as fixture_module

    source = fixture_module.SOURCE
    if change == "duplicate":
        source = source + "\n" + source
    elif change == "separated":
        source = source.replace("\n", "\nAn unrelated instruction.\n", 1)
    else:
        source = source.replace("on line 3", "on line 4")
    monkeypatch.setattr(fixture_module, "SOURCE", source)
    payload, cases = worksheet()
    assert any(
        "complete-source-unit:tests" in issue
        for issue in analyze(payload, cases).issues
    )


@pytest.mark.parametrize(
    "consequent",
    [
        "do not show it on the notice.",
        "never show it on the notice.",
        "the Canada Revenue Agency will not show it on your notice of assessment.",
    ],
)
def test_negated_notice_cannot_use_affirmative_binding(monkeypatch, consequent):
    from tests import test_worksheet_source_clauses as fixture_module

    notice = 'If the amount on line 3 is more than "0," ' + consequent
    monkeypatch.setattr(fixture_module, "SOURCE", ROW + "\n" + notice)
    payload, cases = worksheet()
    payload["rules"][1]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = notice
    assert any(
        "complete-source-unit:tests" in issue
        for issue in analyze(payload, cases).issues
    )
