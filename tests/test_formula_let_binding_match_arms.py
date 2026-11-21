"""Generated match arms are executable branches, not local assignments."""

import pytest
import yaml

from axiom_encode.harness.validator_pipeline import repair_formula_let_bindings


def document(formula):
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "rules": [
                {
                    "name": "amount",
                    "kind": "derived",
                    "versions": [{"formula": formula}],
                }
            ],
        },
        sort_keys=False,
    )


@pytest.mark.parametrize(
    "formula",
    [
        "match eligible:\n    true => income",
        "match eligible:\n    true => income\n    false => 0",
        "match status:\n    qualified => income\n    _ => 0",
        "flag == true",
    ],
)
def test_match_arms_and_equality_remain_byte_identical(formula):
    source = document(formula)
    assert repair_formula_let_bindings(source) == (source, [])


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("subtotal = income + 1\nsubtotal * 2", "(income + 1) * 2"),
        (
            "subtotal = income + 1\nmatch eligible:\n"
            "    true => subtotal\n    false => 0",
            "match eligible:\n    true => (income + 1)\n    false => 0",
        ),
        ("same = flag == true\nsame", "(flag == true)"),
    ],
)
def test_real_assignments_still_inline_without_consuming_match_arms(formula, expected):
    repaired, names = repair_formula_let_bindings(document(formula))
    assert names == ["amount"]
    assert yaml.safe_load(repaired)["rules"][0]["versions"][0]["formula"] == expected
