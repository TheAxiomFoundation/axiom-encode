"""The A/B post-run checks for new inputs in an encoded CFR module."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.test_evals import _canonical_rulespec_content_root

_SPEC = importlib.util.spec_from_file_location(
    "check_state_set_inputs",
    Path(__file__).resolve().parents[1] / "scripts/check_state_set_inputs.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

UNIT = "us/regulation/42/435/733"
SOURCE = (
    "(c)(2) The allowance must be at least $30 a month and must not exceed the "
    "medically needy income standard established under § 435.811, or 133 "
    "percent of the Federal poverty level.\n\n"
    "(d) Other text with $999 that no excerpt reaches.\n"
)
EXCERPT = "medically needy income standard established under § 435.811"


@pytest.fixture
def root(tmp_path):
    root = _canonical_rulespec_content_root(tmp_path / "repos", "us")
    module = root / "regulations" / "42-cfr" / "435" / "811.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "format: rulespec/v1\nrules:\n"
        "  - name: separate_floor\n    kind: derived\n    entity: StateAgency\n"
        "    dtype: Money\n    period: Month\n    versions:\n"
        "      - effective_from: '2014-01-01'\n"
        "        formula: medically_needy_income_standard_amount * 1\n"
    )
    return root


def _module(
    tmp_path: Path,
    *,
    name: str = "medically_needy_income_standard_amount",
    description: str | None = "Standard the State sets under 42 CFR 435.811.",
    root_input: bool = True,
    formula: str | None = None,
    excerpt: str = EXCERPT,
    summary: str = "Summary.",
    dtype: str = "Money",
) -> Path:
    formula = formula or f"max(30, min(need, {name}, 1.33 * fpl))"
    entry = ""
    if root_input:
        entry = f"  - name: {name}\n    entity: Person\n    dtype: {dtype}\n    unit: USD\n    period: Month\n"
        if description is not None:
            entry += f"    description: '{description}'\n"
    path = tmp_path / "candidate.yaml"
    path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        f"  summary: '{summary}'\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {UNIT}\n"
        + (f"inputs:\n{entry}" if entry else "")
        + "rules:\n"
        "  - name: allowance\n    kind: derived\n    entity: Person\n"
        "    dtype: Money\n    unit: USD\n    period: Month\n"
        "    metadata:\n      proof:\n        atoms:\n"
        "          - path: versions[0].formula\n            kind: formula\n"
        "            source:\n"
        f"              corpus_citation_path: {UNIT}\n"
        f'              excerpt: "{excerpt}"\n'
        "    versions:\n      - effective_from: '2014-01-01'\n"
        f"        formula: '{formula}'\n"
    )
    return path


def _baseline(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.yaml"
    path.write_text(
        "format: rulespec/v1\nrules:\n  - name: base\n    kind: derived\n"
        "    entity: Person\n    dtype: Money\n    period: Month\n    versions:\n"
        "      - effective_from: '2014-01-01'\n        formula: need + fpl\n"
    )
    return path


def test_typed_trigger_without_actor_passes(root, tmp_path):
    rows, failures = script.check(root, SOURCE, _module(tmp_path), _baseline(tmp_path))
    assert failures == []
    assert len(rows) == 1
    assert (
        "| medically_needy_income_standard_amount | root | Person/Money/Month |"
        in (rows[0])
    )


def test_baseline_inputs_are_not_new(root, tmp_path):
    rows, _failures = script.check(root, SOURCE, _module(tmp_path), None)
    assert {row.split(" | ")[1] for row in rows} == {
        "medically_needy_income_standard_amount",
        "need",
        "fpl",
    }


def test_literal_only_in_summary_or_excerpt_fails(root, tmp_path):
    module = _module(
        tmp_path,
        formula="min(need, medically_needy_income_standard_amount)",
        summary="At least $30 and 133 percent of the Federal poverty level.",
        excerpt=EXCERPT,
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert sorted(f.split(" in the")[0] for f in failures) == [
        "literal: $30",
        "literal: 133 percent FPL",
    ]


def test_literal_outside_the_cited_paragraph_is_not_required(root, tmp_path):
    _rows, failures = script.check(root, SOURCE, _module(tmp_path), _baseline(tmp_path))
    assert not any("$999" in failure for failure in failures)


def test_input_without_description_fails(root, tmp_path):
    _rows, failures = script.check(
        root, SOURCE, _module(tmp_path, description=None), _baseline(tmp_path)
    )
    assert failures == [
        "description: `medically_needy_income_standard_amount` has no description "
        "naming a provision"
    ]


def test_description_without_provision_fails(root, tmp_path):
    _rows, failures = script.check(
        root,
        SOURCE,
        _module(tmp_path, description="The State standard."),
        _baseline(tmp_path),
    )
    assert len(failures) == 1 and failures[0].startswith("description:")


def test_implicit_identifier_is_an_input_and_fails_description(root, tmp_path):
    rows, failures = script.check(
        root, SOURCE, _module(tmp_path, root_input=False), _baseline(tmp_path)
    )
    assert "| medically_needy_income_standard_amount | implicit | -/-/- |" in rows[0]
    assert failures == [
        "description: `medically_needy_income_standard_amount` has no description "
        "naming a provision",
        "value: `medically_needy_income_standard_amount` is not a number; a "
        "State-set value is",
    ]


def test_exported_name_fails(root, tmp_path):
    module = _module(
        tmp_path,
        name="separate_floor",
        excerpt="the standard established by the agency under § 435.811",
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert failures == [
        "exported: `separate_floor` is a rule name under the RuleSpec root"
    ]


def test_no_trigger_fails(root, tmp_path):
    module = _module(
        tmp_path,
        name="unrelated_amount",
        description="Amount under 42 CFR 435.999.",
        excerpt="(d) Other text with $999 that no excerpt reaches.",
        formula="min(999, unrelated_amount)",
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert failures == [
        "trigger: no actor, State-level or typed trigger holds for `unrelated_amount`"
    ]


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        ("the income standard established by the agency in its State plan", True),
        ("the standard set by the State agency", True),
        ("if the State elects to cover", True),
        ("the agency may specify a standard", True),
        ("the State plan specifies the standard", False),
        ("the standard established by the State plan", False),
        ("the amount set by the agency plan", False),
        ("as specified under section 1902(a)(10) of the Act", False),
        ("the agency must furnish Medicaid as specified in paragraph (b)", False),
        ("established under § 435.811", False),
    ],
)
def test_actor_phrase(excerpt, expected):
    assert script.actor_sets_value(excerpt) is expected


def test_main_exit_code_and_table(root, tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_text(SOURCE)
    module = _module(tmp_path)
    code = script.main(
        [
            "--rulespec-root",
            str(root),
            "--source",
            str(source),
            "--module",
            str(module),
            "--baseline",
            str(_baseline(tmp_path)),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "new inputs: 1; failures: 0" in out
    assert (
        script.main(
            [
                "--rulespec-root",
                str(root),
                "--source",
                str(source),
                "--module",
                str(module),
            ]
        )
        == 1
    )


def _row(rows, name):
    return next(row for row in rows if f"| {name} |" in row)


def test_trigger_is_recorded_per_input(root, tmp_path):
    rows, _ = script.check(root, SOURCE, _module(tmp_path), _baseline(tmp_path))
    assert "| typed:435.811 |" in _row(rows, "medically_needy_income_standard_amount")
    actor_rows, _ = script.check(
        root,
        SOURCE,
        _module(
            tmp_path,
            name="agency_amount",
            excerpt="the allowance",
            description="Set under § 435.733(c).",
        ),
        _baseline(tmp_path),
    )
    assert "| - |" in _row(actor_rows, "agency_amount")


@pytest.mark.parametrize(
    ("source", "trigger"),
    [
        ("(a) A reasonable amount set by the agency applies here.\n", "actor"),
        (
            "(a) The State's AFDC payment standard in effect as of July 16, 1996.\n",
            "state-level",
        ),
        ("(a) The State optional supplement standard applies.\n", "state-level"),
        (
            "(a) The level under the Medicaid State plan as of March 23, 2010.\n",
            "state-level",
        ),
        ("(a) The standard established under § 435.811 applies.\n", "typed:435.811"),
        ("(a) The standard under § 435.811 applies.\n", "-"),
    ],
)
def test_each_trigger_kind(root, tmp_path, source, trigger):
    excerpt = source.split(") ", 1)[1].strip().rstrip(".")
    module = _module(
        tmp_path,
        name="state_value",
        excerpt=excerpt,
        description="Value under 42 CFR 435.811.",
        formula="min(need, state_value)",
    )
    rows, _ = script.check(root, source, module, _baseline(tmp_path))
    assert f"| {trigger} |" in _row(rows, "state_value")


def test_typed_trigger_needs_a_module_for_the_cited_section(root, tmp_path):
    source = "(a) The standard established under § 435.121 applies.\n"
    module = _module(
        tmp_path,
        name="state_value",
        excerpt="The standard established under § 435.121 applies",
        description="Value under 42 CFR 435.121.",
        formula="min(need, state_value)",
    )
    rows, failures = script.check(root, source, module, _baseline(tmp_path))
    assert "| - |" in _row(rows, "state_value")
    assert any(f.startswith("trigger:") for f in failures)


@pytest.mark.parametrize(
    ("description", "ok"),
    [
        ("Standard established under 435.811 for one person.", True),
        ("Standard under § 435.811.", True),
        ("Standard under 42 CFR 435.811.", True),
        ("Standard under 416.1100.", False),
        ("Standard set by the State.", False),
    ],
)
def test_bare_section_of_the_units_part_names_a_provision(
    root, tmp_path, description, ok
):
    _rows, failures = script.check(
        root, SOURCE, _module(tmp_path, description=description), _baseline(tmp_path)
    )
    assert any(f.startswith("description:") for f in failures) is (not ok)


@pytest.mark.parametrize(
    ("name", "description", "flagged"),
    [
        (
            "ssi_home_income_standard",
            "Income standard under 42 CFR 435.725(c)(2)(i).",
            True,
        ),
        (
            "home_income_standard",
            "SSI income standard applied under 42 CFR 435.725.",
            True,
        ),
        ("home_income_standard", "Federal benefit rate under 42 CFR 435.725.", True),
        (
            "home_income_standard",
            "Supplemental Security Income standard, 42 CFR 435.725.",
            True,
        ),
        (
            "optional_supplement_standard",
            "State optional supplement standard, 42 CFR 435.725.",
            False,
        ),
        ("mississippi_standard", "Standard under 42 CFR 435.811.", False),
    ],
)
def test_federal_benefit_standard_is_never_state_set(
    root, tmp_path, name, description, flagged
):
    module = _module(
        tmp_path, name=name, description=description, formula=f"min(need, {name})"
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert any(f.startswith("federal:") for f in failures) is flagged


def test_boolean_input_is_not_a_state_set_value(root, tmp_path):
    module = _module(
        tmp_path,
        name="agency_elects_option",
        dtype="Boolean",
        formula="agency_elects_option",
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert (
        "value: `agency_elects_option` is not a number; a State-set value is"
        in failures
    )


def test_formula_function_names_are_not_inputs(root, tmp_path):
    module = _module(
        tmp_path,
        formula="count_where(need > 0) + medically_needy_income_standard_amount + 30 + 1.33",
    )
    rows, _ = script.check(root, SOURCE, module, None)
    names = {row.split(" | ")[1] for row in rows}
    assert "count_where" not in names
    assert "need" in names


@pytest.mark.parametrize(
    ("source", "excerpt", "trigger"),
    [
        (
            "(a) For other individuals, a reasonable amount set by the agency.\n",
            "For other individuals",
            "actor",
        ),
        (
            "(a) The standard established under 435.811 applies.\n",
            "The standard established under 435.811 applies",
            "-",
        ),
    ],
)
def test_paragraph_actor_and_section_sign_requirement(
    root, tmp_path, source, excerpt, trigger
):
    module = _module(
        tmp_path,
        name="state_value",
        excerpt=excerpt,
        description="Value under 42 CFR 435.811.",
        formula="min(need, state_value)",
    )
    rows, _ = script.check(root, source, module, _baseline(tmp_path))
    assert f"| {trigger} |" in _row(rows, "state_value")
