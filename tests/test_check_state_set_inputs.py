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
SNAP_UNIT = "us/regulation/7/273/9"
SOURCE = (
    "(c)(2) The allowance must be at least $30 a month and must not exceed the "
    "medically needy income standard established by the agency under § 435.811, "
    "or 133 percent of the Federal poverty level.\n\n"
    "(d) Other text with $999 that no excerpt reaches.\n\n"
    "(e) The amount may not exceed the medically needy income standard "
    "established under § 435.811.\n\n"
    "(f) The income standard used to determine eligibility for SSI applies.\n"
)
EXCERPT = "medically needy income standard established by the agency under § 435.811"
SNAP_SOURCE = (
    "(a) Net income is compared with the maximum allotment established under "
    "§ 273.10(e)(4).\n\n"
    "(b) The standard utility allowance set by the State agency applies.\n"
)
DELEGATES = (
    "  - name: standard_delegation\n    kind: source_relation\n"
    "    source_relation:\n      type: delegates\n"
    "      target: us:regulations/42-cfr/435/733#allowance\n"
    "      authority: federal\n"
)


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
    unit: str | None = UNIT,
    relation: str = DELEGATES,
) -> Path:
    formula = formula or f"max(30, min(need, {name}, 1.33 * fpl))"
    entry = ""
    if root_input:
        entry = f"  - name: {name}\n    entity: Person\n    dtype: {dtype}\n    unit: USD\n    period: Month\n"
        if description is not None:
            entry += f"    description: '{description}'\n"
    verification = (
        f"  source_verification:\n    corpus_citation_path: {unit}\n" if unit else ""
    )
    path = tmp_path / "candidate.yaml"
    path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        f"  summary: '{summary}'\n"
        + verification
        + (f"inputs:\n{entry}" if entry else "")
        + "rules:\n"
        + relation
        + "  - name: allowance\n    kind: derived\n    entity: Person\n"
        "    dtype: Money\n    unit: USD\n    period: Month\n"
        "    metadata:\n      proof:\n        atoms:\n"
        "          - path: versions[0].formula\n            kind: formula\n"
        "            source:\n"
        f"              corpus_citation_path: {unit or UNIT}\n"
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


def _row(rows, name):
    return next(row for row in rows if f"| {name} |" in row)


def _kinds(failures):
    return sorted({failure.split(":", 1)[0] for failure in failures})


def _state_value(tmp_path, excerpt, **kwargs):
    kwargs.setdefault("description", "Value under 42 CFR 435.811.")
    return _module(
        tmp_path,
        name="state_value",
        excerpt=excerpt,
        formula="min(need, state_value)",
        **kwargs,
    )


def _main(root, source_path, module, baseline=None):
    argv = [
        "--rulespec-root",
        str(root),
        "--source",
        str(source_path),
        "--module",
        str(module),
    ]
    if baseline is not None:
        argv += ["--baseline", str(baseline)]
    return script.main(argv)


@pytest.fixture
def source_path(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text(SOURCE)
    return path


# --- the accepted case and the checks that do not depend on the trigger ---


def test_state_actor_in_a_verified_excerpt_passes(root, tmp_path):
    rows, failures = script.check(root, SOURCE, _module(tmp_path), _baseline(tmp_path))
    assert failures == []
    assert len(rows) == 1
    assert (
        "| medically_needy_income_standard_amount | root | Person/Money/Month "
        "| actor |" in rows[0]
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
    module = _module(tmp_path, name="separate_floor")
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert failures == [
        "exported: `separate_floor` is a rule name under the RuleSpec root"
    ]


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


def test_main_exit_code_and_table(root, tmp_path, source_path, capsys):
    module = _module(tmp_path)
    assert _main(root, source_path, module, _baseline(tmp_path)) == 0
    assert "new inputs: 1; failures: 0" in capsys.readouterr().out
    # Without a baseline `need` and `fpl` are new inputs too, and fail.
    assert _main(root, source_path, module) == 1


# --- findings 1-3: acceptance is a closed positive condition on the source ---


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        ("the income standard established by the agency in its State plan", True),
        ("the standard set by the State agency", True),
        ("the level specified by the State", True),
        ("if the State elects to cover", True),
        ("the agency may specify a standard", True),
        ("The agency must establish in its State plan the income standard", True),
        ("the State has chosen a higher standard", True),
        ("a standard that the single State agency selects", True),
        ("the amount the agency determines under its State plan", True),
        ("the amount determined by the agency in its State plan", True),
        # No State actor of a verb of the closed list.
        ("the agency determines the amount of the individual's income", False),
        ("the amount determined by the agency", False),
        ("the State plan specifies the standard", False),
        ("the standard established by the State plan", False),
        ("the amount set by the agency plan", False),
        ("as specified under section 1902(a)(10) of the Act", False),
        ("the agency must furnish Medicaid as specified in paragraph (b)", False),
        ("established under § 435.811", False),
        ("the maximum allotment established under § 273.10(e)(4)", False),
        # A possessive is not an actor.
        ("the State's allotment under § 457.608", False),
        ("the State's FMAP", False),
        ("the State's own standard or level", False),
        # A federal actor or a federal formula.
        ("the amount established by the Secretary", False),
        ("the standard set by the Social Security Administration", False),
        ("the rate that a Federal agency establishes", False),
        ("the income standard used to determine eligibility for SSI", False),
        ("the allotment determined under the formula in section 2104(b)", False),
        # Words that only start like a listed verb.
        ("the State election under paragraph (b)", False),
        ("the agency settlement with the provider", False),
        ("the State electronic verification plan", False),
    ],
)
def test_actor_phrase_is_a_closed_list(excerpt, expected):
    assert script.actor_sets_value(excerpt, agency_is_state=True) is expected


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        ("the standard utility allowance set by the State agency", True),
        ("the standard set by the State", True),
        ("the standard established by the agency", False),
        ("the agency may specify a standard", False),
    ],
)
def test_bare_agency_is_an_actor_only_where_it_is_the_state_agency(excerpt, expected):
    assert script.actor_sets_value(excerpt, agency_is_state=False) is expected


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        ("the State's AFDC payment standard in effect as of July 16, 1996", True),
        ("a State's AFDC income standard in effect as of May 1, 1988", True),
        (
            "the effective income level under the Medicaid State plan or waiver "
            "of the State plan as of March 23, 2010 or December 31, 2013",
            True,
        ),
        ("the level under the State plan in effect on July 1, 1997", True),
        # A plan anchor with no date, a date with no plan anchor, a possessive.
        ("the State's AFDC payment standard", False),
        ("the Federal benefit rate in effect as of July 16, 1996", False),
        ("the State's allotment under § 457.608 as of October 1, 2009", False),
        ("the State's FMAP in effect on July 1, 2008", False),
        ("the State optional supplement standard", False),
        ("section 1115 of the Act", False),
    ],
)
def test_plan_level_needs_a_state_plan_anchor_and_a_date(excerpt, expected):
    assert script.plan_level_on_a_date(excerpt) is expected


@pytest.mark.parametrize(
    ("source", "trigger"),
    [
        ("(a) A reasonable amount set by the agency applies here.\n", "actor"),
        (
            "(a) The State's AFDC payment standard in effect as of July 16, 1996.\n",
            "plan-level",
        ),
        (
            "(a) The level under the Medicaid State plan as of March 23, 2010.\n",
            "plan-level",
        ),
        ("(a) The State optional supplement standard applies.\n", "-"),
        ("(a) The standard established under § 435.811 applies.\n", "-"),
        ("(a) The State's allotment under § 457.608 applies.\n", "-"),
    ],
)
def test_each_trigger_kind(root, tmp_path, source, trigger):
    excerpt = source.split(") ", 1)[1].strip().rstrip(".")
    module = _state_value(tmp_path, excerpt)
    rows, failures = script.check(root, source, module, _baseline(tmp_path))
    assert f"| {trigger} |" in _row(rows, "state_value")
    assert ("trigger" in _kinds(failures)) is (trigger == "-")


def test_established_under_a_section_alone_is_no_trigger(root, tmp_path):
    """Former case (b): the phrase names no actor, so it is not accepted."""
    module = _module(
        tmp_path,
        excerpt="medically needy income standard established under § 435.811",
        formula="min(need, medically_needy_income_standard_amount)",
    )
    rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert "| - |" in _row(rows, "medically_needy_income_standard_amount")
    assert _kinds(failures) == ["trigger"]


def test_ssi_income_standard_fails_by_the_positive_condition(root, tmp_path):
    """Owner decision: the SSI standard is not State-set. No word list rejects
    it; the source names no State actor for it."""
    module = _module(
        tmp_path,
        name="ssi_home_income_standard",
        description="Income standard under 42 CFR 435.725(c)(2)(i).",
        excerpt="The income standard used to determine eligibility for SSI applies",
        formula="min(need, ssi_home_income_standard)",
    )
    rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert "| - |" in _row(rows, "ssi_home_income_standard")
    assert _kinds(failures) == ["trigger"]


@pytest.mark.parametrize(
    ("name", "description"),
    [
        ("home_income_standard", "SSI income standard, 42 CFR 435.725."),
        ("home_income_standard", "Federal benefit rate under 42 CFR 435.725."),
        ("fbr_amount", "Amount under 42 CFR 435.725."),
    ],
)
def test_federal_amount_fails_whatever_its_name_or_description(
    root, tmp_path, name, description
):
    module = _module(
        tmp_path,
        name=name,
        description=description,
        excerpt="The income standard used to determine eligibility for SSI applies",
        formula=f"min(need, {name})",
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert _kinds(failures) == ["trigger"]


def test_no_negative_word_list_decides_acceptance(root, tmp_path):
    """The name and description do not decide: only the source text does."""
    module = _module(
        tmp_path,
        name="state_ssi_supplement_standard",
        description="Standard the agency establishes under 42 CFR 435.811.",
        formula="max(30, min(need, state_ssi_supplement_standard, 1.33 * fpl))",
    )
    rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert failures == []
    assert "| actor |" in _row(rows, "state_ssi_supplement_standard")
    assert not hasattr(script, "_FEDERAL_DESCRIPTION")
    assert not hasattr(script, "_FEDERAL_NAME_TOKENS")


def test_snap_maximum_allotment_is_not_an_input(root, tmp_path):
    """7 CFR 273: a federal amount "established under § 273.10(e)(4)"."""
    module = _module(
        tmp_path,
        name="maximum_allotment_amount",
        description="Maximum allotment under 7 CFR 273.10(e)(4).",
        excerpt="the maximum allotment established under § 273.10(e)(4)",
        formula="min(need, maximum_allotment_amount)",
        unit=SNAP_UNIT,
    )
    rows, failures = script.check(root, SNAP_SOURCE, module, _baseline(tmp_path))
    assert "| - |" in _row(rows, "maximum_allotment_amount")
    assert _kinds(failures) == ["scope"]
    assert "outside the allow-list (42 CFR part 435)" in failures[0]


# --- finding 4: only verbatim source text can satisfy the trigger ---


def test_fabricated_excerpt_does_not_pass(root, tmp_path):
    module = _module(
        tmp_path,
        name="federal_chip_allotment_amount",
        description="Allotment set by the State under 42 CFR 435.811.",
        excerpt="set by the State",
        formula="min(need, federal_chip_allotment_amount)",
    )
    rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert "| - |" in _row(rows, "federal_chip_allotment_amount")
    assert _kinds(failures) == ["excerpt", "trigger"]
    assert (
        "excerpt: a proof excerpt cited for `federal_chip_allotment_amount` is "
        'not verbatim in the unit source: "set by the State"' in failures
    )


def test_fabricated_excerpt_fails_the_command(root, tmp_path, source_path, capsys):
    module = _module(
        tmp_path,
        name="federal_chip_allotment_amount",
        excerpt="set by the State",
        formula="min(need, federal_chip_allotment_amount)",
    )
    assert _main(root, source_path, module, _baseline(tmp_path)) == 1
    assert "FAIL excerpt:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "description",
    [
        "Standard established by the agency in its State plan, 42 CFR 435.811.",
        "Level under the Medicaid State plan as of March 23, 2010, § 435.811.",
        "The State sets it under 42 CFR 435.811.",
    ],
)
def test_description_alone_never_satisfies_the_trigger(root, tmp_path, description):
    module = _state_value(
        tmp_path,
        "(d) Other text with $999 that no excerpt reaches.",
        description=description,
    )
    module.write_text(module.read_text().replace("min(need,", "min(999, need,"))
    rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert "| - |" in _row(rows, "state_value")
    assert _kinds(failures) == ["trigger"]


def test_excerpt_match_is_whitespace_normalised_only(root, tmp_path):
    source = "(a) A reasonable amount\n   set by   the agency applies here.\n"
    rows, failures = script.check(
        root,
        source,
        _state_value(tmp_path, "A reasonable amount set by the agency"),
        _baseline(tmp_path),
    )
    assert "| actor |" in _row(rows, "state_value") and failures == []
    rows, failures = script.check(
        root,
        source,
        _state_value(tmp_path, "A reasonable amount is set by the agency"),
        _baseline(tmp_path),
    )
    assert "| - |" in _row(rows, "state_value")
    assert _kinds(failures) == ["excerpt", "trigger"]


def test_a_verified_excerpt_does_not_cover_a_fabricated_one(root, tmp_path):
    module = _module(tmp_path)
    text = module.read_text().replace(
        "    versions:\n      - effective_from: '2014-01-01'\n        formula: 'max",
        "          - path: versions[0].formula\n            kind: formula\n"
        "            source:\n"
        f"              corpus_citation_path: {UNIT}\n"
        '              excerpt: "the allotment the State establishes"\n'
        "    versions:\n      - effective_from: '2014-01-01'\n        formula: 'max",
    )
    assert text != module.read_text()
    module.write_text(text)
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert _kinds(failures) == ["excerpt"]


# --- finding 5: the tool fails closed ---


def _error(capsys):
    captured = capsys.readouterr()
    assert "failures: 0" not in captured.out
    return captured.err


def test_missing_module_is_an_error(root, tmp_path, source_path, capsys):
    assert _main(root, source_path, tmp_path / "absent.yaml") == 2
    assert "ERROR module not found" in _error(capsys)


def test_unparsable_module_is_an_error(root, tmp_path, source_path, capsys):
    module = tmp_path / "candidate.yaml"
    module.write_text("rules: [unclosed\n")
    assert _main(root, source_path, module) == 2
    assert "ERROR module is not parsable YAML" in _error(capsys)


def test_module_that_is_no_mapping_is_an_error(root, tmp_path, source_path, capsys):
    module = tmp_path / "candidate.yaml"
    module.write_text("- a\n- b\n")
    assert _main(root, source_path, module) == 2
    assert "ERROR module is not a YAML mapping" in _error(capsys)


def test_unparsable_yaml_under_the_root_is_an_error(
    root, tmp_path, source_path, capsys
):
    (root / "regulations" / "42-cfr" / "435" / "broken.yaml").write_text(
        "rules:\n  - name: [unclosed\n"
    )
    assert _main(root, source_path, _module(tmp_path), _baseline(tmp_path)) == 2
    err = _error(capsys)
    assert "ERROR RuleSpec root file is not parsable YAML" in err
    assert "broken.yaml" in err


def test_nonexistent_rulespec_root_is_an_error(tmp_path, source_path, capsys):
    code = _main(
        tmp_path / "no-such-root", source_path, _module(tmp_path), _baseline(tmp_path)
    )
    assert code == 2
    assert "ERROR RuleSpec root not found" in _error(capsys)


def test_rulespec_root_without_a_module_is_an_error(tmp_path, source_path, capsys):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    assert _main(empty, source_path, _module(tmp_path), _baseline(tmp_path)) == 2
    assert "ERROR RuleSpec root holds no module" in _error(capsys)


def test_missing_source_is_an_error(root, tmp_path, capsys):
    assert _main(root, tmp_path / "absent.txt", _module(tmp_path)) == 2
    assert "ERROR unit source not found" in _error(capsys)


def test_empty_source_is_an_error(root, tmp_path, capsys):
    source = tmp_path / "empty.txt"
    source.write_text(" \n")
    assert _main(root, source, _module(tmp_path)) == 2
    assert "ERROR unit source text is empty" in _error(capsys)


def test_missing_baseline_is_an_error(root, tmp_path, source_path, capsys):
    code = _main(root, source_path, _module(tmp_path), tmp_path / "absent.yaml")
    assert code == 2
    assert "ERROR baseline not found" in _error(capsys)


# --- finding 6: the tool enforces the allow-list ---


def test_module_without_a_corpus_citation_path_is_an_error(
    root, tmp_path, source_path, capsys
):
    module = _module(tmp_path, unit=None)
    assert _main(root, source_path, module, _baseline(tmp_path)) == 2
    assert "corpus_citation_path" in _error(capsys)
    with pytest.raises(script.ToolError, match="corpus_citation_path"):
        script.check(root, SOURCE, module, _baseline(tmp_path))


@pytest.mark.parametrize(
    "unit",
    [
        SNAP_UNIT,
        "us/regulation/42/436/831",
        "us/regulation/42/457/608",
        "us/statute/42/1396a",
        "us-az/manual/des/faa5/dependent-care-expense/block-4",
    ],
)
def test_state_actor_excerpt_outside_the_allow_list_fails(root, tmp_path, unit):
    """Even a true State actor phrase is not accepted where the rule is not
    emitted."""
    module = _module(
        tmp_path,
        name="standard_utility_allowance_amount",
        description="Allowance under 7 CFR 273.9(d)(6).",
        excerpt="The standard utility allowance set by the State agency applies",
        formula="min(need, standard_utility_allowance_amount)",
        unit=unit,
    )
    rows, failures = script.check(root, SNAP_SOURCE, module, _baseline(tmp_path))
    assert "| - |" in _row(rows, "standard_utility_allowance_amount")
    assert "scope" in _kinds(failures)


def test_unit_outside_the_allow_list_with_no_new_input_passes(root, tmp_path):
    module = _module(tmp_path, root_input=False, formula="need + fpl", unit=SNAP_UNIT)
    assert script.check(root, SNAP_SOURCE, module, _baseline(tmp_path)) == ([], [])


def test_tool_uses_the_prompt_allow_list(root, tmp_path, monkeypatch):
    monkeypatch.setattr(
        script.encoder_prompt, "CFR_STATE_SET_AMOUNT_PARTS", frozenset({("7", "273")})
    )
    monkeypatch.setattr(
        script.encoder_prompt, "CFR_PARTS_WHERE_AGENCY_IS_THE_STATE", frozenset()
    )
    _rows, failures = script.check(root, SOURCE, _module(tmp_path), _baseline(tmp_path))
    assert _kinds(failures) == ["scope"]
    assert script.state_set_trigger("set by the State agency", ("7", "273", "9"))
    assert not script.state_set_trigger("set by the agency", ("7", "273", "9"))
    assert not script.state_set_trigger("set by the State agency", None)


# --- finding 7: the delegation record ---


def test_accepted_input_needs_a_delegates_source_relation(root, tmp_path):
    _rows, failures = script.check(
        root, SOURCE, _module(tmp_path, relation=""), _baseline(tmp_path)
    )
    assert _kinds(failures) == ["relation"]


@pytest.mark.parametrize("relation_type", ["sets", "implements"])
def test_sets_or_implements_is_not_the_delegation_record(root, tmp_path, relation_type):
    relation = DELEGATES.replace("type: delegates", f"type: {relation_type}")
    _rows, failures = script.check(
        root, SOURCE, _module(tmp_path, relation=relation), _baseline(tmp_path)
    )
    assert _kinds(failures) == ["relation"]
