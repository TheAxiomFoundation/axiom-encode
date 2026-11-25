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
    "(e) The amount may not exceed the medically needy income standard "
    "established under § 435.811.\n\n"
    "(f) The income standard used to determine eligibility for SSI applies.\n"
)
EXCERPT = "medically needy income standard established by the agency under § 435.811"
NO_ACTOR_EXCERPT = "medically needy income standard established under § 435.811"
SSI_EXCERPT = "The income standard used to determine eligibility for SSI applies"
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


def _formula(*names: str) -> str:
    """A formula that uses the inputs and holds the literals of SOURCE."""
    return f"max(30, min(need, {', '.join(names)}, 1.33 * fpl))"


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
    formula = formula or _formula(name)
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
        "      - effective_from: '2014-01-01'\n        formula: need + fpl + income\n"
    )
    return path


def _row(rows, name):
    return next(row for row in rows if f"| {name} |" in row)


def _kinds(failures):
    return sorted({failure.split(":", 1)[0] for failure in failures})


def _one(tmp_path, name, excerpt, *, formula=None, **kwargs):
    """A candidate with one new input and one excerpt."""
    kwargs.setdefault("description", "Value under 42 CFR 435.811.")
    return _module(
        tmp_path,
        name=name,
        excerpt=excerpt,
        formula=formula or f"min(need, {name})",
        **kwargs,
    )


def _check(root, tmp_path, module, source=SOURCE, unit=UNIT):
    return script.check(root, unit, source, module, _baseline(tmp_path))


def _main(root, source_path, module, baseline=None, unit=UNIT):
    argv = [
        "--rulespec-root",
        str(root),
        "--unit",
        unit,
        "--source",
        str(source_path),
        "--module",
        str(module),
    ]
    if baseline is not None:
        argv += ["--baseline", str(baseline)]
    return script.main(argv)


def _rule(name: str, formula: str, excerpts: list[str]) -> str:
    atoms = "".join(
        "          - path: versions[0].formula\n            kind: formula\n"
        "            source:\n"
        f"              corpus_citation_path: {UNIT}\n"
        f'              excerpt: "{excerpt}"\n'
        for excerpt in excerpts
    )
    return (
        f"  - name: {name}\n    kind: derived\n    entity: Person\n"
        "    dtype: Money\n    unit: USD\n    period: Month\n"
        "    metadata:\n      proof:\n        atoms:\n" + atoms + "    versions:\n"
        f"      - effective_from: '2014-01-01'\n        formula: '{formula}'\n"
    )


def _multi(tmp_path: Path, names: list[str], rules: list[str]) -> Path:
    """A candidate with several root inputs and hand-written rules."""
    inputs = "".join(
        f"  - name: {name}\n    entity: Person\n    dtype: Money\n    unit: USD\n"
        "    period: Month\n    description: 'Value under 42 CFR 435.811.'\n"
        for name in names
    )
    path = tmp_path / "candidate.yaml"
    path.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {UNIT}\n"
        f"inputs:\n{inputs}rules:\n{DELEGATES}" + "".join(rules)
    )
    return path


@pytest.fixture
def source_path(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text(SOURCE)
    return path


# --- the accepted case and the checks that do not depend on the trigger ---


def test_state_actor_in_a_verified_excerpt_passes(root, tmp_path):
    rows, failures = _check(root, tmp_path, _module(tmp_path))
    assert failures == []
    assert len(rows) == 1
    assert (
        "| medically_needy_income_standard_amount | root | Person/Money/Month "
        "| actor |" in rows[0]
    )


def test_baseline_inputs_are_not_new(root, tmp_path):
    rows, _failures = script.check(root, UNIT, SOURCE, _module(tmp_path), None)
    assert {row.split(" | ")[1] for row in rows} == {
        "medically_needy_income_standard_amount",
        "need",
        "fpl",
    }


def test_input_without_description_fails(root, tmp_path):
    _rows, failures = _check(root, tmp_path, _module(tmp_path, description=None))
    assert failures == [
        "description: `medically_needy_income_standard_amount` has no description "
        "naming a provision"
    ]


def test_description_without_provision_fails(root, tmp_path):
    _rows, failures = _check(
        root, tmp_path, _module(tmp_path, description="The State standard.")
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
    _rows, failures = _check(root, tmp_path, _module(tmp_path, description=description))
    assert any(f.startswith("description:") for f in failures) is (not ok)


def test_implicit_identifier_is_an_input_and_fails_description(root, tmp_path):
    rows, failures = _check(root, tmp_path, _module(tmp_path, root_input=False))
    assert "| medically_needy_income_standard_amount | implicit | -/-/- |" in rows[0]
    assert failures == [
        "description: `medically_needy_income_standard_amount` has no description "
        "naming a provision",
        "value: `medically_needy_income_standard_amount` is not a number; a "
        "State-set value is",
    ]


def test_exported_name_fails(root, tmp_path):
    source = "(a) The separate floor established by the agency applies.\n"
    module = _one(
        tmp_path, "separate_floor", "separate floor established by the agency"
    )
    _rows, failures = _check(root, tmp_path, module, source)
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
    _rows, failures = _check(root, tmp_path, module)
    assert (
        "value: `agency_elects_option` is not a number; a State-set value is"
        in failures
    )


def test_formula_function_names_are_not_inputs(root, tmp_path):
    module = _module(
        tmp_path,
        formula="count_where(need > 0) + medically_needy_income_standard_amount + 30 + 1.33",
    )
    rows, _ = script.check(root, UNIT, SOURCE, module, None)
    names = {row.split(" | ")[1] for row in rows}
    assert "count_where" not in names
    assert "need" in names


def test_main_exit_code_and_table(root, tmp_path, source_path, capsys):
    module = _module(tmp_path)
    assert _main(root, source_path, module, _baseline(tmp_path)) == 0
    assert "new inputs: 1; failures: 0" in capsys.readouterr().out
    # Without a baseline `need` and `fpl` are new inputs too, and fail.
    assert _main(root, source_path, module) == 1


# --- the closed shapes: which noun phrase the State verb acts on ---


@pytest.mark.parametrize(
    ("excerpt", "phrases"),
    [
        (
            "Income at or below the income standard established by the agency "
            "in its State plan",
            [{"income", "standard"}],
        ),
        ("the standard set by the State agency", [{"standard"}]),
        ("the level specified by the State", [{"level"}]),
        ("the agency may specify a standard", [{"standard"}]),
        (
            "The agency must establish in its State plan the income standard "
            "as follows",
            [{"income", "standard"}],
        ),
        ("a higher standard the State has chosen", [{"higher", "standard"}]),
        (
            "The amount of the medically needy income standard for one person "
            "established by the agency under § 435.811",
            [{"medically", "needy", "income", "standard"}],
        ),
        (
            "a special income level the agency has established under § 435.236, "
            "FFP is available only if their income does not exceed 300 percent "
            "of the SSI benefit amount payable under section 1611(b)(1)",
            [{"special", "income", "level"}],
        ),
        ("the amount the agency determines under its State plan", [{"amount"}]),
        ("the amount determined by the agency in its State plan", [{"amount"}]),
        # A share of another quantity: only the State-chosen parameter binds.
        (
            "a percentage of the Federal poverty level that the State elects",
            [{"percentage"}],
        ),
        (
            "the State elects a percentage of the Federal poverty level",
            [{"percentage"}],
        ),
        (
            "the agency sets the standard at 300 percent of the SSI benefit amount",
            [{"standard"}],
        ),
        (
            "300 percent of the SSI benefit amount established by the agency",
            [{"300", "percent"}],
        ),
        # The verb has no noun phrase as its object.
        ("The agency may elect", []),
        ("if the State elects to cover", []),
        ("The agency may elect to disregard the Federal poverty level increase", []),
        (
            "The agency determines whether income exceeds the SSI Federal "
            "benefit rate for individuals covered under the State plan",
            [],
        ),
        (
            "The agency must determine financial eligibility using the Federal "
            "poverty level published by the Secretary for groups covered in "
            "its State plan",
            [],
        ),
        ("the agency determines the amount of the individual's income", []),
        ("the amount determined by the agency", []),
        (
            "The agency may not establish a standard other than the SSI "
            "Federal benefit rate",
            [],
        ),
        (
            "The amount established by the Secretary applies. The State agency "
            "must set aside records",
            [],
        ),
        # No State actor of a verb of the closed list.
        ("the State plan specifies the standard", []),
        ("the standard established by the State plan", []),
        ("the amount set by the agency plan", []),
        ("as specified under section 1902(a)(10) of the Act", []),
        ("the agency must furnish Medicaid as specified in paragraph (b)", []),
        ("the maximum allotment established under § 273.10(e)(4)", []),
        ("the State's allotment under § 457.608", []),
        ("the State's FMAP", []),
        ("the State's own standard or level", []),
        ("the amount established by the Secretary", []),
        ("the standard set by the Social Security Administration", []),
        ("the rate that a Federal agency establishes", []),
        ("the benefit rate the Social Security agency establishes", []),
        ("the income standard used to determine eligibility for SSI", []),
        ("the State election under paragraph (b)", []),
        ("the agency settlement with the provider", []),
        ("each State set of rules", []),
    ],
)
def test_actor_amount_phrases(excerpt, phrases):
    found = script.actor_amount_phrases(excerpt, agency_is_state=True)
    assert found == [frozenset(phrase) for phrase in phrases]


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
    assert bool(script.actor_amount_phrases(excerpt, agency_is_state=False)) is expected


@pytest.mark.parametrize(
    ("excerpt", "phrases"),
    [
        (
            "the State's AFDC payment standard in effect as of July 16, 1996",
            [{"state", "afdc", "payment", "standard", "1996"}],
        ),
        (
            "a State's AFDC income standard in effect as of May 1, 1988, "
            "increased by no more than the percentage increase in the Consumer "
            "Price Index for all urban consumers",
            [{"state", "afdc", "income", "standard", "1988"}],
        ),
        (
            "the effective income level for section 1931 families under the "
            "Medicaid State plan or waiver of the State plan as of March 23, "
            "2010 or December 31, 2013",
            [{"effective", "income", "level", "2010"}],
        ),
        (
            "the level under the State plan in effect on July 1, 1997",
            [{"level", "1997"}],
        ),
        # A plan anchor with no date, a date that is not the plan's date.
        ("the State's AFDC payment standard", []),
        ("the level under the Medicaid State plan", []),
        (
            "eligible under the Medicaid State plan whose income does not "
            "exceed the SSI Federal benefit rate in effect on January 1, 2014",
            [],
        ),
        ("the Federal benefit rate in effect as of July 16, 1996", []),
        ("the State's allotment under § 457.608 as of October 1, 2009", []),
        ("the State's FMAP in effect on July 1, 2008", []),
        ("the State optional supplement standard", []),
        ("section 1115 of the Act", []),
    ],
)
def test_plan_level_phrases(excerpt, phrases):
    assert script.plan_level_phrases(excerpt) == [frozenset(p) for p in phrases]


@pytest.mark.parametrize(
    ("name", "excerpt", "trigger"),
    [
        ("income_standard", "the income standard established by the agency", "actor"),
        (
            "state_income_standard_amount",
            "the income standard set by the State",
            "actor",
        ),
        (
            "percentage",
            "a percentage of the Federal poverty level that the State elects",
            "actor",
        ),
        (
            "afdc_payment_standard_1996",
            "State's AFDC payment standard in effect as of July 16, 1996",
            "plan-level",
        ),
        (
            "income_level",
            "the income level under the Medicaid State plan as of March 23, 2010",
            "plan-level",
        ),
        # The name is not the object of the verb.
        (
            "federal_poverty_level",
            "a percentage of the Federal poverty level that the State elects",
            None,
        ),
        ("ssi_income_standard", "the income standard established by the agency", None),
        ("mnis", "the medically needy income standard established by the agency", None),
        (
            "standard",
            "the income level under the Medicaid State plan as of March 23, 2010",
            None,
        ),
        # A name of free words only binds to nothing.
        ("state_amount", "the amount set by the State", None),
    ],
)
def test_trigger_binds_the_input_name_to_the_object_of_the_verb(name, excerpt, trigger):
    assert script.state_set_trigger(name, excerpt, ("42", "435", "733")) == trigger
    assert script.state_set_trigger(name, excerpt, ("42", "436", "733")) is None
    assert script.state_set_trigger(name, excerpt, None) is None


@pytest.mark.parametrize(
    ("source", "name", "trigger"),
    [
        (
            "(a) A reasonable amount set by the agency applies here.\n",
            "reasonable_amount",
            "actor",
        ),
        (
            "(a) The State's AFDC payment standard in effect as of July 16, 1996.\n",
            "afdc_payment_standard",
            "plan-level",
        ),
        (
            "(a) The level under the Medicaid State plan as of March 23, 2010.\n",
            "level_2010",
            "plan-level",
        ),
        (
            "(a) The State optional supplement standard applies.\n",
            "optional_supplement_standard",
            "-",
        ),
        ("(a) The standard established under § 435.811 applies.\n", "standard", "-"),
        ("(a) The State's allotment under § 457.608 applies.\n", "allotment", "-"),
    ],
)
def test_each_trigger_kind(root, tmp_path, source, name, trigger):
    excerpt = source.split(") ", 1)[1].strip().rstrip(".")
    rows, failures = _check(root, tmp_path, _one(tmp_path, name, excerpt), source)
    assert f"| {trigger} |" in _row(rows, name)
    assert _kinds(failures) == ([] if trigger != "-" else ["trigger"])


def test_established_under_a_section_alone_is_no_trigger(root, tmp_path):
    """Former case (b): the phrase names no actor, so it is not accepted."""
    rows, failures = _check(root, tmp_path, _module(tmp_path, excerpt=NO_ACTOR_EXCERPT))
    assert "| - |" in _row(rows, "medically_needy_income_standard_amount")
    assert _kinds(failures) == ["trigger"]


# --- federal amounts inside 42 CFR 435 (review round 6) ---


def test_ssi_income_standard_fails_by_the_positive_condition(root, tmp_path):
    """Owner decision: the SSI standard is not State-set. No word list rejects
    it; the source names no State actor for it."""
    module = _module(
        tmp_path,
        name="ssi_home_income_standard",
        description="Income standard under 42 CFR 435.725(c)(2)(i).",
        excerpt=SSI_EXCERPT,
    )
    rows, failures = _check(root, tmp_path, module)
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
    module = _module(tmp_path, name=name, description=description, excerpt=SSI_EXCERPT)
    _rows, failures = _check(root, tmp_path, module)
    assert _kinds(failures) == ["trigger"]


def test_no_negative_word_list_decides_acceptance(root, tmp_path):
    """The source text decides: an amount it attributes to the State passes
    whatever words its name holds."""
    source = "(a) Income may not exceed the SSI supplement standard set by the State.\n"
    module = _one(
        tmp_path,
        "ssi_supplement_standard",
        "the SSI supplement standard set by the State",
    )
    rows, failures = _check(root, tmp_path, module, source)
    assert failures == []
    assert "| actor |" in _row(rows, "ssi_supplement_standard")
    assert not hasattr(script, "_FEDERAL_DESCRIPTION")
    assert not hasattr(script, "_FEDERAL_NAME_TOKENS")


SOURCE_725 = (
    "(c)(2) Maintenance needs of spouse. This amount must be based on a "
    "reasonable assessment of need but must not exceed the highest of— "
    "(i) The amount of the income standard used to determine eligibility for "
    "SSI for an individual living in his own home; "
    "(ii) The amount of the highest income standard, in the appropriate "
    "category of age, blindness, or disability, used to determine eligibility "
    "for an optional State supplement; or "
    "(iii) The amount of the medically needy income standard for one person "
    "established by the agency under § 435.811.\n\n"
    "(d) The agency may elect to apply this section to individuals in the group.\n"
)
EXCERPT_725 = (
    "medically needy income standard for one person established by the agency "
    "under § 435.811"
)


def test_one_excerpt_does_not_launder_three_inputs(root, tmp_path):
    """435.725(c)(2): the SSI standard and the Federal benefit rate ride on the
    one actor excerpt of the medically needy income standard."""
    names = ["ssi_income_standard", "ssi_federal_benefit_rate", "mnis"]
    module = _multi(
        tmp_path,
        names,
        [_rule("spouse_ceiling", f"max({', '.join(names)})", [EXCERPT_725])],
    )
    rows, failures = _check(root, tmp_path, module, SOURCE_725)
    for name in names:
        assert "| - |" in _row(rows, name)
        assert any(
            f.startswith(f"trigger: no verified source excerpt makes `{name}`")
            for f in failures
        )


def test_the_bound_input_passes_and_its_rule_mates_fail(root, tmp_path):
    names = ["ssi_income_standard", "medically_needy_income_standard"]
    module = _multi(
        tmp_path,
        names,
        [_rule("spouse_ceiling", f"max({', '.join(names)})", [EXCERPT_725])],
    )
    rows, failures = _check(root, tmp_path, module, SOURCE_725)
    assert "| actor |" in _row(rows, "medically_needy_income_standard")
    assert "| - |" in _row(rows, "ssi_income_standard")
    assert _kinds(failures) == ["trigger"] and len(failures) == 1


def test_one_excerpt_serves_one_input(root, tmp_path):
    """Two names that both fit the phrase: the second has no excerpt of its own."""
    names = ["medically_needy_income_standard", "needy_income_standard"]
    module = _multi(
        tmp_path,
        names,
        [_rule("spouse_ceiling", f"max({', '.join(names)})", [EXCERPT_725])],
    )
    rows, failures = _check(root, tmp_path, module, SOURCE_725)
    assert "| actor |" in _row(rows, "medically_needy_income_standard")
    assert "| - |" in _row(rows, "needy_income_standard")
    assert failures == [
        "trigger: the only excerpt that names `needy_income_standard` already "
        "serves `medically_needy_income_standard`; one excerpt serves one input"
    ]


def test_ssi_input_citing_an_unrelated_actor_phrase_fails(root, tmp_path):
    module = _one(tmp_path, "ssi_income_standard", "The agency may elect")
    rows, failures = _check(root, tmp_path, module, SOURCE_725)
    assert "| - |" in _row(rows, "ssi_income_standard")
    assert _kinds(failures) == ["trigger"]


@pytest.mark.parametrize(
    ("source", "name", "formula"),
    [
        # 435.1005: the agency established the special income level, not the
        # SSI benefit amount that caps it.
        (
            "For beneficiaries eligible under a special income level the agency "
            "has established under § 435.236, FFP is available only if their "
            "income does not exceed 300 percent of the SSI benefit amount "
            "payable under section 1611(b)(1) of the Act.\n",
            "ssi_benefit_amount",
            "income <= 3 * ssi_benefit_amount",
        ),
        (
            "(b) The agency determines whether income exceeds the SSI Federal "
            "benefit rate for individuals covered under the State plan.\n",
            "ssi_federal_benefit_rate",
            None,
        ),
        (
            "(c) The agency must determine financial eligibility using the "
            "Federal poverty level published by the Secretary for groups "
            "covered in its State plan.\n",
            "federal_poverty_level",
            None,
        ),
        # 435.110(c)(2): a price index increase beside a dated AFDC standard.
        (
            "(ii) The State's AFDC income standard in effect as of July 16, "
            "1996, increased by no more than the percentage increase in the "
            "Consumer Price Index for all urban consumers since such date.\n",
            "cpi_u_percentage_increase",
            None,
        ),
        (
            "(b) Individuals eligible under the Medicaid State plan whose "
            "income does not exceed the SSI Federal benefit rate in effect on "
            "January 1, 2014.\n",
            "ssi_fbr_2014",
            None,
        ),
    ],
)
def test_federal_quantity_beside_a_state_phrase_is_not_bound(
    root, tmp_path, source, name, formula
):
    excerpt = source.strip().rstrip(".").split(") ", 1)[-1]
    module = _one(tmp_path, name, excerpt, formula=formula)
    rows, failures = _check(root, tmp_path, module, source)
    assert "| - |" in _row(rows, name)
    assert _kinds(failures) == ["trigger"]


def test_the_state_chosen_level_of_the_same_sentence_is_bound(root, tmp_path):
    """435.1005: the special income level is the amount the agency established."""
    source = (
        "For beneficiaries eligible under a special income level the agency has "
        "established under § 435.236, FFP is available only if their income "
        "does not exceed 300 percent of the SSI benefit amount.\n"
    )
    module = _one(
        tmp_path,
        "special_income_level",
        "a special income level the agency has established under § 435.236",
        formula="income <= min(special_income_level, 3 * ssi_benefit)",
    )
    rows, failures = _check(root, tmp_path, module, source)
    assert "| actor |" in _row(rows, "special_income_level")
    # `ssi_benefit` is a second new input with no phrase of its own.
    assert "| - |" in _row(rows, "ssi_benefit")
    assert any("`ssi_benefit`" in f and f.startswith("trigger:") for f in failures)


def test_snap_maximum_allotment_is_not_an_input(root, tmp_path):
    """7 CFR 273: a federal amount "established under § 273.10(e)(4)"."""
    module = _one(
        tmp_path,
        "maximum_allotment",
        "the maximum allotment established under § 273.10(e)(4)",
        description="Maximum allotment under 7 CFR 273.10(e)(4).",
        unit=SNAP_UNIT,
    )
    rows, failures = _check(root, tmp_path, module, SNAP_SOURCE, SNAP_UNIT)
    assert "| - |" in _row(rows, "maximum_allotment")
    assert _kinds(failures) == ["scope"]
    assert "outside the allow-list (42 CFR part 435)" in failures[0]


# --- only verbatim source text can satisfy the trigger ---


def test_fabricated_excerpt_does_not_pass(root, tmp_path):
    module = _module(
        tmp_path,
        name="federal_chip_allotment_amount",
        description="Allotment set by the State under 42 CFR 435.811.",
        excerpt="set by the State",
    )
    rows, failures = _check(root, tmp_path, module)
    assert "| - |" in _row(rows, "federal_chip_allotment_amount")
    assert _kinds(failures) == ["excerpt", "trigger"]
    assert (
        "excerpt: a proof excerpt cited for `federal_chip_allotment_amount` is "
        'not verbatim in the unit source: "set by the State"' in failures
    )


def test_fabricated_object_phrase_does_not_pass(root, tmp_path):
    module = _module(
        tmp_path,
        name="federal_chip_allotment",
        excerpt="the federal CHIP allotment set by the State",
    )
    _rows, failures = _check(root, tmp_path, module)
    assert _kinds(failures) == ["excerpt", "trigger"]


def test_fabricated_excerpt_fails_the_command(root, tmp_path, source_path, capsys):
    module = _module(
        tmp_path, name="federal_chip_allotment_amount", excerpt="set by the State"
    )
    assert _main(root, source_path, module, _baseline(tmp_path)) == 1
    assert "FAIL excerpt:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "description",
    [
        "The medically needy income standard established by the agency, 42 CFR 435.811.",
        "Level under the Medicaid State plan as of March 23, 2010, § 435.811.",
        "The State sets it under 42 CFR 435.811.",
    ],
)
def test_description_alone_never_satisfies_the_trigger(root, tmp_path, description):
    module = _module(tmp_path, excerpt=NO_ACTOR_EXCERPT, description=description)
    rows, failures = _check(root, tmp_path, module)
    assert "| - |" in _row(rows, "medically_needy_income_standard_amount")
    assert _kinds(failures) == ["trigger"]


def test_excerpt_match_is_whitespace_normalised_only(root, tmp_path):
    source = "(a) A reasonable amount\n   set by   the agency applies here.\n"
    rows, failures = _check(
        root,
        tmp_path,
        _one(tmp_path, "reasonable_amount", "A reasonable amount set by the agency"),
        source,
    )
    assert "| actor |" in _row(rows, "reasonable_amount") and failures == []
    rows, failures = _check(
        root,
        tmp_path,
        _one(tmp_path, "reasonable_amount", "A reasonable amount is set by the agency"),
        source,
    )
    assert "| - |" in _row(rows, "reasonable_amount")
    assert _kinds(failures) == ["excerpt", "trigger"]


def test_a_verified_excerpt_does_not_cover_a_fabricated_one(root, tmp_path):
    name = "medically_needy_income_standard_amount"
    module = _multi(
        tmp_path,
        [name],
        [
            _rule(
                "allowance",
                _formula(name),
                [EXCERPT, "the allotment the State establishes"],
            )
        ],
    )
    _rows, failures = _check(root, tmp_path, module)
    assert _kinds(failures) == ["excerpt"]


# --- literals of the unit source (review round 6, finding 3) ---


def test_literal_only_in_summary_or_excerpt_fails(root, tmp_path):
    module = _module(
        tmp_path,
        formula="min(need, medically_needy_income_standard_amount)",
        summary="At least $30 and 133 percent of the Federal poverty level.",
    )
    _rows, failures = _check(root, tmp_path, module)
    assert failures == [
        "literal: $30 of the unit source is in no formula",
        "literal: 133 percent of the unit source is in no formula",
    ]


def test_literal_in_a_neighbour_paragraph_is_required(root, tmp_path):
    source = (
        "(c) The allowance must be at least $30.\n\n"
        "(d) It must not exceed the standard established by the agency in its "
        "State plan.\n"
    )
    excerpt = "the standard established by the agency in its State plan"
    _rows, failures = _check(
        root, tmp_path, _one(tmp_path, "standard", excerpt), source
    )
    assert failures == ["literal: $30 of the unit source is in no formula"]
    module = _one(tmp_path, "standard", excerpt, formula="max(30, min(need, standard))")
    assert _check(root, tmp_path, module, source)[1] == []


@pytest.mark.parametrize(
    ("wording", "formula_term", "ok"),
    [
        ("133 percent FPL", "1.33 * fpl", True),
        ("133 percent FPL", "133 * fpl / 100", True),
        ("133 percent FPL", "fpl", False),
        ("133 percent FPL", "1133 * fpl", False),
        ("133 1/3 percent of the payment standard", "4 / 3 * fpl", True),
        ("133 1/3 percent of the payment standard", "1.3333 * fpl", True),
        ("133 1/3 percent of the payment standard", "1.33 * fpl", False),
        ("133 1/3 percent of the payment standard", "fpl", False),
        ("300 percent of the SSI benefit amount", "3 * fpl", True),
        ("300 percent of the SSI benefit amount", "fpl", False),
        ("$1,500.50", "1500.50 + fpl", True),
        ("$1,500.50", "1500 + fpl", False),
    ],
)
def test_part_435_literal_wordings(root, tmp_path, wording, formula_term, ok):
    source = (
        f"(c) The higher of {wording} and the standard established by the "
        "agency in its State plan.\n"
    )
    module = _one(
        tmp_path,
        "standard",
        "the standard established by the agency in its State plan",
        formula=f"max({formula_term}, standard)",
    )
    _rows, failures = _check(root, tmp_path, module, source)
    assert _kinds(failures) == ([] if ok else ["literal"])


# --- the tool fails closed ---


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


@pytest.mark.parametrize(
    ("body", "key"),
    [
        ("inputs: nope\nrules: []\n", "inputs"),
        ("rules: nope\n", "rules"),
        ("inputs: nope\nrules: nope\n", "inputs"),
        ("rules:\n  - nope\n", "rules"),
        ("inputs:\n  - nope\nrules: []\n", "inputs"),
    ],
)
def test_inputs_or_rules_of_the_wrong_shape_is_an_error(
    root, tmp_path, source_path, capsys, body, key
):
    module = tmp_path / "candidate.yaml"
    module.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {UNIT}\n" + body
    )
    assert _main(root, source_path, module) == 2
    assert f"ERROR module `{key}` is not a list of mappings" in _error(capsys)


def test_baseline_of_the_wrong_shape_is_an_error(root, tmp_path, source_path, capsys):
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("rules: nope\n")
    assert _main(root, source_path, _module(tmp_path), baseline) == 2
    assert "ERROR baseline `rules` is not a list of mappings" in _error(capsys)


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


# --- the tool enforces the allow-list on the caller's unit ---


def test_unit_is_a_required_argument(root, tmp_path, source_path):
    with pytest.raises(SystemExit) as exit_info:
        script.main(
            [
                "--rulespec-root",
                str(root),
                "--source",
                str(source_path),
                "--module",
                str(_module(tmp_path)),
            ]
        )
    assert exit_info.value.code == 2


def test_module_without_a_corpus_citation_path_is_an_error(
    root, tmp_path, source_path, capsys
):
    module = _module(tmp_path, unit=None)
    assert _main(root, source_path, module, _baseline(tmp_path)) == 2
    assert "corpus_citation_path" in _error(capsys)
    with pytest.raises(script.ToolError, match="corpus_citation_path"):
        _check(root, tmp_path, module)


def test_module_cannot_declare_itself_into_scope(root, tmp_path, capsys):
    """A SNAP unit whose module claims a 42 CFR 435 path does not pass."""
    source = tmp_path / "snap.txt"
    source.write_text(SNAP_SOURCE)
    module = _one(
        tmp_path,
        "standard_utility_allowance",
        "The standard utility allowance set by the State agency",
        unit=UNIT,
    )
    assert script.check(root, UNIT, SNAP_SOURCE, module, _baseline(tmp_path))[1] == []
    assert _main(root, source, module, _baseline(tmp_path), unit=SNAP_UNIT) == 2
    assert "is not the unit `us/regulation/7/273/9`" in _error(capsys)


@pytest.mark.parametrize(
    "unit",
    [
        SNAP_UNIT,
        "us/regulation/42/436/831",
        "us/regulation/42/457/608",
        "us/statute/42/1396a",
        "us-az/manual/des/faa5/dependent-care-expense/block-4",
        "us-ca/regulation/42/435/733",
    ],
)
def test_state_actor_excerpt_outside_the_allow_list_fails(root, tmp_path, unit):
    """Even a true State actor phrase is not accepted where the rule is not
    emitted."""
    module = _one(
        tmp_path,
        "standard_utility_allowance",
        "The standard utility allowance set by the State agency",
        description="Allowance under 7 CFR 273.9(d)(6).",
        unit=unit,
    )
    rows, failures = _check(root, tmp_path, module, SNAP_SOURCE, unit)
    assert "| - |" in _row(rows, "standard_utility_allowance")
    assert _kinds(failures) == ["scope"]


def test_subsection_unit_of_an_allow_listed_part_is_in_scope(root, tmp_path):
    unit = "us/regulation/42/435/733/c/2"
    rows, failures = _check(root, tmp_path, _module(tmp_path, unit=unit), SOURCE, unit)
    assert failures == [] and "| actor |" in rows[0]


def test_unit_outside_the_allow_list_with_no_new_input_passes(root, tmp_path):
    module = _module(tmp_path, root_input=False, formula="need + fpl", unit=SNAP_UNIT)
    assert _check(root, tmp_path, module, SNAP_SOURCE, SNAP_UNIT) == ([], [])


def test_tool_uses_the_prompt_allow_list(root, tmp_path, monkeypatch):
    monkeypatch.setattr(
        script.encoder_prompt, "CFR_STATE_SET_AMOUNT_PARTS", frozenset({("7", "273")})
    )
    monkeypatch.setattr(
        script.encoder_prompt, "CFR_PARTS_WHERE_AGENCY_IS_THE_STATE", frozenset()
    )
    _rows, failures = _check(root, tmp_path, _module(tmp_path))
    assert _kinds(failures) == ["scope"]
    snap = ("7", "273", "9")
    assert script.state_set_trigger(
        "allowance", "the allowance set by the State agency", snap
    )
    assert not script.state_set_trigger(
        "allowance", "the allowance set by the agency", snap
    )


# --- the delegation record ---


def test_accepted_input_needs_a_delegates_source_relation(root, tmp_path):
    _rows, failures = _check(root, tmp_path, _module(tmp_path, relation=""))
    assert _kinds(failures) == ["relation"]


@pytest.mark.parametrize("relation_type", ["sets", "implements"])
def test_sets_or_implements_is_not_the_delegation_record(root, tmp_path, relation_type):
    relation = DELEGATES.replace("type: delegates", f"type: {relation_type}")
    _rows, failures = _check(root, tmp_path, _module(tmp_path, relation=relation))
    assert _kinds(failures) == ["relation"]
