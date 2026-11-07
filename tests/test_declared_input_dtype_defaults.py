"""Generated test inputs take the dtype their module declares (axiom-encode#2052).

The encoder writes test input values in three places: the cases it generates,
the repair that rewrites wrong-typed values, and the repairs that add missing
inputs. Each one guessed an input's type from its name and from how formulas
use it. A Money input named `unauthorized_transfers_before_notice` has neither
a numeric word in its name nor an operator next to it in `min(cap, ...)`, so
the guess said boolean: generated cases gave it `false`, and the wrong-typed
repair then rewrote a model's correct `100` to `false`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import (
    _DECLARED_INPUT_KIND_DEFAULTS,
    _declared_input_default_resolver,
    _declared_input_dtype_kind,
    _default_generated_test_input_value,
    _default_refs_for_missing_input,
    _factual_input_appears_numeric,
    _import_base_to_repo_file,
    _infer_missing_input_default,
    _missing_input_default,
    _wrong_typed_test_input_replacement,
)
from axiom_encode.harness.dependency_stubs import UnsafeRulespecContextPath

BOOLEAN_DTYPES = ("Boolean", "Bool", "boolean", "bool", "Judgment", "judgment")
NUMERIC_DTYPES = (
    "Money",
    "money",
    "Rate",
    "rate",
    "Decimal",
    "Integer",
    "int",
    "Count",
    "Float",
)
TEXT_DTYPES = ("String", "Text", "text")
UNGROUPED_DTYPES = ("Date", "Period", "")
DTYPE_KINDS = {
    **{dtype: "boolean" for dtype in BOOLEAN_DTYPES},
    **{dtype: "numeric" for dtype in NUMERIC_DTYPES},
    **{dtype: "text" for dtype in TEXT_DTYPES},
}

# The rejected candidate from a 12 CFR 1005.6 encode: Money inputs whose names
# carry no numeric word, used inside `min(...)` and as bare branch values.
REG_E_LIABILITY_MODULE = {
    "format": "rulespec/v1",
    "inputs": [
        {"name": "consumer_notified_within_two_business_days", "dtype": "Boolean"},
        {"name": "unauthorized_transfers_before_notice", "dtype": "Money"},
        {
            "name": "unauthorized_transfers_after_two_business_days_before_notice",
            "dtype": "Money",
        },
        {
            "name": "institution_establishes_later_transfers_preventable",
            "dtype": "Boolean",
        },
        {"name": "state_law_or_agreement_imposes_less_liability", "dtype": "Boolean"},
        {"name": "liability_under_this_section", "dtype": "Money"},
        {"name": "state_law_or_agreement_liability_amount", "dtype": "Money"},
    ],
    "rules": [
        {
            "name": "timely_notice_liability",
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "formula": (
                        "if consumer_notified_within_two_business_days:\n"
                        "  min(timely_notice_liability_cap, "
                        "unauthorized_transfers_before_notice)\n"
                        "else: 0"
                    )
                }
            ],
        },
        {
            "name": "untimely_notice_later_liability",
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "formula": (
                        "if institution_establishes_later_transfers_preventable:\n"
                        "  unauthorized_transfers_after_two_business_days_before_notice\n"
                        "else: 0"
                    )
                }
            ],
        },
        {
            "name": "state_law_or_agreement_liability",
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "formula": (
                        "if state_law_or_agreement_imposes_less_liability: "
                        "min(liability_under_this_section, "
                        "state_law_or_agreement_liability_amount) "
                        "else: liability_under_this_section"
                    )
                }
            ],
        },
    ],
}
REG_E_MONEY_INPUTS = (
    "unauthorized_transfers_before_notice",
    "unauthorized_transfers_after_two_business_days_before_notice",
    "liability_under_this_section",
    "state_law_or_agreement_liability_amount",
)


def _module(*, inputs=None, formula="0", rule_dtype="Money"):
    payload = {
        "format": "rulespec/v1",
        "rules": [
            {
                "name": "principal_output",
                "kind": "derived",
                "dtype": rule_dtype,
                "versions": [{"formula": formula}],
            }
        ],
    }
    if inputs is not None:
        payload["inputs"] = inputs
    return payload


def _same_value(left, right) -> bool:
    """Equal, and of the same type, so `False` never passes for `0`."""
    return type(left) is type(right) and left == right


def _matches_kind(value, kind: str) -> bool:
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "numeric":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


class TestDeclaredInputDtypeKind:
    @pytest.mark.parametrize("dtype", sorted(DTYPE_KINDS))
    def test_each_declared_dtype_maps_to_its_group(self, dtype):
        payload = _module(inputs=[{"name": "fact", "dtype": dtype}])

        assert (
            _declared_input_dtype_kind("fact", rules_payload=payload)
            == (DTYPE_KINDS[dtype])
        )

    @pytest.mark.parametrize("dtype", UNGROUPED_DTYPES)
    def test_ungrouped_dtypes_leave_the_choice_to_the_heuristics(self, dtype):
        payload = _module(inputs=[{"name": "fact", "dtype": dtype}])

        assert _declared_input_dtype_kind("fact", rules_payload=payload) is None

    def test_conflicting_duplicate_declarations_decide_nothing(self):
        payload = _module(
            inputs=[
                {"name": "fact", "dtype": "Money"},
                {"name": "fact", "dtype": "Boolean"},
            ]
        )

        assert _declared_input_dtype_kind("fact", rules_payload=payload) is None

    @pytest.mark.parametrize("inputs", [None, {}, "fact", [None, "fact", 3]])
    def test_malformed_input_lists_decide_nothing(self, inputs):
        payload = _module(inputs=inputs)

        assert _declared_input_dtype_kind("fact", rules_payload=payload) is None


class TestRegEMoneyInputs:
    @pytest.mark.parametrize("input_name", REG_E_MONEY_INPUTS)
    def test_money_inputs_default_to_zero(self, input_name):
        value = _default_generated_test_input_value(
            input_name, rules_payload=REG_E_LIABILITY_MODULE
        )

        assert _same_value(value, 0)
        assert _factual_input_appears_numeric(
            input_name, rules_payload=REG_E_LIABILITY_MODULE
        )

    def test_boolean_inputs_still_default_to_false(self):
        value = _default_generated_test_input_value(
            "consumer_notified_within_two_business_days",
            rules_payload=REG_E_LIABILITY_MODULE,
        )

        assert value is False

    def test_wrong_typed_repair_keeps_a_correct_money_value(self):
        replacement = _wrong_typed_test_input_replacement(
            "unauthorized_transfers_before_notice",
            100,
            rules_payload=REG_E_LIABILITY_MODULE,
        )

        assert replacement is None

    def test_wrong_typed_repair_turns_a_false_money_value_into_zero(self):
        replacement = _wrong_typed_test_input_replacement(
            "liability_under_this_section",
            False,
            rules_payload=REG_E_LIABILITY_MODULE,
        )

        assert _same_value(replacement, 0)


class TestDeclarationBeatsNameHeuristics:
    def test_boolean_looking_name_declared_money_is_numeric(self):
        payload = _module(inputs=[{"name": "is_amount_due", "dtype": "Money"}])

        value = _default_generated_test_input_value(
            "is_amount_due", rules_payload=payload
        )

        assert _same_value(value, 0)

    def test_numeric_looking_name_declared_boolean_is_boolean(self):
        payload = _module(
            inputs=[{"name": "household_income_amount", "dtype": "Boolean"}],
            formula="household_income_amount * 2",
        )

        value = _default_generated_test_input_value(
            "household_income_amount", rules_payload=payload
        )

        assert value is False
        assert not _factual_input_appears_numeric(
            "household_income_amount", rules_payload=payload
        )

    def test_filing_status_declared_string_is_text(self):
        payload = _module(inputs=[{"name": "filing_status", "dtype": "String"}])

        value = _default_generated_test_input_value(
            "filing_status", rules_payload=payload
        )

        assert _same_value(value, "")

    def test_declared_count_takes_one_when_positive_counts_are_preferred(self):
        payload = _module(inputs=[{"name": "household_size", "dtype": "Count"}])

        value = _default_generated_test_input_value(
            "household_size", rules_payload=payload, prefer_positive_counts=True
        )

        assert _same_value(value, 1)


class TestUndeclaredBareNumericValues:
    """Issue #2052's own shape: modules with no `inputs` list."""

    def test_rate_rule_branch_values_are_numeric(self):
        payload = _module(
            rule_dtype="Rate",
            formula=(
                "if indexation_freeze_applies == 1:\n"
                "    pre_2025_index_factor\n"
                "else:\n"
                "    current_index_factor"
            ),
        )

        for input_name in ("pre_2025_index_factor", "current_index_factor"):
            value = _default_generated_test_input_value(
                input_name, rules_payload=payload
            )
            assert _same_value(value, 0), input_name

    def test_single_line_branch_values_are_numeric_and_the_condition_is_not(self):
        payload = _module(
            formula="if notice_was_late: later_transfers else: earlier_transfers"
        )

        assert _same_value(
            _default_generated_test_input_value(
                "later_transfers", rules_payload=payload
            ),
            0,
        )
        assert _same_value(
            _default_generated_test_input_value(
                "earlier_transfers", rules_payload=payload
            ),
            0,
        )
        assert (
            _default_generated_test_input_value(
                "notice_was_late", rules_payload=payload
            )
            is False
        )

    def test_whole_formula_of_a_money_rule_is_numeric(self):
        payload = _module(formula="liability_under_this_section")

        value = _default_generated_test_input_value(
            "liability_under_this_section", rules_payload=payload
        )

        assert _same_value(value, 0)

    def test_min_argument_is_numeric(self):
        payload = _module(formula="min(liability_cap, transfers_before_notice)")

        value = _default_generated_test_input_value(
            "transfers_before_notice", rules_payload=payload
        )

        assert _same_value(value, 0)

    def test_judgment_rule_that_returns_an_input_bare_keeps_it_boolean(self):
        payload = _module(rule_dtype="Judgment", formula="consumer_gave_notice")

        value = _default_generated_test_input_value(
            "consumer_gave_notice", rules_payload=payload
        )

        assert value is False

    def test_a_longer_name_that_starts_with_the_input_is_not_a_match(self):
        # `transfers` appears whole in the condition, so the formula reaches
        # the branch-value check, which must not read `transfers_total` as it.
        payload = _module(
            formula="if flag: transfers_total else: sum_where(members, amount, transfers)"
        )

        value = _default_generated_test_input_value("transfers", rules_payload=payload)

        assert value is False

    def test_an_int_rule_returns_its_branch_values_as_numbers(self):
        # The rules engine reads a rule dtype of `int` as Integer.
        payload = _module(rule_dtype="int", formula="if flag: transfers else: 0")

        value = _default_generated_test_input_value("transfers", rules_payload=payload)

        assert _same_value(value, 0)

    def test_an_elif_condition_inside_a_min_argument_stays_boolean(self):
        payload = _module(
            formula=(
                "min(cap_amount, if a_code == 1: base_amount elif late_flag: 5 else: 0)"
            )
        )

        value = _default_generated_test_input_value("late_flag", rules_payload=payload)

        assert value is False
        assert (
            _wrong_typed_test_input_replacement(
                "late_flag", True, rules_payload=payload
            )
            is None
        )

    def test_a_comment_after_a_colon_is_not_a_branch_value(self):
        payload = _module(
            formula="# cap rule: notice_flag\nsum_where(members, amount, notice_flag)"
        )

        value = _default_generated_test_input_value(
            "notice_flag", rules_payload=payload
        )

        assert value is False

    def test_a_branch_value_with_a_trailing_comment_is_numeric(self):
        payload = _module(formula="if flag: transfers  # before notice\nelse: 0")

        value = _default_generated_test_input_value("transfers", rules_payload=payload)

        assert _same_value(value, 0)

    def test_a_hash_inside_a_string_starts_no_comment(self):
        payload = _module(formula='if code == "a#b": transfers else: 0')

        value = _default_generated_test_input_value("transfers", rules_payload=payload)

        assert _same_value(value, 0)


class TestMissingInputDefaults:
    # Each name's guess differs from the declared value, so the declaration
    # can only win by being read.
    @pytest.mark.parametrize(
        ("declared", "input_name"),
        [
            (False, "household_income_amount"),
            (0, "liability_under_this_section"),
            ("", "liability_under_this_section"),
        ],
    )
    def test_a_falsy_declared_default_still_wins(self, declared, input_name):
        assert not _same_value(_infer_missing_input_default(input_name), declared)

        value = _missing_input_default(
            input_name,
            f"us:regulations/12-cfr/1005/6#input.{input_name}",
            declared_default=lambda _ref: declared,
        )

        assert _same_value(value, declared)

    def test_no_declaration_falls_back_to_the_name_guess(self):
        value = _missing_input_default(
            "household_income_amount",
            "us:statutes/1#input.household_income_amount",
            declared_default=lambda _ref: None,
        )

        assert _same_value(
            value, _infer_missing_input_default("household_income_amount")
        )

    def test_resolver_reads_the_declaration_from_the_repo(self, tmp_path):
        repo = tmp_path / "rulespec-us" / "us"
        module = repo / "regulations/12-cfr/1005/6.yaml"
        module.parent.mkdir(parents=True)
        module.write_text(yaml.safe_dump(REG_E_LIABILITY_MODULE, sort_keys=False))
        lookup = _declared_input_default_resolver(repo_path=repo)

        assert _same_value(
            lookup("us:regulations/12-cfr/1005/6#input.liability_under_this_section"),
            0,
        )
        assert (
            lookup(
                "us:regulations/12-cfr/1005/6#input."
                "consumer_notified_within_two_business_days"
            )
            is False
        )
        assert lookup("us:regulations/12-cfr/1005/6#input.not_declared") is None
        assert lookup("us:regulations/12-cfr/1005/7#input.anything") is None
        assert (
            lookup("us:regulations/12-cfr/1005/6#liability_under_this_section") is None
        )

    def test_resolver_falls_back_when_the_path_guard_refuses_the_repo(self, tmp_path):
        # The guard accepts only a jurisdiction folder directly inside a
        # canonical rulespec-<country> checkout; this one is neither.
        repo = tmp_path / "not-a-rulespec-checkout"
        module = repo / "regulations/12-cfr/1005/6.yaml"
        module.parent.mkdir(parents=True)
        module.write_text(yaml.safe_dump(REG_E_LIABILITY_MODULE, sort_keys=False))
        with pytest.raises(UnsafeRulespecContextPath):
            _import_base_to_repo_file("us:regulations/12-cfr/1005/6", repo_path=repo)
        lookup = _declared_input_default_resolver(repo_path=repo)

        assert (
            lookup("us:regulations/12-cfr/1005/6#input.liability_under_this_section")
            is None
        )
        assert _same_value(
            _missing_input_default(
                "liability_under_this_section",
                "us:regulations/12-cfr/1005/6#input.liability_under_this_section",
                declared_default=lookup,
            ),
            _infer_missing_input_default("liability_under_this_section"),
        )

    def test_resolver_prefers_known_payloads(self, tmp_path):
        lookup = _declared_input_default_resolver(
            repo_path=tmp_path,
            known_payloads={"us:regulations/12-cfr/1005/6": REG_E_LIABILITY_MODULE},
        )

        assert _same_value(
            lookup(
                "us:regulations/12-cfr/1005/6#input.unauthorized_transfers_before_notice"
            ),
            0,
        )

    def test_target_input_default_uses_the_declaration(self):
        refs = _default_refs_for_missing_input(
            "liability_under_this_section",
            target_ref="us:regulations/12-cfr/1005/6",
            baseline_inputs={},
            target_input_names={"liability_under_this_section"},
            declared_default=_declared_input_default_resolver(
                repo_path=Path("/nonexistent"),
                known_payloads={"us:regulations/12-cfr/1005/6": REG_E_LIABILITY_MODULE},
            ),
        )

        assert len(refs) == 1
        ref, value = refs[0]
        assert ref == "us:regulations/12-cfr/1005/6#input.liability_under_this_section"
        assert _same_value(value, 0)


# Properties. Names mix words the heuristics read as boolean (is, has, applies),
# words they read as numeric (amount, income, rate) and neutral words, and the
# formulas use the input in boolean, numeric, string and bare positions, so the
# heuristics would answer every way; a declaration must override all of them.
NAME_WORDS = (
    "is",
    "has",
    "any",
    "eligible",
    "applies",
    "amount",
    "income",
    "rate",
    "count",
    "months",
    "household",
    "size",
    "transfers",
    "notice",
    "liability",
    "section",
    "factor",
    "before",
    "under",
)
FORMULA_TEMPLATES = (
    "{n}",
    "if flag: {n} else: 0",
    "if {n}: 1 else: 0",
    "not {n}",
    "{n} and other_fact",
    "min(cap, {n})",
    "{n} + 1",
    '{n} == "category"',
    "max(0, {n} - 1)",
    "if flag:\n  {n}\nelse:\n  0",
    "min(cap, if code == 1: 0 elif {n}: 1 else: 2)",
    "# note: {n}\nif flag: {n}  # value\nelse: 0",
)
RULE_DTYPES = ("Money", "Rate", "Integer", "Judgment", "Boolean")

input_names = st.lists(st.sampled_from(NAME_WORDS), min_size=1, max_size=5).map(
    "_".join
)
grouped_dtypes = st.sampled_from(sorted(DTYPE_KINDS))
scalar_values = st.one_of(
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=6),
)


@settings(max_examples=300, deadline=None)
@given(
    input_name=input_names,
    template=st.sampled_from(FORMULA_TEMPLATES),
    rule_dtype=st.sampled_from(RULE_DTYPES),
    declared_dtype=grouped_dtypes,
    prefer_positive_counts=st.booleans(),
)
def test_a_declaration_decides_the_generated_value(
    input_name, template, rule_dtype, declared_dtype, prefer_positive_counts
):
    """Invariant: a declared input's generated value takes its declared type."""
    kind = DTYPE_KINDS[declared_dtype]
    payload = _module(
        inputs=[{"name": input_name, "dtype": declared_dtype}],
        formula=template.format(n=input_name),
        rule_dtype=rule_dtype,
    )

    value = _default_generated_test_input_value(
        input_name,
        rules_payload=payload,
        prefer_positive_counts=prefer_positive_counts,
    )

    assert _matches_kind(value, kind)
    if kind == "numeric":
        assert value in (0, 1)
    else:
        assert _same_value(value, _DECLARED_INPUT_KIND_DEFAULTS[kind])
    assert _factual_input_appears_numeric(input_name, rules_payload=payload) == (
        kind == "numeric"
    )


@settings(max_examples=300, deadline=None)
@given(
    input_name=input_names,
    template=st.sampled_from(FORMULA_TEMPLATES),
    rule_dtype=st.sampled_from(RULE_DTYPES),
    declared_dtype=grouped_dtypes,
    value=scalar_values,
)
def test_the_wrong_typed_repair_never_breaks_a_declared_value(
    input_name, template, rule_dtype, declared_dtype, value
):
    """Invariant: the repair leaves a value of the declared type alone, and any
    value it writes has the declared type."""
    kind = DTYPE_KINDS[declared_dtype]
    payload = _module(
        inputs=[{"name": input_name, "dtype": declared_dtype}],
        formula=template.format(n=input_name),
        rule_dtype=rule_dtype,
    )

    replacement = _wrong_typed_test_input_replacement(
        input_name, value, rules_payload=payload
    )

    if _matches_kind(value, kind):
        assert replacement is None
    if replacement is not None:
        assert _matches_kind(replacement, kind)


@settings(max_examples=300, deadline=None)
@given(
    input_name=input_names,
    other_names=st.lists(input_names, max_size=4),
    other_dtypes=st.lists(
        st.sampled_from(sorted(DTYPE_KINDS) + list(UNGROUPED_DTYPES)),
        min_size=4,
        max_size=4,
    ),
    template=st.sampled_from(FORMULA_TEMPLATES),
    rule_dtype=st.sampled_from(RULE_DTYPES),
)
def test_declarations_only_change_the_inputs_they_name(
    input_name, other_names, other_dtypes, template, rule_dtype
):
    """Invariant: declaring other inputs never changes an undeclared input's value."""
    formula = template.format(n=input_name)
    bare = _module(formula=formula, rule_dtype=rule_dtype)
    others = [
        {"name": name, "dtype": dtype}
        for name, dtype in zip(other_names, other_dtypes)
        if name != input_name
    ]
    declared_elsewhere = _module(inputs=others, formula=formula, rule_dtype=rule_dtype)

    assert _same_value(
        _default_generated_test_input_value(input_name, rules_payload=bare),
        _default_generated_test_input_value(
            input_name, rules_payload=declared_elsewhere
        ),
    )
    assert _factual_input_appears_numeric(
        input_name, rules_payload=bare
    ) == _factual_input_appears_numeric(input_name, rules_payload=declared_elsewhere)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    input_name=input_names,
    declared_dtype=st.one_of(
        st.none(), st.sampled_from(sorted(DTYPE_KINDS) + list(UNGROUPED_DTYPES))
    ),
    template=st.sampled_from(FORMULA_TEMPLATES),
)
def test_the_repo_resolver_agrees_with_the_payload_reading(
    tmp_path_factory, input_name, declared_dtype, template
):
    """Differential: reading the module from disk gives the same default as
    reading the parsed payload."""
    inputs = (
        None
        if declared_dtype is None
        else [{"name": input_name, "dtype": declared_dtype}]
    )
    payload = _module(inputs=inputs, formula=template.format(n=input_name))
    repo = tmp_path_factory.mktemp("case") / "rulespec-us" / "us"
    module = repo / "statutes/26/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(yaml.safe_dump(payload, sort_keys=False))

    from_disk = _declared_input_default_resolver(repo_path=repo)(
        f"us:statutes/26/1#input.{input_name}"
    )
    kind = _declared_input_dtype_kind(input_name, rules_payload=payload)
    expected = None if kind is None else _DECLARED_INPUT_KIND_DEFAULTS[kind]

    if expected is None:
        assert from_disk is None
    else:
        assert _same_value(from_disk, expected)
