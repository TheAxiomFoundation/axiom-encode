"""Formula token and case-environment reuse must not change an analysis.

The tokenizer and string masking are pure functions of their text, so a bounded
cache returns what a fresh call would. A case's dependency environment is
reused only inside one analysis, only for the same input objects, and only
while their contents are unchanged.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as v

_FORMULA_ALPHABET = st.sampled_from(
    list("abxy_019 .:,=+-*/<>()[]#\"'\\\n\t") + ["if ", "else", "match ", '"""']
)
_FORMULA_TEXTS = st.lists(_FORMULA_ALPHABET, max_size=40).map("".join)


@pytest.fixture(autouse=True)
def empty_formula_caches():
    sc._cached_rulespec_formula_tokens.cache_clear()
    sc._cached_mask_formula_strings_and_comments.cache_clear()
    yield
    sc._cached_rulespec_formula_tokens.cache_clear()
    sc._cached_mask_formula_strings_and_comments.cache_clear()


@settings(max_examples=300, deadline=None)
@given(_FORMULA_TEXTS)
def test_cached_tokens_and_masks_match_a_fresh_scan(text):
    fresh_tokens = sc._tokenize_rulespec_formula(text)
    fresh_mask = sc._blank_formula_strings_and_comments(text)

    assert sc._rulespec_formula_tokens(text) == fresh_tokens
    assert sc._rulespec_formula_tokens(text) == fresh_tokens
    assert sc._mask_formula_strings_and_comments(text) == fresh_mask
    assert sc._mask_formula_strings_and_comments(text) == fresh_mask
    assert len(fresh_mask) == len(text)


def test_token_and_mask_caches_are_reused_bounded_and_skip_large_texts():
    tokens = sc._cached_rulespec_formula_tokens
    masks = sc._cached_mask_formula_strings_and_comments
    formula = 'if label == "a # b": income else: 0  # note'

    first = sc._rulespec_formula_tokens(formula)
    assert sc._rulespec_formula_tokens(formula) is first
    assert tokens.cache_info().hits == 1
    assert first[0] == ("IF", "if", 0, 2)
    assert first[-1] == ("EOF", "", len(formula), len(formula))
    masked = sc._mask_formula_strings_and_comments(formula)
    assert sc._mask_formula_strings_and_comments(formula) is masked
    assert masks.cache_info().hits == 1
    assert "#" not in masked and "a" not in masked[12:19]

    assert sc._rulespec_formula_tokens('"open') is None
    assert sc._rulespec_formula_tokens('"open') is None

    for number in range(4_097):
        sc._rulespec_formula_tokens(f"input_{number}")
        sc._mask_formula_strings_and_comments(f"input_{number}")
    assert tokens.cache_info().currsize == 4_096
    assert masks.cache_info().currsize == 4_096

    tokens.cache_clear()
    masks.cache_clear()
    large = "income + " * 2_000 + "1"
    assert len(large) > 16_384
    assert sc._rulespec_formula_tokens(large) == sc._tokenize_rulespec_formula(large)
    assert sc._mask_formula_strings_and_comments(large) == large
    assert tokens.cache_info().currsize == 0
    assert masks.cache_info().currsize == 0


def _rule(name, formula, dtype="Decimal"):
    return {
        "name": name,
        "kind": "derived",
        "dtype": dtype,
        "versions": [{"formula": formula}],
    }


def _rules():
    return {
        "helper": _rule("helper", "if applies: amount * 2 else: 0"),
        "result": _rule("result", "if applies: helper else: baseline"),
    }


def _case(applies=True, amount=50):
    return {
        "name": "case",
        "period": "2025",
        "input": {"applies": applies, "amount": amount, "baseline": 200},
        "output": {"helper": amount * 2 if applies else 0, "result": 100},
    }


def _count_evaluations(monkeypatch):
    calls = []
    original = sc._evaluate_case_dependency_environment

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(sc, "_evaluate_case_dependency_environment", counted)
    return calls


def _environment(rules, case, environment, **kwargs):
    return sc._case_dependency_environment(
        rules,
        case,
        formula_environment=environment,
        require_asserted_value=kwargs.pop("require_asserted_value", False),
        **kwargs,
    )


def test_memo_reuses_a_fresh_copy_only_inside_one_analysis_scope(monkeypatch):
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}

    _environment(rules, case, environment)
    _environment(rules, case, environment)
    assert len(calls) == 2  # no memo outside an analysis

    with sc._case_dependency_environment_memo_scope():
        first = _environment(rules, case, environment)
        first["helper"] = "changed by a caller"
        second = _environment(rules, case, environment)
        assert len(calls) == 3
        assert second == {"helper": 100, "result": 100}
        assert second is not first
        second["result"] = "changed by a caller"
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
        assert len(calls) == 3
        _environment(rules, case, environment, allowed_names={"helper"})
        _environment(rules, case, environment, require_asserted_value=True)
        _environment(rules, case, environment, allow_bound_row_assertions=True)
        assert len(calls) == 6
        _environment(rules, copy.deepcopy(case), environment)
        assert len(calls) == 7  # an equal but different case object is not reused
        with sc._case_dependency_environment_memo_scope():
            _environment(rules, case, environment)
            assert len(calls) == 8  # a nested analysis starts its own memo
        _environment(rules, case, environment)
        assert len(calls) == 8

    _environment(rules, case, environment)
    assert len(calls) == 9


def _typed(environment):
    return [(name, type(value), value) for name, value in environment.items()]


def _plain(rules):
    return rules


def _declared(rules):
    rules["helper"]["rounding"] = "declared"
    return rules


def _rounded(rules):
    rules["helper"]["rounding"] = "declared"
    rules["helper"] = sc._CurrencyRoundingRule(
        rules["helper"], sc._CurrencyRounding("half_up", 0)
    )
    return rules


@pytest.mark.parametrize(
    "prepare, mutate",
    [
        (_plain, lambda rules, case, environment: case["input"].update(amount=7)),
        (
            _plain,
            lambda rules, case, environment: rules["helper"]["versions"][0].update(
                formula="if applies: amount * 3 else: 0"
            ),
        ),
        # A constant that disagrees with a case input empties the environment.
        (_plain, lambda rules, case, environment: environment.update(baseline=1)),
        # The same mapping contents, but only the rule type resolves rounding.
        (
            _declared,
            lambda rules, case, environment: rules.update(
                helper=sc._CurrencyRoundingRule(
                    rules["helper"], sc._CurrencyRounding("half_up", 0)
                )
            ),
        ),
        # The same mapping contents, but its resolved rounding changes in place.
        (
            _rounded,
            lambda rules, case, environment: setattr(
                rules["helper"],
                "resolved_currency_rounding",
                sc._CurrencyRounding("half_up", 2),
            ),
        ),
    ],
    ids=("case_input", "rule_formula", "constants", "rule_type", "rounding"),
)
def test_memo_evaluates_again_after_any_input_changes_in_place(
    monkeypatch, prepare, mutate
):
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = prepare(_rules()), _case(amount=50.25), {}
    with sc._case_dependency_environment_memo_scope():
        before = _environment(rules, case, environment)
        mutate(rules, case, environment)
        after = _environment(rules, case, environment)
    assert len(calls) == 2
    direct = sc._evaluate_case_dependency_environment(
        rules, case, formula_environment=environment, require_asserted_value=False
    )
    assert _typed(after) == _typed(direct)
    assert _typed(after) != _typed(before)


_NAMES = ("a", "b", "c", "d")
_EXPRESSIONS = st.one_of(
    st.sampled_from(_NAMES),
    st.integers(min_value=-3, max_value=3).map(str),
    st.tuples(
        st.sampled_from(_NAMES), st.sampled_from("+-*"), st.sampled_from(_NAMES)
    ).map(" ".join),
    st.tuples(
        st.sampled_from(_NAMES), st.sampled_from(_NAMES), st.sampled_from(_NAMES)
    ).map(lambda parts: f"if {parts[0]} > 0: {parts[1]} else: {parts[2]}"),
)
_VALUES = st.one_of(st.integers(min_value=-2, max_value=2), st.booleans())
_STEPS = st.lists(
    st.one_of(
        st.tuples(
            st.just("call"),
            st.booleans(),
            st.one_of(st.none(), st.sets(st.sampled_from(_NAMES))),
            st.integers(min_value=0, max_value=1),
        ),
        st.tuples(st.just("input"), st.sampled_from(_NAMES), _VALUES),
        st.tuples(st.just("output"), st.sampled_from(_NAMES), _VALUES),
        st.tuples(st.just("formula"), st.sampled_from(_NAMES[:3]), _EXPRESSIONS),
        st.tuples(st.just("constant"), st.sampled_from(_NAMES), _VALUES),
    ),
    min_size=1,
    max_size=12,
)


@settings(max_examples=150, deadline=None)
@given(
    formulas=st.fixed_dictionaries({name: _EXPRESSIONS for name in _NAMES[:3]}),
    inputs=st.dictionaries(st.sampled_from(_NAMES), _VALUES),
    steps=_STEPS,
)
def test_memoized_environment_matches_direct_evaluation_at_every_step(
    formulas, inputs, steps
):
    rules = {name: _rule(name, formula) for name, formula in formulas.items()}
    cases = [
        {"name": "first", "period": "2025", "input": dict(inputs), "output": {}},
        {"name": "second", "input": dict(inputs), "output": {}},
    ]
    environment: dict[str, object] = {}
    with sc._case_dependency_environment_memo_scope():
        for step in steps:
            kind = step[0]
            if kind == "call":
                _, require_asserted_value, allowed_names, case_index = step
                memoized = _environment(
                    rules,
                    cases[case_index],
                    environment,
                    require_asserted_value=require_asserted_value,
                    allowed_names=allowed_names,
                )
                direct = sc._evaluate_case_dependency_environment(
                    rules,
                    cases[case_index],
                    formula_environment=environment,
                    require_asserted_value=require_asserted_value,
                    allowed_names=allowed_names,
                )
                assert memoized == direct
                assert list(memoized) == list(direct)
                assert [type(value) for value in memoized.values()] == [
                    type(value) for value in direct.values()
                ]
            elif kind == "input":
                cases[0]["input"][step[1]] = step[2]
            elif kind == "output":
                cases[1]["output"][step[1]] = step[2]
            elif kind == "formula":
                rules[step[1]]["versions"][0]["formula"] = step[2]
            else:
                environment[step[1]] = step[2]


_FIXTURE = Path(__file__).parent / "fixtures/source_path_shadow"
_CITATION = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"


def test_analysis_with_memo_matches_analysis_without_it(monkeypatch):
    payload = yaml.safe_load((_FIXTURE / "candidate.yaml").read_text())
    cases = yaml.safe_load((_FIXTURE / "cases.yaml").read_text())
    source = (_FIXTURE / "source.txt").read_text()
    calls = _count_evaluations(monkeypatch)

    def analyze():
        return sc.analyze_complete_source_unit(
            yaml.safe_dump(payload),
            source,
            corpus_citation_path=_CITATION,
            test_cases=copy.deepcopy(cases),
            extract_numeric_occurrences=functools.partial(
                v.extract_typed_numeric_inventory_occurrences_from_text,
                profile="legacy",
            ),
            extract_numeric_grounding_occurrences=functools.partial(
                v.extract_typed_numeric_occurrences_from_text, profile="legacy"
            ),
            extract_named_scalars=v.extract_named_scalar_occurrences,
            numeric_value_is_grounded=v.numeric_value_is_grounded,
        )

    memoized = analyze()
    memoized_calls = len(calls)
    with monkeypatch.context() as patch:
        patch.setattr(sc, "_CASE_DEPENDENCY_ENVIRONMENT_MEMO", _NoMemo())
        direct = analyze()

    assert memoized == direct
    assert memoized.issues
    assert 0 < memoized_calls < len(calls) - memoized_calls


class _NoMemo:
    """Stand in for the memo context variable with the memo always off."""

    def set(self, _value):
        return None

    def reset(self, _token):
        return None

    def get(self):
        return None
