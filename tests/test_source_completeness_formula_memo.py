"""Formula token and case-environment reuse must not change an analysis.

The tokenizer and string masking are pure functions of their text, so a bounded
cache returns what a fresh call would. The memo reuses a case's dependency
environment only inside one analysis, only for the same input objects, and only
when their contents, Decimal settings and observed native float behavior match.
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import decimal
import functools
import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as v

_FORMULA_ALPHABET = st.sampled_from(
    list("abxy_019 .:,=+-*/<>()[]#\"'\\\n\t") + ["if ", "else", "match ", '"""']
)
_FORMULA_TEXTS = st.lists(_FORMULA_ALPHABET, max_size=40).map("".join)


@pytest.fixture(autouse=True)
def stable_exception_mask_reader(monkeypatch, request):
    """Ordinary memo tests can run even without a verified native fenv layout."""

    original = sc._floating_point_exception_mask_reader
    if "hardware_rounding" not in request.fixturenames:
        monkeypatch.setattr(
            sc, "_floating_point_exception_mask_reader", lambda: lambda: True
        )
    return original


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

    for number in range(1_025):
        sc._rulespec_formula_tokens(f"input_{number}")
        sc._mask_formula_strings_and_comments(f"input_{number}")
    assert tokens.cache_info().currsize == 1_024
    assert masks.cache_info().currsize == 1_024

    tokens.cache_clear()
    masks.cache_clear()
    largest = "x" * sc._FORMULA_TEXT_CACHE_CHARACTERS
    sc._rulespec_formula_tokens(largest)
    sc._mask_formula_strings_and_comments(largest)
    assert tokens.cache_info().currsize == 1
    assert masks.cache_info().currsize == 1
    tokens.cache_clear()
    masks.cache_clear()
    large = "income + " * 500 + "1"
    assert len(large) > sc._FORMULA_TEXT_CACHE_CHARACTERS
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


def test_memo_never_shares_a_value_a_caller_can_change(monkeypatch):
    calls = _count_evaluations(monkeypatch)
    rules = {"out": _rule("out", "items")}
    case = {"name": "case", "input": {"items": [1, 2]}, "output": {}}
    environment = {}
    with sc._case_dependency_environment_memo_scope():
        first = _environment(rules, case, environment)
        assert first == {"out": [1, 2]}
        first["out"].append(99)
        second = _environment(rules, case, environment)
    assert len(calls) == 2
    assert second == {"out": [1, 2]}
    assert second["out"] is not first["out"]


def test_memo_evaluates_inputs_too_deep_to_fingerprint(monkeypatch):
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}
    nested: list = []
    for _ in range(100_000):
        nested = [nested]
    case["output"]["deep"] = nested
    with sc._case_dependency_environment_memo_scope():
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
    assert len(calls) == 2


@contextlib.contextmanager
def _integer_string_limit():
    original = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4_300)
        yield
    finally:
        sys.set_int_max_str_digits(original)


@settings(max_examples=20, deadline=None)
@example(exponent=5_000)
@given(exponent=st.integers(min_value=4_301, max_value=5_500))
def test_memo_matches_fresh_with_an_ignored_integer_too_large_to_fingerprint(exponent):
    rules = {"out": _rule("out", "1")}
    case = {"input": {}, "output": {"deep": 10**exponent}}
    environment = {}
    with _integer_string_limit():
        direct = sc._evaluate_case_dependency_environment(
            rules, case, formula_environment=environment, require_asserted_value=False
        )
        assert direct == {"out": 1}
        with sc._case_dependency_environment_memo_scope():
            assert _environment(rules, case, environment) == direct
            assert _environment(rules, case, environment) == direct


@pytest.mark.parametrize(
    "error", (ValueError, TypeError, RuntimeError, OSError, MemoryError, RecursionError)
)
def test_memo_evaluates_fresh_when_an_ignored_value_cannot_be_fingerprinted(
    monkeypatch, error
):
    class Unprintable:
        def __repr__(self):
            raise error("unused value cannot be fingerprinted")

    rules = {"out": _rule("out", "1")}
    case = {"input": {}, "output": {"ignored": Unprintable()}}
    environment = {}
    direct = sc._evaluate_case_dependency_environment(
        rules, case, formula_environment=environment, require_asserted_value=False
    )
    assert direct == {"out": 1}
    calls = _count_evaluations(monkeypatch)
    with sc._case_dependency_environment_memo_scope():
        assert _environment(rules, case, environment) == direct
        assert _environment(rules, case, environment) == direct
    assert len(calls) == 2


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


@pytest.mark.parametrize("require_asserted_value", (False, True))
def test_memo_evaluates_again_when_the_decimal_precision_changes(
    monkeypatch, require_asserted_value
):
    calls = _count_evaluations(monkeypatch)
    rules = {"out": _rule("out", "-x")}
    case = {"name": "case", "input": {"x": 12345}, "output": {"out": -12345}}
    environment = {}
    with sc._case_dependency_environment_memo_scope():
        with decimal.localcontext(prec=5):
            before = _environment(
                rules,
                case,
                environment,
                require_asserted_value=require_asserted_value,
            )
        with decimal.localcontext(prec=3):
            after = _environment(
                rules,
                case,
                environment,
                require_asserted_value=require_asserted_value,
            )
            direct = sc._evaluate_case_dependency_environment(
                rules,
                case,
                formula_environment=environment,
                require_asserted_value=require_asserted_value,
            )
    assert len(calls) == 3  # both memoized calls evaluated, plus the direct one
    assert before == {"out": decimal.Decimal("-12345")}
    assert _typed(after) == _typed(direct)
    assert after == (
        {} if require_asserted_value else {"out": decimal.Decimal("-1.23E+4")}
    )


@pytest.mark.parametrize(
    "change",
    (
        lambda context: setattr(context, "prec", 7),
        lambda context: setattr(context, "rounding", decimal.ROUND_DOWN),
        lambda context: setattr(context, "Emin", -99),
        lambda context: setattr(context, "Emax", 99),
        lambda context: setattr(context, "capitals", 0),
        lambda context: setattr(context, "clamp", 1),
        lambda context: context.traps.__setitem__(decimal.Inexact, True),
    ),
    ids=("prec", "rounding", "Emin", "Emax", "capitals", "clamp", "traps"),
)
def test_memo_keeps_a_separate_entry_for_each_decimal_setting(monkeypatch, change):
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}
    with sc._case_dependency_environment_memo_scope():
        _environment(rules, case, environment)
        with decimal.localcontext() as context:
            change(context)
            _environment(rules, case, environment)
            assert len(calls) == 2  # the changed setting misses
            _environment(rules, case, environment)
            assert len(calls) == 2  # and the same setting hits
        _environment(rules, case, environment)
        assert len(calls) == 2  # the original setting still hits


def test_memo_still_hits_after_a_decimal_flag_changes(monkeypatch):
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}
    with sc._case_dependency_environment_memo_scope():
        with decimal.localcontext() as context:
            context.clear_flags()
            _environment(rules, case, environment)
            context.flags[decimal.Inexact] = True
            _environment(rules, case, environment)
    assert len(calls) == 1


@pytest.fixture(scope="module")
def hardware_rounding():
    """Probe C rounding modes without assuming an architecture's FE_* values."""

    try:
        library = ctypes.CDLL(None)
        get_rounding = library.fegetround
        set_rounding = library.fesetround
    except (AttributeError, OSError, TypeError, RuntimeError):
        pytest.skip("C floating-point rounding API unavailable")
    if sc._floating_point_behavior() is None:
        pytest.skip(
            "The memo evaluates every call fresh on this host, since it can't "
            "confirm that native floating-point traps are masked"
        )
    get_rounding.argtypes = ()
    get_rounding.restype = ctypes.c_int
    set_rounding.argtypes = (ctypes.c_int,)
    set_rounding.restype = ctypes.c_int
    original = get_rounding()
    if original < 0:
        pytest.skip("C floating-point rounding mode unavailable")
    modes = []
    try:
        # x86 and ARM encodings; only use modes the host accepts and reports.
        for mode in (0, 0x400, 0x800, 0xC00, 0x400000, 0x800000, 0xC00000):
            if set_rounding(mode) == 0 and get_rounding() == mode:
                modes.append(mode)
    finally:
        assert set_rounding(original) == 0
    if len(modes) < 2:
        pytest.skip("Host does not expose multiple floating-point rounding modes")
    return library, modes


@pytest.mark.parametrize("error", (AttributeError, OSError, TypeError, RuntimeError))
def test_hardware_rounding_fixture_skips_when_the_host_api_is_unavailable(
    monkeypatch, error
):
    def unavailable(_name):
        raise error("rounding query unavailable")

    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    with pytest.raises(pytest.skip.Exception, match="rounding API unavailable"):
        hardware_rounding.__wrapped__()


@contextlib.contextmanager
def _hardware_rounding_mode(library, mode):
    original = library.fegetround()
    try:
        assert library.fesetround(mode) == 0
        yield
    finally:
        assert library.fesetround(original) == 0


def test_memo_preserves_money_assertion_comparison_at_each_hardware_rounding(
    monkeypatch, hardware_rounding
):
    library, modes = hardware_rounding
    calls = _count_evaluations(monkeypatch)
    rules = {"x": _rule("x", "0.10000000000000002", "Money")}
    case = {"name": "case", "period": "2025", "input": {}, "output": {"x": 0.1}}
    environment = {}
    with sc._case_dependency_environment_memo_scope():
        for index, mode in enumerate(modes):
            with _hardware_rounding_mode(library, mode):
                memoized = _environment(
                    rules, case, environment, require_asserted_value=True
                )
                repeated = _environment(
                    rules, case, environment, require_asserted_value=True
                )
                direct = sc._evaluate_case_dependency_environment(
                    rules,
                    case,
                    formula_environment=environment,
                    require_asserted_value=True,
                )
            assert _typed(memoized) == _typed(direct)
            assert _typed(repeated) == _typed(direct)
            assert len(calls) == 2 * (index + 1)  # each mode misses once, then hits


@settings(max_examples=75, deadline=None, database=None)
@example(mode_indexes=[0, 1], require_asserted_value=True)
@given(
    mode_indexes=st.lists(st.integers(0, 3), min_size=2, max_size=8),
    require_asserted_value=st.booleans(),
)
def test_memo_matches_fresh_evaluation_across_hardware_rounding_sequences(
    hardware_rounding, mode_indexes, require_asserted_value
):
    library, modes = hardware_rounding
    rules = {"x": _rule("x", "0.10000000000000002", "Money")}
    case = {"name": "case", "period": "2025", "input": {}, "output": {"x": 0.1}}
    environment = {}
    with sc._case_dependency_environment_memo_scope():
        for index in mode_indexes:
            with _hardware_rounding_mode(library, modes[index % len(modes)]):
                memoized = _environment(
                    rules,
                    case,
                    environment,
                    require_asserted_value=require_asserted_value,
                )
                direct = sc._evaluate_case_dependency_environment(
                    rules,
                    case,
                    formula_environment=environment,
                    require_asserted_value=require_asserted_value,
                )
            assert _typed(memoized) == _typed(direct)


@pytest.mark.parametrize("error", (ValueError, RuntimeError))
def test_memo_evaluates_fresh_when_float_behavior_is_unknown(monkeypatch, error):
    def unavailable(*args):
        raise error("float behavior cannot be packed")

    monkeypatch.setattr(sc.struct, "pack", unavailable)
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}
    with sc._case_dependency_environment_memo_scope():
        first = _environment(rules, case, environment)
        second = _environment(rules, case, environment)
    assert len(calls) == 2
    assert _typed(first) == _typed(second)
    assert first == {"helper": 100, "result": 100}


@pytest.mark.parametrize("error", (AttributeError, OSError, TypeError, RuntimeError))
def test_memo_evaluates_fresh_without_a_host_api(
    monkeypatch, stable_exception_mask_reader, error
):
    def unavailable(_name):
        raise error("rounding query unavailable")

    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    monkeypatch.setattr(
        sc,
        "_floating_point_exception_mask_reader",
        stable_exception_mask_reader.__wrapped__,
    )
    assert sc._floating_point_behavior() is None
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}
    with sc._case_dependency_environment_memo_scope():
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
    assert len(calls) == 2


@pytest.mark.parametrize("reader", (None, lambda: False))
def test_memo_evaluates_fresh_when_native_traps_cannot_be_ruled_out(
    monkeypatch, reader
):
    monkeypatch.setattr(sc, "_floating_point_exception_mask_reader", lambda: reader)

    def unsafe_operands():
        pytest.fail("Unsafe float probe must not execute")
        yield

    monkeypatch.setattr(sc, "_FLOAT_BEHAVIOR_OPERANDS", unsafe_operands())
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}
    with sc._case_dependency_environment_memo_scope():
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
    assert len(calls) == 2


def test_memo_evaluates_fresh_when_reading_native_masks_raises(monkeypatch):
    def unreadable():
        raise RuntimeError("native mask read denied")

    monkeypatch.setattr(sc, "_floating_point_exception_mask_reader", lambda: unreadable)
    calls = _count_evaluations(monkeypatch)
    rules, case, environment = _rules(), _case(), {}
    with sc._case_dependency_environment_memo_scope():
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
        assert _environment(rules, case, environment) == {"helper": 100, "result": 100}
    assert len(calls) == 2


def test_decimal_partitions_can_retain_hits_after_inputs_and_settings_return(
    monkeypatch,
):
    calls = _count_evaluations(monkeypatch)
    rules = {"out": _rule("out", "-x")}
    case = {"name": "case", "input": {"x": 12345}, "output": {}}
    environment = {}
    counts, results = [], []
    with sc._case_dependency_environment_memo_scope():
        for precision, value in ((5, 12345), (3, 23456), (5, 12345)):
            case["input"]["x"] = value
            with decimal.localcontext(prec=precision):
                results.append(_environment(rules, case, environment))
            counts.append(len(calls))
    assert counts == [1, 2, 2]
    assert results == [
        {"out": decimal.Decimal("-12345")},
        {"out": decimal.Decimal("-2.35E+4")},
        {"out": decimal.Decimal("-12345")},
    ]


_NAMES = ("a", "b", "c", "d")
_EXPRESSIONS = st.one_of(
    st.sampled_from(_NAMES),
    st.sampled_from(_NAMES).map(lambda name: f"-{name}"),
    st.integers(min_value=-3, max_value=3).map(str),
    st.tuples(
        st.sampled_from(_NAMES), st.sampled_from("+-*"), st.sampled_from(_NAMES)
    ).map(" ".join),
    st.tuples(
        st.sampled_from(_NAMES), st.sampled_from(_NAMES), st.sampled_from(_NAMES)
    ).map(lambda parts: f"if {parts[0]} > 0: {parts[1]} else: {parts[2]}"),
)
_VALUES = st.one_of(
    st.integers(min_value=-2, max_value=2), st.booleans(), st.just(12345)
)
_STEPS = st.lists(
    st.one_of(
        st.tuples(
            st.just("call"),
            st.booleans(),
            st.one_of(st.none(), st.sets(st.sampled_from(_NAMES))),
            st.integers(min_value=0, max_value=1),
        ),
        st.tuples(st.just("precision"), st.sampled_from((1, 2, 28))),
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
    with (
        sc._case_dependency_environment_memo_scope(),
        decimal.localcontext() as context,
    ):
        for step in steps:
            kind = step[0]
            if kind == "precision":
                context.prec = step[1]
            elif kind == "call":
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


@settings(max_examples=150, deadline=None)
@given(
    value=st.integers(min_value=-(10**7), max_value=10**7),
    precisions=st.lists(st.sampled_from((1, 2, 3, 5, 28)), min_size=2, max_size=6),
    require_asserted_value=st.booleans(),
)
def test_memoized_environment_matches_direct_evaluation_at_every_precision(
    value, precisions, require_asserted_value
):
    rules = {"out": _rule("out", "-x")}
    case = {"name": "case", "input": {"x": value}, "output": {"out": -value}}
    environment: dict[str, object] = {}
    with sc._case_dependency_environment_memo_scope():
        for precision in precisions:
            with decimal.localcontext(prec=precision):
                memoized = _environment(
                    rules,
                    case,
                    environment,
                    require_asserted_value=require_asserted_value,
                )
                direct = sc._evaluate_case_dependency_environment(
                    rules,
                    case,
                    formula_environment=environment,
                    require_asserted_value=require_asserted_value,
                )
            assert _typed(memoized) == _typed(direct)


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
