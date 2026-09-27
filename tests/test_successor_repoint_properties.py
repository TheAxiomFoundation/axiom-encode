"""Property-based tests for the successor repoint's pure functions.

Each test states one invariant that must hold for every input:

1. Lexer partition: the code and non-code segments of a formula concatenate
   back to the formula.
2. Identity rename: renaming a symbol to itself changes nothing.
3. Non-code preservation: a rename never changes a string, docstring or
   comment, and it changes exactly the bare uses in code, never a member
   access (checked against formulas whose pieces are classified by
   construction, independently of the lexer).
4. Round trip: when ``new`` has no bounded use in a formula, renaming
   ``old -> new -> old`` returns the formula.
5. Completeness: after ``old -> new`` no bounded use of ``old`` remains, and
   the uses of ``new`` (with their literal subscripts) are exactly the former
   uses of ``old`` and ``new``.
6. Envelope: every accepted concept map is injective with disjoint ``from``
   and ``to`` sets, and the request digest depends on neither JSON key order
   nor which loader parsed it.
7. Concept proof, differential: ``prove_concept_map`` accepts a legacy and a
   successor version ladder exactly when a day-by-day comparison over the
   successor window finds the same table keys and values on every day, and the
   proved keys are exactly the keys defined on every day of that window.
8. Window flags: the pre- and post-window behavior-change flags equal their
   geometric definition over the dependent's formula versions.
9. Receipt identity: the digest ignores mapping key order and changes when any
   field changes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import yaml
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from axiom_encode.successor_repoint import (
    ENVELOPE_SCHEMA,
    SuccessorRepointError,
    _formula_code_segments,
    _formula_symbol_uses,
    load_repoint_request_bytes,
    load_repoint_request_payload,
    prove_concept_map,
    receipt_identity_sha256,
    replace_formula_symbol,
)

PROPERTY_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# ---------------------------------------------------------------------------
# Formula lexing and renaming
# ---------------------------------------------------------------------------

# A small alphabet so names collide with each other, with member access, with
# subscripts and with every lexer delimiter.
_NAMES = st.from_regex(r"[a-c][a-c0-9_]{0,2}", fullmatch=True)
_FORMULA_TOKENS = st.one_of(
    _NAMES,
    st.sampled_from(
        [
            " ",
            "\n",
            ".",
            "x.",
            "[",
            "]",
            "[0]",
            "[1]",
            "[ 2 ]",
            "[-1]",
            "0",
            "7",
            "_",
            "(",
            ")",
            "+",
            '"',
            "'",
            '"""',
            "#",
            "\\",
            "\\'",
            '\\"',
        ]
    ),
)
_FORMULAS = st.lists(_FORMULA_TOKENS, max_size=40).map("".join)


def _non_code(formula: str) -> list[str]:
    return [segment for code, segment in _formula_code_segments(formula) if not code]


@PROPERTY_SETTINGS
@given(formula=_FORMULAS)
def test_lexer_segments_partition_the_formula(formula):
    segments = _formula_code_segments(formula)
    assert "".join(segment for _code, segment in segments) == formula
    assert all(segment for _code, segment in segments)


@PROPERTY_SETTINGS
@given(formula=_FORMULAS, name=_NAMES)
def test_renaming_a_symbol_to_itself_is_a_no_op(formula, name):
    assert replace_formula_symbol(formula, name, name) == formula


@PROPERTY_SETTINGS
@given(formula=_FORMULAS, old=_NAMES, new=_NAMES)
def test_a_rename_never_touches_strings_docstrings_or_comments(formula, old, new):
    renamed = replace_formula_symbol(formula, old, new)
    assert _non_code(renamed) == _non_code(formula)


@PROPERTY_SETTINGS
@given(formula=_FORMULAS, old=_NAMES, new=_NAMES)
def test_a_rename_round_trips_when_the_target_is_unused(formula, old, new):
    assume(old != new)
    assume(not _formula_symbol_uses(formula, new))
    renamed = replace_formula_symbol(formula, old, new)
    assert replace_formula_symbol(renamed, new, old) == formula


def _sorted_uses(uses):
    return sorted(uses, key=lambda use: (use is None, use if use is not None else 0))


@PROPERTY_SETTINGS
@given(formula=_FORMULAS, old=_NAMES, new=_NAMES)
def test_a_rename_moves_every_use_and_keeps_its_subscript(formula, old, new):
    assume(old != new)
    renamed = replace_formula_symbol(formula, old, new)
    assert _formula_symbol_uses(renamed, old) == ()
    assert _sorted_uses(_formula_symbol_uses(renamed, new)) == _sorted_uses(
        _formula_symbol_uses(formula, old) + _formula_symbol_uses(formula, new)
    )


# An independent oracle: build formulas from pieces whose class is known by
# construction (code, a quoted string, a docstring, a comment), with code made
# of bare uses, member accesses and subscripted uses.  The engine lexer
# (axiom-rules-engine af6e4ea, src/formula.rs:175-310) treats strings,
# docstrings and ``#`` comments as non-code, and ``x.name`` is a member access,
# not a use of ``name``; a rename must change exactly the bare uses in code.

_SEPARATORS = st.sampled_from([" ", " + ", ")", "(", ", ", "\n", " * "])


@st.composite
def _code_pieces(draw, old: str, new: str, *, after_comment: bool):
    names = st.sampled_from([old, new, "other", "x_" + old])
    parts: list[tuple[str, str]] = []
    if after_comment:
        parts.append(("\n", "\n"))
    for _ in range(draw(st.integers(min_value=1, max_value=5))):
        kind = draw(st.sampled_from(["use", "member", "subscript", "literal"]))
        name = draw(names)
        if kind == "use":
            renamed = new if name == old else name
            parts.append((name, renamed))
        elif kind == "member":
            parts.append((f"x.{name}", f"x.{name}"))
        elif kind == "subscript":
            key = draw(st.integers(min_value=0, max_value=3))
            renamed = new if name == old else name
            parts.append((f"{name}[{key}]", f"{renamed}[{key}]"))
        else:
            parts.append(("7", "7"))
        separator = draw(_SEPARATORS)
        parts.append((separator, separator))
    return parts


_QUOTED_TEXT = st.text(alphabet=st.sampled_from(list("abc_ .#[]0")), max_size=8)


@st.composite
def _non_code_piece(draw, old: str):
    kind = draw(st.sampled_from(["double", "single", "docstring", "comment"]))
    body = draw(st.lists(st.sampled_from([old, "x", " ", "#", "[1]", "."]), max_size=4))
    text = "".join(body) + draw(_QUOTED_TEXT)
    if kind == "double":
        escape = draw(st.sampled_from(["", '\\"', "\\\\"]))
        piece = f'"{text}{escape}{old}"'
    elif kind == "single":
        escape = draw(st.sampled_from(["", "\\'", "\\\\"]))
        piece = f"'{text}{escape}{old}'"
    elif kind == "docstring":
        # A lone quote inside a docstring opens nothing.
        inner = draw(st.sampled_from(["", " ' ", ' "q" ', " ' \" "]))
        piece = f'"""{text}{inner}\n{old}"""'
    else:
        piece = f"# {text} {old}"
    return kind, piece


@st.composite
def _classified_formulas(draw):
    old = draw(_NAMES)
    new = draw(_NAMES.filter(lambda name: name != old))
    original: list[str] = []
    expected: list[str] = []
    after_comment = False
    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        for before, after in draw(_code_pieces(old, new, after_comment=after_comment)):
            original.append(before)
            expected.append(after)
        kind, piece = draw(_non_code_piece(old))
        original.append(piece)
        expected.append(piece)
        after_comment = kind == "comment"
    if after_comment:
        original.append("\n")
        expected.append("\n")
    return old, new, "".join(original), "".join(expected)


@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=_classified_formulas())
def test_a_rename_changes_exactly_the_bare_uses_in_code(case):
    old, new, formula, expected = case
    assert replace_formula_symbol(formula, old, new) == expected


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

_CONCEPTS = st.from_regex(r"[a-d][a-d_]{0,2}", fullmatch=True)


def _envelope(pairs) -> dict:
    return {
        "schema": ENVELOPE_SCHEMA,
        "legacy_primary": "us/policies/irs/legacy-table.yaml",
        "successor_primary": "us/policies/irs/page-15.yaml",
        "dependents": ["us/statutes/26/32.yaml"],
        "concept_map": [{"from": old, "to": new} for old, new in pairs],
        "program_scope_updates": [],
    }


@PROPERTY_SETTINGS
@given(pairs=st.lists(st.tuples(_CONCEPTS, _CONCEPTS), min_size=1, max_size=6))
def test_an_accepted_concept_map_is_injective_and_never_chains(pairs):
    payload = _envelope(pairs)
    try:
        request = load_repoint_request_payload(payload)
    except SuccessorRepointError:
        olds = [old for old, _new in pairs]
        news = [new for _old, new in pairs]
        assert (
            any(old == new for old, new in pairs)
            or len(set(olds)) != len(olds)
            or len(set(news)) != len(news)
            or set(olds) & set(news)
        )
        return
    olds = [pair.old for pair in request.concept_map]
    news = [pair.new for pair in request.concept_map]
    assert len(set(olds)) == len(olds)
    assert len(set(news)) == len(news)
    assert not set(olds) & set(news)
    assert request.concept_renames == dict(pairs)


@PROPERTY_SETTINGS
@given(
    pairs=st.lists(
        st.tuples(_CONCEPTS, _CONCEPTS),
        min_size=1,
        max_size=6,
        unique_by=lambda p: p[0],
    ),
    order=st.randoms(use_true_random=False),
)
def test_the_request_digest_ignores_key_order_and_loader(pairs, order):
    payload = _envelope(pairs)
    try:
        request = load_repoint_request_payload(payload)
    except SuccessorRepointError:
        return
    keys = list(payload)
    order.shuffle(keys)
    shuffled = {key: payload[key] for key in keys}
    from_bytes = load_repoint_request_bytes(json.dumps(shuffled).encode("utf-8"))
    assert from_bytes.sha256 == request.sha256
    assert from_bytes.canonical_bytes == request.canonical_bytes
    assert request.sha256 == hashlib.sha256(request.canonical_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Concept proof: differential against a day-by-day comparison
# ---------------------------------------------------------------------------

_EPOCH = date(2026, 1, 1)
_KEYS = (0, 1, 2)


@st.composite
def _ladders(draw, *, first_start: int | None = None):
    """A contiguous version ladder as (start, end_or_None, values) day offsets."""

    count = draw(st.integers(min_value=1, max_value=4))
    start = (
        draw(st.integers(min_value=0, max_value=20))
        if first_start is None
        else first_start
    )
    widths = draw(
        st.lists(st.integers(min_value=1, max_value=15), min_size=count, max_size=count)
    )
    open_ended = draw(st.booleans())
    keys = draw(
        st.lists(st.sampled_from(_KEYS), min_size=1, max_size=3, unique=True).map(
            sorted
        )
    )
    ladder = []
    cursor = start
    for position, width in enumerate(widths):
        last = position == count - 1
        end = None if (last and open_ended) else cursor + width - 1
        if draw(st.booleans()):
            keys = draw(
                st.lists(
                    st.sampled_from(_KEYS), min_size=1, max_size=3, unique=True
                ).map(sorted)
            )
        values = {key: draw(st.integers(min_value=0, max_value=2)) for key in keys}
        ladder.append((cursor, end, values))
        if end is not None:
            cursor = end + 1
    return ladder


def _day(offset: int) -> str:
    return (_EPOCH + timedelta(days=offset)).isoformat()


def _rule(name: str, ladder, *, indexed: bool) -> dict:
    versions = []
    for start, end, values in ladder:
        version: dict[str, object] = {"effective_from": _day(start)}
        if end is not None:
            version["effective_to"] = _day(end)
        if indexed:
            version["values"] = dict(values)
        else:
            version["formula"] = str(values[min(values)])
        versions.append(version)
    rule: dict[str, object] = {
        "name": name,
        "kind": "parameter",
        "dtype": "Money",
        "unit": "USD",
        "versions": versions,
    }
    if indexed:
        rule["indexed_by"] = "child_count"
    return rule


def _module(rule: dict) -> bytes:
    payload = {"format": "rulespec/v1", "module": {}, "rules": [rule]}
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


def _active(ladder, day: int, *, indexed: bool):
    for start, end, values in ladder:
        if start <= day and (end is None or day <= end):
            return dict(values) if indexed else {0: values[min(values)]}
    return None


def _day_by_day(legacy, successor, *, indexed: bool):
    """Return (accepted, keys) by comparing every day of the successor window."""

    window_start = successor[0][0]
    window_end = successor[-1][1]
    boundaries = [start for start, _e, _v in legacy + successor] + [
        end + 1 for _s, end, _v in legacy + successor if end is not None
    ]
    # Past every boundary both ladders are constant, so one more day decides.
    horizon = window_end if window_end is not None else max(boundaries) + 1
    common: set[int] | None = None
    for day in range(window_start, horizon + 1):
        old = _active(legacy, day, indexed=indexed)
        new = _active(successor, day, indexed=indexed)
        if old is None or new is None or old != new:
            return False, None
        common = set(new) if common is None else common & set(new)
    return True, tuple(sorted(common or ()))


def _prove(legacy_raw: bytes, successor_raw: bytes):
    request = load_repoint_request_payload(
        {
            "schema": ENVELOPE_SCHEMA,
            "legacy_primary": "us/policies/irs/legacy-table.yaml",
            "successor_primary": "us/policies/irs/page-15.yaml",
            "dependents": ["us/statutes/26/32.yaml"],
            "concept_map": [{"from": "legacy_amounts", "to": "successor_amounts"}],
            "program_scope_updates": [],
        }
    )
    return prove_concept_map(
        legacy_raw=legacy_raw,
        successor_raw=successor_raw,
        request=request,
        dependent_raws={},
    )


@st.composite
def _ladder_pairs(draw):
    legacy = draw(_ladders())
    if draw(st.booleans()):
        # Re-slice the legacy values over a successor window so equality is
        # common; optionally perturb one cell.
        successor = draw(_ladders(first_start=draw(st.integers(0, 25))))
        resliced = []
        for start, end, _values in successor:
            source = _active(legacy, start, indexed=True)
            resliced.append((start, end, dict(source) if source else {0: 0}))
        if draw(st.booleans()) and resliced:
            index = draw(st.integers(0, len(resliced) - 1))
            start, end, values = resliced[index]
            key = draw(st.sampled_from(sorted(values)))
            values = {**values, key: values[key] + 1}
            resliced[index] = (start, end, values)
        return legacy, resliced
    return legacy, draw(_ladders())


@settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(pair=_ladder_pairs(), indexed=st.booleans())
def test_the_concept_proof_agrees_with_a_day_by_day_comparison(pair, indexed):
    legacy, successor = pair
    expected, expected_keys = _day_by_day(legacy, successor, indexed=indexed)
    try:
        proofs = _prove(
            _module(_rule("legacy_amounts", legacy, indexed=indexed)),
            _module(_rule("successor_amounts", successor, indexed=indexed)),
        )
    except SuccessorRepointError:
        accepted = False
    else:
        accepted = True
    assert accepted is expected, (legacy, successor, indexed)
    if accepted:
        (proof,) = proofs.proofs
        assert proof.keys == expected_keys
        assert proof.window_start == _day(successor[0][0])
        end = successor[-1][1]
        assert proof.window_end == (_day(end) if end is not None else None)


# ---------------------------------------------------------------------------
# Window flags
# ---------------------------------------------------------------------------


@st.composite
def _use_windows(draw):
    count = draw(st.integers(min_value=1, max_value=4))
    windows = []
    for _ in range(count):
        start = draw(st.integers(min_value=-10, max_value=40))
        end = draw(st.one_of(st.none(), st.integers(min_value=start, max_value=60)))
        windows.append((start, end))
    return windows


@PROPERTY_SETTINGS
@given(windows=_use_windows(), successor_open=st.booleans())
def test_window_flags_match_their_geometric_definition(windows, successor_open):
    successor_end = None if successor_open else 30
    ladder = [(0, successor_end, {0: 1})]
    legacy = [(-20, None, {0: 1})]
    rules = []
    for index, (start, end) in enumerate(windows):
        version: dict[str, object] = {
            "effective_from": _day(start),
            "formula": "legacy_amounts[0] + 1",
        }
        if end is not None:
            version["effective_to"] = _day(end)
        rules.append(
            {
                "name": f"use_{index}",
                "kind": "derived",
                "entity": "TaxUnit",
                "dtype": "Money",
                "period": "Year",
                "versions": [version],
            }
        )
    dependent = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "imports": ["us:policies/irs/legacy-table"],
            "module": {},
            "rules": rules,
        },
        sort_keys=False,
    ).encode("utf-8")
    request = load_repoint_request_payload(
        {
            "schema": ENVELOPE_SCHEMA,
            "legacy_primary": "us/policies/irs/legacy-table.yaml",
            "successor_primary": "us/policies/irs/page-15.yaml",
            "dependents": ["us/statutes/26/32.yaml"],
            "concept_map": [{"from": "legacy_amounts", "to": "successor_amounts"}],
            "program_scope_updates": [],
        }
    )
    proofs = prove_concept_map(
        legacy_raw=_module(_rule("legacy_amounts", legacy, indexed=True)),
        successor_raw=_module(_rule("successor_amounts", ladder, indexed=True)),
        request=request,
        dependent_raws={"us/statutes/26/32.yaml": dependent},
    )
    distinct = set(windows)
    assert proofs.pre_window_behavior_change is any(
        start < 0 for start, _end in distinct
    )
    assert proofs.post_window_behavior_change is (
        successor_end is not None
        and any(end is None or end > successor_end for _start, end in distinct)
    )
    assert proofs.behavior_change_outside_successor_window is (
        proofs.pre_window_behavior_change or proofs.post_window_behavior_change
    )
    assert len(proofs.dependent_use_windows) == len(distinct)


# ---------------------------------------------------------------------------
# Receipt identity
# ---------------------------------------------------------------------------

_JSON_SCALARS = st.one_of(
    st.none(), st.booleans(), st.integers(-5, 5), st.text(max_size=4)
)
_JSON = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(max_size=3), children, max_size=3),
    ),
    max_leaves=12,
)


def _reordered(value, rng):
    if isinstance(value, dict):
        keys = list(value)
        rng.shuffle(keys)
        return {key: _reordered(value[key], rng) for key in keys}
    if isinstance(value, list):
        return [_reordered(item, rng) for item in value]
    return value


@PROPERTY_SETTINGS
@given(
    payload=st.dictionaries(st.text(max_size=4), _JSON, min_size=1, max_size=5),
    rng=st.randoms(use_true_random=False),
)
def test_the_receipt_digest_ignores_key_order(payload, rng):
    assert receipt_identity_sha256(_reordered(payload, rng)) == receipt_identity_sha256(
        payload
    )


@PROPERTY_SETTINGS
@given(
    payload=st.dictionaries(st.text(max_size=4), _JSON, min_size=1, max_size=5),
    data=st.data(),
)
def test_the_receipt_digest_changes_when_any_field_changes(payload, data):
    key = data.draw(st.sampled_from(sorted(payload)))
    replacement = data.draw(_JSON)
    assume(
        json.dumps(replacement, sort_keys=True)
        != json.dumps(payload[key], sort_keys=True)
    )
    changed = {**payload, key: replacement}
    assert receipt_identity_sha256(changed) != receipt_identity_sha256(payload)
