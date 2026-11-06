"""Round 20: forward source indexes and the complete Hebrew-script route gate."""

from __future__ import annotations

import bisect
import sys
import unicodedata
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness import source_completeness as completeness
from tests.hebrew_gate import open_hebrew_gate
from tests.test_source_completeness_hebrew_rendering_audit import _route_accepted


@given(
    st.lists(st.integers(), max_size=200),
    st.lists(st.tuples(st.integers(), st.integers()), max_size=200),
)
@settings(max_examples=200, deadline=None)
def test_forward_start_cursor_agrees_with_bisect(starts, raw_bounds):
    """Every sorted index and every pair of forward bounds has the old answer."""

    starts.sort()
    lows = sorted(min(pair) for pair in raw_bounds)
    highs = sorted(max(pair) for pair in raw_bounds)
    cursor = hebrew._StartCursor(starts)
    for low, high in zip(lows, highs):
        index = bisect.bisect_left(starts, low)
        expected = index < len(starts) and starts[index] < high
        assert cursor.contains(low, high) is expected


@pytest.mark.parametrize("bounds", [((2, 4), (1, 5)), ((2, 4), (2, 3))])
def test_forward_start_cursor_asserts_when_a_query_bound_moves_back(bounds):
    cursor = hebrew._StartCursor([1, 3, 5])
    cursor.contains(*bounds[0])
    with pytest.raises(AssertionError):
        cursor.contains(*bounds[1])


class _ReadCountingStarts(Sequence[int]):
    """Count actual sequence reads without changing the indexed values."""

    def __init__(self, starts: Sequence[int]) -> None:
        self.starts = starts
        self.reads = 0

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index):
        self.reads += 1
        return self.starts[index]


@pytest.mark.parametrize("repetitions", (8_000, 16_000, 32_000))
@pytest.mark.parametrize("family", ("disclaimer", "claim-denial"))
def test_disclaimer_source_queries_read_the_three_arrays_linearly(
    monkeypatch, repetitions, family
):
    """Growing disclaimer and claim/denial arrays use bounded reads per citation."""

    arrays = []
    original = hebrew._StartCursor

    def counting_cursor(starts):
        counted = _ReadCountingStarts(starts)
        arrays.append(counted)
        return original(counted)

    monkeypatch.setattr(hebrew, "_StartCursor", counting_cursor)
    frame = (
        "סעיף 2 נותר בעינו ו"
        if family == "disclaimer"
        else "הטענה שהתשלום מחושב בהתאם לסעיף 2 שגוי ו"
    )
    assert hebrew._linked_source_instruments(frame * repetitions + "סוף.") == {}
    # _linked_source_instruments builds claim, denial and disclaimer cursors
    # in that order. Disclaimer refusal short-circuits the other two queries.
    reads = [array.reads for array in arrays]
    expected = (
        [0, 0, repetitions]
        if family == "disclaimer"
        else [repetitions, 2 * repetitions - 1, 0]
    )
    assert reads == expected
    print(f"round20 {family} n={repetitions}: claims/denials/disclaimers={reads}")


def _in_hebrew_block(code: int) -> bool:
    return 0x0590 <= code <= 0x05FF or 0xFB1D <= code <= 0xFB4F


# Derive this set using the interpreter's own NFKC rather than asserting a
# Unicode-version-dependent count. CI uses Python 3.13 and local runs use 3.14.
HEBREW_ROUTE_CODEPOINTS = tuple(
    code
    for code in range(sys.maxunicode + 1)
    if _in_hebrew_block(code)
    or any(
        _in_hebrew_block(ord(char)) for char in unicodedata.normalize("NFKC", chr(code))
    )
)


def test_script_gate_covers_raw_blocks_and_all_nfkc_preimages():
    assert set(range(0x0590, 0x0600)) <= set(HEBREW_ROUTE_CODEPOINTS)
    assert set(range(0xFB1D, 0xFB50)) <= set(HEBREW_ROUTE_CODEPOINTS)
    assert set(range(0x2135, 0x2139)) <= set(HEBREW_ROUTE_CODEPOINTS)
    for code in HEBREW_ROUTE_CODEPOINTS:
        assert hebrew.holds_hebrew_script(chr(code)), f"U+{code:04X}"
    for text in ("English words", "Deutsch: äöüß", "section 2K", "section 2K"):
        assert not hebrew.holds_hebrew_script(text)
    print(f"round20 Hebrew route code points={len(HEBREW_ROUTE_CODEPOINTS)}")


_ROUTE_CASES = (
    (
        "runtime-gap",
        completeness._reason_names_source_bound_runtime_gap,
        "us-nj/statute/54a:8-6(c) cannot be encoded until the runtime has a "
        "document-submission capability for the required federal return copy.",
        "(c) The director may require the taxpayer to submit a copy of the "
        "taxpayer's federal return.",
        {"corpus_citation_path": "us-nj/statute/54a:8-6", "path": ("c",)},
    ),
    (
        "louisiana-insufficiency",
        completeness._reason_dependency_is_source_bound,
        "R.S. 47:32 lacks the formula.",
        "(a) The amount is determined under R.S. 47:32.",
        {"corpus_citation_path": "us-la/statute/47:295", "path": ("a",)},
    ),
    (
        "same-act",
        completeness._reason_names_missing_same_act_dependency,
        "Miss. Code section 27-7-5(1)(b)(ii)(7) provides the 2030 rate except "
        "as otherwise provided in Section 2 of this act; no executable RuleSpec "
        "output for Section 2 of this act is supplied in the available context.",
        "For calendar year 2030 and later, except as otherwise provided in "
        "Section 2 of this act, the rate shall be three percent (3%).",
        {
            "corpus_citation_path": "us-ms/statute/27-7-5",
            "path": ("1", "b", "ii", "7"),
        },
    ),
)


@pytest.mark.parametrize("name,read,reason,source,kwargs", _ROUTE_CASES)
@pytest.mark.parametrize("position", ("reason", "source"))
def test_every_hebrew_block_or_nfkc_preimage_refuses_each_english_route(
    name, read, reason, source, kwargs, position
):
    """All three routes refuse in both positions, with their positive control."""

    assert read(reason, source, **kwargs), name
    for code in HEBREW_ROUTE_CODEPOINTS:
        marked_reason = reason + " " + chr(code) if position == "reason" else reason
        marked_source = source + " " + chr(code) if position == "source" else source
        # The Hebrew code point opens the gate for its candidate.
        with open_hebrew_gate():
            assert not read(marked_reason, marked_source, **kwargs), (
                name,
                position,
                f"U+{code:04X}",
            )


@pytest.mark.parametrize("name,read,reason,source,kwargs", _ROUTE_CASES)
@pytest.mark.parametrize("position", ("reason", "source"))
def test_public_route_gate_keeps_raw_marks_before_folding(
    name, read, reason, source, kwargs, position
):
    """The public analyzer retains the gate when its reason copy drops marks."""

    path = kwargs["path"]
    if name == "same-act":
        reason = reason.replace("27-7-5(1)(b)(ii)(7)", "27-7-5(a)")
        source = "(a) " + source
        path = ("a",)
    row = {"citation": kwargs["corpus_citation_path"], "path": path}
    assert _route_accepted(row, reason, source), name
    for code in HEBREW_ROUTE_CODEPOINTS:
        marked_reason = reason + " " + chr(code) if position == "reason" else reason
        marked_source = source + " " + chr(code) if position == "source" else source
        assert not _route_accepted(row, marked_reason, marked_source), (
            name,
            position,
            f"U+{code:04X}",
        )
