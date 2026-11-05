"""Invariants of the Armenian ARLIS amendment-history recall mask.

The recognizer may only ever *remove* numeric recall obligations that sit in
a fully validated, standalone, depth-zero ledger. These tests pin:

* a differential against an independent, regex-free reference parser of the
  same grammar (the encoder prompt's contract);
* mask-only behaviour: recognized spans become spaces in place, so recall-text
  positions keep their source origins;
* operative residue, nesting and malformed tokens never lose an obligation;
* a work bound stated in operation counts, not elapsed time;
* a differential over every repository fixture, listing each changed verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAD = "Շահառուին վճարել 500 դրամ:"

_DASHES = "-֊‐‑‒–—―−﹘﹣－"
_DOTS = ".․"
_ACTIONS = ("փոփ", "լրաց", "խմբ")
_ASCII_DIGITS = "0123456789"


# ---------------------------------------------------------------------------
# Independent reference parser (no regular expressions).
# ---------------------------------------------------------------------------


def _is_digits(text: str) -> bool:
    # ASCII only: the prompt's `DD.MM.YY` and `N` are ASCII digits.
    return bool(text) and all(character in _ASCII_DIGITS for character in text)


def _ref_action(token: str) -> bool:
    if token and token[-1] in _DOTS:
        token = token[:-1]
    return token.lower() in _ACTIONS


def _ref_date(token: str) -> bool:
    if len(token) not in (8, 10) or token[2] not in _DOTS or token[5] not in _DOTS:
        return False
    return _is_digits(token[:2]) and _is_digits(token[3:5]) and _is_digits(token[6:])


def _ref_law_identifier(token: str) -> bool:
    if token.startswith("ՀՕ"):
        rest = token[2:]
    elif token.startswith("Հ"):
        rest = token[1:]
    else:
        return False
    # DASH number [DASH number] DASH Ն
    if len(rest) < 4 or rest[0] not in _DASHES or rest[-1] != "Ն":
        return False
    if rest[-2] not in _DASHES:
        return False
    middle = rest[1:-2]
    dashes = [index for index, character in enumerate(middle) if character in _DASHES]
    if not dashes:
        return _is_digits(middle)
    if len(dashes) > 1:
        return False
    return _is_digits(middle[: dashes[0]]) and _is_digits(middle[dashes[0] + 1 :])


def _is_separator(character: str) -> bool:
    return character.isspace() or character in ",;"


def _ref_history(history: str) -> bool:
    """Tokens split on maximal separator runs; a citation is date WS identifier."""

    tokens: list[tuple[str, str]] = []  # (separator before token, token)
    index = 0
    trailing = ""
    while index < len(history):
        start = index
        while index < len(history) and _is_separator(history[index]):
            index += 1
        separator = history[start:index]
        start = index
        while index < len(history) and not _is_separator(history[index]):
            index += 1
        if start < index:
            tokens.append((separator, history[start:index]))
        else:
            trailing = separator
    # Only whitespace may lead or trail the token sequence.
    if not tokens or not all(c.isspace() for c in tokens[0][0] + trailing):
        return False
    seen_action = seen_citation = False
    position = 0
    while position < len(tokens):
        token = tokens[position][1]
        if _ref_action(token):
            seen_action = True
            position += 1
            continue
        if _ref_date(token) and position + 1 < len(tokens):
            separator, identifier = tokens[position + 1]
            if all(c.isspace() for c in separator) and _ref_law_identifier(identifier):
                seen_citation = True
                position += 2
                continue
        return False
    return seen_action and seen_citation


def _ref_body(body: str) -> bool:
    text = body.strip()
    # Article label: digits[(dot)digits] DASH (ին|րդ)
    index = 0
    while index < len(text) and text[index] in _ASCII_DIGITS:
        index += 1
    if index == 0:
        return False
    if index < len(text) and text[index] in _DOTS:
        index += 1
        start = index
        while index < len(text) and text[index] in _ASCII_DIGITS:
            index += 1
        if index == start:
            return False
    if index >= len(text) or text[index] not in _DASHES:
        return False
    index += 1
    if text[index : index + 2] not in ("ին", "րդ"):
        return False
    index += 2
    start = index
    while index < len(text) and text[index].isspace():
        index += 1
    if index == start or not text.startswith("հոդվածը", index):
        return False
    index += len("հոդվածը")
    start = index
    while index < len(text) and text[index].isspace():
        index += 1
    if index == start or index >= len(text):
        return False
    return _ref_history(text[index:])


def _ref_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    depth = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character == "(":
            line_start = text.rfind("\n", 0, index) + 1
            closing = index + 1
            while closing < len(text) and text[closing] not in "()":
                closing += 1
            standalone_start = all(c in " \t" for c in text[line_start:index])
            if depth == 0 and standalone_start and closing < len(text):
                after = closing + 1
                while after < len(text) and text[after] in " \t":
                    after += 1
                ends_line = (
                    after == len(text)
                    or text[after] == "\n"
                    or text[after : after + 2] == "\r\n"
                )
                if (
                    text[closing] == ")"
                    and ends_line
                    and _ref_body(text[index + 1 : closing])
                ):
                    spans.append((index + 1, closing))
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        index += 1
    return spans


def _impl_spans(text: str) -> list[tuple[int, int]]:
    return list(sc._armenian_amendment_history_numeric_recall_spans(text))


# ---------------------------------------------------------------------------
# Generators.
# ---------------------------------------------------------------------------

_dash = st.sampled_from(_DASHES)
_dot = st.sampled_from(_DOTS)
_digits = st.from_regex(r"[0-9]{1,3}", fullmatch=True)
_separator = st.sampled_from([" ", "\t", ", ", " , ", "; ", ",", ";", "\n", ",\n"])


@st.composite
def _action(draw):
    word = draw(st.sampled_from(_ACTIONS))
    if draw(st.booleans()):
        word = word[0].upper() + word[1:]
    return word + draw(st.sampled_from(["", ".", "․"]))


@st.composite
def _citation(draw):
    day = draw(st.from_regex(r"[0-9]{2}", fullmatch=True))
    month = draw(st.from_regex(r"[0-9]{2}", fullmatch=True))
    year = draw(st.from_regex(r"[0-9]{2}|[0-9]{4}", fullmatch=True))
    dot = draw(_dot)
    prefix = draw(st.sampled_from(["Հ", "ՀՕ"]))
    number = draw(_digits)
    if draw(st.booleans()):
        number += draw(_dash) + draw(_digits)
    space = draw(st.sampled_from([" ", "  ", "\n", "\t"]))
    return (
        f"{day}{dot}{month}{dot}{year}{space}{prefix}{draw(_dash)}{number}"
        f"{draw(_dash)}Ն"
    )


@st.composite
def _valid_ledger(draw):
    label = draw(_digits)
    if draw(st.booleans()):
        label += draw(_dot) + draw(_digits)
    label += draw(_dash) + draw(st.sampled_from(["ին", "րդ"]))
    items = [draw(_action()), draw(_citation())]
    items += draw(st.lists(st.one_of(_action(), _citation()), max_size=4))
    items = draw(st.permutations(items))
    history = items[0] + "".join(draw(_separator) + item for item in items[1:])
    return f"({label} հոդվածը {history})"


_NOISE = st.sampled_from(
    [
        "",
        "x",
        "500",
        "(",
        ")",
        ",",
        ";",
        " ",
        "\n",
        "\r\n",
        "հօ",
        "Ն",
        "-",
        "ՀՕ-1-Ն",
        "07.12.22",
        "7.12.22",
        "փոփ",
        "․",
        "Ա",
        "١",
    ]
)


@st.composite
def _mutated_ledger(draw):
    ledger = draw(_valid_ledger())
    for _ in range(draw(st.integers(min_value=1, max_value=3))):
        position = draw(st.integers(min_value=0, max_value=len(ledger)))
        width = draw(st.integers(min_value=0, max_value=3))
        ledger = ledger[:position] + draw(_NOISE) + ledger[position + width :]
    return ledger


@st.composite
def _single_kind_ledger(draw):
    """A ledger whose history drops every action or every citation."""
    label = draw(_digits) + draw(_dash) + draw(st.sampled_from(["ին", "րդ"]))
    item = draw(st.sampled_from([_action(), _citation()]))
    items = draw(st.lists(item, min_size=1, max_size=4))
    history = items[0] + "".join(draw(_separator) + entry for entry in items[1:])
    return f"({label} հոդվածը {history})"


_NON_ASCII_DIGITS = st.sampled_from(["٠", "١", "٧", "۵", "५", "２", "৩"])


@st.composite
def _non_ascii_digit_ledger(draw):
    """A valid ledger with one of its ASCII digits swapped for a Unicode digit."""
    ledger = draw(_valid_ledger())
    positions = [index for index, char in enumerate(ledger) if char in _ASCII_DIGITS]
    index = draw(st.sampled_from(positions))
    return ledger[:index] + draw(_NON_ASCII_DIGITS) + ledger[index + 1 :]


@st.composite
def _source(draw):
    ledger = draw(
        st.one_of(
            _valid_ledger(),
            _mutated_ledger(),
            _single_kind_ledger(),
            _non_ascii_digit_ledger(),
        )
    )
    before = draw(st.sampled_from(["", "\n", "(ա\n", "(ա)\n", " ", "x "]))
    after = draw(st.sampled_from(["", "\n", "\r\n", " \n", ")", " x", "\n)"]))
    return f"{LEAD}\n{before}{ledger}{after}"


_SETTINGS = settings(
    max_examples=600,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Differential and property tests.
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_source())
def test_recognizer_matches_independent_reference_parser(source):
    assert _impl_spans(source) == _ref_spans(source)


@_SETTINGS
@given(_valid_ledger())
def test_every_generated_valid_standalone_ledger_is_masked(ledger):
    source = f"{LEAD}\n{ledger}"
    start = source.index(ledger)
    assert _impl_spans(source) == [(start + 1, start + len(ledger) - 1)]


@_SETTINGS
@given(_source())
def test_masking_preserves_length_and_every_unmasked_character(source):
    spans = _impl_spans(source)
    masked = sc._mask_numeric_spans(source, spans)
    assert len(masked) == len(source)
    covered = {index for start, end in spans for index in range(start, end)}
    for index, (before, after) in enumerate(zip(source, masked, strict=True)):
        assert after == (" " if index in covered else before)


@_SETTINGS
@given(_source())
def test_spans_are_depth_zero_standalone_parentheticals(source):
    for start, end in _impl_spans(source):
        # Only the body is blanked; both parentheses stay in the recall text.
        assert source[start - 1] == "(" and source[end] == ")"
        assert "(" not in source[start:end]
        assert ")" not in source[start:end]
        start -= 1
        depth = 0
        for character in source[:start]:
            if character == "(":
                depth += 1
            elif character == ")":
                depth = max(0, depth - 1)
        assert depth == 0
        line_start = source.rfind("\n", 0, start) + 1
        assert source[line_start:start].strip(" \t") == ""


@_SETTINGS
@given(
    _valid_ledger(),
    st.sampled_from(
        [
            ", լրացուցիչ 500 դրամ",
            " ուժի մեջ է մտնում 2024 թվականի հուլիսի 1-ից",
            " 750",
            "; 12",
        ]
    ),
)
def test_operative_residue_keeps_every_obligation(ledger, residue):
    source = f"{LEAD}\n{ledger[:-1]}{residue})"
    assert _impl_spans(source) == []
    assert sc.authoritative_numeric_recall_text(source) == source


@_SETTINGS
@given(_valid_ledger(), st.integers(min_value=1, max_value=3))
def test_nested_ledger_keeps_every_obligation(ledger, depth):
    source = f"{LEAD}\n" + "(ա\n" * depth + ledger + "\n)" * depth
    assert _impl_spans(source) == []
    assert sc.authoritative_numeric_recall_text(source) == source


@_SETTINGS
@given(_valid_ledger())
def test_recall_positions_stay_on_source_characters(ledger):
    source = f"{LEAD}\n{ledger}"
    source_map = sc._NumericRecallSourceMap(list(range(len(source))))
    recall = sc.authoritative_numeric_recall_text(source, _source_map=source_map)
    start = recall.index("500")
    assert source_map.source_span(start, start + 3) == (start, start + 3)
    assert recall.split("\n", 1)[0] == LEAD
    assert recall.split("\n", 1)[1].strip("( )") == ""


def test_lowercase_identifier_differs_from_reference_only_if_both_reject():
    source = f"{LEAD}\n(1-ին հոդվածը փոփ. 07.12.22 հօ-538-ն)"
    assert _impl_spans(source) == _ref_spans(source) == []


# ---------------------------------------------------------------------------
# Work bound in operation counts.
# ---------------------------------------------------------------------------


class _CountingPattern:
    def __init__(self, pattern):
        self._pattern = pattern
        self.calls = 0

    def __getattr__(self, name):
        attribute = getattr(self._pattern, name)

        def counted(*args, **kwargs):
            self.calls += 1
            return attribute(*args, **kwargs)

        return counted


@pytest.mark.parametrize("candidates", (1, 10, 100))
def test_regex_work_is_bounded_per_candidate(monkeypatch, candidates):
    names = (
        "_ARMENIAN_AMENDMENT_HISTORY_ARTICLE",
        "_ARMENIAN_AMENDMENT_HISTORY_ACTION",
        "_ARMENIAN_AMENDMENT_HISTORY_CITATION",
        "_ARMENIAN_AMENDMENT_HISTORY_SEQUENCE",
    )
    counters = {}
    for name in names:
        counters[name] = _CountingPattern(getattr(sc, name))
        monkeypatch.setattr(sc, name, counters[name])
    ledger = "(1-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)"
    source = LEAD + ("\n" + ledger) * candidates

    assert len(_impl_spans(source)) == candidates
    # One article match and at most one action, citation and sequence check
    # per standalone candidate: work grows with the candidates, not with any
    # rescanning of the text.
    assert all(counter.calls <= candidates for counter in counters.values())


def test_depth_map_is_built_once_and_never_for_texts_without_candidates(
    monkeypatch,
):
    class _CountingString(str):
        iterations = 0

        def __iter__(self):
            type(self).iterations += 1
            return super().__iter__()

    without = _CountingString(LEAD + " (inline 1-ին հոդվածը)")
    assert _impl_spans(without) == []
    assert _CountingString.iterations == 0

    ledger = "(1-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)"
    many = _CountingString(LEAD + ("\n" + ledger) * 50)
    assert len(_impl_spans(many)) == 50
    assert _CountingString.iterations == 1


# ---------------------------------------------------------------------------
# Differential over the repository's fixtures.
# ---------------------------------------------------------------------------

# Every fixture whose numeric recall text changes because of this recognizer,
# as (path, number of masked ledgers). Adding a fixture that changes must be a
# deliberate edit here.
EXPECTED_CHANGED_FIXTURE_VERDICTS: dict[str, int] = {}
_FIXTURE_ROOTS = ("tests/fixtures", "benchmarks")
_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv"}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _fixture_texts(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    yield raw
    if path.suffix == ".json":
        try:
            yield from _strings(json.loads(raw))
        except ValueError:
            return
    elif path.suffix == ".jsonl":
        for line in raw.splitlines():
            try:
                yield from _strings(json.loads(line))
            except ValueError:
                continue


def test_fixture_differential_lists_every_changed_verdict():
    changed: dict[str, int] = {}
    scanned = 0
    for root in _FIXTURE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
                continue
            scanned += 1
            masked = sum(len(_impl_spans(text)) for text in _fixture_texts(path))
            if masked:
                changed[path.relative_to(REPO_ROOT).as_posix()] = masked
    assert scanned > 0
    assert changed == EXPECTED_CHANGED_FIXTURE_VERDICTS


def test_masked_ledger_keeps_earlier_session_law_history_demanded():
    """The history chain before a masked ledger stays a recall obligation.

    This checks the outcome, not the call order: because the parentheses are
    kept, the blanked ledger cannot make the chain look terminal whichever of
    the two steps runs first.
    """
    source = (
        "The credit is $162 for 65 years.\n\n"
        "L.1976, c.47, s. 54A:1-2; amended 2003, c.246, s.39.\n"
        "(1-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)\n"
    )
    without_ledger = source.split("(1-ին")[0]
    cleaned = sc.authoritative_numeric_recall_text(source)

    # The history chain is not terminal in the source, so it stays demanded,
    # exactly as on main; only the ledger itself is masked.
    assert "L.1976, c.47, s. 54A:1-2; amended 2003, c.246, s.39." in cleaned
    assert "538" not in cleaned
    assert cleaned.startswith(without_ledger.rstrip("\n"))


@pytest.mark.parametrize("width", (1_000, 50_000, 200_000))
def test_long_whitespace_run_inside_body_is_handled(width):
    # The article pattern runs on the stripped body with no lazy quantifier,
    # so a long whitespace run before a stray token is a single linear scan.
    # Python's re exposes no step counter; this pins the result on large input.
    rejected = f"{LEAD}\n(1-ին հոդվածը փոփ.{' ' * width}x)\n"
    accepted = f"{LEAD}\n(1-ին հոդվածը փոփ.{' ' * width}07.12.22 ՀՕ-538-Ն)\n"
    assert _impl_spans(rejected) == _ref_spans(rejected) == []
    assert _impl_spans(accepted) == _ref_spans(accepted) != []


@pytest.mark.parametrize(
    "reference",
    ("section", "Section", "paragraph", "page", "article", "Article", "§"),
)
def test_masked_ledger_never_joins_a_reference_word_to_the_next_line(reference):
    source = (
        f"as provided in {reference}\n"
        "(1-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)\n"
        "2.5 percent of income applies."
    )
    control = source.replace("(1-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)", "(x)")
    cleaned = sc.authoritative_numeric_recall_text(source)

    assert "538" not in cleaned
    assert ("2.5" in cleaned) == (
        "2.5" in sc.authoritative_numeric_recall_text(control)
    )
    # The eval coverage path removes cross-references after masking; with the
    # parentheses blanked too, "section" there would swallow 2.5.
    from axiom_encode.harness.evals import _numeric_occurrence_source_text

    assert "2.5" in _numeric_occurrence_source_text(
        sc.mask_armenian_amendment_history(source)
    )


@pytest.mark.parametrize(
    "ledger",
    (
        "(1-ին հոդվածը փոփ. ٠٧.١٢.٢٢ ՀՕ-٥٣٨-Ն)",
        "(١-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)",
        "(1-ին հոդվածը փոփ. 07.12.22ՀՕ-538-Ն)",
    ),
)
def test_shapes_outside_the_prompt_grammar_keep_their_numbers(ledger):
    source = f"{LEAD}\n{ledger}"
    assert _impl_spans(source) == []
    assert sc.authoritative_numeric_recall_text(source) == source


@pytest.mark.parametrize(
    "ledger",
    (
        "(293.1-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)",
        "(1-ին հոդվածը ՓՈՓ. 07.12.22 ՀՕ-538-Ն)",
        "(1-ին հոդվածը Լրաց 07.12.22 ՀՕ-538-Ն)",
    ),
)
def test_prompt_described_variants_are_masked(ledger):
    source = f"{LEAD}\n{ledger}"
    assert _impl_spans(source) != []


def test_prompt_described_dash_and_dot_variants_are_masked():
    source = f"{LEAD}\n(1–ին հոդվածը փոփ․ 07․12․22 ՀՕ—538—Ն)"
    assert _impl_spans(source) == _ref_spans(source) != []


def test_comma_right_after_article_label_keeps_numbers():
    source = f"{LEAD}\n(1-ին հոդվածը, փոփ. 07.12.22 ՀՕ-538-Ն)"
    assert _impl_spans(source) == _ref_spans(source) == []


# ---------------------------------------------------------------------------
# Examples that pin each guard of the recognizer on its own.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ledger",
    (
        # Citation without any action: the action-search guard.
        "(1-ին հոդվածը 22.12.10 ՀՕ-17-Ն)",
        "(1-ին հոդվածը 22.12.10 ՀՕ-17-Ն, 23.01.11 Հ-5-Ն)",
        # Action without any citation: the citation-search guard.
        "(1-ին հոդվածը փոփ.)",
        "(1-ին հոդվածը փոփ., լրաց)",
        # Non-ASCII digits in exactly one numeric field each.
        "(1-ին հոդվածը փոփ. ٢٢.١٢.١٠ ՀՕ-17-Ն)",
        "(1-ին հոդվածը փոփ. 22.12.2٠1٠ ՀՕ-17-Ն)",
        "(1-ին հոդվածը փոփ. 22.12.10 ՀՕ-١٧-Ն)",
        "(1-ին հոդվածը փոփ. 22.12.10 ՀՕ-17-٣-Ն)",
        "(١-ին հոդվածը փոփ. 22.12.10 ՀՕ-17-Ն)",
        "(1․٢-ին հոդվածը փոփ. 22.12.10 ՀՕ-17-Ն)",
    ),
)
def test_each_recognizer_guard_keeps_its_numbers(ledger):
    source = f"{LEAD}\n{ledger}"
    assert _ref_spans(source) == []
    assert _impl_spans(source) == []
    assert sc.authoritative_numeric_recall_text(source) == source
    assert sc.mask_armenian_amendment_history(source) == source


def test_stray_list_marker_before_ledger_does_not_nest_it():
    # The depth map clamps at zero, so an unmatched `1)` list marker cannot
    # push a later standalone ledger to depth -1 and lose it; the list
    # amount stays demanded.
    ledger = "(1-ին հոդվածը փոփ. 07.12.22 ՀՕ-538-Ն)"
    source = f"1) Ամսական 90000 դրամ\n{ledger}"
    start = source.index(ledger)

    assert _impl_spans(source) == [(start + 1, start + len(ledger) - 1)]
    assert "90000" in sc.authoritative_numeric_recall_text(source)


# ---------------------------------------------------------------------------
# Prompt parity for the dash and dot characters.
# ---------------------------------------------------------------------------


def _armenian_prompt_bullet() -> str:
    from axiom_encode.prompts.encoder import ENCODER_PROMPT

    start = ENCODER_PROMPT.index("- In Armenian ARLIS text")
    return ENCODER_PROMPT[start:][: ENCODER_PROMPT[start:].index("\n- ", 1)]


def _codepoints_named(text: str) -> set[str]:
    import re

    named: set[str] = set()
    for match in re.finditer(r"U\+([0-9A-F]{4})(?: to U\+([0-9A-F]{4}))?", text):
        first = int(match.group(1), 16)
        last = int(match.group(2) or match.group(1), 16)
        named.update(chr(code) for code in range(first, last + 1))
    return named


def _characters_matching(pattern: str) -> set[str]:
    import re

    compiled = re.compile(pattern)
    return {
        chr(code)
        for code in range(0x110000)
        if not 0xD800 <= code <= 0xDFFF and compiled.fullmatch(chr(code))
    }


def test_prompt_names_exactly_the_dashes_the_recognizer_accepts():
    bullet = " ".join(_armenian_prompt_bullet().split())
    dash_sentence = bullet[
        bullet.index("Each `-` above may be") : bullet.index("and no other dash")
    ]
    named = _codepoints_named(dash_sentence) | {"-"}  # "the hyphen-minus"
    assert "hyphen-minus" in dash_sentence

    assert named == _characters_matching(sc._ARMENIAN_AMENDMENT_HISTORY_DASH)
    for dash in named:
        source = f"{LEAD}\n(1{dash}ին հոդվածը փոփ. 07.12.22 ՀՕ{dash}538{dash}Ն)"
        assert _impl_spans(source) != [], hex(ord(dash))


def test_prompt_names_exactly_the_dots_the_recognizer_accepts():
    bullet = " ".join(_armenian_prompt_bullet().split())
    dot_sentence = bullet[
        bullet.index("each `.` may be") : bullet.index("Whitespace must follow")
    ]
    named = _codepoints_named(dot_sentence) | {"."}

    assert named == _characters_matching(sc._ARMENIAN_AMENDMENT_HISTORY_DOT)


# Characters a dot position must reject, plus the two it must accept. The
# Armenian full stop U+0589 is the likeliest wrong addition.
_DOT_PROBES = (".", "\u2024", "\u0589", "\u00b7", "\uff0e", "\u2027", "\u06d4")


def _ledger_with_dots(*, label: str, action: str, first: str, second: str) -> str:
    return f"(1{label}2-ին հոդվածը փոփ{action} 07{first}12{second}22 ՀՕ-538-Ն)"


@pytest.mark.parametrize("position", ("label", "action", "first", "second"))
def test_every_dot_position_accepts_exactly_the_prompt_dots(position):
    # Drives each of the four places a dot appears with real ledgers, so a
    # dot added to any one of them alone fails here.
    for dot in _DOT_PROBES:
        dots = {"label": ".", "action": ".", "first": ".", "second": "."}
        dots[position] = dot
        source = f"{LEAD}\n{_ledger_with_dots(**dots)}"
        accepted = _impl_spans(source) != []
        assert accepted == (dot in {".", "\u2024"}), (position, hex(ord(dot)))
        assert accepted == (_ref_spans(source) != [])


@pytest.mark.parametrize("dash", ("\u2e3a", "\u2e17", "\u301c", "\u1806", "\u2053"))
def test_a_dash_the_prompt_does_not_name_keeps_the_numbers(dash):
    source = f"{LEAD}\n(1-ին հոդվածը փոփ. 07.12.22 ՀՕ{dash}538{dash}Ն)"
    assert _impl_spans(source) == []
    assert dash not in " ".join(_armenian_prompt_bullet().split())
