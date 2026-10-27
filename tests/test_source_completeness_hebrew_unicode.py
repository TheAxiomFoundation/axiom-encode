"""Round 15: Unicode properties, rather than a list of invisible examples."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from axiom_encode.harness import _unicode_default_ignorables as unicode_data
from axiom_encode.harness import hebrew_source_text as hebrew

_SPEC = importlib.util.spec_from_file_location(
    "_hebrew_unicode_cases",
    Path(__file__).with_name("test_source_completeness_hebrew.py"),
)
assert _SPEC and _SPEC.loader
cases = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cases
_SPEC.loader.exec_module(cases)


def _content(branch: str, reason: str) -> str:
    return cases._deferral_content(branch, "placeholder").replace(
        '"placeholder"', json.dumps(reason)
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IGNORABLES = tuple(
    code
    for low, high in unicode_data.DEFAULT_IGNORABLE_RANGES
    for code in range(low, high + 1)
)


@pytest.mark.parametrize("version", unicode_data._SUPPORTED_VERSIONS)
def test_default_ignorable_table_matches_the_unicode_source_file(version):
    """The generated ranges and their receipt match the unmodified UCD file."""

    assert unicode_data.UNICODE_VERSION == unicodedata.unidata_version
    table = importlib.import_module(
        f"axiom_encode.harness._unicode_default_ignorables_{version.replace('.', '_')}"
    )
    assert table.UNICODE_VERSION == version
    path = (
        ROOT / "tests" / "fixtures" / "unicode" / f"DerivedCoreProperties-{version}.txt"
    )
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == table.DERIVED_CORE_PROPERTIES_SHA256
    independently_parsed = tuple(
        (int(low, 16), int(high or low, 16))
        for low, high in re.findall(
            r"^([0-9A-F]+)(?:\.\.([0-9A-F]+))?\s*; Default_Ignorable_Code_Point\s*#",
            data.decode(),
            re.MULTILINE,
        )
    )
    assert independently_parsed == table.DEFAULT_IGNORABLE_RANGES
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_default_ignorable_table.py"),
            "--unicode-version",
            version,
            "--check",
        ],
        check=True,
        timeout=30,
    )


def test_default_ignorables_exclude_visible_text_and_all_whitespace():
    assert len(DEFAULT_IGNORABLES) == 4174
    assert set(DEFAULT_IGNORABLES) <= hebrew.folded_marks()
    visible = "־׀׃׆אבג123ABCabc \t\n\r\v\f\u0085\u00a0\u1680\u2000\u2028\u2029\u202f\u205f\u3000"
    assert not set(map(ord, visible)) & hebrew.folded_marks()
    assert hebrew.fold_marks(visible) is visible
    assert not any(chr(code).isspace() for code in DEFAULT_IGNORABLES)
    # The Hebrew gate also preserves pure English and German text exactly.
    for text in ("not\u17b4 missing section 2", "nicht\u115f verfügbar"):
        assert hebrew.fold_marks(text) is text


NEGATORS = tuple(
    dict.fromkeys(
        cases._NEGATORS
        + cases._OTHER_NEGATORS
        + cases._LINK_NEGATORS
        + cases._OTHER_LINK_NEGATORS
        + (
            "אינה",
            "אינם",
            "אינן",
            "איני",
            "אינך",
            "אינכם",
            "אינכן",
            "איננו",
            "איננה",
            "אינני",
            "אל",
            "אף",
            "בלי",
            "מבלי",
        )
    )
)
SIGNALS = (
    "חסר",
    "חסרים",
    "חסרה",
    "חסרות",
    "טרם קודד",
    "טרם קודדה",
    "טרם קודדו",
    "עדיין לא קודד",
    "עדיין אינו מקודד",
    "עדיין אינה מקודדת",
    "עדיין אינם מקודדים",
    "עדיין אינן מקודדות",
    "בלתי זמין",
    "בלתי זמינה",
    "בלתי זמינים",
    "בלתי זמינות",
    "דורש",
    "דורשת",
    "דורשים",
    "דורשות",
    "תלוי ב",
    "תלויה ב",
    "תלויים ב",
    "תלויות ב",
    "עד אשר יהיה זמין",
    "עד שיקודד",
    "בלי",
    "ללא",
    "מבלי",
    "לא ניתן לחשב",
    "לא אפשר לקודד",
    "אי אפשר לקודד",
)
LABELS = tuple(
    f"({label})"
    for label in ("א", "ב", "ה1", "י״א", "יב׳", "1", "2א", "999", "א)(1)(א)(1)(א)(1")
)


def _literal_test_tokens():
    """Include every token spelled in the pinned Hebrew test tables."""

    literals = (
        hebrew.fold_marks(node.value)
        for node in ast.walk(ast.parse(Path(cases.__file__).read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    negators, signals, labels = set(NEGATORS), set(SIGNALS), set(LABELS)
    negators.update(row[0] for row in cases.ROUND_15_NEGATOR_TRANSLATIONS)
    for text in literals:
        for pattern in (
            hebrew._REASON_NEGATION,
            hebrew._NEGATIONS,
            hebrew._OTHER_NEGATION,
        ):
            negators.update(match.group() for match in pattern.finditer(text))
        signals.update(match.group() for match in hebrew._SIGNAL.finditer(text))
        labels.update(
            f"({match.group(1)})"
            for match in hebrew._CHAIN_LABEL.finditer(text)
            if hebrew.parse_outline_label(match.group(1))
        )
    return tuple(sorted(negators)), tuple(sorted(signals)), tuple(sorted(labels))


NEGATORS, SIGNALS, LABELS = _literal_test_tokens()


@pytest.mark.parametrize(
    "words", (NEGATORS, SIGNALS, LABELS), ids=("negators", "signals", "labels")
)
def test_every_default_ignorable_at_every_position_preserves_the_reader_input(words):
    """Exhaust the Cartesian product, including every unassigned ignorable.

    Every guard consumes this same folded input. Equality of that input pins
    its verdict, paths and full issues without repeating millions of identical
    analyzer invocations; the integrated test below independently pins the
    fold's use in each guard and original-offset mapping.
    """

    marks = tuple(map(chr, DEFAULT_IGNORABLES))
    for word in words:
        expected = word + " חסר"
        expected_lines = "\n".join([expected] * len(marks))
        for position in range(len(word) + 1):
            before, after = expected[:position], expected[position:]
            # Each line has exactly one insertion. Batching the lines keeps
            # the complete Cartesian product practical in the normal suite.
            marked = "\n".join(before + mark + after for mark in marks)
            assert hebrew.fold_marks(marked) == expected_lines, (word, position)


@pytest.mark.parametrize("code", DEFAULT_IGNORABLES, ids=lambda code: f"U+{code:04X}")
def test_every_default_ignorable_reaches_the_integrated_reason_source_and_outline_guards(
    code,
):
    """Every property member preserves the full verdict and issue strings."""

    mark = chr(code)
    reason = "ל" + mark + "א חסר סעיף 2"
    plain_reason = "לא חסר סעיף 2"
    assert cases._analyze(_content("a", reason), cases.REVIEW_SOURCE) == cases._analyze(
        _content("a", plain_reason), cases.REVIEW_SOURCE
    )
    marked_source = "(א) ל" + mark + "א מחושב בהתאם לסעיף 2."
    plain_source = "(א) לא מחושב בהתאם לסעיף 2."
    content = _content("a", "חסר סעיף 2")
    assert cases._analyze(content, marked_source) == cases._analyze(
        content, plain_source
    )
    signal = "ח" + mark + "סר סעיף 2"
    assert cases._analyze(_content("a", signal), cases.REVIEW_SOURCE) == cases._analyze(
        content, cases.REVIEW_SOURCE
    )
    marked_source = "(א" + mark + ") הוראה."
    branches = cases.recognize_source_structure(
        marked_source, corpus_citation_path=cases.CITATION
    )
    assert [(branch.path, branch.label) for branch in branches] == [(("a",), "(א)")]
    assert all(
        marked_source[branch.start : branch.end].strip() == branch.text
        for branch in branches
    )
    assert cases._analyze(cases._UNENCODED_CONTENT, marked_source) == cases._analyze(
        cases._UNENCODED_CONTENT, "(א) הוראה."
    )


@pytest.mark.parametrize(("branch", "source", "reason"), cases._OFFSET_PAIRS)
@pytest.mark.parametrize(
    "mark",
    (
        "\u05b9",
        "\u200f",
        "\u2063",
        "\U000110bd",
        "\U000e0100",
        "\U000e007f",
        "\ud804\udcbd",
    ),
)
def test_the_reviewers_35_marked_offset_pairs_preserve_paths_slices_and_issues(
    branch, source, reason, mark
):
    def marked(text):
        return "".join(char + (mark if "א" <= char <= "ת" else "") for char in text)

    marked_source, marked_reason = marked(source), marked(reason)
    plain = cases.recognize_source_structure(
        source, corpus_citation_path=cases.CITATION
    )
    branches = cases.recognize_source_structure(
        marked_source, corpus_citation_path=cases.CITATION
    )
    assert [(item.path, item.label) for item in branches] == [
        (item.path, item.label) for item in plain
    ]
    assert all(
        marked_source[item.start : item.end].strip() == item.text for item in branches
    )
    assert cases._analyze(
        _content(branch, marked_reason), marked_source
    ) == cases._analyze(_content(branch, reason), source)
