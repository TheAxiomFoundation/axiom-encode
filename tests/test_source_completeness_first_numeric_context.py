"""Known protected lexical contexts veto first-numeric recovery; UNKNOWN does not."""

import dataclasses
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness

CONTEXTS = [
    (
        "outer_ascii_quote",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) Heading. "For purposes of this chapter— (1) eligible people."',
        122,
        True,
        [78, 92, 122],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "outer_single_quote",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. ‘For purposes of this chapter— (1) eligible people.’",
        122,
        True,
        [78, 92, 122],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "outer_curly_quote",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. “For purposes of this chapter— (1) eligible people.”",
        122,
        True,
        [78, 92, 122],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "balanced_parenthesis",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. (For purposes of this chapter— (1) eligible people.)",
        122,
        True,
        [78, 92, 122],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "balanced_square_group",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. [For purposes of this chapter— (1) eligible people.]",
        122,
        True,
        [78, 92, 122],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "balanced_curly_group",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. {For purposes of this chapter— (1) eligible people.}",
        122,
        True,
        [78, 92, 122],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "literal_example_prose",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. The following is a quoted example: ‘For purposes of this chapter— (1) eligible "
        "people.’",
        157,
        True,
        [78, 127, 157],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "backtick_code",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. `For purposes of this chapter— (1) eligible people.`",
        122,
        False,
        [78, 92, 122],
        ["OUTSIDE", "UNKNOWN", "UNKNOWN"],
    ),
    (
        "triple_backtick_code",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. ```For purposes of this chapter— (1) eligible people.```",
        124,
        False,
        [78, 94, 124],
        ["OUTSIDE", "UNKNOWN", "UNKNOWN"],
    ),
    (
        "escaped_ascii_quotes",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) Heading. \\"For purposes of this chapter— (1) eligible people.\\"',
        123,
        False,
        [78, 93, 123],
        ["OUTSIDE", "UNKNOWN", "UNKNOWN"],
    ),
    (
        "guillemet_quote",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. «For purposes of this chapter— (1) eligible people.»",
        122,
        True,
        [78, 92, 122],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "double_low_quote",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. „For purposes of this chapter— (1) eligible people.“",
        122,
        False,
        [78, 92, 122],
        ["OUTSIDE", "UNKNOWN", "UNKNOWN"],
    ),
    (
        "editorial_group",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. [Editor's example: For purposes of this chapter— (1) eligible people.]",
        140,
        True,
        [78, 110, 140],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
    (
        "reporting_language",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) Heading. The witness quoted the following draft text: "For purposes of this chapter— (1) '
        'eligible people."',
        167,
        True,
        [78, 137, 167],
        ["OUTSIDE", "PROTECTED", "PROTECTED"],
    ),
]

PRESERVED = [
    (
        "angle_before_chapeau",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Income < 50 qualifies. For purposes of this chapter— (1) eligible people.",
        135,
    ),
    (
        "backtick_before_chapeau",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) A `code` label applies. For purposes of this chapter— (1) eligible people.",
        136,
    ),
    (
        "escape_before_chapeau",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) A \\ escape occurs. For purposes of this chapter— (1) eligible people.",
        131,
    ),
    (
        "unquoted_reporting",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Commentary. A draft says For purposes of this chapter— (1) eligible people.",
        137,
    ),
    (
        "unquoted_indirect_reporting",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) This reports that the agency wrote For purposes of this chapter— (1) eligible people.",
        147,
    ),
    (
        "operator_less_than",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) income < 50 qualifies.",
        112,
    ),
    (
        "operator_greater_than",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) income > 50 qualifies.",
        112,
    ),
    (
        "later_backtick",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people. A later `code` example is unrelated.",
        112,
    ),
    (
        "later_escape",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people. A later \\ escape is unrelated.",
        112,
    ),
    (
        "later_unclosed_quote",
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) For purposes of this chapter— (1) eligible people. A later "unfinished quote',
        112,
    ),
]

LEGACY_CASES = [
    (
        "plain plain ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "plain plain ordinary end."],
            },
        },
    ),
    (
        "plain (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[6, 13]], "plain         ordinary end."],
            },
        },
    ),
    (
        "plain [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[6, 12]], "plain        ordinary end."],
            },
        },
    ),
    (
        "plain {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[6, 12]], "plain        ordinary end."],
            },
        },
    ),
    (
        "plain “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[6, 14]]},
            "later_context": {
                "kind": "return",
                "value": [[[6, 14]], "plain          ordinary end."],
            },
        },
    ),
    (
        'plain "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[6, 14]]},
            "later_context": {
                "kind": "return",
                "value": [[[6, 14]], "plain          ordinary end."],
            },
        },
    ),
    (
        "plain ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[6, 14]]},
            "later_context": {
                "kind": "return",
                "value": [[[6, 14]], "plain          ordinary end."],
            },
        },
    ),
    (
        "plain «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[6, 14]]},
            "later_context": {
                "kind": "return",
                "value": [[[6, 14]], "plain          ordinary end."],
            },
        },
    ),
    (
        "plain employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "plain employers' ordinary end."],
            },
        },
    ),
    (
        "plain worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "plain worker's ordinary end."],
            },
        },
    ),
    (
        "plain (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[6, 16]], "plain            ordinary end."],
            },
        },
    ),
    (
        "plain [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[6, 17]], "plain             ordinary end."],
            },
        },
    ),
    (
        "(group) plain ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7]], "        plain ordinary end."],
            },
        },
    ),
    (
        "(group) (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "(group) [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 14]], "               ordinary end."],
            },
        },
    ),
    (
        "(group) {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 14]], "               ordinary end."],
            },
        },
    ),
    (
        "(group) “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[8, 16]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        '(group) "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[8, 16]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        "(group) ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[8, 16]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        "(group) «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[8, 16]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        "(group) employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7]], "        employers' ordinary end."],
            },
        },
    ),
    (
        "(group) worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7]], "        worker's ordinary end."],
            },
        },
    ),
    (
        "(group) (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 18]], "                   ordinary end."],
            },
        },
    ),
    (
        "(group) [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 7], [8, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "[note] plain ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6]], "       plain ordinary end."],
            },
        },
    ),
    (
        "[note] (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 14]], "               ordinary end."],
            },
        },
    ),
    (
        "[note] [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 13]], "              ordinary end."],
            },
        },
    ),
    (
        "[note] {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 13]], "              ordinary end."],
            },
        },
    ),
    (
        "[note] “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        '[note] "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "[note] ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "[note] «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "[note] employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6]], "       employers' ordinary end."],
            },
        },
    ),
    (
        "[note] worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6]], "       worker's ordinary end."],
            },
        },
    ),
    (
        "[note] (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[8, 16]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "[note] [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 18]], "                   ordinary end."],
            },
        },
    ),
    (
        "{word} plain ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6]], "       plain ordinary end."],
            },
        },
    ),
    (
        "{word} (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 14]], "               ordinary end."],
            },
        },
    ),
    (
        "{word} [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 13]], "              ordinary end."],
            },
        },
    ),
    (
        "{word} {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 13]], "              ordinary end."],
            },
        },
    ),
    (
        "{word} “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        '{word} "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "{word} ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "{word} «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[7, 15]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "{word} employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6]], "       employers' ordinary end."],
            },
        },
    ),
    (
        "{word} worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6]], "       worker's ordinary end."],
            },
        },
    ),
    (
        "{word} (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[8, 16]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "{word} [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 6], [7, 18]], "                   ordinary end."],
            },
        },
    ),
    (
        "“worker” plain ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         plain ordinary end."],
            },
        },
    ),
    (
        "“worker” (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        "“worker” [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "“worker” {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "“worker” “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        '“worker” "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "“worker” ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "“worker” «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "“worker” employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         employers' ordinary end."],
            },
        },
    ),
    (
        "“worker” worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         worker's ordinary end."],
            },
        },
    ),
    (
        "“worker” (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [10, 18]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "“worker” [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        '"worker" plain ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         plain ordinary end."],
            },
        },
    ),
    (
        '"worker" (group) ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        '"worker" [note] ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        '"worker" {word} ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        '"worker" “worker” ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        '"worker" "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        '"worker" ‘worker’ ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        '"worker" «worker» ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        '"worker" employers\' ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         employers' ordinary end."],
            },
        },
    ),
    (
        '"worker" worker\'s ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         worker's ordinary end."],
            },
        },
    ),
    (
        '"worker" (“worker”) ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [10, 18]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        '"worker" [one (two)] ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        "‘worker’ plain ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         plain ordinary end."],
            },
        },
    ),
    (
        "‘worker’ (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        "‘worker’ [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "‘worker’ {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "‘worker’ “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        '‘worker’ "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "‘worker’ ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "‘worker’ «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "‘worker’ employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         employers' ordinary end."],
            },
        },
    ),
    (
        "‘worker’ worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         worker's ordinary end."],
            },
        },
    ),
    (
        "‘worker’ (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [10, 18]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "‘worker’ [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        "«worker» plain ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         plain ordinary end."],
            },
        },
    ),
    (
        "«worker» (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 16]], "                 ordinary end."],
            },
        },
    ),
    (
        "«worker» [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "«worker» {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 15]], "                ordinary end."],
            },
        },
    ),
    (
        "«worker» “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        '«worker» "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "«worker» ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "«worker» «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "«worker» employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         employers' ordinary end."],
            },
        },
    ),
    (
        "«worker» worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8]], "         worker's ordinary end."],
            },
        },
    ),
    (
        "«worker» (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8], [10, 18]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "«worker» [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[0, 8]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 8], [9, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        "employers' plain ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "employers' plain ordinary end."],
            },
        },
    ),
    (
        "employers' (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[11, 18]], "employers'         ordinary end."],
            },
        },
    ),
    (
        "employers' [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[11, 17]], "employers'        ordinary end."],
            },
        },
    ),
    (
        "employers' {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[11, 17]], "employers'        ordinary end."],
            },
        },
    ),
    (
        "employers' “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[11, 19]], "employers'          ordinary end."],
            },
        },
    ),
    (
        'employers\' "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[11, 19]], "employers'          ordinary end."],
            },
        },
    ),
    (
        "employers' ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[11, 19]], "employers'          ordinary end."],
            },
        },
    ),
    (
        "employers' «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[11, 19]], "employers'          ordinary end."],
            },
        },
    ),
    (
        "employers' employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "employers' employers' ordinary end."],
            },
        },
    ),
    (
        "employers' worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "employers' worker's ordinary end."],
            },
        },
    ),
    (
        "employers' (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[12, 20]]},
            "later_context": {
                "kind": "return",
                "value": [[[11, 21]], "employers'            ordinary end."],
            },
        },
    ),
    (
        "employers' [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[11, 22]], "employers'             ordinary end."],
            },
        },
    ),
    (
        "worker's plain ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "worker's plain ordinary end."],
            },
        },
    ),
    (
        "worker's (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[9, 16]], "worker's         ordinary end."],
            },
        },
    ),
    (
        "worker's [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[9, 15]], "worker's        ordinary end."],
            },
        },
    ),
    (
        "worker's {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[9, 15]], "worker's        ordinary end."],
            },
        },
    ),
    (
        "worker's “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[9, 17]], "worker's          ordinary end."],
            },
        },
    ),
    (
        'worker\'s "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[9, 17]], "worker's          ordinary end."],
            },
        },
    ),
    (
        "worker's ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[9, 17]], "worker's          ordinary end."],
            },
        },
    ),
    (
        "worker's «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[9, 17]]},
            "later_context": {
                "kind": "return",
                "value": [[[9, 17]], "worker's          ordinary end."],
            },
        },
    ),
    (
        "worker's employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "worker's employers' ordinary end."],
            },
        },
    ),
    (
        "worker's worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[], "worker's worker's ordinary end."],
            },
        },
    ),
    (
        "worker's (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[10, 18]]},
            "later_context": {
                "kind": "return",
                "value": [[[9, 19]], "worker's            ordinary end."],
            },
        },
    ),
    (
        "worker's [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[9, 20]], "worker's             ordinary end."],
            },
        },
    ),
    (
        "(“worker”) plain ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10]], "           plain ordinary end."],
            },
        },
    ),
    (
        "(“worker”) (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 18]], "                   ordinary end."],
            },
        },
    ),
    (
        "(“worker”) [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "(“worker”) {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 17]], "                  ordinary end."],
            },
        },
    ),
    (
        "(“worker”) “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9], [11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        '(“worker”) "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[1, 9], [11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "(“worker”) ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9], [11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "(“worker”) «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9], [11, 19]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "(“worker”) employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10]], "           employers' ordinary end."],
            },
        },
    ),
    (
        "(“worker”) worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10]], "           worker's ordinary end."],
            },
        },
    ),
    (
        "(“worker”) (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9], [12, 20]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 21]], "                      ordinary end."],
            },
        },
    ),
    (
        "(“worker”) [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": [[1, 9]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 10], [11, 22]], "                       ordinary end."],
            },
        },
    ),
    (
        "[one (two)] plain ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11]], "            plain ordinary end."],
            },
        },
    ),
    (
        "[one (two)] (group) ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 19]], "                    ordinary end."],
            },
        },
    ),
    (
        "[one (two)] [note] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 18]], "                   ordinary end."],
            },
        },
    ),
    (
        "[one (two)] {word} ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 18]], "                   ordinary end."],
            },
        },
    ),
    (
        "[one (two)] “worker” ordinary end.",
        {
            "annual": {"kind": "return", "value": [[12, 20]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        '[one (two)] "worker" ordinary end.',
        {
            "annual": {"kind": "return", "value": [[12, 20]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        "[one (two)] ‘worker’ ordinary end.",
        {
            "annual": {"kind": "return", "value": [[12, 20]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        "[one (two)] «worker» ordinary end.",
        {
            "annual": {"kind": "return", "value": [[12, 20]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 20]], "                     ordinary end."],
            },
        },
    ),
    (
        "[one (two)] employers' ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11]], "            employers' ordinary end."],
            },
        },
    ),
    (
        "[one (two)] worker's ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11]], "            worker's ordinary end."],
            },
        },
    ),
    (
        "[one (two)] (“worker”) ordinary end.",
        {
            "annual": {"kind": "return", "value": [[13, 21]]},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 22]], "                       ordinary end."],
            },
        },
    ),
    (
        "[one (two)] [one (two)] ordinary end.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [[[0, 11], [12, 23]], "                        ordinary end."],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) Heading. "For purposes of this chapter— (1) eligible people."',
        {
            "annual": {"kind": "return", "value": [[91, 143]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 143],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. ‘For purposes of this chapter— (1) eligible people.’",
        {
            "annual": {"kind": "return", "value": [[91, 143]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 143],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. “For purposes of this chapter— (1) eligible people.”",
        {
            "annual": {"kind": "return", "value": [[91, 143]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 143],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. (For purposes of this chapter— (1) eligible people.)",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 143],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. [For purposes of this chapter— (1) eligible people.]",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 143],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. {For purposes of this chapter— (1) eligible people.}",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 143],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. The following is a quoted example: ‘For purposes of this chapter— (1) eligible "
        "people.’",
        {
            "annual": {"kind": "return", "value": [[126, 178]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [126, 178],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    Heading. The following is a quoted "
                    "example:                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. `For purposes of this chapter— (1) eligible people.`",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. ```For purposes of this chapter— (1) eligible people.```",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) Heading. \\"For purposes of this chapter— (1) eligible people.\\"',
        {
            "annual": {"kind": "return", "value": [[92, 145]]},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. «For purposes of this chapter— (1) eligible people.»",
        {
            "annual": {"kind": "return", "value": [[91, 143]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 143],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. „For purposes of this chapter— (1) eligible people.“",
        {
            "annual": {"kind": "return", "value": None},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Heading. [Editor's example: For purposes of this chapter— (1) eligible people.]",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [91, 161],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    "
                    "Heading.                                                                       ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) Heading. The witness quoted the following draft text: "For purposes of this chapter— (1) '
        'eligible people."',
        {
            "annual": {"kind": "return", "value": [[136, 188]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [136, 188],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    Heading. The witness quoted the following draft "
                    "text:                                                     ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Commentary. A draft says For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [137, 140],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    Commentary. A draft says For purposes of this chapter—     "
                    "eligible people.",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) This reports that the agency wrote For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [147, 150],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    This reports that the agency wrote For purposes of this "
                    "chapter—     eligible people.",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) income < 50 qualifies.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) income > 50 qualifies.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people. A later `code` example is unrelated.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people. A later \\ escape is unrelated.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        '(g) For purposes of this chapter— (1) eligible people. A later "unfinished quote',
        {
            "annual": {"kind": "return", "value": None},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Income < 50 qualifies. For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) A `code` label applies. For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) A \\ escape occurs. For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people.\n"
        "\n"
        "(h) Later `token`. For purposes of this chapter— (1) eligible helpers.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people.\n"
        "\n"
        "(h) Later \\ escape. For purposes of this chapter— (1) eligible helpers.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people.\n"
        "\n"
        '"\n'
        "(h) Heading. For purposes of this chapter— (1) eligible helpers.\n"
        '"',
        {
            "annual": {"kind": "return", "value": [[134, 202]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [112, 115],
                        [134, 202],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    For purposes of this chapter—     eligible people.\n"
                    "\n"
                    "                                                                    ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people.\n"
        "\n"
        '"\n'
        '(h) Heading."\n'
        "For purposes of this chapter— (1) eligible helpers.",
        {
            "annual": {"kind": "return", "value": [[134, 149]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [112, 115],
                        [134, 149],
                        [180, 183],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    For purposes of this chapter—     eligible people.\n"
                    "\n"
                    "               \n"
                    "For purposes of this chapter—     eligible helpers.",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people.\n"
        "\n"
        "(\n"
        "(h) Heading.) For purposes of this chapter— (1) eligible helpers.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [112, 115],
                        [134, 149],
                        [180, 183],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    For purposes of this chapter—     eligible people.\n"
                    "\n"
                    "                For purposes of this chapter—     eligible "
                    "helpers.",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people.\n"
        "\n"
        "[\n"
        "(h) Heading. For purposes of this chapter— (1) eligible helpers.\n"
        "]",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [112, 115],
                        [134, 202],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    For purposes of this chapter—     eligible people.\n"
                    "\n"
                    "                                                                    ",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) For purposes of this chapter— (1) eligible people.\n"
        "\n"
        "(h) A “worker” (including helpers) qualifies. For purposes of this chapter— (1) eligible "
        "helpers.",
        {
            "annual": {"kind": "return", "value": [[140, 148]]},
            "later_context": {
                "kind": "return",
                "value": [
                    [
                        [0, 3],
                        [13, 16],
                        [26, 29],
                        [39, 42],
                        [52, 55],
                        [65, 68],
                        [78, 81],
                        [112, 115],
                        [134, 137],
                        [140, 148],
                        [149, 168],
                        [210, 213],
                    ],
                    "    Duty a.\n"
                    "\n"
                    "    Duty b.\n"
                    "\n"
                    "    Duty c.\n"
                    "\n"
                    "    Duty d.\n"
                    "\n"
                    "    Duty e.\n"
                    "\n"
                    "    Duty f.\n"
                    "\n"
                    "    For purposes of this chapter—     eligible people.\n"
                    "\n"
                    "    A                              qualifies. For purposes of this "
                    "chapter—     eligible helpers.",
                ],
            },
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Income < 50 qualifies. For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) <tag>plain text</tag>. For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) Income > 50 qualifies. For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(a) Duty a.\n"
        "\n"
        "(b) Duty b.\n"
        "\n"
        "(c) Duty c.\n"
        "\n"
        "(d) Duty d.\n"
        "\n"
        "(e) Duty e.\n"
        "\n"
        "(f) Duty f.\n"
        "\n"
        "(g) <!-- a reported note --> For purposes of this chapter— (1) eligible people.",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": [[], ""]},
        },
    ),
    (
        '"unfinished',
        {
            "annual": {"kind": "return", "value": None},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "(unfinished",
        {
            "annual": {"kind": "return", "value": None},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "wrong]",
        {
            "annual": {"kind": "return", "value": None},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        '“mixed "quotes”',
        {
            "annual": {"kind": "return", "value": None},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "`code`",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "\\escape",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        "x < 50",
        {
            "annual": {"kind": "return", "value": []},
            "later_context": {"kind": "return", "value": None},
        },
    ),
    (
        None,
        {
            "annual": {
                "kind": "error",
                "message": "'NoneType' object is not iterable",
                "type": "TypeError",
            },
            "later_context": {
                "kind": "error",
                "message": "'NoneType' object is not iterable",
                "type": "TypeError",
            },
        },
    ),
    (
        42,
        {
            "annual": {
                "kind": "error",
                "message": "'int' object is not iterable",
                "type": "TypeError",
            },
            "later_context": {
                "kind": "error",
                "message": "'int' object is not iterable",
                "type": "TypeError",
            },
        },
    ),
    (
        b"bytes",
        {
            "annual": {
                "kind": "error",
                "message": "'in <string>' requires string as left operand, not int",
                "type": "TypeError",
            },
            "later_context": {
                "kind": "error",
                "message": "'in <string>' requires string as left operand, not int",
                "type": "TypeError",
            },
        },
    ),
]


def records(source):
    return [
        dataclasses.asdict(branch)
        for branch in completeness.recognize_source_structure(
            source, corpus_citation_path="us/regulation/12/1006/14"
        )
    ]


@pytest.mark.parametrize(
    "name,source,target,protected,points,statuses",
    CONTEXTS,
    ids=[row[0] for row in CONTEXTS],
)
def test_original_contexts_are_explicitly_protected_or_unknown(
    name, source, target, protected, points, statuses
):
    branches = records(source)
    admitted = any(b["start"] == target and len(b["path"]) > 1 for b in branches)
    assert admitted is not protected
    assert (
        list(completeness._source_context_checkpoints(source, points).values())
        == statuses
    )


@pytest.mark.parametrize("name,source,target", PRESERVED)
def test_unknown_prefix_later_syntax_and_reporting_preserve_native_admission(
    name, source, target
):
    assert any(b["start"] == target and len(b["path"]) > 1 for b in records(source))


def capture(function, value):
    try:
        import json

        return {"kind": "return", "value": json.loads(json.dumps(function(value)))}
    except Exception as error:
        return {"kind": "error", "type": type(error).__name__, "message": str(error)}


@pytest.mark.parametrize("value,expected", LEGACY_CASES)
def test_legacy_annual_and_later_heading_results_and_errors(value, expected):
    assert (
        capture(completeness._annual_source_quoted_spans, value) == expected["annual"]
    )
    assert (
        capture(completeness._later_heading_source_context, value)
        == expected["later_context"]
    )


@settings(max_examples=25, deadline=None, derandomize=True, database=None)
@given(
    prefix=st.sampled_from(
        [
            "plain ",
            "(group) ",
            "[note] ",
            "{word} ",
            '"worker" ',
            "“worker” ",
            "worker's ",
        ]
    ),
    suffix=st.sampled_from(
        ["later `code`", "later <tag>", 'later "unfinished', "later ]"]
    ),
)
def test_later_unknown_or_invalid_text_never_changes_earlier_checkpoints(
    prefix, suffix
):
    source = prefix + "target "
    points = tuple(range(len(source)))
    assert completeness._source_context_checkpoints(
        source, points
    ) == completeness._source_context_checkpoints(source + suffix, points)


@pytest.mark.parametrize("count", [4, 32, 256])
def test_one_checkpoint_walk_stops_at_last_needed_character(count):
    text = "plain " * count + "target" + ' later "unterminated' * 32
    points = (0, 2, len("plain " * count))
    visits = []
    calls = []

    class CountedText(str):
        def __iter__(self):
            for index, character in enumerate(super().__iter__()):
                visits.append(index)
                yield character

    original = completeness._source_quote_group_walk

    def observed(value, checkpoints=()):
        calls.append(tuple(checkpoints))
        return original(value, checkpoints)

    with patch.object(completeness, "_source_quote_group_walk", observed):
        assert set(
            completeness._source_context_checkpoints(CountedText(text), points).values()
        ) == {"OUTSIDE"}
    assert visits == list(range(points[-1] + 1))
    assert len(calls) == 1


@pytest.mark.parametrize("count", [4, 32, 256])
def test_first_numeric_builds_one_checkpoint_index(count):
    prefix = "\n\n".join(
        "(" + chr(n) + ") Duty " + chr(n) + "." for n in range(ord("a"), ord("g"))
    )
    text = (
        prefix
        + "\n\n(g) "
        + ("For purposes of this chapter— (1) eligible people. " * count)
    )
    original = completeness._source_context_checkpoints
    calls = []

    def observed(source, positions):
        calls.append(tuple(positions))
        return original(source, positions)

    with patch.object(completeness, "_source_context_checkpoints", observed):
        assert any(b["path"] == ("g", "1") for b in records(text))
    assert len(calls) == 1


def test_legacy_cache_api_remains_available():
    assert completeness._annual_source_quoted_spans.cache_info().maxsize == 256


def test_legacy_stopiteration_error_is_not_converted_by_checkpoint_seam():
    class Stops(str):
        def __getitem__(self, key):
            raise StopIteration("SOURCE_STOP_SENTINEL")

    completeness._annual_source_quoted_spans.cache_clear()
    for function in (
        completeness._annual_source_quoted_spans,
        completeness._later_heading_source_context,
    ):
        with pytest.raises(StopIteration, match="SOURCE_STOP_SENTINEL"):
            function(Stops("xy"))
