"""Regex fragments for Hebrew prefixes joined by visible punctuation.

A maqaf or any of the three printed hyphens below joins a Hebrew prefix
to the following prefix or word.  Readers accept the joint in place; they
never fold it away, so source punctuation and offsets are retained.
"""

import re

PREFIX_JOINT_CHARS = "\u05be-\u2010\u2011"
PREFIX_JOINT = "[" + re.escape(PREFIX_JOINT_CHARS) + "]"
HEBREW_PREFIX_LETTERS = "והבכלמשד"
# A reader cannot restart at a suffix just after a joint.  Otherwise ``מ־לא``
# would be read as the standalone negator ``לא`` although ``מלא`` is "full".
# A leading or standalone hyphen remains ordinary punctuation; the preceding
# Hebrew prefix is what makes a joint part of the word.
PREFIX_BOUNDARY = rf"(?<![א-תA-Za-z0-9_])(?<![{HEBREW_PREFIX_LETTERS}]{PREFIX_JOINT})"


def prefix_word(prefix: str) -> str:
    """A literal prefix chain, with an optional joint after each letter."""

    return "".join(re.escape(letter) + PREFIX_JOINT + "?" for letter in prefix)


def prefix_chain(letters: str, maximum: int | None = None) -> str:
    """Zero or more prefix letters from ``letters``, retaining visible joints."""

    repeats = "*" if maximum is None else f"{{0,{maximum}}}"
    return rf"(?:[{re.escape(letters)}]{PREFIX_JOINT}?){repeats}"
