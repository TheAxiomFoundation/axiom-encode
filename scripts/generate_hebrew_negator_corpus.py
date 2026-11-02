#!/usr/bin/env python3
"""Receipt the distinct Hebrew words in the pinned Israeli statute snapshot.

Usage: python scripts/generate_hebrew_negator_corpus.py SNAPSHOT [--check]

The public-law JSONL stays outside Git. The fixture contains its complete word
vocabulary, and short statutory contexts for the finite, manually audited list
below. Expected contextual readings come from the statutory contexts and their
meanings. The source computation vocabulary uses the shared grammatical
generator; the audited reason readings and semantic exclusions below keep cable,
imprisonment, and other homographs among the negative controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from axiom_encode.harness import hebrew_negators as grammar
from axiom_encode.harness.hebrew_prefixes import prefix_word

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/hebrew_negator_corpus.json"
SNAPSHOT_SHA256 = "fd5f1e4d6fb941c32cfbf2ef4f248146c29baf41906d1cdb3d6c7437b86d0ac1"
WORD = re.compile("[א-ת]+")

# Manually read against the contexts emitted below. These words express a
# negative, prohibition, refusal, failure, or exception in the cited statutes.
# Context-dependent particles/verbs are recorded separately, rather than being
# declared negative because the production regex happens to accept them.
UNCONDITIONAL_NEGATORS = frozenset(
    """
    אין אינה אינו אינם אינן איננה איננו אילולא אלמלא שאילולא באין בלא בלי בלתי דלא הבלתי
    ואין ואינה ואינו ואינם ואיננה ובאין ובלא ובלי ובלתי וטרם ולא
    ולולא וללא ולמעט ולהוציא ושאינה ושאינו ושאינם ושלא
    טרם כבלתי כשטרם כשלא לא לאו לולא ללא למעט לסרב מבלי מלבד
    שאין שאינה שאינו שאינם שאינן שאיננה שאיננו שבלי שטרם שלא שלולא
    שסירב סירב סירבה האסורה איסור הנמנעים יוחרגו נשללו שהוחרג
    נמנעה הנמנע אסר שוללת השוללת נמנע נבצר שיימנע תסרב נכשל כשל
    נכשלו שכשלו תימנע זולת בהיעדר בהעדר
    """.split()
)

# These resemble prefixed particles but are ordinary lexical words. Each reason
# is a semantic exclusion, not an allowance for a known false positive.
EXCLUSIONS = {
    **dict.fromkeys(
        "מלא המלא למלא מלמלא ממלא לממלא הממלא שממלא מלאו ומלאו שמלאו".split(),
        "The מלא root means full, fill, fulfil, or reach an age; it is not prefixed לא.",
    ),
    **dict.fromkeys(
        "מלאי המלאי ומלאי כמלאי למלאי הבלאי והבלאי בבלאי".split(),
        "Inventory (מלאי) or wear (בלאי), not a prefixed negative particle.",
    ),
    **dict.fromkeys("מאי במאי".split(), "May, the month, not prefixed אי."),
    **dict.fromkeys("דופן בדופן".split(), "Wall (דופן), not prefixed פן."),
    **dict.fromkeys("האף מהאף".split(), "The nose, not negative-polarity אף."),
    "שמאל": "Left, not relative ש plus prohibitive אל.",
    "כאל": "As/to a financial institution; אל is a preposition here.",
    "לאל": "Medical אל-וסת (amenorrhea), not a jussive/prohibitive clause.",
    "לאין": "The medical noun phrase לאין אונות, not an operative negative clause.",
    **dict.fromkeys(
        "מאסר במאסר למאסר המאסר ממאסר".split(),
        "Imprisonment, not prepositional מ plus the verb אסר (forbade).",
    ),
    "בכבל": "Cable transmission, not prefixes ב/כ plus the particle בל.",
    "בטרם": "Temporal before; the legacy English guard also excludes before.",
    "ובטרם": "And before; the legacy English guard also excludes before.",
    **dict.fromkeys(
        "ואי באי שאי לאי".split(),
        "Nominal אי (non-/in-) or the homograph באי כוחם; only אי אפשר/ניתן is a clause negator.",
    ),
    "לאף": "Negative-polarity any in לאף אחד; not an independent clause negator.",
    "שאף": "Negative-polarity any before אינו; not an independent clause negator.",
    "בחסר": "Understated or short (ביתר או בחסר, מכירה בחסר), not ב plus the absence noun חוסר.",
    **dict.fromkeys(
        "היעדר העדר חוסר מחוסר כמחוסר".split(),
        "A bare or causal absence noun (absence of, for lack of); only ב with the noun in construct means without.",
    ),
    "בהעדרה": "In its absence: the pronoun suffix is the absent thing, so no later word is negated.",
}

CONTEXT_DEPENDENT = frozenset(
    "אי אל אף ואף בל הלא להוציא מעולם מנוע מנועה למנוע חוץ פרט".split()
)

# Round 19: a source also reads these heads as negating its link, but only
# over a computation complement (without calculation, prevents calculation,
# instead of calculation), as English's source guard reads ``without
# calculat*``, ``prevented`` and ``rather than``. Their statutory homographs
# (a missing person, blocked income, at a place, exclusive) take no
# computation complement. Grammar lists the heads; the statutes supply the
# contexts. Bare מחוסר before an indefinite complement is causal ("for lack
# of"), so it reads only in a relative clause there.
SOURCE_GUARDED_HEADS = frozenset(
    (*grammar.SOURCE_PRIVATIVE_HEADS, *grammar.SOURCE_CONSTRUCT_HEADS)
) - {"מחוסר"}
ORDINARY_PREFIXES = tuple(
    conjunction + subordinator
    for conjunction in ("", "ו")
    for subordinator in ("", "ש", "כש", "מש", "לכש", "בש", "ד")
)
# The Hebrew of English's determin*, comput*/calculat*, appl*, impos* and
# aris*, as nouns with their suffixes and as infinitives.
SOURCE_COMPUTATION_WORD = re.compile(
    r"\s+(?:(?:כל|שום|אותו|אותה|אותם|אותן)\s+)?"
    rf"(?:{prefix_word('ה')})?(?:{grammar._COMPUTATION_NOUNS}"
    rf"|{grammar._COMPUTATION_VERBS})(?![א-ת])"
)


def source_guarded_head(token: str) -> str | None:
    """The guarded head ``token`` spells after an ordinary prefix chain, if any."""

    for prefix in sorted(ORDINARY_PREFIXES, key=len):
        head = token[len(prefix) :]
        if token.startswith(prefix) and head in SOURCE_GUARDED_HEADS:
            return head
        if token.startswith(prefix) and head == "מחוסר" and set(prefix) & {"ש", "ד"}:
            return head
    return None


def _source_reading(token: str, after: str, negator: bool) -> bool:
    """The source reading: a negator, or a guarded head over a computation."""

    if negator:
        return True
    return source_guarded_head(token) is not None and bool(
        SOURCE_COMPUTATION_WORD.match(after)
    )


AUDITED_WORDS = UNCONDITIONAL_NEGATORS | EXCLUSIONS.keys() | CONTEXT_DEPENDENT


def statutory_text(text: str) -> str:
    """Remove vocalization independently, before splitting the statute's words.

    Otherwise בכֶבֶל (cable) would incorrectly contribute בכ, ב, and ל. The
    snapshot contains Hebrew Mn marks and Cf direction marks; neither the
    production fold nor the production negator grammar is imported here.
    """

    return "".join(
        character
        for character in text
        if not (
            unicodedata.category(character) == "Cf"
            or (
                "\u0590" <= character <= "\u05ff"
                and unicodedata.category(character) == "Mn"
            )
        )
    )


def _reading(token: str, before: str, after: str) -> tuple[bool, str]:
    if before and before[-1].isdigit():
        return (
            False,
            "A Hebrew section-number suffix attached to digits, not a lexical negator.",
        )
    if token in UNCONDITIONAL_NEGATORS:
        return (
            True,
            "Negative/prohibitive/exclusion reading in the cited statutory context.",
        )
    if token in EXCLUSIONS:
        return False, EXCLUSIONS[token]
    if token == "אי":
        negative = bool(re.match(r"\s+(?:אפשר|ניתן)(?![א-ת])", after))
        return negative, (
            "אי אפשר/ניתן means cannot."
            if negative
            else "Nominal אי (non-/in-) is outside the clause-negator construction."
        )
    if token == "אל":
        return (
            False,
            "Prepositional to a report, assignee, or spouse; no imperfect verb follows.",
        )
    if token in {"אף", "ואף"}:
        if re.search(r"(?:^|\s)[וש]*על\s*$", before):
            return (
                False,
                "Concessive על אף (despite), even when a later clause contains לא.",
            )
        for connective, meaning in (
            ("אם", "even if"),
            ("שלא", "although not"),
            ("בשונה", "even in deviation from"),
            ("כי", "although"),
        ):
            if re.match(r"\s+" + connective + r"(?![א-ת])", after):
                return False, f"Concessive {meaning}, not an independent negator."
        negative = bool(re.match(r"\s+לא(?![א-ת])", after))
        return negative, (
            "Negative polarity: not even / nor."
            if negative
            else "Affirmative even/also, without an independent negative reading."
        )
    if token == "בל":
        return False, "טופס בל/50 abbreviates National Insurance; not the particle בל."
    if token == "הלא":
        return True, "Attributive הלא מנוצל means the unused (real-estate portion)."
    if token == "להוציא":
        for governor, explanation in (
            ("הורשו", "Authorized to issue policies; not except."),
            ("ניתן", "Able to issue a building permit; not except."),
            ("רשאית(?: קופה)?", "Authorized spending by a health fund; not except."),
            ("שלא", "A negated spending infinitive, not the conjunction except."),
            (
                "כדי",
                "A purpose infinitive to remove from a category, not the conjunction except.",
            ),
        ):
            if re.search(governor + r"\s*$", before):
                return False, explanation
        return True, "Except/exclude, rather than issuing or spending."
    if token == "מעולם":
        return True, "מעולם לא היתה means was never; לא licenses negative polarity."
    if token in {"מנוע", "מנועה", "למנוע"}:
        negative = bool(re.match(r"\s+מ[א-ת]", after))
        return negative, (
            "Prevented from / prevent from; a following מ complement licenses the reading."
            if negative
            else "Engine or an unlicensed prevent infinitive, not the prevented-from construction."
        )
    if token == "חוץ":
        # חוץ as foreign in תושב/מדינת/הכנסות חוץ is not except. Only a
        # remaining מ complement can license the exception construction.
        has_complement = bool(re.match(r"[ \t]+מ[א-ת]", after))
        foreign_head = bool(
            re.search(
                r"(?:תושב(?:י|ת)?|מדינ(?:ת|ות)|הכנסות|מסי|מטבע|עיתונאי)\s*$",
                before,
            )
        )
        if has_complement and foreign_head:
            previous = re.search(r"([א-ת]+)\s*$", before)
            if previous and previous.group(1) not in {
                "תושב",
                "תושבי",
                "תושבת",
                "מדינת",
                "מדינות",
                "הכנסות",
                "מסי",
                "מטבע",
                "עיתונאי",
            }:
                return (
                    False,
                    "A prefixed foreign-head noun phrase; the following מ is not except.",
                )
            return (
                False,
                "Foreign in a headed noun phrase; the following מ belongs to another word.",
            )
        if re.match(r"־[א-ת]", after):
            return False, "External/non- compound adjective; not the exception except."
        negative = has_complement and not foreign_head
        return negative, (
            "Except, with a following מ complement."
            if negative
            else "Foreign/external or a compound adjective, not the exception except."
        )
    if token == "פרט":
        negative = bool(re.match(r"[ \t]+ל[א-ת]", after))
        return negative, (
            "Except, with a following ל complement."
            if negative
            else "An item/detail, without the except-for ל construction."
        )
    raise AssertionError(token)


def generate(snapshot: Path) -> dict:
    data = snapshot.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise ValueError(f"Unexpected snapshot SHA256: {digest}")
    rows = [json.loads(line) for line in data.splitlines()]
    counts: Counter[str] = Counter()
    contexts: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        text = statutory_text(row.get("body") or "")
        for match in WORD.finditer(text):
            token = match.group()
            counts[token] += 1
            guarded = source_guarded_head(token) is not None
            if token not in AUDITED_WORDS and not guarded:
                continue
            low, high = max(0, match.start() - 100), min(len(text), match.end() + 120)
            before, after = text[low : match.start()], text[match.end() : high]
            if token in AUDITED_WORDS:
                negative, explanation = _reading(token, before, after)
            else:
                negative = False
                explanation = (
                    "A source-only head: it negates a link only over a computation complement."
                    if SOURCE_COMPUTATION_WORD.match(after)
                    else "A source-only head with no computation complement: its statutory sense."
                )
            source_negative = _source_reading(token, after, negative)
            if token in AUDITED_WORDS and source_negative != negative:
                explanation += (
                    " In a source, a guarded head over a computation complement."
                )
            if before and before[-1].isdigit():
                source_negative = negative
            if (
                sum(
                    context["explanation"] == explanation for context in contexts[token]
                )
                >= 3
            ):
                continue
            contexts[token].append(
                {
                    "citation": row["citation_path"],
                    "text": text[low:high],
                    "start": match.start() - low,
                    "end": match.end() - low,
                    "negator": negative,
                    "source_negator": source_negative,
                    "explanation": explanation,
                }
            )
    return {
        "snapshot": {
            "filename": snapshot.name,
            "sha256": digest,
            "bytes": len(data),
            "records": len(rows),
            "text_records": sum(bool(row.get("body")) for row in rows),
        },
        "generator": "scripts/generate_hebrew_negator_corpus.py",
        "tokenization": "Strip Hebrew Mn vocalization marks and Cf direction marks independently, then take every maximal [א-ת]+ run in a non-null body.",
        "distinct_token_count": len(counts),
        "vocabulary": sorted(counts),
        "probe_negators": sorted(
            (UNCONDITIONAL_NEGATORS | {"בל", "להוציא"}) & counts.keys()
        ),
        # Each word before " חישוב": the source also reads its guarded heads.
        # ``דוחות`` is the verb "reject" unless a noun context applies, and a
        # clause-initial probe has none (round 23).
        "source_probe_negators": sorted(
            ((UNCONDITIONAL_NEGATORS | {"בל", "להוציא"}) & counts.keys())
            | {word for word in counts if source_guarded_head(word) is not None}
        ),
        "audited_tokens": [
            {"token": token, "occurrences": counts[token], "contexts": contexts[token]}
            for token in sorted(contexts)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = json.dumps(generate(args.snapshot), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if FIXTURE.read_text() != output:
            raise SystemExit("Hebrew negator corpus fixture is out of date")
    else:
        temporary = FIXTURE.with_suffix(".json.tmp")
        temporary.write_text(output)
        temporary.replace(FIXTURE)


if __name__ == "__main__":
    main()
