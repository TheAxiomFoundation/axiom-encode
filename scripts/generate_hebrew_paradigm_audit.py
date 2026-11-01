#!/usr/bin/env python3
"""Append generated round-20 family frames without rewriting prior audit rows.

The frames refer to grammatical slots, rather than copying all combinations
of forms, prefixes and visible joints into JSON. The test expands those slots.
Reason verdicts are receipted separately against the unchanged source-only
boundary at 0acff3551; --baseline-source points to that commit's src directory.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from axiom_encode.harness import hebrew_negators as grammar
from axiom_encode.harness.hebrew_source_text import fold_marks

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/hebrew_rendering_audit.json"
BASELINE = "0acff35515e7bdb7d2a97fd6b048f9e131ba53e9"


def english_frames(role: str, family: str) -> tuple[str, str]:
    """Faithful canonical English, in the order its finite guard recognizes."""

    end = " in accordance with section 2."
    if role in {"computation", "computational-processing"}:
        noun = {
            "קביעה": "determination",
            "היקבעות": "determination",
            "חישוב": "calculation",
            "עיבוד": "computational processing",
            "החלה": "application",
            "יישום": "application",
            "ישום": "application",
            "תחולה": "application",
            "הטלה": "imposition",
            "השתה": "imposition",
            "חיוב": "imposition of liability",
            "היווצרות": "arising",
            "הווצרות": "arising",
            "התהוות": "arising",
            "צמיחה": "arising",
        }[family]
        return (
            f"(a) The payment shall be without {noun}{end}",
            f"(a) The payment shall be with {noun}{end}",
        )
    if role in {"claim", "claim-denial"}:
        negative = (
            f"(a) The official denies the claim that the payment is calculated{end}"
        )
        return negative, negative.replace("denies", "affirms")
    if role in {
        "postposed-claim",
        "postposed-correct",
        "postposed-false",
        "false-phrase",
    }:
        negative = f"(a) It is false that the payment is computed{end}"
        return negative, negative.replace("false", "true")
    if role in {"disclaimer", "continued-disclaimer"}:
        negative = (
            "(a) The provisions of section 2 remain unaffected and the payment "
            f"is calculated{end}"
        )
        return negative, negative.replace("unaffected", "effective")
    if role in {"relative-computation", "infinitive-computation"}:
        negative = f"(a) The payment is fixed rather than calculated{end}"
        return negative, negative.replace("rather than", "and")
    if role == "phrase":
        negative = (
            f"(a) The official disclaims calculation{end}"
            if family == "התנער"
            else f"(a) The calculation was excluded from the payment{end}"
        )
        return negative, negative.replace("disclaims", "affirms").replace(
            "excluded from", "included in"
        )
    if role == "clause-negator":
        negative = f"(a) The payment is prevented from being calculated{end}"
        return negative, negative.replace("prevented from", "capable of")
    # An absent calculation, exemption from it, and a privative adjective all
    # say the payment is without a calculation. This preserves the guard's
    # canonical computation noun rather than its unrecognized bare "absent".
    negative = f"(a) The payment shall be without calculation{end}"
    return negative, negative.replace("without", "with")


def family_frames() -> list[dict]:
    """One generated-frame class for each family in each applicable role."""

    rows = []

    def add(registry, family, slot, role):
        rows.append(
            {
                "id": f"r20-{role}-{family}",
                "cls": f"{role}-{family}",
                "registry": registry,
                "family": family,
                "slot": slot,
                "role": role,
                "source_verdict": False,
                "positive_verdict": True,
                "english": list(english_frames(role, family)),
            }
        )

    for family in grammar.COMPUTATION_NOUN_PARADIGMS:
        add("noun", family.name, "forms", "computation")
    add("noun", "עיבוד", "forms", "computational-processing")
    add("adjective", "חישובי", "forms", "computational-adjective")
    add("adjective", "מוחל", "absolute", "passive-computation")
    for family in grammar.COMPUTATION_VERB_PARADIGMS:
        add("verb", family.name, "future", "relative-computation")
        excluded = [
            form
            for form in family.spellings("future")
            if form not in grammar.COMPUTATION_RELATIVE_FUTURES
        ]
        if excluded:
            rows[-1]["excluded_forms"] = {
                form: "With relative ש, the statutory noun rehabilitation."
                for form in excluded
            }
        add("verb", family.name, "infinitive", "infinitive-computation")
    for family in grammar.NOUN_PARADIGMS.values():
        if set(family.forms) <= set(grammar.CLAIM_NOUNS):
            add("noun", family.name, "forms", "claim")
            add("noun", family.name, "forms", "postposed-claim")
    for registry_name, families in (
        ("verb", grammar.VERB_PARADIGMS),
        ("adjective", grammar.ADJECTIVE_PARADIGMS),
    ):
        for family in families.values():
            if set(family.forms) <= set(grammar.SOURCE_CLAUSE_NEGATORS):
                add(registry_name, family.name, "forms", "clause-negator")
            if set(family.forms) <= set(grammar.SOURCE_PRIVATIVE_HEADS):
                add(registry_name, family.name, "forms", "privative")
    for family in ("התנער", "הוצא"):
        add("verb", family, "forms", "phrase")
    for family in ("שקרי", "שגוי"):
        add("adjective", family, "forms", "false-phrase")
    add("verb", "כפר", "forms", "claim-denial")
    for family in ("נותר", "עמד", "נשאר"):
        add("verb", family, "forms", "disclaimer")
    add("verb", "המשיך", "forms", "continued-disclaimer")
    for family in ("נפקד", "נעדר", "חסר"):
        add("adjective", family, "forms", "absent-predicate")
    add("adjective", "פטור", "absolute", "from-complement")
    add("adjective", "פטור", "construct", "construct-complement")
    add("adjective", "שונה", "forms", "from-complement")
    add("verb", "הבדיל", "forms", "from-complement")
    add("adjective", "נכון", "forms", "postposed-correct")
    for family in ("שקרי", "שגוי"):
        add("adjective", family, "forms", "postposed-false")
    add("verb", "נבצר", "forms", "postposed-false")
    return rows


def reason_forms() -> list[str]:
    return sorted(
        {
            form
            for registry in (
                grammar.NOUN_PARADIGMS,
                grammar.VERB_PARADIGMS,
                grammar.ADJECTIVE_PARADIGMS,
            )
            for family in registry.values()
            for form in family.forms
        }
    )


def reason_frames(form: str) -> tuple[str, str, str]:
    return (
        f"{form} חישוב חסר סעיף 2",
        f"חסר {form} סעיף 2",
        f"חסר סעיף 2 {form} חישוב",
    )


def baseline_reasons(source: Path) -> list[list]:
    """Ask the old reason reader, never the newly generated source grammar."""

    code = """
import json, os, sys, types
from pathlib import Path
# Import the exact old source reader and its local dependencies without the
# unrelated encoder backend imports in the package's convenience __init__.
package_root = Path(os.environ['PYTHONPATH']) / 'axiom_encode'
for name, path in (('axiom_encode', package_root),
                   ('axiom_encode.harness', package_root / 'harness')):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
from axiom_encode.harness.hebrew_source_text import reason_names_source_bound_dependency
forms = json.load(sys.stdin)
source = '(א) התשלום מחושב בהתאם לסעיף 2.'
rows = []
for form in forms:
    frames = (f'{form} חישוב חסר סעיף 2', f'חסר {form} סעיף 2',
              f'חסר סעיף 2 {form} חישוב')
    rows.append([form, [reason_names_source_bound_dependency(
        reason, source, corpus_citation_path='il/statute/test-law/section-1')
        for reason in frames]])
print(json.dumps(rows, ensure_ascii=False))
"""
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(source.resolve()), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(reason_forms(), ensure_ascii=False),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
        env=environment,
    )
    return json.loads(result.stdout)


# --- Round 21: spelling closure ---------------------------------------------
#
# Every snapshot word one ``ו`` or ``י`` from a generated form, under each of
# the 14 ordinary prefix chains, is generated or excluded here with one named
# reason. A word is a neighbour of a form when deleting one ``ו`` or ``י`` from
# either gives the other. ``--snapshot`` recomputes the inventory from the
# pinned 1,633-record snapshot, and ``--check --snapshot`` fails on a new
# neighbour with no class. The attributive verbs of the claim reader stay out:
# a missed attributive spelling keeps a claim reading and refuses, never admits.
HOMOGRAPH = "homograph"
OTHER_LEXEME = "other lexeme"
DEFECTIVE_ACTIVE = "defective active"
NEIGHBOUR_EXCLUSIONS = {
    "שנת": (OTHER_LEXEME, "year of"),
    "שנים": (OTHER_LEXEME, "years"),
    "עבודה": (OTHER_LEXEME, "work"),
    "חייב": (HOMOGRAPH, "obliged, owes (חַיָּב)"),
    "חולים": (OTHER_LEXEME, "the sick"),
    "שנה": (OTHER_LEXEME, "year"),
    "החל": (HOMOGRAPH, "began; starting from (הֵחֵל)"),
    "יותר": (OTHER_LEXEME, "more (יוֹתֵר)"),
    "יקבע": (DEFECTIVE_ACTIVE, "will determine (יִקְבַּע)"),
    "שומה": (OTHER_LEXEME, "assessment"),
    "חוב": (OTHER_LEXEME, "debt"),
    "שני": (OTHER_LEXEME, "second; two of"),
    "שנות": (OTHER_LEXEME, "years of"),
    "הוציא": (HOMOGRAPH, "spent, took out (active)"),
    "חובה": (OTHER_LEXEME, "duty"),
    "יותרו": (HOMOGRAPH, "will be allowed (יוּתְּרוּ)"),
    "מחלות": (OTHER_LEXEME, "diseases"),
    "חשב": (DEFECTIVE_ACTIVE, "calculated (חִשֵּׁב); accountant"),
    "חולה": (OTHER_LEXEME, "ill; patient"),
    "יקבעו": (DEFECTIVE_ACTIVE, "will determine"),
    "לחולי": (OTHER_LEXEME, "for the sick of"),
    "מחלת": (OTHER_LEXEME, "disease of"),
    "פיטורים": (OTHER_LEXEME, "dismissal"),
    "מצא": (OTHER_LEXEME, "found"),
    "המשך": (OTHER_LEXEME, "continuation"),
    "מחובת": (OTHER_LEXEME, "from the duty of"),
    "מניעה": (OTHER_LEXEME, "impediment (noun); the listed head is construct מניעת"),
    "שום": (OTHER_LEXEME, "any (a listed quantifier)"),
    "תקבע": (DEFECTIVE_ACTIVE, "will determine (feminine)"),
    "החולה": (OTHER_LEXEME, "the patient"),
    "קבעה": (HOMOGRAPH, "she determined (active)"),
    "החלו": (HOMOGRAPH, "began (הֵחֵלּוּ)"),
    "עמוד": (OTHER_LEXEME, "column; page; spine"),
    "טעונה": (OTHER_LEXEME, "requires"),
    "ליישום": (OTHER_LEXEME, "for implementation: ל with the noun"),
    "עבוד": (OTHER_LEXEME, "שעבוד, a lien: ש is part of the word"),
    "חייבה": (HOMOGRAPH, "charged (active)"),
    "יצא": (OTHER_LEXEME, "went out"),
    "עמידה": (OTHER_LEXEME, "compliance; standing"),
    "תחל": (HOMOGRAPH, "will begin (תָּחֵל)"),
    "דווחו": (HOMOGRAPH, "were reported (verb)"),
    "חובו": (OTHER_LEXEME, "his debt"),
    "חול": (OTHER_LEXEME, "weekday; sand"),
    "טעונות": (OTHER_LEXEME, "require"),
    "מחייבת": (HOMOGRAPH, "requires (active)"),
    "אמרה": (HOMOGRAPH, "she said; saying (אִמְרָה)"),
    "הוציאה": (HOMOGRAPH, "spent (active)"),
    "משת": (OTHER_LEXEME, "משמשת, serves: מש is part of the word"),
    "נטלי": (OTHER_LEXEME, "דנטלי, dental: ד is part of the word"),
    "דוחו": (OTHER_LEXEME, "דו״חות, reports, cut at the gershayim"),
    "הניחה": (OTHER_LEXEME, "left behind"),
    "חולות": (OTHER_LEXEME, "sick (feminine plural)"),
    "חוסר": (OTHER_LEXEME, "lack (noun), read only as בחוסרו של"),
    "טעונת": (OTHER_LEXEME, "requiring"),
    "יחל": (HOMOGRAPH, "will begin"),
    "יפטור": (HOMOGRAPH, "will exempt (active)"),
    "ישונה": (HOMOGRAPH, "will be amended"),
    "לקיום": (OTHER_LEXEME, "for fulfilment"),
    "מחוב": (OTHER_LEXEME, "than the debt"),
    "מחייבות": (HOMOGRAPH, "require (active)"),
    "מחייבים": (HOMOGRAPH, "require (active)"),
    "מחשב": (DEFECTIVE_ACTIVE, "calculates; computer"),
    "מחשבים": (DEFECTIVE_ACTIVE, "calculate; computers"),
    "מחשבת": (DEFECTIVE_ACTIVE, "calculates (feminine)"),
    "נטל": (OTHER_LEXEME, "burden"),
    "נטלה": (HOMOGRAPH, "she took"),
    "נכיון": (OTHER_LEXEME, "discount"),
    "עמדות": (OTHER_LEXEME, "positions"),
    "עמדת": (OTHER_LEXEME, "position of"),
    "תשתנה": (HOMOGRAPH, "will change"),
    "דווח": (HOMOGRAPH, "was reported"),
    "דווחה": (HOMOGRAPH, "was reported (feminine)"),
    "דחיה": (OTHER_LEXEME, "deferral (noun)"),
    "הונחה": (HOMOGRAPH, "was tabled"),
    "הוציאו": (HOMOGRAPH, "spent (active plural)"),
    "המשויך": (OTHER_LEXEME, "associated"),
    "הנחיות": (OTHER_LEXEME, "guidelines"),
    "ושמה": (OTHER_LEXEME, "and places (וְשָׂמָה)"),
    "חיוביים": (OTHER_LEXEME, "positive"),
    "חישב": (HOMOGRAPH, "calculated (active)"),
    "חסם": (HOMOGRAPH, "blocked (active); block"),
    "יוציא": (HOMOGRAPH, "will spend (active)"),
    "יחוב": (HOMOGRAPH, "will owe"),
    "יחייב": (HOMOGRAPH, "will oblige (active)"),
    "יחסם": (OTHER_LEXEME, "שיחסם, whose ratio"),
    "יחשב": (DEFECTIVE_ACTIVE, "will calculate"),
    "יצאו": (OTHER_LEXEME, "went out (plural)"),
    "לחייבו": (OTHER_LEXEME, "to charge him: an infinitive with an object suffix"),
    "מחייב": (HOMOGRAPH, "requires (active)"),
    "מנוע": (OTHER_LEXEME, "barred: the passive participle, outside active מנע"),
    "מנועה": (OTHER_LEXEME, "barred (feminine): the passive participle"),
    "מניע": (OTHER_LEXEME, "motive"),
    "נעדרו": (HOMOGRAPH, "were absent (verb)"),
    "עמיד": (OTHER_LEXEME, "resistant"),
    "פיטורי": (OTHER_LEXEME, "dismissal of"),
    "קים": (OTHER_LEXEME, "שקים, cheques: ש is part of the word"),
    "תחייב": (HOMOGRAPH, "will bind (active)"),
    "תחשב": (HOMOGRAPH, "will be deemed (Nif'al); will calculate"),
}
# The holam alternates of every verb family that writes a holam, each added
# or refused with its reason. A family absent here writes no holam in a slot.
HOLAM_ALTERNATES = (
    ("כפר", "future", "יכפורו", "added", "plene plural beside יכפרו"),
    ("כפר", "future", "תכפרנה", "added", "defective feminine plural beside תכפורנה"),
    ("כפר", "future", "יכפר", "refused", "homograph: will atone (Pi'el)"),
    ("כפר", "future", "תכפר", "refused", "homograph: will atone (Pi'el)"),
    ("כפר", "present", "כפרת", "refused", "homograph: atonement of"),
    ("כפר", "present", "כפרים", "refused", "homograph: villages"),
    ("כפר", "present", "כפרות", "refused", "homograph: atonements"),
    ("עמד", "future", "יעמודו", "added", "plene plural beside יעמדו"),
    ("עמד", "future", "תעמדנה", "added", "defective feminine plural beside תעמודנה"),
    ("עמד", "future", "יעמד", "refused", "homograph: will set up (Hif'il)"),
    ("עמד", "future", "תעמד", "refused", "homograph: will set up (Hif'il)"),
    ("עמד", "present", "עמדת", "refused", "homograph: position of"),
    (
        "עמד",
        "present",
        "עמדים",
        "refused",
        "not an accepted spelling: the participle writes its holam",
    ),
    ("עמד", "present", "עמדות", "refused", "homograph: positions"),
    ("צמח", "present", "צמחת", "refused", "homograph: you grew"),
    ("צמח", "present", "צמחים", "refused", "homograph: plants"),
    (
        "צמח",
        "present",
        "צמחות",
        "refused",
        "not an accepted spelling: the participle writes its holam",
    ),
    (
        "דחה",
        "present",
        "דחים",
        "refused",
        "not an accepted spelling: the participle writes its holam",
    ),
    ("דחה", "present", "דחות", "refused", "homograph: postponements"),
    ("מנע", "present", "מנעת", "refused", "homograph: you prevented"),
    (
        "מנע",
        "present",
        "מנעים",
        "refused",
        "not an accepted spelling: the participle writes its holam",
    ),
    (
        "מנע",
        "present",
        "מנעות",
        "refused",
        "not an accepted spelling: the participle writes its holam",
    ),
    ("חל", "future", "יחל", "refused", "homograph: will begin"),
    ("חל", "future", "תחל", "refused", "homograph: will begin"),
    ("חל", "future", "יחלו", "refused", "homograph: will begin (plural)"),
    ("חל", "future", "תחלנה", "refused", "homograph: will begin (feminine plural)"),
    ("קם", "future", "תקומנה", "added", "plene feminine plural beside תקמנה"),
    ("קם", "future", "יקם", "refused", "homograph: will establish (Hif'il)"),
    ("קם", "future", "תקם", "refused", "homograph: will establish (Hif'il)"),
    ("קם", "future", "יקמו", "refused", "homograph: will establish (plural)"),
    ("נותר", "past", "נתר", "refused", "homograph: jumped"),
    ("נותר", "future", "ייוותר", "added", "full spelling beside יוותר"),
    ("נותר", "future", "ייוותרו", "added", "full spelling beside יוותרו"),
    ("נותר", "future", "יותר", "refused", "homograph: more"),
    ("נותר", "future", "יותרו", "refused", "homograph: will be allowed"),
    ("נוצר", "past", "נצר", "refused", "homograph: guarded; scion"),
    ("נוצר", "future", "ייצר", "refused", "homograph: will produce"),
)
# Defective spellings of listed words that stay out, each a homograph.
DEFECTIVE_EXCLUSIONS = (
    "יקבע",
    "יקבעו",
    "תקבע",
    "יחשב",
    "תחשב",
    "מחשב",
    "מחשבת",
    "מחשבים",
    "חשב",
    "עמדת",
    "עמדות",
    "דווח",
    "דווחו",
    "אמרה",
    "עבוד",
    "יחוב",
)


def generated_forms() -> dict[str, list[str]]:
    """Every generated form with its families: the negator and computation grammar."""

    forms: dict[str, set[str]] = collections.defaultdict(set)
    for kind, registry in (
        ("noun", grammar.NOUN_PARADIGMS),
        ("adjective", grammar.ADJECTIVE_PARADIGMS),
        ("verb", grammar.VERB_PARADIGMS),
    ):
        for name, family in registry.items():
            for form in family.forms:
                forms[form].add(f"{kind}:{name}")
    return {form: sorted(families) for form, families in forms.items()}


def _vav_yod_deletions(word: str) -> set[str]:
    return {word[:i] + word[i + 1 :] for i, char in enumerate(word) if char in "וי"}


def neighbour_inventory(snapshot: Path) -> list[dict]:
    """Snapshot words one ``ו`` or ``י`` from a generated form, with counts."""

    forms = generated_forms()
    deleted: dict[str, set[str]] = collections.defaultdict(set)
    for form in forms:
        for shorter in _vav_yod_deletions(form):
            deleted[shorter].add(form)
    tokens: collections.Counter[str] = collections.Counter()
    for line in snapshot.read_text().splitlines():
        record = json.loads(line)
        for field in ("body", "heading"):
            tokens.update(re.findall(r"[א-ת]+", fold_marks(record.get(field) or "")))
    chains = sorted(set(grammar.ORDINARY_PREFIX_CHAINS), key=len, reverse=True)
    counts: collections.Counter[str] = collections.Counter()
    near: dict[str, set[str]] = {}
    for token, count in tokens.items():
        splits = [
            token[len(chain) :]
            for chain in chains
            if token.startswith(chain) and len(token) - len(chain) >= 2
        ]
        # A token that is a generated form under some prefix chain is that form.
        generated = [word for word in splits if word in forms]
        for word in generated or splits:
            close = {form for form in _vav_yod_deletions(word) if form in forms}
            close |= deleted.get(word, set())
            close.discard(word)
            if close:
                counts[word] += count
                near[word] = close
    return [
        {"word": word, "count": count, "near": sorted(near[word])}
        for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def classify_neighbours(inventory: list[dict]) -> list[dict]:
    forms = generated_forms()
    rows = []
    for row in inventory:
        word = row["word"]
        if word in forms:
            rows.append({**row, "class": "generated", "reason": ""})
        elif word in NEIGHBOUR_EXCLUSIONS:
            kind, reason = NEIGHBOUR_EXCLUSIONS[word]
            rows.append({**row, "class": kind, "reason": reason})
        else:
            rows.append({**row, "class": "unclassified", "reason": ""})
    return rows


def spelling_closure(snapshot: Path | None, pinned: dict | None) -> dict:
    """The round-21 fixture object: the classified inventory and the tables."""

    if snapshot is not None:
        inventory = neighbour_inventory(snapshot)
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    elif pinned is not None:
        inventory = [
            {key: row[key] for key in ("word", "count", "near")}
            for row in pinned["neighbours"]
        ]
        digest = pinned["snapshot_sha256"]
    else:
        raise SystemExit("--snapshot is required to build the neighbour inventory")
    return {
        "snapshot_sha256": digest,
        "neighbours": classify_neighbours(inventory),
        "holam_alternates": [
            {
                "family": family,
                "slot": slot,
                "form": form,
                "decision": decision,
                "reason": reason,
            }
            for family, slot, form, decision, reason in HOLAM_ALTERNATES
        ],
        "defective_exclusions": list(DEFECTIVE_EXCLUSIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="the pinned statute snapshot (JSONL) for the round-21 neighbour sweep",
    )
    args = parser.parse_args()
    original = FIXTURE.read_text()
    fixture = json.loads(original)
    generated = {
        "generator": "scripts/generate_hebrew_paradigm_audit.py",
        "reason_baseline": BASELINE,
        "frames": family_frames(),
    }
    if args.baseline_source:
        generated["reason_verdicts"] = baseline_reasons(args.baseline_source)
    elif "round20" in fixture:
        generated["reason_verdicts"] = fixture["round20"]["reason_verdicts"]
    else:
        parser.error("--baseline-source is required to receipt the reason verdicts")
    closure = spelling_closure(args.snapshot, fixture.get("round21"))
    if args.check:
        if fixture.get("round20") != generated:
            raise SystemExit("Hebrew paradigm audit fixture is out of date")
        if [row[0] for row in generated["reason_verdicts"]] != reason_forms():
            raise SystemExit("Hebrew reason paradigm receipt is out of date")
        unclassified = [
            row["word"]
            for row in closure["neighbours"]
            if row["class"] == "unclassified"
        ]
        if unclassified:
            raise SystemExit(f"Unclassified spelling neighbours: {unclassified}")
        if fixture.get("round21") != closure:
            raise SystemExit("Hebrew spelling closure fixture is out of date")
        return
    # Preserve every existing line and row. Only our appended objects are
    # replaced on regeneration; the original 1,213 target rows remain
    # byte-identical.
    marker = ',\n  "round20": '
    if marker in original:
        prefix = original.split(marker, 1)[0]
    else:
        prefix = original.rstrip()[:-1].rstrip()
    payload = json.dumps(generated, ensure_ascii=False, indent=2)
    payload = payload.replace("\n", "\n  ")
    closure_payload = json.dumps(closure, ensure_ascii=False, indent=2)
    closure_payload = closure_payload.replace("\n", "\n  ")
    temporary = FIXTURE.with_suffix(".json.tmp")
    temporary.write_text(
        prefix + marker + payload + ',\n  "round21": ' + closure_payload + "\n}\n"
    )
    temporary.replace(FIXTURE)


if __name__ == "__main__":
    main()
