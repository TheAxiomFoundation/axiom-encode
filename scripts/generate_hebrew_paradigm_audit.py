#!/usr/bin/env python3
"""Append generated round-20 family frames without rewriting prior audit rows.

The frames refer to grammatical slots, rather than copying all combinations
of forms, prefixes and visible joints into JSON. The test expands those slots.
Reason verdicts are receipted separately against the unchanged source-only
boundary at 0acff3551; --baseline-source points to that commit's src directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from axiom_encode.harness import hebrew_negators as grammar

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
            for form in family.future
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
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
    if args.check:
        if fixture.get("round20") != generated:
            raise SystemExit("Hebrew paradigm audit fixture is out of date")
        if [row[0] for row in generated["reason_verdicts"]] != reason_forms():
            raise SystemExit("Hebrew reason paradigm receipt is out of date")
        return
    # Preserve every existing line and row. Only our appended object is replaced
    # on regeneration; the original 1,213 target rows remain byte-identical.
    marker = ',\n  "round20": '
    if marker in original:
        prefix = original.split(marker, 1)[0]
    else:
        prefix = original.rstrip()[:-1].rstrip()
    payload = json.dumps(generated, ensure_ascii=False, indent=2)
    payload = payload.replace("\n", "\n  ")
    temporary = FIXTURE.with_suffix(".json.tmp")
    temporary.write_text(prefix + marker + payload + "\n}\n")
    temporary.replace(FIXTURE)


if __name__ == "__main__":
    main()
