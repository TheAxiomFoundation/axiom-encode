#!/usr/bin/env python3
"""Check whether a case's provision text carries the evidence for its defect.

``provision.txt`` is resolved from the module's first ``corpus_citation_path``
only, so the text that shows the pre-fix module is wrong can live under a
different citation. This tool records, per case, whether strings tied to the
change appear in the shipped provision:

``evidence_in_provision``
    ``present``  at least one post-side string (below) appears in the provision.
    ``absent``   strings were tested and no post-side string appears.
    ``unknown``  nothing could be tested, or the case is metadata-only
                 (``artifacts_shipped`` false: no shipped files to test).

``evidence_check``
    The method name, the rules tested, every string tested with its origin,
    side and outcome, and the reason for the status.

Strings tested, for the rules named in ``locator.rule_names``:

* ``added_excerpt``: the ``source.excerpt`` of each proof atom in
  ``post_fix.yaml`` that the fix added (no pre-fix atom has the same
  ``path``, cited provision and excerpt), or that belongs to a rule the fix
  added.
* ``context_excerpt``: the excerpt of an unchanged atom whose ``path`` names a
  rule field the fix changed (a ``versions[i].<field>`` or a rule-level key
  other than ``name``, ``metadata`` and ``source``). It counts toward the
  status only when the fix added no post-side ``added_excerpt`` or
  ``added_value``; otherwise it is recorded with ``counts`` false.
* ``added_value`` and ``removed_value``: numbers, ISO dates and code-like
  identifiers (``9903.01.77``, ``165(d)``) in the changed fields, present on
  only one side of the fix. A number is tested in its written form, with
  thousands separators, and for a fraction below one as a percentage
  (``0.15`` as ``15 percent``, ``15 per cent`` and ``15%``); a date as
  ``November 14, 2025`` and ``14 November 2025``. Integers below 10 are not
  tested.
* ``triage_quote``: spans of eight or more words (and 20 or more characters)
  that the triage record quotes (between double, curly or single quotation
  marks) in the reader's
  ``pre_fix_wrong_because`` and notes and the verifier's justification and
  notes, excluding spans that look like code.

Matching casefolds both sides and collapses whitespace runs to one space. A
number, date or code form must also not touch another digit (``15`` does not
match inside ``1.15`` or ``150``). An excerpt or quote with an ellipsis
(``...`` or ``…``) matches when its fragments occur in order, each starting
within ``MAX_ELISION_CHARS`` (300) characters of the end of the one before.
The test is mechanical: it does not read the provision for meaning.

Side. ``removed_value`` strings, and any excerpt or quote that contains a
removed value, are pre-side: they support the pre-fix module (the printed
``$143`` of 7 CFR 273.10 in ``us-015``). Everything else is post-side. When
only pre-side strings appear, the reason is ``provision_supports_pre_fix``.

Usage::

    uv run python benchmarks/verifier/real_defects_v0/tools/check_evidence.py \\
        --corpus-dir benchmarks/verifier/real_defects_v0 [--write]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

METHOD = "changed_rule_strings_v1"
ORIGINS = (
    "added_excerpt",
    "context_excerpt",
    "added_value",
    "removed_value",
    "triage_quote",
)
NORMALIZATION = (
    "casefold; whitespace runs collapsed to one space; number, date and code "
    "forms must not touch another digit"
)
QUOTE_FIELDS = (
    "pre_fix_wrong_because",
    "triage_notes",
    "verifier_justification",
    "verifier_notes",
)
MIN_QUOTE_WORDS = 8
UNCOMPARED_RULE_KEYS = frozenset({"name", "metadata", "source"})
MAX_ELISION_CHARS = 300
_ELLIPSIS = re.compile(r"\.\.\.|\u2026")
STATUSES = ("present", "absent", "unknown")

_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_NUMBER = re.compile(r"(?<![\w.])\d[\d_,]*(?:\.\d+)?(?![\w])")
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DOTTED_CODE = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w])")
_PAREN_CODE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[A-Za-z]?(?:\([0-9A-Za-z]{1,5}\))+")
_QUOTED = (
    re.compile(r'"([^"\n]{20,})"'),
    re.compile(r"“([^”\n]{20,})”"),
    re.compile(r"(?:(?<=[\s(\[:])|^)'([^'\n]{20,}?)'(?=[\s.,;:)\]]|$)"),
)
_CODE_HINT = re.compile(r"`|=>|==|!=|<=|>=|\w_\w|\bformula\b|\bversions\[")


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def load_module(text: str) -> dict[str, Any]:
    try:
        doc = yaml.load(text, Loader=_LOADER)  # noqa: S506 (safe loader)
    except yaml.YAMLError:
        return {}
    return doc if isinstance(doc, dict) else {}


def rules_by_name(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rules = doc.get("rules")
    if not isinstance(rules, list):
        return {}
    return {
        str(rule["name"]): rule
        for rule in rules
        if isinstance(rule, dict) and rule.get("name") is not None
    }


def proof_atoms(rule: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not rule:
        return []
    proof = (rule.get("metadata") or {}).get("proof") or {}
    atoms = proof.get("atoms") if isinstance(proof, dict) else None
    return [atom for atom in atoms or [] if isinstance(atom, dict)]


def _atom_key(atom: dict[str, Any]) -> tuple[str, str, str]:
    """An excerpt atom's identity: its path, cited provision and excerpt."""

    source = atom.get("source") if isinstance(atom.get("source"), dict) else {}
    return (
        str(atom.get("path") or ""),
        str(source.get("corpus_citation_path") or ""),
        " ".join(str(source.get("excerpt") or "").split()),
    )


def changed_fields(
    pre: dict[str, Any] | None, post: dict[str, Any] | None
) -> dict[str, tuple[Any, Any]]:
    """Rule fields whose value differs, as ``{path: (pre_value, post_value)}``.

    Paths are rule-level keys or ``versions[i].<field>``. ``metadata`` (the
    proof atoms) is compared separately, and ``source`` is skipped: it is a
    citation label, and the corpus treats citation-string changes as
    provenance, not meaning. A rule present on one side only yields every
    field, with ``None`` on the missing side.
    """

    pre = pre or {}
    post = post or {}
    changed: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(pre) | set(post)):
        if key in UNCOMPARED_RULE_KEYS:
            continue
        if key == "versions":
            before = pre.get("versions") or []
            after = post.get("versions") or []
            for index in range(max(len(before), len(after))):
                old = before[index] if index < len(before) else {}
                new = after[index] if index < len(after) else {}
                old = old if isinstance(old, dict) else {"value": old}
                new = new if isinstance(new, dict) else {"value": new}
                for field in sorted(set(old) | set(new)):
                    if old.get(field) != new.get(field):
                        changed[f"versions[{index}].{field}"] = (
                            old.get(field),
                            new.get(field),
                        )
        elif pre.get(key) != post.get(key):
            changed[key] = (pre.get(key), post.get(key))
    return changed


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return json.dumps(value, sort_keys=True, default=str)


def value_tokens(text: str) -> set[str]:
    """Numbers, ISO dates and code-like identifiers in ``text``."""

    tokens: set[str] = set()
    for match in _DATE.finditer(text):
        tokens.add(match.group(0))
    stripped = _DATE.sub(" ", text)
    for pattern in (_DOTTED_CODE, _PAREN_CODE):
        for match in pattern.finditer(stripped):
            tokens.add(match.group(0))
        stripped = pattern.sub(" ", stripped)
    for match in _NUMBER.finditer(stripped):
        raw = match.group(0).replace("_", "").replace(",", "")
        try:
            number = Decimal(raw)
        except InvalidOperation:
            continue
        if number == number.to_integral_value() and abs(number) < 10:
            continue
        tokens.add(_canonical_number(number))
    return tokens


def _canonical_number(number: Decimal) -> str:
    text = format(number.normalize(), "f")
    return text


def value_forms(token: str) -> list[str]:
    """The written forms a value token is searched for in the provision."""

    date = _DATE.fullmatch(token)
    if date:
        try:
            day = dt.date(int(date[1]), int(date[2]), int(date[3]))
        except ValueError:
            return [token]
        month = day.strftime("%B")
        return [
            token,
            f"{month} {day.day}, {day.year}",
            f"{day.day} {month} {day.year}",
        ]
    if _DOTTED_CODE.fullmatch(token) or _PAREN_CODE.fullmatch(token):
        return [token]
    try:
        number = Decimal(token)
    except InvalidOperation:
        return [token]
    forms = [token]
    if number == number.to_integral_value() and abs(number) >= 1000:
        forms.append(f"{int(number):,}")
    elif abs(number) >= 1000:
        whole, _, frac = token.partition(".")
        forms.append(f"{int(whole):,}.{frac}")
    if 0 < number < 1:
        percent = _canonical_number(number * 100)
        forms += [f"{percent} percent", f"{percent} per cent", f"{percent}%"]
    return forms


def _form_pattern(form: str) -> re.Pattern[str]:
    escaped = re.escape(normalize(form))
    head = r"(?<![0-9])(?<![0-9][.,])" if form[:1].isdigit() else ""
    tail = r"(?![0-9]|[.,][0-9])" if form[-1:].isdigit() else ""
    return re.compile(head + escaped + tail)


def triage_quotes(case: dict[str, Any]) -> list[tuple[str, str]]:
    """``(field, span)`` for quoted prose spans in the triage record."""

    triage = case.get("triage") or {}
    found: list[tuple[str, str]] = []
    for field in QUOTE_FIELDS:
        text = triage.get(field) or ""
        for pattern in _QUOTED:
            for match in pattern.finditer(text):
                span = match.group(1).strip()
                if len(span.split()) < MIN_QUOTE_WORDS or _CODE_HINT.search(span):
                    continue
                found.append((f"triage.{field}", span))
    return found


def check_case(
    case: dict[str, Any],
    pre_text: str | None,
    post_text: str | None,
    provision_text: str | None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(evidence_in_provision, evidence_check)`` for one case."""

    check: dict[str, Any] = {
        "method": METHOD,
        "normalization": NORMALIZATION,
        "rules_tested": [],
        "rules_missing": [],
        "strings": [],
        "post_side_tested": 0,
        "post_side_matched": 0,
        "pre_side_tested": 0,
        "pre_side_matched": 0,
        "reason": "",
    }
    if pre_text is None or post_text is None or provision_text is None:
        check["reason"] = "metadata_only"
        return "unknown", check
    pre_rules = rules_by_name(load_module(pre_text))
    post_rules = rules_by_name(load_module(post_text))
    provision = normalize(provision_text)
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    removed_tokens: set[str] = set()
    before: set[tuple[str, str]] = set()
    after: set[tuple[str, str]] = set()
    excerpts: list[tuple[str, str, dict[str, Any]]] = []

    def add(origin: str, text: str, where: str, side: str, extra: dict) -> None:
        key = (side, origin, text)
        if key in entries:
            if where not in entries[key]["where"]:
                entries[key]["where"].append(where)
            return
        entries[key] = {
            "origin": origin,
            "side": side,
            "text": text,
            "where": [where],
            **extra,
        }

    for name in (case.get("locator") or {}).get("rule_names") or []:
        pre_rule = pre_rules.get(name)
        post_rule = post_rules.get(name)
        if pre_rule is None and post_rule is None:
            check["rules_missing"].append(name)
            continue
        check["rules_tested"].append(name)
        changed = changed_fields(pre_rule, post_rule)
        for path, (old, new) in changed.items():
            before |= {(f"{name}.{path}", t) for t in value_tokens(_flatten(old))}
            after |= {(f"{name}.{path}", t) for t in value_tokens(_flatten(new))}
        old_atoms = {_atom_key(atom) for atom in proof_atoms(pre_rule)}
        whole_rule_new = pre_rule is None
        for atom in proof_atoms(post_rule):
            source = atom.get("source") or {}
            excerpt = source.get("excerpt") if isinstance(source, dict) else None
            if not isinstance(excerpt, str) or not excerpt.strip():
                continue
            atom_path = str(atom.get("path") or "")
            if whole_rule_new or _atom_key(atom) not in old_atoms:
                origin = "added_excerpt"
            elif atom_path in changed:
                origin = "context_excerpt"
            else:
                continue
            excerpts.append(
                (
                    origin,
                    excerpt.strip(),
                    f"{name}.{atom_path}",
                    {"cites": source.get("corpus_citation_path")},
                )
            )
    # A value is added or removed when it occurs on one side only, across the
    # changed fields of every named rule (a value that moved between rules is
    # neither).
    old_tokens = {token for _where, token in before}
    new_tokens = {token for _where, token in after}
    for where, token in sorted(after):
        if token not in old_tokens:
            add("added_value", token, where, "post", {})
    for where, token in sorted(before):
        if token not in new_tokens:
            removed_tokens.add(token)
            add("removed_value", token, where, "pre", {})
    for origin, excerpt, where, extra in excerpts:
        side = "pre" if _contains_any(excerpt, removed_tokens) else "post"
        add(origin, excerpt, where, side, extra)
    for field, span in triage_quotes(case):
        side = "pre" if _contains_any(span, removed_tokens) else "post"
        add("triage_quote", span, field, side, {})

    # Context excerpts (unchanged atoms on a changed field) count only when the
    # fix added no excerpt or value of its own: when it did, the added text is
    # what the correction rests on, and the unchanged text stood beside the
    # defect.
    fix_added = any(
        e["side"] == "post" and e["origin"] in {"added_excerpt", "added_value"}
        for e in entries.values()
    )
    for entry in sorted(
        entries.values(),
        key=lambda e: (ORIGINS.index(e["origin"]), e["side"], e["text"]),
    ):
        if entry["origin"] in {"added_value", "removed_value"}:
            forms = value_forms(entry["text"])
            entry["forms"] = forms
            entry["matched"] = any(_form_pattern(f).search(provision) for f in forms)
        else:
            entry["matched"] = text_in(entry["text"], provision)
        entry["counts"] = not (entry["origin"] == "context_excerpt" and fix_added)
        check["strings"].append(entry)
        if not entry["counts"]:
            continue
        check[f"{entry['side']}_side_tested"] += 1
        if entry["matched"]:
            check[f"{entry['side']}_side_matched"] += 1

    if not check["strings"]:
        check["reason"] = "nothing_to_test"
        return "unknown", check
    if check["post_side_matched"]:
        check["reason"] = "post_side_string_found"
        return "present", check
    if check["pre_side_matched"]:
        check["reason"] = "provision_supports_pre_fix"
    else:
        check["reason"] = "no_tested_string_found"
    return "absent", check


def text_in(text: str, provision: str) -> bool:
    """Whether a quoted string occurs in the (normalized) provision.

    An ellipsis marks elided source text, so a string with one matches when
    its fragments occur in order, each within ``MAX_ELISION_CHARS`` of the
    end of the one before.
    """

    fragments = [normalize(part) for part in _ELLIPSIS.split(text)]
    fragments = [part for part in fragments if part]
    if not fragments:
        return False
    start = provision.find(fragments[0])
    while start >= 0:
        end = start + len(fragments[0])
        for part in fragments[1:]:
            found = provision.find(part, end)
            if found < 0 or found - end > MAX_ELISION_CHARS:
                break
            end = found + len(part)
        else:
            return True
        start = provision.find(fragments[0], start + 1)
    return False


def _contains_any(text: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    normalized = normalize(text)
    return any(
        _form_pattern(form).search(normalized)
        for token in sorted(tokens)
        for form in value_forms(token)
    )


def check_case_dir(case_dir: Path, case: dict[str, Any]) -> tuple[str, dict]:
    files = {
        name: case_dir / name
        for name in ("pre_fix.yaml", "post_fix.yaml", "provision.txt")
    }
    if not case.get("artifacts_shipped", True) or not all(
        path.exists() for path in files.values()
    ):
        return check_case(case, None, None, None)
    texts = {name: path.read_text(encoding="utf-8") for name, path in files.items()}
    return check_case(
        case, texts["pre_fix.yaml"], texts["post_fix.yaml"], texts["provision.txt"]
    )


def with_evidence(
    case: dict[str, Any], status: str, check: dict[str, Any]
) -> dict[str, Any]:
    """The case with both evidence keys placed after ``provision_resolution``."""

    out: dict[str, Any] = {}
    for key, value in case.items():
        if key in {"evidence_in_provision", "evidence_check"}:
            continue
        out[key] = value
        if key == "provision_resolution":
            out["evidence_in_provision"] = status
            out["evidence_check"] = check
    if "evidence_in_provision" not in out:
        out["evidence_in_provision"] = status
        out["evidence_check"] = check
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--write", action="store_true", help="write the fields into each case.json"
    )
    args = parser.parse_args(argv)
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for path in sorted((args.corpus_dir / "cases").glob("*/case.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        status, check = check_case_dir(path.parent, case)
        counts[status] += 1
        reasons[check["reason"]] += 1
        if args.write:
            text = json.dumps(
                with_evidence(case, status, check), indent=2, ensure_ascii=False
            )
            path.write_text(text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "by_status": dict(sorted(counts.items())),
                "by_reason": dict(sorted(reasons.items())),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
