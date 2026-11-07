"""Complete Schedule declaration ownership, separate from actual observation.

Names corroborate a reviewed factual contract; they do not prove fact truth.
No projected document or unaccounted declaration receives snapshot eligibility.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from axiom_encode.bound_schedule_conditions import (
    ScheduleConditionRecords,
    recognize_schedule_conditions,
)
from axiom_encode.bound_schedule_structure import BoundScheduleStructure

from .bound_form_operands import _number, _tree
from .imported_export_evidence import _named, _payload


def _tokens(name):
    return {
        w if w in {"has", "is", "was"} else w.removesuffix("s")
        for w in name.lower().split("_")
    }


def _role(name, words):
    tokens = _tokens(name)
    return (
        not tokens & {"no", "not", "never", "without", "ineligible"}
        and _tokens("_".join(words)) <= tokens
    )


# Closed reviewed factual vocabulary, independently checked against source roles.
# This corroborates a caller mapping; it does not authenticate factual truth.
_FACT_CONTRACTS = {
    "use": (
        "is_completing_home_renovation_schedule",
        "This individual is completing this schedule; does not assert eligibility or completion compliance.",
    ),
    "paper": ("is_filing_paper_return", "This individual is filing a paper return."),
    "space": (
        "needs_additional_space",
        "The individual completing this form needs more space for its entries.",
    ),
    "eligible": (
        "has_eligible_home_renovation_expenses",
        "The individual has eligible home renovation expenses as an externally supplied form premise, not derived from positive amount or age.",
    ),
    "residence": (
        "expenses_relate_to_return_page_one_residence",
        "The claimant answers whether expenses on line1 relate to residence address entered on return page1; no address equality inferred.",
    ),
    "relationship": (
        "has_spouse_or_common_law_partner_at_reference_date",
        "The claimant has the explicitly identified spouse/common-law partner at the medical reference date. Reciprocal fixture descriptions identify two distinct claimants; no engine relation is inferred.",
    ),
    "separate": (
        "occupied_separate_principal_residences_at_reference_date",
        "Claimant and identified spouse/common-law partner occupied separate principal residences on the explicit reference date.",
    ),
    "medical": (
        "principal_residence_separation_was_for_medical_reasons",
        "The stated separate principal residences at the reference date were for medical reasons.",
    ),
    "date": (
        "medical_separation_reference_date",
        "Date at which all three medical-separation premise facts are asserted; not the date of this execution.",
    ),
    "gst": (
        "gst_hst_number_is_applicable",
        "For this explicitly identified sales-slip/contract entry, GST/HST number is applicable; no claimant-entry ownership or complete table inferred.",
    ),
}


def _fact_role(item, role, return_page):
    expected_name, expected_description = _FACT_CONTRACTS[role]
    names = {expected_name}
    if role == "residence":
        words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }
        names = {expected_name.replace("page_one", "page_" + str(return_page))}
        if return_page in words:
            names.add(expected_name.replace("page_one", "page_" + words[return_page]))
        expected_description = expected_description.replace(
            "page1;", f"page{return_page};"
        )
    if item.get("name") not in names:
        return False
    if "description" not in item:
        return True
    description = item["description"]
    return (
        isinstance(description, str)
        and " ".join(description.split()) == expected_description
    )


def _expression(value):
    node = _tree(value)
    return ast.dump(node) if node is not None else None


def _entered_role(item, row):
    """Corroborate the reviewed opaque fact mapping, not its factual truth."""
    name = item.get("name", "")
    description = item.get("description", "")
    if not isinstance(description, str) or not description.strip():
        return False
    description = re.sub(r"\s+", " ", description).lower()
    if re.search(
        r"\b(?:not|never|without)\s+(?:the\s+)?(?:total|eligible|government|assistance|expenses|claimed)\b",
        description,
    ):
        return False
    if row == 1:
        return (
            _role(name, {"total", "eligible", "expense", "line", "1"})
            and "assistance" not in _tokens(name)
            and "total eligible expenses" in description
            and "government assistance" not in description
        )
    if row == 2:
        return (
            _role(name, {"government", "assistance", "line", "2"})
            and "government assistance received or receivable" in description
            and re.search(r"included in .+line\s*1\b", description) is not None
        )
    if row == 4:
        return (
            _role(name, {"expense", "claimed", "other", "individual", "line", "4"})
            and re.search(r"expenses included in .+line\s*3\b", description) is not None
            and "claimed by other eligible individuals" in description
        )
    return False


def _formula_proof(rule, form, conditions):
    atoms = rule.get("metadata", {}).get("proof", {}).get("atoms")
    if not isinstance(atoms, list) or len(atoms) != 3:
        return None
    by_path = {a.get("path"): a for a in atoms if isinstance(a, dict)}
    if len(by_path) != 3:
        return None
    result = None
    for field in ("formula", "effective_from", "effective_to"):
        atom = by_path.get("versions[0]." + field)
        if not isinstance(atom, dict) or set(atom) != {"path", "kind", "source"}:
            return None
        source = atom.get("source")
        if (
            not isinstance(source, dict)
            or set(source) != {"corpus_citation_path", "excerpt"}
            or source["corpus_citation_path"] != form.citation
        ):
            return None
        excerpt = source["excerpt"]
        if not isinstance(excerpt, str) or not excerpt:
            return None
        if field == "formula":
            if atom["kind"] != "formula" or form.source.count(excerpt) != 1:
                return None
            result = excerpt
        else:
            if atom["kind"] != "effective_period" or excerpt != str(form.year):
                return None
            date_span = conditions.medical.reference_date_span
            year_end = date_span[1]
            allowed = [form.header_year_span, (year_end - 4, year_end)]
            if form.wrapper_year_span is not None:
                allowed.append(form.wrapper_year_span)
            if [
                m.span() for m in re.finditer(re.escape(excerpt), form.source)
            ] != sorted(allowed):
                return None
    return result


@dataclass(frozen=True)
class ScheduleBinding:
    form: BoundScheduleStructure
    conditions: ScheduleConditionRecords
    content_sha256: str
    namespace: str | None
    row_owners: tuple[tuple[int, str], ...]
    roles: tuple[tuple[str, str], ...]
    input_roles: tuple[tuple[str, str], ...]
    formulas: tuple[tuple[str, str], ...]
    entered: tuple[tuple[str, str], ...]

    def matches(self, *, content, branch, rule, principal_rules, case, leaf, trace):
        if (
            hashlib.sha256(content.encode()).hexdigest() != self.content_sha256
            or not isinstance(rule, dict)
            or not isinstance(case, dict)
            or trace
        ):
            return False
        payload = _payload(content)
        rules = _named(payload, "rules") if payload else None
        if (
            rules is None
            or {n: r for n, r in rules.items() if r.get("kind") == "derived"}
            != principal_rules
        ):
            return False
        owner = next(
            (
                name
                for number, name in self.row_owners
                if next(r for r in self.form.rows if r.number == number).span
                == (branch.start, branch.end)
            ),
            None,
        )
        if (
            owner != rule.get("name")
            or rules.get(owner) != rule
            or self.form.source[branch.start : branch.end] != branch.text
        ):
            return False
        if (
            _expression(leaf) != dict(self.formulas).get(owner)
            or case.get("period") != self.period
        ):
            return False
        facts = self.case_facts(case)
        if facts is None:
            return False
        needed = self.reached_inputs(owner, rules)
        return needed is not None and all(
            n in facts and _number(facts[n]) is not None for n in needed
        )

    @property
    def period(self):
        return {
            "period_kind": "tax_year",
            "start": f"{self.form.year}-01-01",
            "end": f"{self.form.year}-12-31",
        }

    def case_facts(self, case):
        values = case.get("input")
        if not isinstance(values, dict):
            return None
        result = {}
        for key, value in values.items():
            if not isinstance(key, str):
                return None
            if "#" in key:
                if self.namespace is None or not key.startswith(
                    self.namespace + "#input."
                ):
                    return None
                name = key[len(self.namespace + "#input.") :]
            else:
                name = key.removeprefix("input.")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in result:
                return None
            result[name] = value
        return result

    def reached_inputs(self, name, rules):
        inputs = {n for _, n in self.input_roles} | {n for _, n in self.entered}
        reached, active = set(), set()

        def visit(n):
            if n in inputs:
                reached.add(n)
                return True
            if n in {"period_end", "period_start", "min"}:
                return True
            if n in active or n not in rules:
                return False
            active.add(n)
            node = _tree(rules[n]["versions"][0]["formula"])
            valid = node is not None and all(
                visit(child.id)
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            )
            active.remove(n)
            return valid

        return reached if visit(name) else None


def bind_schedule(
    form: BoundScheduleStructure,
    content: str,
    sources: Mapping[str, str | None],
    *,
    root_relative: str | None = None,
) -> ScheduleBinding | None:
    """Closed static graph; no execution evidence or condition clearance."""
    try:
        conditions = recognize_schedule_conditions(form)
        if conditions is None or sources.get(form.citation) != form.source:
            return None
        namespace = None
        if root_relative is not None:
            if (
                re.fullmatch(
                    r"[a-z][a-z0-9-]*/policies/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.yaml",
                    root_relative,
                )
                is None
            ):
                return None
            path = PurePosixPath(root_relative)
            namespace = (
                path.parts[0]
                + ":"
                + PurePosixPath(*path.parts[1:]).with_suffix("").as_posix()
            )
        payload = _payload(content)
        if (
            payload is None
            or set(payload)
            - {"format", "module", "units", "inputs", "rules", "imports"}
            or payload.get("imports", []) != []
        ):
            return None
        rules, inputs, units = (
            _named(payload, k) for k in ("rules", "inputs", "units")
        )
        if (
            rules is None
            or inputs is None
            or units != {"CAD": {"name": "CAD", "kind": "currency", "minor_units": 2}}
            or set(rules) & set(inputs)
            or (set(rules) | set(inputs)) & {"period_start", "period_end", "min"}
        ):
            return None
        module = payload.get("module")
        if (
            not isinstance(module, dict)
            or module.get("proof_validation") != {"required": True}
            or module.get("source_verification")
            != {
                "corpus_citation_path": form.citation,
                "source_sha256": form.source_sha256,
            }
        ):
            return None
        proof = {}
        for n, r in rules.items():
            if (
                set(r)
                - {
                    "name",
                    "kind",
                    "dtype",
                    "source",
                    "unit",
                    "entity",
                    "period",
                    "versions",
                    "metadata",
                }
                or r.get("metadata", {}).keys() != {"proof"}
                or r["metadata"]["proof"].keys() != {"atoms"}
            ):
                return None
            if (
                r.get("kind") not in {"derived", "parameter"}
                or len(r.get("versions", [])) != 1
            ):
                return None
            v = r["versions"][0]
            if (
                set(v) != {"effective_from", "effective_to", "formula"}
                or v["effective_from"] != f"{form.year}-01-01"
                or v["effective_to"] != f"{form.year}-12-31"
                or _tree(v["formula"]) is None
            ):
                return None
            p = _formula_proof(r, form, conditions)
            if p is None:
                return None
            proof[n] = p
        for i in inputs.values():
            if (
                set(i) - {"name", "dtype", "unit", "entity", "period", "description"}
                or i.get("period") != "Year"
                or not isinstance(i.get("description"), str)
            ):
                return None
        rows = {r.number: r for r in form.rows}
        if set(rows) != set(range(1, 8)) or rows[5].field_number != 60360:
            return None
        text = {n: form.source[slice(*r.span)] for n, r in rows.items()}
        caption = {
            n: re.sub(r"\s+", " ", form.source[slice(*r.caption_span)])
            for n, r in rows.items()
        }
        if (
            caption[2]
            != "Amount of government assistance received or receivable included in the amount on line 1 ^"
            or caption[4]
            != "Amount of eligible expenses included on line 3 being claimed by other eligible individuals ^"
        ):
            return None
        cap_match = re.fullmatch(
            r"Line 1 minus line 2 \(maximum \$(\d[\d,]*)\) = \^", caption[3]
        )
        rate_match = re.fullmatch(r"Applicable rate (\d+(?:\.\d+)?)%", caption[6])
        if (
            cap_match is None
            or rate_match is None
            or not re.fullmatch(r"Line 3 minus line 4 [A-Za-z '-]+ = \^", caption[5])
            or not re.fullmatch(
                r"Line 5 multiplied by the percentage from line 6 Enter this amount on line 47900 of your return\. [A-Za-z '-]+ = \^",
                caption[7],
            )
        ):
            return None
        if any(
            re.search(
                r"\b(?:if|when|not|never|provided|otherwise|subject|but|however|less|plus|unless|except|after|before|subtract|deduct|pay|claim|ignore)\b",
                caption[n],
                re.I,
            )
            for n in (5, 7)
        ):
            return None
        used = set()
        owners = {}
        entered = {}
        roles = {}
        input_roles = {}

        def pick(kind, dtype, formula, excerpt, entity="Person", words=()):
            found = [
                n
                for n, r in rules.items()
                if n not in used
                and r.get("kind") == kind
                and r.get("dtype") == dtype
                and proof[n] == excerpt
                and _expression(r["versions"][0]["formula"]) == _expression(formula)
                and (
                    kind == "parameter"
                    or (r.get("entity") == entity and r.get("period") == "Year")
                )
                and r.get("unit") == ("CAD" if dtype == "Money" else None)
                and (not words or _role(n, words))
            ]
            if len(found) != 1:
                raise ValueError("Nonunique source owner")
            used.add(found[0])
            return found[0]

        for n in (1, 2, 4):
            found = []
            for name, r in rules.items():
                node = _tree(r["versions"][0]["formula"])
                if (
                    isinstance(node, ast.Name)
                    and node.id in inputs
                    and proof[name] == text[n]
                ):
                    i = inputs[node.id]
                    if (
                        i.get("dtype") == "Money"
                        and i.get("unit") == "CAD"
                        and i.get("entity") == "Person"
                        and _entered_role(i, n)
                    ):
                        found.append((name, node.id))
            if len(found) != 1:
                return None
            name, inp = found[0]
            owners[n] = pick("derived", "Money", inp, text[n])
            entered[name] = inp
        roles["cap"] = pick(
            "parameter", "Money", int(cap_match[1].replace(",", "")), text[3]
        )
        roles["rate"] = pick(
            "parameter", "Rate", str(_number(rate_match[1]) / 100), text[6]
        )

        def group(a, b):
            return form.source[rows[a].span[0] : rows[b].span[1]]

        owners[3] = pick(
            "derived",
            "Money",
            f"min({roles['cap']}, {owners[1]} - {owners[2]})",
            group(1, 3),
        )
        owners[5] = pick("derived", "Money", f"{owners[3]} - {owners[4]}", group(3, 5))
        owners[6] = roles["rate"]
        owners[7] = pick(
            "derived", "Money", f"{owners[5]} * {roles['rate']}", group(5, 7)
        )

        def fact(role, words, dtype="Boolean", entity="Person"):
            found = [
                n
                for n, i in inputs.items()
                if n not in entered.values()
                and n not in input_roles.values()
                and _fact_role(i, role, conditions.residence.return_page)
                and i.get("dtype") == dtype
                and i.get("entity") == entity
                and "unit" not in i
            ]
            if len(found) != 1:
                raise ValueError("Ambiguous factual role")
            input_roles[role] = found[0]
            return found[0]

        use = fact("use", {"completing", "schedule"})
        paper = fact("paper", {"filing", "paper", "return"})
        space = fact("space", {"need", "additional", "space"})
        eligible = fact("eligible", {"has", "eligible", "expense"})
        res = fact("residence", {"expense", "relate", "return", "residence"})
        spouse = fact("relationship", {"has", "spouse", "partner", "reference", "date"})
        separate = fact(
            "separate",
            {"occupied", "separate", "principal", "residence", "reference", "date"},
        )
        medical = fact("medical", {"separation", "medical", "reason"})
        day = fact("date", {"medical", "separation", "reference", "date"}, "Date")
        gst = fact("gst", {"gst", "hst", "number", "applicable"}, entity="SourceEntry")
        raw = form.source

        def paragraph(start):
            return raw[raw.index(start) :].split("\n\n", 1)[0]

        receipt = paragraph("Attach a copy of this schedule to your paper return.")
        if (
            re.sub(r"\s+", " ", receipt)
            != "Attach a copy of this schedule to your paper return. Do not include your receipts, but keep them in case the Canada Revenue Agency asks to see them later."
        ):
            return None

        def obligation(role, formula, excerpt, words, entity="Person"):
            roles[role] = pick("derived", "Judgment", formula, excerpt, entity, words)

        obligation(
            "completion",
            eligible,
            "Complete this schedule if you have eligible home renovation expenses.",
            {"schedule", "completion", "required"},
        )
        obligation(
            "paper",
            f"{use} and {paper}",
            receipt,
            {"paper", "return", "schedule", "attachment", "required"},
        )
        obligation("retain", use, receipt, {"receipt", "retention", "required"})
        obligation(
            "exclude", use, receipt, {"receipt", "exclusion", "submission", "required"}
        )
        obligation(
            "sheet",
            space,
            "If you need more space, attach an additional sheet of paper.",
            {"sheet", "attachment", "required"},
        )
        residence = raw[slice(*conditions.residence.span)]
        obligation(
            "residence_yes",
            res,
            residence,
            {"residence", "yes", "checkbox", "required"},
        )
        obligation(
            "residence_no",
            f"not {res}",
            residence,
            {"residence", "address", "reporting", "required"},
        )
        med = raw[slice(*conditions.medical.span)]
        if conditions.medical.reference_date.isoformat() != f"{form.year}-12-31":
            return None
        obligation(
            "medical",
            f"{spouse} and {separate} and {medical} and {day} == period_end",
            med,
            {"medical", "separation", "permission", "applie"},
        )
        obligation(
            "spouse_address",
            roles["medical"],
            med,
            {"spouse", "residence", "address", "reporting", "required"},
        )
        obligation(
            "gst",
            gst,
            "Supplier or contractor - GST/HST Number (if applicable) ^",
            {"supplier", "gst", "hst", "number", "reporting", "required"},
            "SourceEntry",
        )
        if used != set(rules) or set(entered.values()) | set(
            input_roles.values()
        ) != set(inputs):
            return None
        return ScheduleBinding(
            form,
            conditions,
            hashlib.sha256(content.encode()).hexdigest(),
            namespace,
            tuple(sorted(owners.items())),
            tuple(sorted(roles.items())),
            tuple(sorted(input_roles.items())),
            tuple(
                (n, _expression(r["versions"][0]["formula"])) for n, r in rules.items()
            ),
            tuple(sorted(entered.items())),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        RecursionError,
    ):
        return None
