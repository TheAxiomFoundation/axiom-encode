"""Prior-report reference frames; static descriptors are not runtime authority."""

from __future__ import annotations

from dataclasses import dataclass

from axiom_encode.printed_source_rows import recognize_printed_rows


@dataclass(frozen=True)
class PriorFormRole:
    span: tuple[int, int]
    part: str
    principal: str
    copy_input: str
    annual_inputs: tuple[str, ...]
    selector: str = "t936_completed_2024"
    reference_year: int = 2024


_SPANS = {5792: "a", 5836: "a", 6189: "b", 6234: "b"}


def prior_form_reference_binding(
    policy: dict, text: str, citation: str, case: dict, span: tuple[int, int]
) -> PriorFormRole | None:
    """Bind both complete copy and entered-historical-total routes.

    This certifies a declaration contract, never the accuracy or authenticity
    of earlier tax returns. Current-year tax rules do not recompute old totals.
    """
    from . import source_completeness as s
    from .local_pair_source_evidence import matches_expected_formula

    try:
        if (
            type(span) is not tuple
            or len(span) != 2
            or any(type(v) is not int for v in span)
            or span[0] not in _SPANS
            or span[1] != span[0] + 4
        ):
            return None
        if case.get("period") != {
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        }:
            return None
        if (
            recognize_printed_rows(text, allow_multipage=True) is None
            or text[span[0] : span[1]] != "2024"
        ):
            return None
        part = _SPANS[span[0]]
        line = 16 if part == "a" else 19
        row_span, note_span = (
            ((5745, 5861), (6541, 6817))
            if part == "a"
            else ((6142, 6259), (6817, 7090))
        )
        row, note = (text[a:b].strip() for a, b in (row_span, note_span))
        expected_row = (
            "Amount from line 16 in Part 3 of Form T936 for 2024 (if you did not complete Form T936 for 2024, see note (1)) + 15"
            if part == "a"
            else "Amount from line 19 of Part 3 of Form T936 for 2024. (if you did not complete Form T936 for 2024, see note (2)) + 20"
        )
        expected_note = (
            "(1) To calculate your total investment expenses from previous years, complete Part 1 of Form T936 for each year from 1988 to 2024 in which you had investment expenses (do not complete line 7 for 1988 to 1991). Add the amounts from line A and enter the total on line 15 above."
            if part == "a"
            else "(2) To calculate your total investment income from previous years, complete Part 2 of Form T936 for each year from 1988 to 2024 in which you had investment income (do not complete line 13 for 1988 to 1991). Add the amounts from line B and enter the total on line 20 above."
        )
        if (
            " ".join(row.split()) != expected_row
            or " ".join(note.split()) != expected_note
        ):
            return None
        rules, inputs = policy["rules"], policy["inputs"]
        declarations = [*rules, *inputs]
        names = [r["name"] for r in declarations]
        if len(names) != len(set(names)):
            return None
        by_name = {r["name"]: r for r in declarations}
        selector = "t936_completed_2024"
        copy_input = f"t936_2024_part3_line_{line}"
        annual_inputs = tuple(
            f"t936_completed_{year}_part_{part}" for year in range(1988, 2025)
        )
        descriptions = {
            selector: (
                "Boolean",
                "A completed2024 T936 is available for the source-stated copying route.",
            ),
            copy_input: (
                "Money",
                f"Exact literal prior2024 Part3 line{line} copied as the source directs; no silent line correction.",
            ),
        }
        for year, name in zip(range(1988, 2025), annual_inputs, strict=True):
            description = f"Explicit completed{year} Part{part.upper()} total. Zero must be supplied when no amount; no default. Historical law is not recomputed using2025 rules."
            if year <= 1991:
                description += " Source note omits line7/13 for1988..1991."
            descriptions[name] = ("Money", description)
        for name, (dtype, description) in descriptions.items():
            r = by_name[name]
            if (
                r not in inputs
                or any(
                    k in r
                    for k in (
                        "formula",
                        "indexed_by",
                        "default",
                        "rounding",
                        "versions",
                        "values",
                    )
                )
                or r.get("dtype") != dtype
                or r.get("entity") != "Person"
                or r.get("period") != "Year"
                or r.get("description") != description
                or r.get("unit") != ("CAD" if dtype == "Money" else None)
            ):
                return None
        principal = f"t936_previous_{part}"
        r = by_name[principal]
        if (
            r not in rules
            or any(
                r.get(k) != v
                for k, v in {
                    "kind": "derived",
                    "dtype": "Money",
                    "unit": "CAD",
                    "entity": "Person",
                    "period": "Year",
                }.items()
            )
            or any(k in r for k in ("rounding", "default", "indexed_by"))
        ):
            return None
        index = s._selected_rule_formula_version_index(r, case)
        if index is None:
            return None
        v = r["versions"][index]
        if (v.get("effective_from"), v.get("effective_to")) != (
            "2025-01-01",
            "2025-12-31",
        ):
            return None
        expected = f"if {selector}:\n  {copy_input}\nelse:\n  " + " + ".join(
            annual_inputs
        )
        if not matches_expected_formula(v.get("formula"), expected):
            return None
        atoms = set(s._rule_source_excerpt_atoms(r))
        required = [
            ("formula", row),
            ("formula", note),
            ("effective_from", text.splitlines()[0]),
            ("effective_to", text.splitlines()[0]),
        ]
        if not all(
            any(
                path == f"versions[{index}].{field}"
                and cite == citation
                and excerpt.strip() == owned
                for path, cite, excerpt in atoms
            )
            for field, owned in required
        ):
            return None
        return PriorFormRole(span, part, principal, copy_input, annual_inputs)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return None
