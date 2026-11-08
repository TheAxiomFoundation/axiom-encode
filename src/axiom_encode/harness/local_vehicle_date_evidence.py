"""Static vehicle-date source roles. Descriptors confer no execution authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from axiom_encode.printed_source_rows import recognize_printed_rows


@dataclass(frozen=True)
class VehicleDateRole:
    span: tuple[int, int]
    year: int
    subtype: str
    lower: date | None
    upper: date | None
    cap: int
    selector: str
    principal: str = "t936_asset_included_uncollectible_portion"


_ROLES = (
    (3630, 2022, "ordinary", None, "2022-01-01", 30000),
    (3669, 2022, "ordinary", "2022-01-01", "2023-01-01", 34000),
    (3708, 2023, "ordinary", "2023-01-01", "2024-01-01", 36000),
    (3747, 2024, "ordinary", "2024-01-01", "2025-01-01", 37000),
    (3937, 2022, "zepv", "2022-01-01", "2023-01-01", 59000),
    (3979, 2022, "zepv", "2023-01-01", None, 61000),
)


def vehicle_date_role_binding(
    policy: dict, text: str, citation: str, case: dict, span: tuple[int, int]
) -> VehicleDateRole | None:
    """Bind a full ordered selector and propagation path, not a sampled year.

    This closed source profile is deliberately not a generic year exception.
    The caller must separately prove current authenticated execution and the
    exact source-map occurrence before using a returned descriptor.
    """
    from . import source_completeness as s

    try:
        period = case.get("period")
        if period != {
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        }:
            return None
        rows = [r for r in _ROLES if span == (r[0], r[0] + 4)]
        if len(rows) != 1 or recognize_printed_rows(text, allow_multipage=True) is None:
            return None
        start, year, subtype, lower, upper, cap = rows[0]
        if text[start : start + 4] != str(year):
            return None
        header = text.splitlines()[0]
        paragraph_start = text.index("the uncollectible portion of proceeds")
        paragraph_end = text.index("after 2022)", paragraph_start) + len("after 2022)")
        paragraph = text[paragraph_start:paragraph_end]
        split = paragraph.index("zero-emission")
        owner = (
            paragraph[:split].removesuffix("and ").strip()
            if subtype == "ordinary"
            else paragraph[split:]
        )
        rules = policy["rules"]
        inputs = policy["inputs"]
        declarations = [*rules, *inputs]
        names = [d["name"] for d in declarations]
        if len(names) != len(set(names)):
            return None
        by_name = {d["name"]: d for d in declarations}
        env = s._constant_rule_environment(policy)
        env.update(
            s._declared_date_table_environment(
                policy, authoritative_citation_path=citation
            )
        )
        selected = s._formula_environment_for_case(env, case)

        def proves(rule: dict, index: int, field: str, excerpt: str) -> bool:
            return any(
                p == f"versions[{index}].{field}"
                and c == citation
                and e.strip() == excerpt.strip()
                for p, c, e in s._rule_source_excerpt_atoms(rule)
            )

        def version(
            name: str, dtype: str, kind: str, excerpt: str, field: str = "formula"
        ) -> dict | None:
            r = by_name[name]
            if (
                r not in rules
                or r.get("kind") != kind
                or r.get("dtype") != dtype
                or any(k in r for k in ("default", "rounding"))
                or (dtype != "Money" and r.get("unit") is not None)
            ):
                return None
            indexes = (
                s._selected_temporal_version_indexes(env[name], case)
                if field == "values" and name in env
                else ()
            )
            i = (
                (indexes[0] if len(indexes) == 1 else None)
                if field == "values"
                else s._selected_rule_formula_version_index(r, case)
            )
            if i is None:
                return None
            v = r["versions"][i]
            if (v.get("effective_from"), v.get("effective_to")) != (
                "2025-01-01",
                "2025-12-31",
            ):
                return None
            if not (
                proves(r, i, field, excerpt)
                and all(
                    proves(r, i, k, header) for k in ("effective_from", "effective_to")
                )
            ):
                return None
            return v

        descriptions = {
            "t936_asset_acquisition_date": (
                "Date",
                "Actual acquisition calendar date, not disposition or claim date.",
            ),
            "t936_asset_acquisition_cost": (
                "Money",
                "Acquisition cost used solely for stated passenger-vehicle threshold.",
            ),
            "t936_asset_uncollectible_portion": (
                "Money",
                "Entered uncollectible portion of proceeds of this depreciable-property disposition; not computed from an invented recovery formula.",
            ),
            "t936_asset_is_timber_resource": (
                "Boolean",
                "Source classification timber resource.",
            ),
            "t936_asset_is_ordinary_passenger_vehicle": (
                "Boolean",
                "Passenger vehicle other than the separately treated ZEPV class.",
            ),
            "t936_asset_is_zero_emission_passenger_vehicle": (
                "Boolean",
                "Separately stated ZEPV classification; no claim the source defines historical category admission.",
            ),
        }
        for name, (dtype, description) in descriptions.items():
            r = by_name[name]
            if (
                r not in inputs
                or r.get("dtype") != dtype
                or r.get("description") != description
                or r.get("entity") != "SourceEntry"
                or r.get("period") != "Year"
                or any(
                    k in r
                    for k in (
                        "default",
                        "formula",
                        "indexed_by",
                        "rounding",
                        "versions",
                        "values",
                    )
                )
                or (dtype != "Money" and r.get("unit") is not None)
            ):
                return None
            if dtype == "Money" and r.get("unit") != "CAD":
                return None
        if (
            version("t936_source_year", "Integer", "parameter", header) is None
            or type(selected.get("t936_source_year")) is not int
            or selected["t936_source_year"] != 2025
        ):
            return None
        dates = {f"t936_start_{y}": f"{y}-01-01" for y in range(2022, 2027)}
        dates["t936_march_18_2019"] = "2019-03-18"
        for name, expected in dates.items():
            table = selected.get(name)
            if (
                version(name, "Date", "parameter", paragraph, "values") is None
                or type(table) is not s._DeclaredDateTable
                or table.selected_index_value != 2025
                or table.indexed_by != "t936_source_year"
                or table.cells != ((2025, date.fromisoformat(expected)),)
            ):
                return None
        caps = {
            30000: "$30,000 for vehicles acquired prior to 2022",
            34000: "$34,000 for vehicles acquired in 2022",
            36000: "$36,000 for vehicles\nacquired in 2023",
            37000: "$37,000 for vehicles acquired in 2024",
            38000: "$38,000 for vehicles acquired in 2025",
            55000: "$55,000 for ZEPV acquired after March 18, 2019",
            59000: "$59,000 for ZEPV acquired\nin 2022",
            61000: "$61,000 for ZEPV acquired after 2022",
        }
        for value, excerpt in caps.items():
            name = f"t936_vehicle_cost_{value}"
            if (
                excerpt not in paragraph
                or version(name, "Money", "parameter", excerpt) is None
                or by_name[name].get("unit") != "CAD"
                or s._rulespec_runtime_decimal(selected.get(name)) != Decimal(value)
            ):
                return None

        def body(cost: int) -> str:
            return f"  if t936_asset_acquisition_cost > t936_vehicle_cost_{cost}:\n    true\n  else:\n    false"

        ordinary = (
            "\n".join(
                f"{'if' if n == 0 else 'elif'} t936_asset_acquisition_date < t936_start_{y}[t936_source_year]:\n{body(c)}"
                for n, (y, c) in enumerate(
                    zip(
                        range(2022, 2027),
                        (30000, 34000, 36000, 37000, 38000),
                        strict=True,
                    )
                )
            )
            + "\nelse:\n  false"
        )
        zepv = (
            "\n".join(
                f"{'if' if n == 0 else 'elif'} t936_asset_acquisition_date {op} {name}[t936_source_year]:\n{body(c)}"
                for n, (name, op, c) in enumerate(
                    (
                        ("t936_start_2023", ">=", 61000),
                        ("t936_start_2022", ">=", 59000),
                        ("t936_march_18_2019", ">", 55000),
                    )
                )
            )
            + "\nelse:\n  false"
        )
        selector = f"t936_{'ordinary_vehicle' if subtype == 'ordinary' else 'zepv'}_cost_excluded"
        v = version(selector, "Boolean", "derived", owner)
        if v is None or v.get("formula", "").strip() != (
            ordinary if subtype == "ordinary" else zepv
        ):
            return None
        principal = "t936_asset_included_uncollectible_portion"
        pv = version(principal, "Money", "derived", paragraph)
        expected = "if t936_asset_is_timber_resource:\n  0\nelif t936_asset_is_zero_emission_passenger_vehicle and t936_zepv_cost_excluded == true:\n  0\nelif t936_asset_is_ordinary_passenger_vehicle and t936_ordinary_vehicle_cost_excluded == true:\n  0\nelse:\n  t936_asset_uncollectible_portion"
        if (
            pv is None
            or pv.get("formula", "").strip() != expected
            or by_name[principal].get("unit") != "CAD"
        ):
            return None
        if any(
            by_name[n].get("entity") != "SourceEntry"
            or by_name[n].get("period") != "Year"
            for n in (selector, principal)
        ):
            return None
        return VehicleDateRole(
            span,
            year,
            subtype,
            date.fromisoformat(lower) if lower else None,
            date.fromisoformat(upper) if upper else None,
            cap,
            selector,
        )
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        return None
