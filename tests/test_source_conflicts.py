"""The source-conflict compare step (``axiom-encode compare-sources``)."""

from __future__ import annotations

import copy
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from axiom_encode import source_conflicts
from axiom_encode.source_conflicts import cli as compare_cli
from axiom_encode.source_conflicts.artifact import BOOL, NUMERIC, Artifact
from axiom_encode.source_conflicts.compare import compare, render_record
from axiom_encode.source_conflicts.engine import (
    Engine,
    EngineError,
    pinned_engine_commit,
)
from axiom_encode.source_conflicts.expressions import ExpressionError, parse_constraint
from axiom_encode.source_conflicts.gates import LinkGateError, check_link
from axiom_encode.source_conflicts.link import (
    _CONSTRAINT_KEYS,
    _DERIVED_KEYS,
    _FACT_KEYS,
    _FIXED_KEYS,
    _PERIOD_KEYS,
    _SAMPLING_KEYS,
    _SIDE_KEYS,
    _TOP_KEYS,
    LinkError,
    load_link,
    parse_link,
)
from axiom_encode.source_conflicts.values import (
    combination,
    generate_values,
    grid_size,
    select_indices,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "source_conflicts"
RULESPEC = FIXTURES / "rulespec-us"
LINK = FIXTURES / "work_requirement.link.yaml"
EXAMPLE = (
    REPO
    / "docs"
    / "source-conflicts"
    / "examples"
    / "us-medicaid-community-engagement.link.json"
)
A = "us:statutes/999/1/a#"
B = "us:regulations/99-cfr/1/2#"
REPOSITORY = {"commit": "0" * 40, "dirty": False}


def document() -> dict:
    return yaml.safe_load(LINK.read_text())


def artifact_payload(side: str) -> dict:
    return json.loads((FIXTURES / "artifacts" / f"{side}.json").read_text())


class FakeEngine:
    """Evaluates the checked-in fixture artifacts as the engine would."""

    commit = "f" * 40
    _OPS = {
        "gte": lambda a, b: a >= b,
        "gt": lambda a, b: a > b,
        "lte": lambda a, b: a <= b,
        "lt": lambda a, b: a < b,
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
    }

    def __init__(self, payloads: dict[str, dict] | None = None):
        self.payloads = payloads or {
            "us/statutes/999/1/a.yaml": artifact_payload("a"),
            "us/regulations/99-cfr/1/2.yaml": artifact_payload("b"),
        }
        self.compiled: dict[Path, dict] = {}
        self.requests: list[dict] = []

    def compile(self, module: Path, rulespec_root: Path, output: Path) -> dict:
        payload = self.payloads[module.relative_to(rulespec_root).as_posix()]
        self.compiled[output] = payload
        return payload

    def _value(self, program, node, inputs):
        kind = node["kind"]
        if kind == "literal":
            value = node["value"]["value"]
            return value if isinstance(value, bool) else Decimal(str(value))
        if kind == "input":
            return inputs[node["name"]]
        if kind == "parameter_lookup":
            parameter = next(
                p for p in program["parameters"] if p["name"] == node["parameter"]
            )
            return Decimal(str(parameter["versions"][0]["values"]["0"]["value"]))
        if kind == "derived":
            rule = next(d for d in program["derived"] if d["name"] == node["name"])
            return self._value(program, rule["expr"], inputs)
        if kind in ("and", "or"):
            values = [self._value(program, item, inputs) for item in node["items"]]
            return all(values) if kind == "and" else any(values)
        if kind == "not":
            return not self._value(program, node["item"], inputs)
        if kind == "add":
            return sum(
                (self._value(program, item, inputs) for item in node["items"]),
                Decimal(0),
            )
        if kind == "if":
            branch = (
                "then_expr"
                if self._value(program, node["condition"], inputs)
                else "else_expr"
            )
            return self._value(program, node[branch], inputs)
        left = self._value(program, node["left"], inputs)
        right = self._value(program, node["right"], inputs)
        if kind == "comparison":
            return self._OPS[node["op"]](left, right)
        return {"mul": left * right, "div": left / right if right else Decimal(0)}[kind]

    def run(self, artifact: Path, request: dict) -> dict:
        self.requests.append(request)
        payload = self.compiled[artifact]
        program = payload["program"]
        slots = {
            entry["canonical_request_name"]: entry["slot"]
            for entry in payload["metadata"]["input_catalog"]
        }
        by_entity: dict[str, dict] = {}
        for item in request["dataset"]["inputs"]:
            value = item["value"]["value"]
            by_entity.setdefault(item["entity_id"], {})[slots[item["name"]]] = (
                value if isinstance(value, bool) else Decimal(str(value))
            )
        results = []
        for query in request["queries"]:
            outputs = {}
            for name in query["outputs"]:
                rule = next(d for d in program["derived"] if d["id"] == name)
                holds = self._value(
                    program, rule["expr"], by_entity[query["entity_id"]]
                )
                outputs[name] = {
                    "kind": "judgment",
                    "outcome": "holds" if holds else "not_holds",
                }
            results.append({"entity_id": query["entity_id"], "outputs": outputs})
        return {"results": results}


def run_compare(
    doc: dict | None = None, tmp_path: Path | None = None, engine=None
) -> dict:
    link = parse_link(doc if doc is not None else document())
    return compare(
        link,
        rulespec_root=RULESPEC,
        engine=engine or FakeEngine(),
        workdir=tmp_path or Path("/nonexistent"),
        repository=REPOSITORY,
    )


def artifacts() -> dict[str, Artifact]:
    return {side: Artifact(artifact_payload(side), "2027-01-01") for side in ("a", "b")}


# -- link loader ------------------------------------------------------------


def test_fixture_and_example_links_load():
    link = load_link(LINK)
    assert link.id == "us:fixture-work-requirement-statute-vs-regulation"
    assert link.sides["b"].guards == (B + "individual_in_scope",)
    assert link.facts["work_hours"].values is None
    assert link.derived[A + "input.monthly_total_hours"].op == "sum"
    for path in (EXAMPLE, *sorted(FIXTURES.glob("hr1_*.link.json"))):
        assert load_link(path).sides["a"].module == "us/statutes/42/1396a/xx.yaml"


def test_link_hash_ignores_file_format_and_number_spelling(tmp_path):
    doc = document()
    as_json = tmp_path / "link.json"
    as_json.write_text(json.dumps(doc))
    assert load_link(as_json).sha256 == load_link(LINK).sha256
    doc["fixed"][B + "input.minimum_wage_rate"]["value"] = 7.25
    assert parse_link(doc).sha256 == load_link(LINK).sha256
    doc["fixed"][B + "input.minimum_wage_rate"]["value"] = "7.5"
    assert parse_link(doc).sha256 != load_link(LINK).sha256


def mutate(path: list, value=None, *, delete: bool = False) -> dict:
    doc = document()
    target = doc
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return doc


@pytest.mark.parametrize(
    "doc,message",
    [
        (mutate(["format"], "v0"), "link.format must be"),
        (mutate(["id"], "No Jurisdiction"), "link.id must be a stable lowercase id"),
        (
            mutate(["id"], "fixture-without-prefix"),
            "link.id must be a stable lowercase id",
        ),
        (mutate(["b"], delete=True), "link lacks the required key 'b'"),
        (mutate(["a", "combine"], "first"), "a.combine must be one of any, all"),
        (mutate(["a", "module"], "../outside.yaml"), "a.module must be a relative"),
        (mutate(["a", "module"], "/abs/a.yaml"), "a.module must be a relative"),
        (mutate(["a", "outputs"], []), "a.outputs must be a non-empty list"),
        (mutate(["a", "extra"], 1), "a has the unknown key 'extra'"),
        (
            mutate(["facts", "work_hours", "kind"], "money"),
            "facts.work_hours.kind must be",
        ),
        (mutate(["facts", "work_hours", "b"], delete=True), "must map to an input"),
        (
            mutate(["facts", "work_hours", "values"], ["0", True]),
            "must all be decimal values",
        ),
        (mutate(["facts", "work_hours", "values"], ["1", "1.0"]), "repeats a value"),
        (mutate(["facts", "work_hours", "step"], "0"), "step must be a positive"),
        (
            mutate(["facts", "Work-Hours"], {"kind": "bool", "a": "x"}),
            "a fact name must match",
        ),
        (
            mutate(["facts", "agency_counts_income_as_hours", "min"], 0),
            "min does not apply to a bool fact",
        ),
        (
            mutate(["derived", A + "input.monthly_total_hours", "op"], "product"),
            "op must be one of sum",
        ),
        (
            mutate(
                ["derived", A + "input.monthly_total_hours", "of"],
                ["work_hours", "missing"],
            ),
            "names 'missing', which is not a fact",
        ),
        (
            mutate(
                ["derived", A + "input.monthly_total_hours", "of"],
                ["work_hours", "agency_counts_income_as_hours"],
            ),
            "names the bool fact",
        ),
        (
            mutate(["fixed", B + "input.minimum_wage_rate", "value"], True),
            "value must be a decimal value",
        ),
        (
            mutate(
                ["fixed", B + "input.documented_hours"],
                {"kind": "decimal", "value": "1"},
            ),
            "is supplied by both facts.work_hours and fixed",
        ),
        (
            mutate(
                ["facts", "other_hours"],
                {"kind": "decimal", "b": B + "input.documented_hours"},
            ),
            "is mapped by both facts.work_hours and facts.other_hours",
        ),
        (
            mutate(["constraints", 0, "rule"], "monthly_income >= wage * 7.25"),
            "'wage', which is not a fact",
        ),
        (
            mutate(["period", "end"], "2027-01-01"),
            "period.end must be after period.start",
        ),
        (
            mutate(["period", "start"], "January"),
            "period.start must be a YYYY-MM-DD date",
        ),
        (mutate(["guards", "c"], ["x"]), "guards has the unknown key 'c'"),
        (
            mutate(["sampling"], {"max_combinations": 0}),
            "max_combinations must be an integer from 1 to",
        ),
        (
            mutate(["sampling"], {"max_combinations": 10**7}),
            "max_combinations must be an integer from 1 to",
        ),
        (
            mutate(["period", "start"], "٢٠٢٧-01-01"),
            "period.start must be a YYYY-MM-DD date",
        ),
        (
            mutate(["fixed", B + "input.minimum_wage_rate", "value"], "٧"),
            "value must be a decimal value",
        ),
        (mutate(["sampling"], {"seed": -1}), "seed must be a non-negative integer"),
    ],
)
def test_invalid_links_are_refused_with_a_clear_message(doc, message):
    with pytest.raises(
        LinkError, match=message.replace("(", r"\(").replace(")", r"\)")
    ):
        parse_link(doc)


def test_loader_reports_every_problem_at_once():
    doc = mutate(["id"], "bad")
    doc["a"]["combine"] = "first"
    del doc["entity"]
    with pytest.raises(LinkError) as caught:
        parse_link(doc)
    assert len(caught.value.problems) >= 3


def test_unreadable_link_files_are_link_errors(tmp_path):
    with pytest.raises(LinkError, match="cannot read"):
        load_link(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(LinkError, match="cannot parse"):
        load_link(bad)
    other = tmp_path / "link.txt"
    other.write_text("{}")
    with pytest.raises(LinkError, match="must end in"):
        load_link(other)


def test_schema_lists_the_keys_the_loader_accepts():
    schema = json.loads(
        (Path(source_conflicts.__file__).parent / "link.schema.json").read_text()
    )
    defs = schema["$defs"]
    pairs = [
        (schema, _TOP_KEYS),
        (defs["side"], _SIDE_KEYS),
        (defs["fact"], _FACT_KEYS),
        (defs["derived"], _DERIVED_KEYS),
        (defs["fixed"], _FIXED_KEYS),
        (defs["constraint"], _CONSTRAINT_KEYS),
        (schema["properties"]["period"], _PERIOD_KEYS),
        (schema["properties"]["sampling"], _SAMPLING_KEYS),
    ]
    for node, keys in pairs:
        assert set(node["properties"]) == set(keys)
        assert set(node.get("required", [])) == {
            key for key, needed in keys.items() if needed
        }
        assert node["additionalProperties"] is False
    assert schema["properties"]["format"]["const"] == source_conflicts.LINK_FORMAT


# -- constraint expressions -------------------------------------------------

KINDS = {
    "income": "decimal",
    "hours": "decimal",
    "count": "integer",
    "flag": "bool",
    "other": "bool",
}


@pytest.mark.parametrize(
    "text,case,expected",
    [
        ("income >= hours * 7.25", {"income": "580", "hours": "80"}, True),
        ("income >= hours * 7.25", {"income": "579.99", "hours": "80"}, False),
        ("income >= 7.25 * hours", {"income": "0", "hours": "0"}, True),
        ("income - hours * 2 + 1 > 0", {"income": "3", "hours": "2"}, False),
        ("-income + 10 < hours", {"income": "3", "hours": "8"}, True),
        ("income + income == 2 * income", {"income": "3"}, True),
        ("hours + count <= 744", {"hours": "700.5", "count": 44}, False),
        ("count != 3", {"count": 3}, False),
        ("flag == true", {"flag": True}, True),
        ("flag != other", {"flag": True, "other": True}, False),
        ("false == flag", {"flag": False}, True),
    ],
)
def test_constraints_evaluate_linear_comparisons(text, case, expected):
    assert parse_constraint(text, KINDS).holds(case) is expected


@pytest.mark.parametrize(
    "text,message",
    [
        ("income * hours >= 1", "multiplies two facts"),
        ("income >= wage", "'wage', which is not a fact"),
        ("income >= 1 >= hours", "exactly one comparison"),
        ("income + 1", "exactly one comparison"),
        ("__import__('os').system('true') == 1", "unsupported text"),
        ("income >= hours ** 2", "lacks a number or a fact"),
        ("income >= (hours)", "unsupported text"),
        ("income / 2 >= 1", "unsupported text"),
        ("flag >= true", "orders bool values"),
        ("flag == 1", "mixes bool and numeric"),
        ("income + flag >= 1", "mixes bool and numeric"),
        ("income == true", "mixes bool and numeric"),
        ("1 >= 0", "names no fact"),
        ("income >= ", "is empty"),
        ("income hours >= 1", "lacks '+' or '-'"),
        ("income * 2 * 3 >= 1", "more than two operands"),
        ("", "non-empty string"),
    ],
)
def test_constraints_outside_the_expression_form_are_refused(text, message):
    with pytest.raises(
        ExpressionError,
        match=message.replace("(", r"\(").replace("+", r"\+").replace("*", r"\*"),
    ):
        parse_constraint(text, KINDS)


# -- artifact view ----------------------------------------------------------


def test_artifact_closure_kinds_and_thresholds():
    b = artifacts()["b"]
    derived, parameters, slots = b.closure([B + "hours_path_met"])
    assert derived == [
        "hours_path_met",
        "counted_work_hours",
        "monthly_income_threshold",
    ]
    assert parameters == ["required_monthly_hours"]
    assert set(slots) == {
        "individual_is_covered",
        "service_hours",
        "agency_counts_income_as_hours",
        "income_for_month",
        "minimum_wage_rate",
        "documented_hours",
    }
    classes = b.input_classes(derived)
    assert classes["individual_is_covered"] == {BOOL}
    assert classes["agency_counts_income_as_hours"] == {BOOL}
    assert classes["income_for_month"] == {NUMERIC}
    assert classes["service_hours"] == {NUMERIC}

    fixed = {"minimum_wage_rate": "7.25", "individual_is_covered": True}
    found = {(t.value, t.rule, t.inputs) for t in b.thresholds(derived, fixed)}
    # 580 is no literal of the module: it folds from the fixed wage times the 80-hour parameter.
    assert (
        Decimal("580.00"),
        B + "counted_work_hours",
        frozenset({"income_for_month"}),
    ) in found
    assert (
        Decimal(80),
        B + "hours_path_met",
        frozenset(
            {
                "service_hours",
                "agency_counts_income_as_hours",
                "income_for_month",
                "documented_hours",
            }
        ),
    ) in found
    # Without the fixed wage the income threshold is not a constant.
    assert {t.value for t in b.thresholds(derived, {})} == {Decimal(80)}


def test_artifact_reads_the_version_in_force_at_the_period_start():
    payload = artifact_payload("a")
    parameter = payload["program"]["parameters"][0]
    parameter["versions"] = [
        {
            "effective_from": "0001-01-01",
            "effective_to": "2026-12-31",
            "values": {"0": {"kind": "integer", "value": 80}},
        },
        {
            "effective_from": "2027-01-01",
            "values": {"0": {"kind": "integer", "value": 100}},
        },
    ]
    for start, expected in (("2026-06-01", 80), ("2027-01-01", 100)):
        artifact = Artifact(payload, start)
        derived, _, _ = artifact.closure([A + "work_requirement_met"])
        assert [t.value for t in artifact.thresholds(derived, {})] == [
            Decimal(expected)
        ]


# -- gates ------------------------------------------------------------------


def gate_problems(doc: dict) -> list[str]:
    with pytest.raises(LinkGateError) as caught:
        check_link(parse_link(doc), artifacts())
    return caught.value.problems


def test_fixture_link_passes_the_gates_without_warnings():
    assert check_link(load_link(LINK), artifacts()) == []


def test_gate_names_must_resolve_in_the_compiled_artifacts():
    doc = document()
    doc["a"]["outputs"] = [A + "no_such_rule"]
    doc["guards"]["b"] = [B + "no_such_guard"]
    doc["facts"]["work_hours"]["b"] = B + "input.no_such_input"
    doc["fixed"]["us:nowhere#input.x"] = {"kind": "bool", "value": True}
    problems = gate_problems(doc)
    assert any(
        "a.outputs" in p and "no_such_rule" in p and "not a derived rule" in p
        for p in problems
    )
    assert any("b.guards" in p and "no_such_guard" in p for p in problems)
    assert any("facts.work_hours.b" in p and "not an input" in p for p in problems)
    assert any(
        "fixed.us:nowhere#input.x: not an input of either" in p for p in problems
    )


def test_gate_every_input_the_rules_read_needs_a_supplier():
    doc = document()
    del doc["fixed"][B + "input.minimum_wage_rate"]
    del doc["derived"]
    problems = gate_problems(doc)
    assert problems == [
        f"side a: input '{A}input.monthly_total_hours' is read by the compared rules "
        "but no fact, derived input or fixed input supplies it",
        f"side b: input '{B}input.minimum_wage_rate' is read by the compared rules "
        "but no fact, derived input or fixed input supplies it",
    ]


def test_gate_kinds_must_match_how_the_rules_use_an_input():
    doc = document()
    doc["facts"]["agency_counts_income_as_hours"] = {
        "kind": "decimal",
        "b": B + "input.agency_counts_income_as_hours",
    }
    doc["fixed"][B + "input.minimum_wage_rate"] = {"kind": "bool", "value": True}
    problems = gate_problems(doc)
    assert any(
        "facts.agency_counts_income_as_hours: kind decimal does not match side b" in p
        for p in problems
    )
    assert any(
        f"fixed.{B}input.minimum_wage_rate: kind bool does not match side b" in p
        for p in problems
    )


def test_gate_outputs_must_be_yes_no_rules_of_the_link_entity_and_period():
    doc = document()
    doc["b"]["outputs"] = [B + "counted_work_hours"]
    assert any("dtype decimal" in p for p in gate_problems(doc))
    doc = document()
    doc["entity"] = "Household"
    assert any("rule of entity Person" in p for p in gate_problems(doc))
    doc = document()
    doc["period"]["period_kind"] = "tax_year"
    assert any("has period Month" in p for p in gate_problems(doc))


def test_gate_warns_about_supplied_inputs_no_compared_rule_reads():
    doc = document()
    doc["b"]["outputs"] = [B + "individual_in_scope"]
    warnings = check_link(parse_link(doc), artifacts())
    assert (
        f"side b: input '{B}input.documented_hours' (facts) is not read by the compared rules"
        in warnings
    )


def test_gate_missing_module_file(tmp_path):
    doc = document()
    doc["a"]["module"] = "us/statutes/999/1/absent.yaml"
    with pytest.raises(
        LinkGateError, match="a.module: us/statutes/999/1/absent.yaml is not a file"
    ):
        run_compare(doc, tmp_path)


@pytest.mark.parametrize(
    "rules,message",
    [
        (
            ["monthly_income >= work_hours * 7.25", "monthly_income > 1000"],
            r"constraints\[1\] \(monthly_income > 1000\) removes every one of the 224",
        ),
        (["work_hours < 1", "work_hours > 1"], "together remove every one"),
    ],
)
def test_gate_a_constraint_that_removes_every_combination_fails(
    tmp_path, rules, message
):
    doc = document()
    doc["constraints"] = [{"rule": rule} for rule in rules]
    with pytest.raises(LinkGateError, match=message):
        run_compare(doc, tmp_path)


# -- values and sampling ----------------------------------------------------


def fact(**fields):
    doc = document()
    doc["facts"]["probe"] = {"kind": "decimal", "b": B + "input.x", **fields}
    return parse_link(doc).facts["probe"]


def test_generated_values_are_zero_and_each_threshold_with_its_neighbours():
    thresholds = [Decimal(80), Decimal("580.00"), Decimal(80)]
    assert generate_values(fact(), thresholds) == [
        "0",
        "79",
        "80",
        "81",
        "579",
        "580",
        "581",
    ]
    assert generate_values(fact(step="0.01"), [Decimal(580)]) == [
        "0",
        "579.99",
        "580",
        "580.01",
    ]
    assert generate_values(fact(max=80), thresholds) == ["0", "79", "80"]
    assert generate_values(fact(min=-1), [Decimal(0)]) == ["-1", "0", "1"]
    assert generate_values(fact(), [Decimal(0)]) == ["0", "1"]
    assert generate_values(fact(), []) == ["0"]
    assert generate_values(fact(kind="integer"), [Decimal(19), Decimal("52.4")]) == [
        0,
        18,
        19,
        20,
        52,
        53,
    ]
    doc = document()
    assert generate_values(
        parse_link(doc).facts["agency_counts_income_as_hours"], [Decimal(1)]
    ) == [False, True]


def test_grid_order_matches_the_listed_facts_with_the_last_fastest():
    lists = [["a", "b"], [1, 2, 3], [False, True]]
    assert grid_size(lists) == 12
    assert [combination(i, lists) for i in (0, 1, 2, 11)] == [
        ("a", 1, False),
        ("a", 1, True),
        ("a", 2, False),
        ("b", 3, True),
    ]


def test_sampling_is_seeded_sorted_and_only_used_above_the_cap():
    assert select_indices(12, 12, 0) == (list(range(12)), False)
    first, sampled = select_indices(10**12, 500, 7)
    assert sampled and len(first) == 500 and first == sorted(set(first))
    assert first[0] == 0 and first[-1] == 10**12 - 1
    assert select_indices(10**12, 500, 7) == (first, True)
    assert select_indices(10**12, 500, 8)[0] != first
    # Pinned: the sample must not move between Python versions or releases.
    assert select_indices(1000, 5, 0)[0] == [0, 99, 122, 980, 999]
    assert select_indices(1000, 1, 0) == ([0], True)


# -- compare ----------------------------------------------------------------


def test_compare_finds_the_conflict_from_generated_values(tmp_path):
    record = run_compare(tmp_path=tmp_path)
    assert record["format"] == "axiom-source-conflict-record/v1"
    assert record["id"] == "us:fixture-work-requirement-statute-vs-regulation"
    assert record["review_state"] == "unreviewed"
    assert record["conflict"] is True
    assert record["link"] == {"sha256": load_link(LINK).sha256}
    assert record["rulespec"] == REPOSITORY
    assert record["engine"] == {"commit": "f" * 40, "version": "0.2.2"}
    assert record["sampled"] is False
    assert record["warnings"] == []
    assert record["counts"] == {
        "grid": 224,
        "combinations": 224,
        "removed_by_constraints": 128,
        "removed_that_differ": 13,
        "run": 96,
        "skipped_by_guard": 0,
        "differ": 4,
        "witness_groups": 1,
    }
    facts = record["facts"]
    assert facts["work_hours"]["values"] == ["0", "79", "80", "81"]
    assert facts["work_hours"]["values_from"] == "modules"
    assert facts["work_hours"]["read_by"] == ["a", "b"]
    assert facts["monthly_income"]["values"] == [
        "0",
        "79",
        "80",
        "81",
        "579",
        "580",
        "581",
    ]
    assert facts["monthly_income"]["read_by"] == ["b"]
    assert {"value": "580", "side": "b", "rule": B + "counted_work_hours"} in facts[
        "monthly_income"
    ]["thresholds"]
    assert facts["agency_counts_income_as_hours"]["values"] == [False, True]
    assert "thresholds" not in facts["agency_counts_income_as_hours"]

    (group,) = record["witness_groups"]
    assert group["direction"] == {"a": False, "b": True}
    assert group["deciding_rules"] == {
        "a": [A + "work_requirement_met"],
        "b": [B + "hours_path_met"],
    }
    assert group["one_side_facts"] == {
        "monthly_income": "nonzero",
        "agency_counts_income_as_hours": True,
    }
    assert group["count"] == 4
    assert group["witness"]["facts"] == {
        "work_hours": "0",
        "service_hours": "79",
        "monthly_income": "79",
        "agency_counts_income_as_hours": True,
    }
    assert group["witness"]["a"] == {
        "result": False,
        "outputs": {A + "work_requirement_met": False},
    }
    assert group["witness"]["b"] == {
        "result": True,
        "outputs": {B + "hours_path_met": True},
    }
    assert group["fact_values"]["monthly_income"] == ["79", "80", "81", "579"]


def test_record_carries_the_proof_of_the_deciding_rules_on_both_sides(tmp_path):
    record = run_compare(tmp_path=tmp_path)
    a_rules, b_rules = record["sides"]["a"]["rules"], record["sides"]["b"]["rules"]
    assert list(a_rules) == [
        A + "monthly_hours_requirement",
        A + "work_requirement_met",
    ]
    assert a_rules[A + "work_requirement_met"] == {
        "kind": "derived",
        "source": "Fixture statute (a)",
        "corpus_citation_path": None,
        "proof_atoms": [
            {
                "kind": "formula",
                "corpus_citation_path": "us/statute/999/1/a",
                "excerpt": "works not less than 80 hours in the month",
            }
        ],
        "depends_on": [A + "monthly_hours_requirement"],
    }
    assert b_rules[B + "counted_work_hours"]["proof_atoms"][0]["excerpt"] == (
        "may divide the monthly income by the minimum wage"
    )
    assert b_rules[B + "counted_work_hours"]["depends_on"] == [
        B + "monthly_income_threshold"
    ]
    assert b_rules[B + "required_monthly_hours"]["kind"] == "parameter"
    # The guard is no deciding rule, so its rules are not in the proof.
    assert B + "individual_in_scope" not in b_rules
    assert len(record["sides"]["a"]["module_sha256"]) == 64
    assert record["assumptions"] == [
        {
            "where": f"derived.{A}input.monthly_total_hours",
            "note": "The statute takes total hours as an input; the link defines it as work plus service hours.",
        },
        {
            "where": "constraints[0]",
            "note": "Work hours are paid at least the minimum wage.",
        },
    ]


def test_the_step_never_sets_a_review_state_other_than_unreviewed(tmp_path):
    assert source_conflicts.REVIEW_STATES == (
        "unreviewed",
        "source_conflict",
        "encoding_error",
        "refinement_or_option",
    )
    doc = document()
    doc["facts"]["agency_counts_income_as_hours"]["values"] = [False]
    for candidate in (document(), doc):
        assert run_compare(candidate, tmp_path)["review_state"] == "unreviewed"


def test_no_difference_gives_a_record_without_witnesses(tmp_path):
    doc = document()
    doc["facts"]["agency_counts_income_as_hours"]["values"] = [False]
    record = run_compare(doc, tmp_path)
    assert record["conflict"] is False
    assert record["witness_groups"] == []
    assert record["counts"]["differ"] == 0
    assert record["sides"]["b"]["rules"] == {}
    assert record["facts"]["agency_counts_income_as_hours"]["values_from"] == "link"


def test_constraint_report_counts_what_each_constraint_removes(tmp_path):
    record = run_compare(tmp_path=tmp_path)
    assert record["constraints"] == [
        {
            "rule": "monthly_income >= work_hours * 7.25",
            "note": "Work hours are paid at least the minimum wage.",
            "removed": 128,
            "removed_by_this_alone": 128,
            "removed_that_differ": 13,
        }
    ]
    doc = document()
    doc["constraints"].append({"rule": "service_hours <= 80"})
    record = run_compare(doc, tmp_path)
    first, second = record["constraints"]
    assert (first["removed"], first["removed_by_this_alone"]) == (128, 96)
    assert (second["removed"], second["removed_by_this_alone"]) == (56, 24)
    assert record["counts"]["removed_by_constraints"] == 152
    assert record["counts"]["run"] == 72


def test_without_the_constraint_impossible_households_become_witnesses(tmp_path):
    doc = document()
    del doc["constraints"]
    record = run_compare(doc, tmp_path)
    assert record["counts"]["removed_by_constraints"] == 0
    assert record["counts"]["differ"] == 17
    directions = [group["direction"] for group in record["witness_groups"]]
    # 80 work hours with no income: the statute says met, the regulation counts $0 / $7.25.
    assert {"a": True, "b": False} in directions and {
        "a": False,
        "b": True,
    } in directions
    weights = [
        sum(
            1
            for value in group["witness"]["facts"].values()
            if value is not False and value != "0"
        )
        for group in record["witness_groups"]
    ]
    assert weights == sorted(weights)


def test_witnesses_group_by_direction_deciding_rules_and_one_side_facts(tmp_path):
    doc = document()
    del doc["constraints"]
    doc["b"]["outputs"].append(B + "individual_in_scope")
    doc["b"]["combine"] = "all"
    record = run_compare(doc, tmp_path)
    keys = [
        (
            group["direction"]["a"],
            group["direction"]["b"],
            tuple(group["deciding_rules"]["b"]),
            tuple(group["one_side_facts"].items()),
        )
        for group in record["witness_groups"]
    ]
    assert len(keys) == len(set(keys)) > 1
    assert {tuple(group["one_side_facts"]) for group in record["witness_groups"]} == {
        ("monthly_income", "agency_counts_income_as_hours")
    }
    assert {
        group["one_side_facts"]["monthly_income"] for group in record["witness_groups"]
    } == {"zero", "nonzero"}
    assert (
        sum(group["count"] for group in record["witness_groups"])
        == record["counts"]["differ"]
    )
    for group in record["witness_groups"]:
        # combine all: a false side is decided by the outputs that do not hold.
        if not group["direction"]["b"]:
            assert group["deciding_rules"]["b"] == [B + "hours_path_met"]
        else:
            assert group["deciding_rules"]["b"] == doc["b"]["outputs"]


def test_households_that_fail_a_guard_are_skipped(tmp_path):
    doc = document()
    del doc["fixed"][B + "input.individual_is_covered"]
    doc["facts"]["covered"] = {"kind": "bool", "b": B + "input.individual_is_covered"}
    record = run_compare(doc, tmp_path)
    assert record["counts"]["run"] == 192
    assert record["counts"]["skipped_by_guard"] == 96
    assert record["counts"]["differ"] == 4
    assert all(
        group["witness"]["facts"]["covered"] for group in record["witness_groups"]
    )


def test_a_large_grid_is_sampled_and_the_record_says_so(tmp_path):
    doc = document()
    doc["sampling"] = {"max_combinations": 50, "seed": 3}
    record = run_compare(doc, tmp_path)
    assert record["sampled"] is True
    assert record["sampling"] == {
        "method": "sha256-counter/v1",
        "max_combinations": 50,
        "seed": 3,
    }
    assert record["counts"]["grid"] == 224
    assert record["counts"]["combinations"] == 50
    assert record["counts"]["removed_by_constraints"] + record["counts"]["run"] == 50
    assert render_record(run_compare(doc, tmp_path)) == render_record(record)
    doc["sampling"]["seed"] = 4
    assert run_compare(doc, tmp_path)["counts"] != record["counts"]


def test_the_same_inputs_give_byte_identical_records(tmp_path):
    first = render_record(run_compare(tmp_path=tmp_path))
    second = render_record(run_compare(tmp_path=tmp_path / "other"))
    assert first == second
    assert first.endswith("}\n")
    assert json.loads(first)["review_state"] == "unreviewed"
    # The link's key order and file format do not change the record.
    doc = document()
    reordered = {key: doc[key] for key in reversed(list(doc))}
    assert render_record(run_compare(reordered, tmp_path)) == first


def test_requests_send_each_side_only_its_own_inputs(tmp_path):
    engine = FakeEngine()
    run_compare(tmp_path=tmp_path, engine=engine)
    a_request, b_request = engine.requests
    assert a_request["mode"] == "fast"
    assert {item["name"] for item in a_request["dataset"]["inputs"]} == {
        A + "input.monthly_total_hours",
        A + "input.subject_to_requirement",
    }
    assert len(b_request["queries"]) == 224
    query = b_request["queries"][0]
    assert query["period"] == {
        "period_kind": "month",
        "start": "2027-01-01",
        "end": "2027-02-01",
    }
    assert query["outputs"] == [B + "hours_path_met", B + "individual_in_scope"]
    totals = {
        item["entity_id"]: item["value"]
        for item in a_request["dataset"]["inputs"]
        if item["name"] == A + "input.monthly_total_hours"
    }
    # case:86 is work 79, service 0, ...; the derived input is their sum.
    assert totals["case:0"] == {"kind": "decimal", "value": "0"}
    assert totals["case:223"] == {"kind": "decimal", "value": "162"}


# -- engine pin and CLI -----------------------------------------------------


def test_engine_pin_comes_from_the_workflow_toolchain_file(tmp_path):
    with pytest.raises(EngineError, match="pins no engine commit"):
        pinned_engine_commit(tmp_path)
    (tmp_path / ".axiom").mkdir()
    toolchain = tmp_path / ".axiom" / "workflow-toolchain.toml"
    toolchain.write_text(
        '[workflow_toolchain]\naxiom_rules_engine_ref = "' + "a" * 40 + '"\n'
    )
    assert pinned_engine_commit(tmp_path) == (
        "a" * 40,
        ".axiom/workflow-toolchain.toml",
    )
    toolchain.write_text('[workflow_toolchain]\naxiom_rules_engine_ref = "main"\n')
    with pytest.raises(EngineError, match="full lowercase 40-hex"):
        pinned_engine_commit(tmp_path)


@pytest.fixture
def cli_engine(monkeypatch):
    bound = {}

    def bind(checkout, commit):
        bound.update(checkout=checkout, commit=commit)
        return Path("/nonexistent/axiom-rules-engine")

    monkeypatch.setattr(compare_cli, "bind_engine", bind)
    monkeypatch.setattr(compare_cli, "Engine", lambda binary, commit: FakeEngine())
    monkeypatch.setattr(
        compare_cli, "pinned_engine_commit", lambda root: ("c" * 40, "toolchain")
    )
    return bound


def cli_args(tmp_path, link=LINK):
    return [
        "--link",
        str(link),
        "--rulespec-root",
        str(RULESPEC),
        "--axiom-rules-engine-path",
        str(tmp_path),
        "--out",
        str(tmp_path / "out" / "record.json"),
    ]


def test_cli_writes_the_record_and_exits_zero_on_a_conflict(
    tmp_path, capsys, cli_engine
):
    assert compare_cli.run(cli_args(tmp_path)) == 0
    assert cli_engine["commit"] == "c" * 40
    record = json.loads((tmp_path / "out" / "record.json").read_text())
    assert record["conflict"] is True and record["review_state"] == "unreviewed"
    out = capsys.readouterr().out
    assert "224 combinations, 128 removed by constraints, 96 run" in out
    assert "4 differ, 1 witness groups" in out
    assert (
        compare_cli.run([*cli_args(tmp_path), "--axiom-rules-engine-ref", "d" * 40])
        == 0
    )
    assert cli_engine["commit"] == "d" * 40


def test_cli_exits_zero_without_a_conflict(tmp_path, cli_engine):
    doc = document()
    doc["facts"]["agency_counts_income_as_hours"]["values"] = [False]
    link = tmp_path / "link.json"
    link.write_text(json.dumps(doc))
    assert compare_cli.run(cli_args(tmp_path, link)) == 0
    assert (
        json.loads((tmp_path / "out" / "record.json").read_text())["conflict"] is False
    )


def test_cli_exits_one_on_an_invalid_link_or_a_gate_failure(
    tmp_path, capsys, cli_engine
):
    doc = document()
    doc["id"] = "bad"
    link = tmp_path / "link.json"
    link.write_text(json.dumps(doc))
    assert compare_cli.run(cli_args(tmp_path, link)) == 1
    assert "link.id must be" in capsys.readouterr().err

    doc = document()
    doc["a"]["outputs"] = [A + "no_such_rule"]
    link.write_text(json.dumps(doc))
    assert compare_cli.run(cli_args(tmp_path, link)) == 1
    assert "the link failed its gates" in capsys.readouterr().err
    assert not (tmp_path / "out" / "record.json").exists()


def test_cli_exits_three_when_the_engine_cannot_be_bound(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        compare_cli, "pinned_engine_commit", lambda root: ("c" * 40, "toolchain")
    )
    assert compare_cli.run(cli_args(tmp_path)) == 3
    assert "engine-bind" in capsys.readouterr().err


def test_entrypoint_dispatches_compare_sources(tmp_path, monkeypatch, cli_engine):
    from axiom_encode import entrypoint

    monkeypatch.setattr(
        "sys.argv", ["axiom-encode", "compare-sources", *cli_args(tmp_path)]
    )
    assert entrypoint.main() == 0
    assert (tmp_path / "out" / "record.json").is_file()


# -- the real engine --------------------------------------------------------


def test_real_engine_gives_the_record_the_fixture_artifacts_give(real_engine, tmp_path):
    checkout = tmp_path / "rulespec-us"
    shutil.copytree(RULESPEC, checkout)
    link = load_link(LINK)
    workdir = tmp_path / "work"
    workdir.mkdir()
    record = compare(
        link,
        rulespec_root=checkout,
        engine=Engine(real_engine.binary, "f" * 40),
        workdir=workdir,
        repository=REPOSITORY,
    )
    expected = run_compare(tmp_path=tmp_path)
    assert record["counts"]["differ"] == 4
    for volatile in (record, expected):
        volatile["engine"]["version"] = None
    assert copy.deepcopy(record) == expected


# -- findings of the review -------------------------------------------------


def test_a_key_that_appears_twice_is_refused(tmp_path):
    as_yaml = tmp_path / "link.yaml"
    as_yaml.write_text(LINK.read_text() + "constraints: []\n")
    with pytest.raises(LinkError, match="the key 'constraints' appears twice"):
        load_link(as_yaml)
    nested = tmp_path / "nested.yaml"
    text = LINK.read_text()
    assert "  service_hours:\n    kind: decimal\n" in text
    nested.write_text(
        text.replace(
            "  service_hours:\n    kind: decimal\n",
            "  work_hours:\n    kind: decimal\n",
        )
    )
    with pytest.raises(LinkError, match="the key 'work_hours' appears twice"):
        load_link(nested)
    as_json = tmp_path / "link.json"
    as_json.write_text('{"format": "x", "id": "us:a", "id": "us:b"}')
    with pytest.raises(LinkError, match="the key 'id' appears twice"):
        load_link(as_json)


def test_link_hash_changes_with_the_order_of_the_facts():
    doc = document()
    reordered = copy.deepcopy(doc)
    reordered["facts"] = {
        name: doc["facts"][name] for name in reversed(list(doc["facts"]))
    }
    assert parse_link(reordered).sha256 != parse_link(doc).sha256
    top_level = {key: doc[key] for key in reversed(list(doc))}
    assert parse_link(top_level).sha256 == parse_link(doc).sha256


def test_a_rule_with_no_version_in_force_has_no_formula():
    payload = artifact_payload("a")
    rule = payload["program"]["derived"][0]
    rule["versions"][0]["effective_from"] = "2028-01-01"
    artifact = Artifact(payload, "2027-01-01")
    assert artifact.expression(rule) is None
    assert artifact.without_version(["work_requirement_met"]) == [
        A + "work_requirement_met"
    ]
    with pytest.raises(
        LinkGateError, match="has no formula version in force on 2027-01-01"
    ):
        check_link(parse_link(document()), {"a": artifact, "b": artifacts()["b"]})
    later = Artifact(payload, "2028-01-01")
    assert later.without_version(["work_requirement_met"]) == []
    assert later.expression(rule) == rule["versions"][0]["expr"]


def test_constant_folding_of_a_long_rule_chain_is_linear():
    def number(value):
        return {"kind": "literal", "value": {"kind": "integer", "value": value}}

    payload = artifact_payload("a")
    program = payload["program"]
    program["derived"].append(
        {
            "name": "r0",
            "id": "us:x#r0",
            "expr": {"kind": "mul", "left": number(1), "right": number(7)},
        }
    )
    for index in range(1, 400):
        previous = {"kind": "derived", "name": f"r{index - 1}"}
        program["derived"].append(
            {
                "name": f"r{index}",
                "id": f"us:x#r{index}",
                "expr": {
                    "kind": "sub",
                    "left": {"kind": "mul", "left": previous, "right": number(2)},
                    "right": previous,
                },
            }
        )
    artifact = Artifact(payload, "2027-01-01")
    # Without a memo this takes 2**400 steps.
    assert artifact.fold({"kind": "derived", "name": "r399"}, {}) == Decimal(7)
    unknown = {
        "kind": "sub",
        "left": number(1),
        "right": {"kind": "input", "name": "x"},
    }
    assert artifact.fold(unknown, {}) is None
    assert artifact.fold(unknown, {"x": "3"}) == Decimal(-2)
    by_zero = {"kind": "div", "left": number(1), "right": number(0)}
    assert artifact.fold(by_zero, {}) is None


def test_gate_a_fact_with_no_value_fails(tmp_path):
    doc = document()
    doc["facts"]["monthly_income"].update(min=5, max=6)
    with pytest.raises(LinkGateError, match="facts.monthly_income: no value is left"):
        run_compare(doc, tmp_path)


def test_gate_guards_that_skip_every_household_fail(tmp_path):
    doc = document()
    doc["fixed"][B + "input.individual_is_covered"]["value"] = False
    with pytest.raises(
        LinkGateError, match="the guards skip every one of the 96 households"
    ):
        run_compare(doc, tmp_path)


def test_gate_constraint_message_names_the_sample(tmp_path):
    doc = document()
    doc["sampling"] = {"max_combinations": 1}
    doc["constraints"] = [{"rule": "work_hours >= 1"}]
    with pytest.raises(LinkGateError, match="1 combinations sampled from 224"):
        run_compare(doc, tmp_path)


def test_an_input_no_compared_rule_reads_does_not_make_a_fact_two_sided(tmp_path):
    payload = artifact_payload("a")
    payload["metadata"]["input_catalog"].append(
        {
            "slot": "unused_flag",
            "canonical_request_name": A + "input.unused_flag",
            "request_names": [A + "input.unused_flag"],
        }
    )
    engine = FakeEngine(
        {
            "us/statutes/999/1/a.yaml": payload,
            "us/regulations/99-cfr/1/2.yaml": artifact_payload("b"),
        }
    )
    doc = document()
    doc["facts"]["agency_counts_income_as_hours"]["a"] = A + "input.unused_flag"
    record = run_compare(doc, tmp_path, engine)
    assert record["warnings"] == [
        f"side a: input '{A}input.unused_flag' (facts) is not read by the compared rules"
    ]
    assert record["facts"]["agency_counts_income_as_hours"]["read_by"] == ["b"]
    (group,) = record["witness_groups"]
    assert group["one_side_facts"] == {
        "monthly_income": "nonzero",
        "agency_counts_income_as_hours": True,
    }
