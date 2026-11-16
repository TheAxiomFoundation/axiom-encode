"""Pure arithmetic/selection tests; no process authority or source clearance."""

from copy import deepcopy
from decimal import Decimal, localcontext

import pytest

from axiom_encode.harness.actual_local_relation_evidence import (
    _checked_sum,
    _related_ids,
)

FRAME = {"start": "2025-01-01", "end": "2025-12-31"}


def edge(provider, **interval):
    return {
        "name": "relation",
        "tuple": [provider, "root"],
        "interval": interval or FRAME.copy(),
    }


def select(edges):
    return _related_ids(
        edges,
        relation="relation",
        root="root",
        interval=FRAME,
        current_slot=1,
        related_slot=0,
    )


def test_exact_sum_and_empty_set_ignore_ambient_rounding():
    with localcontext() as context:
        context.prec = 1
        assert _checked_sum([0, 70]) == 70
        assert _checked_sum(
            [
                Decimal("70.12345678901234567890123456"),
                Decimal("0.00000000000000000000000001"),
            ]
        ) == Decimal("70.12345678901234567890123457")
        assert _checked_sum([]) == Decimal(0)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        1.0,
        "70",
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("1e-29"),
        2**96,
        Decimal("1e29"),
    ],
)
def test_unsupported_numeric_value_never_rounded_into_credit(value):
    assert _checked_sum([value]) is None


def test_intermediate_overflow_does_not_hide_behind_final_cancellation():
    assert _checked_sum([2**96 - 1, 1, -1]) is None
    assert _checked_sum([2**96 - 1]) == Decimal(2**96 - 1)


def test_full_containment_order_and_nonascii_identity():
    edges = [edge("é"), edge("z"), edge("a")]
    assert select(edges) == ("a", "z", "é")
    assert select(list(reversed(edges))) == ("a", "z", "é")
    assert select([edge("partial", start="2025-02-01", end="2025-12-31")]) == ()
    assert select([edge("wide", start="2024-01-01", end="2026-12-31")]) == ("wide",)
    assert select([]) == ()


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "same_tuple_other_interval",
        "other_relation",
        "missing_interval",
        "bad_date",
        "reverse",
        "extra_slot",
        "empty_id",
    ],
)
def test_unsupported_edges_decline_evidence(change):
    a = edge("provider")
    edges = [a]
    if change == "duplicate":
        edges.append(deepcopy(a))
    elif change == "same_tuple_other_interval":
        edges.append(edge("provider", start="2024-01-01", end="2026-12-31"))
    elif change == "other_relation":
        a["name"] = "other"
    elif change == "missing_interval":
        a.pop("interval")
    elif change == "bad_date":
        a["interval"]["start"] = "2025-02-30"
    elif change == "reverse":
        a["interval"]["end"] = "2024-12-31"
    elif change == "extra_slot":
        a["tuple"].append("third")
    elif change == "empty_id":
        a["tuple"][0] = ""
    assert select(edges) is None


def test_different_root_does_not_contribute():
    a = edge("provider")
    a["tuple"][1] = "unrelated"
    assert select([a]) == ()


def test_ambient_exponent_limits_and_traps_do_not_change_sum():
    from decimal import Inexact, Rounded

    with localcontext() as context:
        context.Emax = 2
        context.Emin = -2
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        assert _checked_sum([1000]) == Decimal(1000)
        assert _checked_sum([Decimal("0.00001")]) == Decimal("0.00001")


def test_oversized_coefficient_declines_without_integer_conversion():
    assert _checked_sum([Decimal("9" * 5000)]) is None


@pytest.mark.parametrize(
    "relation,root",
    [
        (1, "root"),
        ("relation", 1),
        ({}, "root"),
        ("relation", {}),
        ("", "root"),
        ("relation", ""),
        ("relation", "\ud800"),
    ],
)
def test_empty_edge_set_still_requires_valid_identities(relation, root):
    assert (
        _related_ids(
            [],
            relation=relation,
            root=root,
            interval=FRAME,
            current_slot=1,
            related_slot=0,
        )
        is None
    )


def test_surrogate_provider_is_not_a_wire_identity():
    assert select([edge("\ud800")]) is None


def _actual_case():
    import json
    from pathlib import Path

    import yaml

    value = json.loads(
        (Path(__file__).parent / "fixtures/t936_provider_replay.json").read_text()
    )
    return yaml.safe_load(value["policy_yaml"]), value["request"], value["response"]


def test_authored_replay_of_all31_instances_is_not_capture_authority():
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values

    result = _replay_values(*_actual_case())
    assert len(result) == 31
    assert result[("t936_asset_included_uncollectible_portion", "related_0")] == 0
    assert result[("t936_asset_included_uncollectible_portion", "related_1")] == 70
    assert result[("t936_uncollectible_dispositions_total", "case")] == 70
    assert result[("cumulative_net_investment_loss_t936_part_3_line_c", "case")] == 660


@pytest.mark.parametrize(
    "change",
    [
        "swap_providers",
        "missing_provider",
        "wrong_provider_id",
        "wrong_currency",
        "missing_edge",
        "duplicate_edge",
        "partial_interval",
        "wrong_aggregate",
        "wrong_chart",
        "wrong_prior_copy",
    ],
)
def test_same_final_number_cannot_hide_provider_or_root_mutations(change):
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values

    policy, request, response = _actual_case()
    trace = response["results"][0]["trace"]
    provider_keys = [
        k
        for k, n in trace.items()
        if n["name"] == "t936_asset_included_uncollectible_portion"
    ]
    if change == "swap_providers":
        trace[provider_keys[0]]["value"], trace[provider_keys[1]]["value"] = (
            trace[provider_keys[1]]["value"],
            trace[provider_keys[0]]["value"],
        )
    elif change == "missing_provider":
        trace.pop(provider_keys[0])
    elif change == "wrong_provider_id":
        trace[provider_keys[0]]["entity_id"] = "invented"
    elif change == "wrong_currency":
        trace[provider_keys[0]]["unit"] = "USD"
    elif change == "missing_edge":
        request["dataset"]["relations"].pop()
    elif change == "duplicate_edge":
        request["dataset"]["relations"].append(
            deepcopy(request["dataset"]["relations"][0])
        )
    elif change == "partial_interval":
        request["dataset"]["relations"][-1]["interval"]["start"] = "2025-02-01"
    else:
        name = {
            "wrong_aggregate": "t936_uncollectible_dispositions_total",
            "wrong_chart": "federal_2025_t936_chart_a_line_3",
            "wrong_prior_copy": "t936_previous_a",
        }[change]
        node = next(n for n in trace.values() if n["name"] == name)
        node["value"] = {"kind": "decimal", "value": "999"}
    with pytest.raises(ValueError):
        _replay_values(policy, request, response)


@pytest.mark.parametrize(
    "change",
    [
        "cycle",
        "judgment_currency",
        "numeric_input_declared_boolean",
        "missing_query_root",
    ],
)
def test_authored_graph_types_and_query_completeness(change):
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values

    policy, request, response = _actual_case()
    if change == "cycle":
        names = ["t936_should_keep_copy", "t936_should_attach_copy"]
        for index, name in enumerate(names):
            next(r for r in policy["rules"] if r["name"] == name)["versions"][0][
                "formula"
            ] = names[1 - index]
    elif change == "judgment_currency":
        next(
            n
            for n in response["results"][0]["trace"].values()
            if n["kind"] == "judgment"
        )["unit"] = "USD"
    elif change == "numeric_input_declared_boolean":
        next(r for r in policy["inputs"] if r["name"] == "t936_annuity_taxable_56_1_d")[
            "dtype"
        ] = "Boolean"
    else:
        response["results"][0]["trace"].pop(request["queries"][0]["outputs"][0])
    with pytest.raises(ValueError):
        _replay_values(policy, request, response)


@pytest.mark.parametrize("number", [-(1 << 63), (1 << 63) - 1])
def test_engine_integer_wire_bounds(number):
    from axiom_encode.harness.actual_local_relation_evidence import _wire_value

    assert _wire_value({"kind": "integer", "value": number}) == number


@pytest.mark.parametrize("number", [-(1 << 63) - 1, (1 << 63)])
def test_integer_wire_outside_i64_declines(number):
    from axiom_encode.harness.actual_local_relation_evidence import _wire_value

    with pytest.raises(ValueError):
        _wire_value({"kind": "integer", "value": number})


@pytest.mark.parametrize(
    "change", ["omit", "duplicate", "substitute", "provider_to_root", "unreachable"]
)
def test_selected_dependency_edges_are_not_just_equal_values(change):
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values

    policy, request, response = _actual_case()
    trace = response["results"][0]["trace"]
    node = next(
        n
        for n in trace.values()
        if n["name"] == "t936_uncollectible_dispositions_total"
    )
    if change == "omit":
        node["dependencies"].pop()
    elif change == "duplicate":
        node["dependencies"].append(node["dependencies"][0])
    elif change == "substitute":
        node["dependencies"][0] = node["dependencies"][1]
    elif change == "provider_to_root":
        node["dependencies"][0] = node["dependencies"][0].split("@entity:")[0]
    else:
        # Remove an independent requested instruction but retain its trace node.
        key = next(
            k
            for k, n in trace.items()
            if n["name"] == "t936_should_first_complete_chart_a"
        )
        request["queries"][0]["outputs"].remove(key)
    with pytest.raises(ValueError):
        _replay_values(policy, request, response)


def test_local_relation_shape_only_selects_snapshot():
    from axiom_encode.harness.actual_local_relation_evidence import (
        local_relation_snapshot_eligible,
    )

    policy, _, _ = _actual_case()
    assert local_relation_snapshot_eligible(policy)


@pytest.mark.parametrize(
    "change",
    [
        "imports",
        "composition",
        "duplicate",
        "bool_arity",
        "same_entity",
        "foreign_provider",
        "filter",
        "unit",
    ],
)
def test_snapshot_shape_declines_unsupported_profiles(change):
    from axiom_encode.harness.actual_local_relation_evidence import (
        local_relation_snapshot_eligible,
    )

    policy, _, _ = _actual_case()
    relation = next(r for r in policy["rules"] if r.get("kind") == "data_relation")
    aggregate = next(
        r
        for r in policy["rules"]
        if r["name"] == "t936_uncollectible_dispositions_total"
    )
    if change == "imports":
        policy["imports"] = ["ca:other#amount"]
    elif change == "composition":
        policy["module"]["kind"] = "composition"
    elif change == "duplicate":
        policy["rules"].append(deepcopy(aggregate))
    elif change == "bool_arity":
        relation["data_relation"]["arity"] = True
    elif change == "same_entity":
        relation["data_relation"]["arguments"] = ["Person", "Person"]
    elif change == "foreign_provider":
        aggregate["versions"][0]["formula"] = "sum(other.provider)"
    elif change == "filter":
        aggregate["versions"][0]["formula"] = (
            "sum(t936_depreciable_disposition_for_return.t936_asset_included_uncollectible_portion, where=True)"
        )
    else:
        aggregate["unit"] = "USD"
    assert not local_relation_snapshot_eligible(policy)


@pytest.mark.parametrize(
    "change", ["omit", "extra", "index", "value", "version", "source", "duplicate"]
)
def test_parameter_read_lineage_cannot_be_inferred_from_final_amount(change):
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values

    policy, request, response = _actual_case()
    trace = response["results"][0]["trace"]
    node = next(
        n
        for n in trace.values()
        if any(r["name"] == "t936_start_2022" for r in n.get("parameter_reads", []))
    )
    read = next(r for r in node["parameter_reads"] if r["name"] == "t936_start_2022")
    if change == "omit":
        node["parameter_reads"].remove(read)
    elif change == "extra":
        node["parameter_reads"].append({**read, "name": "unread"})
    elif change == "index":
        read["index"] = 0
    elif change == "value":
        read["value"] = {"kind": "date", "value": "2023-01-01"}
    elif change == "version":
        read["effective_from"] = "2024-01-01"
    elif change == "source":
        read["source"] = "different"
    else:
        node["parameter_reads"].append(deepcopy(read))
    with pytest.raises(ValueError):
        _replay_values(policy, request, response)
