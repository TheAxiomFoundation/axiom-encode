"""Independent provider attribution using historical actual895 records."""

import copy
import json
from pathlib import Path

import pytest

from axiom_encode.harness.capital_gift_execution import ENGINE_PIN, _compiled, _record
from axiom_encode.harness.capital_gift_provider_refusal import observe_provider_refusal
from tests.test_capital_gift_refusals import EXECUTION
from tests.test_capital_gift_refusals import bound as _bound


@pytest.fixture(scope="module")
def bound():
    return _bound.__wrapped__()


DATA = json.loads(
    (Path(__file__).parent / "fixtures/t1170-provider-refusals.json").read_text()
)["records"]
NEGATIVE = [r for r in DATA if r["call"]["returncode"]]
POSITIVE = [r for r in DATA if not r["call"]["returncode"]]


@pytest.mark.parametrize("record", NEGATIVE)
def test_actual_two_provider_positions_are_independently_attributed(bound, record):
    observed = observe_provider_refusal(
        bound, EXECUTION["artifact"], ENGINE_PIN, record["case"], record["call"]
    )
    assert observed.provider.endswith(str(observed.provider_index + 1))
    assert observed.provider != observed.root
    assert observed.principal in record["call"]["stderr"]


@pytest.mark.parametrize("record", POSITIVE)
def test_actual_positive_counterparts_keep_exact_trace_validation(bound, record):
    derived, parameters = _compiled(bound, EXECUTION["artifact"], ENGINE_PIN)
    assert _record(bound, derived, parameters, record["case"], record["call"])


@pytest.mark.parametrize(
    "mutation",
    [
        "second_failure",
        "wrong_kind",
        "reverse_edge",
        "error",
        "missing_fact",
        "false_root",
    ],
)
def test_coherent_or_transport_counterexamples_do_not_become_provider_evidence(
    bound, mutation
):
    original = NEGATIVE[-2] if mutation == "false_root" else NEGATIVE[0]
    case, call = copy.deepcopy(original["case"]), copy.deepcopy(original["call"])
    if mutation == "error":
        call["stderr"] += "provider=forged\n"
    elif mutation == "wrong_kind":
        call["request"]["dataset"]["inputs"][0]["entity"] = "Person"
    elif mutation == "reverse_edge":
        call["request"]["dataset"]["relations"][0]["tuple"].reverse()
    else:
        target = next(k for k in case["input"] if "relation." in k)
        if mutation == "second_failure":
            name = next(
                k for k in case["input"][target][1] if k.endswith("qualified_donee")
            )
            case["input"][target][1][name] = False
            provider = f"{case['entity_id']}-{target}-2"
            next(
                r
                for r in call["request"]["dataset"]["inputs"]
                if r["entity_id"] == provider and r["name"] == name
            )["value"] = {"kind": "bool", "value": False}
        elif mutation == "missing_fact":
            name = next(
                k for k in case["input"][target][1] if k.endswith("qualified_donee")
            )
            del case["input"][target][1][name]
            provider = f"{case['entity_id']}-{target}-2"
            call["request"]["dataset"]["inputs"] = [
                r
                for r in call["request"]["dataset"]["inputs"]
                if not (r["entity_id"] == provider and r["name"] == name)
            ]
        else:
            name = next(
                k
                for k in case["input"]
                if k.endswith("nonprescribed_partnership_exchange_applies")
            )
            case["input"][name] = False
            next(r for r in call["request"]["dataset"]["inputs"] if r["name"] == name)[
                "value"
            ] = {"kind": "bool", "value": False}
    with pytest.raises((ValueError, KeyError)):
        observe_provider_refusal(bound, EXECUTION["artifact"], ENGINE_PIN, case, call)


def test_actual_all_eight_provider_failures_have_exact_identity_pairs(bound):
    from axiom_encode.harness.capital_gift_provider_refusal import (
        provider_refusal_pairs,
    )

    negatives = tuple(
        observe_provider_refusal(
            bound, EXECUTION["artifact"], ENGINE_PIN, r["case"], r["call"]
        )
        for r in NEGATIVE
    )
    pairs = provider_refusal_pairs(
        bound,
        EXECUTION["artifact"],
        ENGINE_PIN,
        [r["case"] for r in POSITIVE],
        [r["call"] for r in POSITIVE],
        negatives,
    )
    assert len(pairs) == 8
    assert {p.provider_index for p in pairs} == {0, 1}
    assert len({p.target for p in pairs}) == 4


def test_each_missing_positive_family_refuses_even_when_other_roots_are_valid(bound):
    from axiom_encode.harness.capital_gift_provider_refusal import (
        provider_refusal_pairs,
    )

    negatives = tuple(
        observe_provider_refusal(
            bound, EXECUTION["artifact"], ENGINE_PIN, r["case"], r["call"]
        )
        for r in NEGATIVE
    )
    for omitted in POSITIVE:
        remaining = [r for r in POSITIVE if r is not omitted]
        with pytest.raises(ValueError, match="same-provider"):
            provider_refusal_pairs(
                bound,
                EXECUTION["artifact"],
                ENGINE_PIN,
                [r["case"] for r in remaining],
                [r["call"] for r in remaining],
                negatives,
            )


def test_private_consumer_revalidates_provider_partitions(tmp_path, monkeypatch):
    """Synthetic custody; the embedded process records are historical actuals."""
    from dataclasses import replace

    from axiom_encode.harness import capital_gift_capture as capture
    from tests import test_capital_gift_execution_capture as fixture
    from tests.test_capital_gift_refusal_capture import mixed, stage_all

    monkeypatch.setattr(
        fixture,
        "CASES",
        copy.deepcopy(fixture.CASES) + [copy.deepcopy(r["case"]) for r in DATA],
    )
    expanded = copy.deepcopy(fixture.ACTUAL)
    expanded["calls"] += [copy.deepcopy(r["call"]) for r in DATA]
    monkeypatch.setattr(fixture, "ACTUAL", expanded)
    cap, cases, calls = mixed.__wrapped__(tmp_path, monkeypatch)
    stage_all(cap, cases, calls)
    observation = capture.observation(cap.evidence(), fixture.STATIC["policy_yaml"])
    assert observation is not None
    assert len(observation.refusals) == 18
    assert len(observation.provider_refusals) == 8
    assert len({p.negative_key for p in observation.provider_pairs}) == 8
    # Forging a stored error after successful staging must not survive the
    # consumer's independent revalidation of the raw provider process record.
    cap._capital_gift_refusals[-1] = replace(
        cap._capital_gift_refusals[-1], stderr="division by zero\n"
    )
    assert capture.observation(cap.evidence(), fixture.STATIC["policy_yaml"]) is None


def test_actual_valid_multiquery_positive_cannot_replace_exact_query_pair(bound):
    import yaml

    from axiom_encode.harness.capital_gift_provider_refusal import (
        provider_refusal_pairs,
    )

    cases = yaml.safe_load(EXECUTION["companion_yaml"])
    case, call = cases[7], EXECUTION["calls"][7]
    derived, parameters = _compiled(bound, EXECUTION["artifact"], ENGINE_PIN)
    positive = _record(bound, derived, parameters, case, call)
    assert len(positive.outputs) > 1
    negative = observe_provider_refusal(
        bound,
        EXECUTION["artifact"],
        ENGINE_PIN,
        NEGATIVE[0]["case"],
        NEGATIVE[0]["call"],
    )
    assert negative.target in {name.split("#")[-1] for name, _ in positive.outputs}
    with pytest.raises(ValueError, match="same-provider"):
        provider_refusal_pairs(
            bound, EXECUTION["artifact"], ENGINE_PIN, [case], [call], [negative]
        )
