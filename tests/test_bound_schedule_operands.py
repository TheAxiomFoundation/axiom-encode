from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from axiom_encode.bound_schedule_structure import recognize_bound_schedule_structure
from axiom_encode.harness.bound_schedule_operands import bind_schedule

FIXTURE = Path(__file__).parent / "fixtures/nb_schedule"
SOURCE = (FIXTURE / "source.txt").read_text()
CONTENT = (FIXTURE / "diagnostic.yaml").read_text()
PAYLOAD = yaml.safe_load(CONTENT)
CITATION = PAYLOAD["module"]["source_verification"]["corpus_citation_path"]
ROOT = "ca/policies/cra/t1-2025/nb-s12-seniors-home-renovation-credit.yaml"


def bind(payload=None, source=SOURCE, root=ROOT):
    form = recognize_bound_schedule_structure(source, CITATION)
    if form is None:
        return None
    return bind_schedule(
        form,
        CONTENT if payload is None else yaml.safe_dump(payload),
        {CITATION: source},
        root_relative=root,
    )


def entered(payload):
    return [r for r in payload["rules"] if r["name"].startswith("entered_")]


def test_complete_parent_partition_and_local_namespace():
    binding = bind()
    assert binding is not None
    assert len(binding.row_owners) == 7
    assert len(binding.entered) == 3
    assert len(binding.input_roles) == 10
    assert (
        binding.namespace
        == "ca:policies/cra/t1-2025/nb-s12-seniors-home-renovation-credit"
    )
    assert bind(root="elsewhere/../target.yaml") is None


def test_coherent_entered_expenses_assistance_reversal_is_refused():
    payload = deepcopy(PAYLOAD)
    a, b, _ = entered(payload)
    a["versions"][0]["formula"], b["versions"][0]["formula"] = (
        b["versions"][0]["formula"],
        a["versions"][0]["formula"],
    )
    assert bind(payload) is None


@pytest.mark.parametrize(
    "description",
    [
        "",
        "Government assistance received or receivable included in claimant line1.",
        "Not total eligible expenses.",
    ],
)
def test_entered_fact_description_empty_or_conflicting_refuses(description):
    payload = deepcopy(PAYLOAD)
    payload["inputs"][0]["description"] = description
    assert bind(payload) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_rule",
        "extra_input",
        "duplicate_rule",
        "default_fact",
        "wrong_entity",
        "wrong_date_type",
        "unknown_version",
        "wrong_cap",
        "swap_subtraction",
        "wrong_alias",
        "extra_metadata",
    ],
)
def test_closed_original_declarations_refuse_mutations(mutation):
    p = deepcopy(PAYLOAD)
    if mutation == "extra_rule":
        r = deepcopy(p["rules"][-1])
        r["name"] = "unrelated_copy"
        p["rules"].append(r)
    elif mutation == "extra_input":
        i = deepcopy(p["inputs"][-1])
        i["name"] = "unrelated_fact"
        p["inputs"].append(i)
    elif mutation == "duplicate_rule":
        p["rules"].append(deepcopy(p["rules"][0]))
    elif mutation == "default_fact":
        p["inputs"][0]["default"] = 0
    elif mutation == "wrong_entity":
        p["inputs"][-1]["entity"] = "Person"
    elif mutation == "wrong_date_type":
        next(i for i in p["inputs"] if i["dtype"] == "Date")["dtype"] = "Text"
    elif mutation == "unknown_version":
        p["rules"][0]["versions"].append(deepcopy(p["rules"][0]["versions"][0]))
    elif mutation == "wrong_cap":
        p["rules"][0]["versions"][0]["formula"] = 20000
    elif mutation == "swap_subtraction":
        r = next(r for r in p["rules"] if r["name"].endswith("line_60360"))
        a, b = r["versions"][0]["formula"].split(" - ")
        r["versions"][0]["formula"] = b + " - " + a
    elif mutation == "wrong_alias":
        next(
            r
            for r in p["rules"]
            if r["name"] == "spouse_residence_address_reporting_required"
        )["versions"][0]["formula"] = "additional_sheet_attachment_required"
    elif mutation == "extra_metadata":
        p["rules"][0]["metadata"]["ignored_behavior"] = {"default": 1}
    assert bind(p) is None


@pytest.mark.parametrize(
    "tail",
    ["if income is low", "do not claim", "less assistance", "plus a recovery amount"],
)
def test_operative_row_label_cannot_acquire_ownership(tail):
    source = SOURCE.replace(
        "Home renovation expenses = ^", "Home renovation expenses " + tail + " = ^"
    )
    p = deepcopy(PAYLOAD)
    # Update all exact source proof bytes/hash to ensure rejection is semantic,
    # rather than an incidental stale proof or stale module hash.
    import hashlib

    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    for r in p["rules"]:
        r["source"] = r["source"].replace(
            "Home renovation expenses = ^", "Home renovation expenses " + tail + " = ^"
        )
        for a in r["metadata"]["proof"]["atoms"]:
            a["source"]["excerpt"] = a["source"]["excerpt"].replace(
                "Home renovation expenses = ^",
                "Home renovation expenses " + tail + " = ^",
            )
    assert bind(p, source) is None


@pytest.mark.parametrize("index", range(3, 13))
@pytest.mark.parametrize("defect", ["description", "prefix", "suffix"])
def test_all_factual_roles_reject_opposite_or_unknown_contracts(index, defect):
    p = yaml.safe_load(CONTENT)
    fact = p["inputs"][index]
    old = fact["name"]
    if defect == "description":
        fact["description"] = "The opposite of this premise is true."
    else:
        new = "non_" + old if defect == "prefix" else old + "_except_medical"
        # Rename formulas coherently so rejection is not a missing input.
        content = yaml.safe_dump(p, sort_keys=False).replace(old, new)
        p = yaml.safe_load(content)
    form = recognize_bound_schedule_structure(SOURCE, CITATION)
    assert (
        bind_schedule(form, yaml.safe_dump(p, sort_keys=False), {CITATION: SOURCE})
        is None
    )
