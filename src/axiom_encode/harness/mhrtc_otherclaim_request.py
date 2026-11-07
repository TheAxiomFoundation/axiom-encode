"""Auxiliary OtherClaim queries over an unchanged principal-case dataset.

These public request plans confer no source authority. The private observer must
bind each actual execution to the principal case and check its typed result.
"""

import copy
from pathlib import Path

from .mhrtc_request import _PREFIX, MhrtcRequestPlan, reconstruct_mhrtc_request

_BUNDLES = {
    "qualification": (
        ("other_claim_source_eligibility", "judgment", None),
        ("other_selected_for_individual", "judgment", None),
    ),
    "capacity": (("other_allocated_local_capacity", "decimal", "CAD"),),
}


def reconstruct_mhrtc_other_claim_request(
    case: dict,
    compiled_payload: dict,
    *,
    pipe,
    rules_file: Path,
    case_index: int,
    other_claim_id: str,
    witness_kind: str,
) -> MhrtcRequestPlan:
    """Query a relevant other claimant without changing any principal facts.

    Relevance here checks only declared identity associations. Qualification and
    capacity are computed by the engine, never inferred from the request plan.
    """
    if type(witness_kind) is not str or witness_kind not in _BUNDLES:
        raise ValueError("fixed OtherClaim witness kind required")
    if type(other_claim_id) is not str or not other_claim_id.strip():
        raise ValueError("explicit OtherClaim target required")
    principal = reconstruct_mhrtc_request(
        case,
        compiled_payload,
        pipe=pipe,
        rules_file=rules_file,
        case_index=case_index,
    )
    if principal.request is None:
        raise ValueError("OtherClaim witness requires a principal graph case")
    rows = [
        r for r in case["tables"].get("OtherClaim", []) if r["id"] == other_claim_id
    ]
    if len(rows) != 1:
        raise ValueError("exactly one explicit OtherClaim target row required")
    target = rows[0]
    derived = {r["id"]: r for r in compiled_payload["program"]["derived"]}
    root_kind = derived[principal.derived_outputs[0]]["entity"]
    root = case["entity_id"]
    edges = {(e["name"], tuple(e["tuple"])) for e in case["relations"]}
    parents = list(case["tables"].get("Claim", []))
    if root_kind == "Claim":
        root_rows = [r for r in parents if r["id"] == root]
        if len(root_rows) > 1:
            raise ValueError("duplicate principal Claim row")
        combined = {"id": root, **case["input"]}
        for key, value in (root_rows[0] if root_rows else {}).items():
            if key in combined and combined[key] != value:
                raise ValueError("conflicting principal Claim facts")
            combined[key] = value
        parents = [r for r in parents if r["id"] != root] + [combined]
    relevant = False
    for parent in parents:
        if (
            _PREFIX + "relation.candidate_other_claims",
            (other_claim_id, parent["id"]),
        ) not in edges:
            continue
        if root_kind == "Family":
            if (_PREFIX + "relation.claim_records", (parent["id"], root)) not in edges:
                continue
            filer = case["input"].get(_PREFIX + "input.filing_claimant_key")
            if (
                type(filer) is not int
                or parent.get(_PREFIX + "input.claimant_key") != filer
            ):
                continue
        elif parent["id"] != root:
            continue
        required = (
            parent.get(_PREFIX + "input.renovation_key"),
            parent.get(_PREFIX + "input.claimant_key"),
            target.get(_PREFIX + "input.other_renovation_key"),
            target.get(_PREFIX + "input.other_claimant_key"),
        )
        if all(type(value) is int for value in required):
            renovation, claimant, other_renovation, other_claimant = required
            relevant |= renovation == other_renovation and claimant != other_claimant
    if not relevant:
        raise ValueError("OtherClaim target is not associated with this principal")
    outputs = []
    for name, dtype, unit in _BUNDLES[witness_kind]:
        key = _PREFIX + name
        declaration = derived.get(key, {})
        versions = declaration.get("versions")
        if (
            declaration.get("entity") != "OtherClaim"
            or declaration.get("dtype") != dtype
            or declaration.get("unit") != unit
            or declaration.get("period") != "Year"
            or type(versions) is not list
            or len(versions) != 1
            or type(versions[0]) is not dict
            or versions[0].get("effective_from") != "2025-01-01"
            or versions[0].get("effective_to") != "2025-12-31"
        ):
            raise ValueError("OtherClaim witness output differs from its declared role")
        outputs.append(key)
    request = copy.deepcopy(principal.request)
    request["queries"] = [
        {
            "entity_id": other_claim_id,
            "period": copy.deepcopy(principal.request["queries"][0]["period"]),
            "outputs": outputs,
        }
    ]
    return MhrtcRequestPlan(request, tuple(outputs), ())
