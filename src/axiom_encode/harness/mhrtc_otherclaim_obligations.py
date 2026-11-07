"""Enumerate structurally relevant OtherClaim witnesses, not legal eligibility.

The frozen graph supplies associations. Missing facts on a linked candidate are
errors, not permission to skip a witness. Unlinked records are not invented edges.
"""

from .mhrtc_request import _PREFIX, reconstruct_mhrtc_request


def _integer(row, name):
    value = row.get(_PREFIX + "input." + name)
    if type(value) is not int:
        raise ValueError("explicit integer association fact required: " + name)
    return value


def mhrtc_otherclaim_obligations(case, payload, *, pipe, rules_file, case_index):
    """Return target IDs in frozen table order; qualify every returned target."""
    principal = reconstruct_mhrtc_request(
        case, payload, pipe=pipe, rules_file=rules_file, case_index=case_index
    )
    if principal.request is None:
        return ()
    rows = case["tables"].get("OtherClaim", [])
    if not rows:
        return ()
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unique OtherClaim target rows required")
    derived = {row["id"]: row for row in payload["program"]["derived"]}
    root_kind = derived[principal.derived_outputs[0]]["entity"]
    root_id = case["entity_id"]
    edges = {(edge["name"], tuple(edge["tuple"])) for edge in case["relations"]}
    parents = list(case["tables"].get("Claim", []))
    if root_kind == "Claim":
        matches = [row for row in parents if row["id"] == root_id]
        if len(matches) > 1:
            raise ValueError("duplicate root Claim")
        merged = {"id": root_id, **case["input"]}
        for key, value in (matches[0] if matches else {}).items():
            if key in merged and (
                type(merged[key]) is not type(value) or merged[key] != value
            ):
                raise ValueError("conflicting root Claim facts")
            merged[key] = value
        parents = [merged]
    elif root_kind != "Family":
        raise ValueError("supported principal entity required")
    if root_kind == "Family":
        linked = [
            parent["id"]
            for parent in parents
            if (_PREFIX + "relation.claim_records", (parent["id"], root_id)) in edges
            and any(
                (
                    _PREFIX + "relation.candidate_other_claims",
                    (target["id"], parent["id"]),
                )
                in edges
                for target in rows
            )
        ]
        if len(linked) != len(set(linked)):
            raise ValueError("unique linked parent Claim rows required")
    relevant = set()
    for target in rows:
        for parent in parents:
            if (
                _PREFIX + "relation.candidate_other_claims",
                (target["id"], parent["id"]),
            ) not in edges:
                continue
            if root_kind == "Family":
                if (
                    _PREFIX + "relation.claim_records",
                    (parent["id"], root_id),
                ) not in edges:
                    continue
                if _integer(parent, "claimant_key") != _integer(
                    case["input"], "filing_claimant_key"
                ):
                    continue
            claimant = _integer(parent, "claimant_key")
            renovation = _integer(parent, "renovation_key")
            other_claimant = _integer(target, "other_claimant_key")
            other_renovation = _integer(target, "other_renovation_key")
            if claimant != other_claimant and renovation == other_renovation:
                relevant.add(target["id"])
    return tuple(row["id"] for row in rows if row["id"] in relevant)
