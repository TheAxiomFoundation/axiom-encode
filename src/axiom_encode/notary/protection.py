"""Fail-closed checks of the control-plane protections required by v33.

Read these from GitHub immediately before issuing authority. Repository files
and workflow inputs cannot assert that these protections are enabled.
"""

from __future__ import annotations

from .identity import IdentityRefusal


def _rules(ruleset: dict, *, repository: str, ref: str, bypass: list[dict]) -> dict:
    if (
        ruleset.get("source_type") != "Repository"
        or ruleset.get("source") != repository
        or ruleset.get("target") != "branch"
        or ruleset.get("enforcement") != "active"
        or ruleset.get("conditions") != {"ref_name": {"include": [ref], "exclude": []}}
        or ruleset.get("bypass_actors") != bypass
        or not isinstance(ruleset.get("rules"), list)
    ):
        raise IdentityRefusal("ruleset_identity_or_bypass")
    rules = {}
    for rule in ruleset["rules"]:
        if (
            not isinstance(rule, dict)
            or not isinstance(rule.get("type"), str)
            or rule["type"] in rules
        ):
            raise IdentityRefusal("ruleset_malformed")
        rules[rule["type"]] = rule.get("parameters", {})
    return rules


def require_chain_protection(
    *,
    repository: str,
    chain_app_id: int,
    writer_ruleset: dict,
    integrity_ruleset: dict,
) -> None:
    # Two rulesets are necessary. Giving the App a bypass in a combined ruleset
    # would also let that App bypass the no-force/no-delete/linear requirements.
    if (
        type(chain_app_id) is not int
        or chain_app_id <= 0
        or writer_ruleset.get("id") == integrity_ruleset.get("id")
    ):
        raise IdentityRefusal("chain_protection_identity")
    writer = _rules(
        writer_ruleset,
        repository=repository,
        ref="refs/heads/chain",
        bypass=[
            {
                "actor_id": chain_app_id,
                "actor_type": "Integration",
                "bypass_mode": "always",
            }
        ],
    )
    integrity = _rules(
        integrity_ruleset, repository=repository, ref="refs/heads/chain", bypass=[]
    )
    if (
        not {"creation", "update"} <= writer.keys()
        or not {"deletion", "non_fast_forward", "required_linear_history"}
        <= integrity.keys()
    ):
        raise IdentityRefusal("chain_protection_missing")
    if (
        not isinstance(writer["update"], dict)
        or writer["update"].get("update_allows_fetch_and_merge") is not False
    ):
        raise IdentityRefusal("chain_upstream_update_exception")


def require_lane_protection(
    ruleset: dict,
    *,
    repository: str,
    ref: str,
    lane_app_id: int,
    check_name: str,
) -> None:
    rules = _rules(ruleset, repository=repository, ref=ref, bypass=[])
    status = rules.get("required_status_checks", {})
    if (
        type(lane_app_id) is not int
        or lane_app_id <= 0
        or not check_name
        or "pull_request" not in rules
        or not {"deletion", "non_fast_forward"} <= rules.keys()
        or not isinstance(status, dict)
        or status.get("strict_required_status_checks_policy") is not True
        or not isinstance(status.get("required_status_checks"), list)
        or {"context": check_name, "integration_id": lane_app_id}
        not in status["required_status_checks"]
    ):
        raise IdentityRefusal("lane_admission_protection_missing")


def require_environment(
    body: dict,
    branches: dict,
    *,
    name: str,
    protected_branch: str,
    reviewer_ids: frozenset[int],
) -> None:
    if (
        body.get("name") != name
        or body.get("can_admins_bypass") is not False
        or body.get("deployment_branch_policy")
        != {"protected_branches": False, "custom_branch_policies": True}
        or not isinstance(body.get("protection_rules"), list)
        or not isinstance(branches.get("branch_policies"), list)
    ):
        raise IdentityRefusal("environment_identity_or_bypass")
    policies = branches["branch_policies"]
    if (
        len(policies) != 1
        or policies[0].get("name") != protected_branch
        or policies[0].get("type") != "branch"
        or branches.get("total_count") != 1
    ):
        raise IdentityRefusal("environment_branch_policy")
    rules = [
        r for r in body["protection_rules"] if r.get("type") == "required_reviewers"
    ]
    if len(rules) != 1 or rules[0].get("prevent_self_review") is not True:
        raise IdentityRefusal("environment_review_policy")
    reviewers = rules[0].get("reviewers")
    if not isinstance(reviewers, list) or not reviewers:
        raise IdentityRefusal("environment_reviewers")
    actual = []
    for reviewer in reviewers:
        if reviewer.get("type") != "User" or not isinstance(
            reviewer.get("reviewer"), dict
        ):
            raise IdentityRefusal("environment_reviewers")
        identity = reviewer["reviewer"].get("id")
        if type(identity) is not int or identity not in reviewer_ids:
            raise IdentityRefusal("environment_reviewers")
        actual.append(identity)
    if set(actual) != set(reviewer_ids) or len(actual) != len(set(actual)):
        raise IdentityRefusal("environment_reviewers")


def require_bootstrap_lock(
    ruleset: dict,
    *,
    repository: str,
    ref: str,
    bootstrap_actor_id: int,
) -> None:
    # The ceremony supplies a dedicated one-person GitHub team as the sole
    # bypass actor. Membership must be audited separately against the custodian.
    rules = _rules(
        ruleset,
        repository=repository,
        ref=ref,
        bypass=[
            {
                "actor_id": bootstrap_actor_id,
                "actor_type": "Team",
                "bypass_mode": "pull_request",
            }
        ],
    )
    if (
        not isinstance(rules.get("update"), dict)
        or rules["update"].get("update_allows_fetch_and_merge") is not False
    ):
        raise IdentityRefusal("bootstrap_lane_not_locked")
