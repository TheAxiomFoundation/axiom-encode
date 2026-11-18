from copy import deepcopy

import pytest

from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.protection import (
    require_chain_protection,
    require_environment,
    require_lane_protection,
)

from .lineage_fixtures import LANE


def ruleset(repo, ref, names, *, bypass=(), identity=1):
    return {
        "id": identity,
        "source_type": "Repository",
        "source": repo,
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": list(bypass),
        "conditions": {"ref_name": {"include": [ref], "exclude": []}},
        "rules": [
            {
                "type": name,
                **(
                    {"parameters": {"update_allows_fetch_and_merge": False}}
                    if name == "update"
                    else {}
                ),
            }
            for name in names
        ],
    }


@pytest.fixture
def chain():
    bypass = [{"actor_id": 123, "actor_type": "Integration", "bypass_mode": "always"}]
    return dict(
        repository=LANE + "-notary",
        chain_app_id=123,
        writer_ruleset=ruleset(
            LANE + "-notary", "refs/heads/chain", ["creation", "update"], bypass=bypass
        ),
        integrity_ruleset=ruleset(
            LANE + "-notary",
            "refs/heads/chain",
            ["deletion", "non_fast_forward", "required_linear_history"],
            identity=2,
        ),
    )


def test_two_chain_rulesets_keep_app_subject_to_integrity(chain):
    require_chain_protection(**chain)


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_upstream_fetch_and_merge_exception_refuses(chain, value):
    chain["writer_ruleset"]["rules"][1]["parameters"][
        "update_allows_fetch_and_merge"
    ] = value
    with pytest.raises(IdentityRefusal, match="upstream_update_exception"):
        require_chain_protection(**chain)


@pytest.mark.parametrize(
    "mutation",
    [
        "app-integrity-bypass",
        "admin-writer-bypass",
        "missing-integrity",
        "disabled",
        "wrong-ref",
        "same-rule",
        "missing-update",
    ],
)
def test_chain_controls_fail_closed(chain, mutation):
    if mutation == "app-integrity-bypass":
        chain["integrity_ruleset"]["bypass_actors"] = chain["writer_ruleset"][
            "bypass_actors"
        ]
    elif mutation == "admin-writer-bypass":
        chain["writer_ruleset"]["bypass_actors"].append(
            {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}
        )
    elif mutation == "missing-integrity":
        chain["integrity_ruleset"]["rules"].pop()
    elif mutation == "disabled":
        chain["writer_ruleset"]["enforcement"] = "disabled"
    elif mutation == "wrong-ref":
        chain["integrity_ruleset"]["conditions"]["ref_name"]["include"] = [
            "refs/heads/other"
        ]
    elif mutation == "same-rule":
        chain["integrity_ruleset"]["id"] = 1
    else:
        chain["writer_ruleset"]["rules"].pop()
    with pytest.raises(IdentityRefusal):
        require_chain_protection(**chain)


@pytest.fixture
def lane():
    body = ruleset(
        LANE, "refs/heads/main", ["pull_request", "deletion", "non_fast_forward"]
    )
    body["rules"].append(
        {
            "type": "required_status_checks",
            "parameters": {
                "strict_required_status_checks_policy": True,
                "required_status_checks": [
                    {"context": "Axiom notary admission", "integration_id": 456}
                ],
            },
        }
    )
    return body


@pytest.mark.parametrize(
    "mutation", [None, "unbound-check", "non-strict", "bypass", "no-pull-request"]
)
def test_lane_merge_authority_is_app_bound_and_fresh(lane, mutation):
    if mutation == "unbound-check":
        lane["rules"][-1]["parameters"]["required_status_checks"][0][
            "integration_id"
        ] = None
    elif mutation == "non-strict":
        lane["rules"][-1]["parameters"]["strict_required_status_checks_policy"] = False
    elif mutation == "bypass":
        lane["bypass_actors"] = [
            {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}
        ]
    elif mutation == "no-pull-request":
        lane["rules"].pop(0)
    args = dict(
        repository=LANE,
        ref="refs/heads/main",
        lane_app_id=456,
        check_name="Axiom notary admission",
    )
    if mutation is None:
        require_lane_protection(lane, **args)
    else:
        with pytest.raises(IdentityRefusal):
            require_lane_protection(lane, **args)


def environment():
    return {
        "name": "notary-signing",
        "can_admins_bypass": False,
        "deployment_branch_policy": {
            "protected_branches": False,
            "custom_branch_policies": True,
        },
        "protection_rules": [
            {
                "type": "required_reviewers",
                "prevent_self_review": True,
                "reviewers": [{"type": "User", "reviewer": {"id": 123}}],
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "admin-bypass",
        "self-review",
        "wrong-reviewer",
        "no-reviewer",
        "wrong-env",
        "branch-wildcard",
        "tag",
    ],
)
def test_environment_cannot_silently_degrade(mutation):
    body = deepcopy(environment())
    branches = {
        "total_count": 1,
        "branch_policies": [{"type": "branch", "name": "main"}],
    }
    if mutation == "admin-bypass":
        body["can_admins_bypass"] = True
    elif mutation == "self-review":
        body["protection_rules"][0]["prevent_self_review"] = False
    elif mutation == "wrong-reviewer":
        body["protection_rules"][0]["reviewers"][0]["reviewer"]["id"] = 999
    elif mutation == "no-reviewer":
        body["protection_rules"][0]["reviewers"] = []
    elif mutation == "wrong-env":
        body["name"] = "production-signing"
    elif mutation == "branch-wildcard":
        branches["branch_policies"][0]["name"] = "*"
    elif mutation == "tag":
        branches["branch_policies"][0]["type"] = "tag"
    args = dict(
        name="notary-signing", protected_branch="main", reviewer_ids=frozenset([123])
    )
    if mutation is None:
        require_environment(body, branches, **args)
    else:
        with pytest.raises(IdentityRefusal):
            require_environment(body, branches, **args)


@pytest.mark.parametrize("rebase", [True, None, 0, "false"])
def test_rebase_merge_is_refused_before_admission(rebase):
    from axiom_encode.notary.protection import require_merge_methods

    with pytest.raises(IdentityRefusal, match="unsupported_merge_methods"):
        require_merge_methods(
            {
                "full_name": LANE,
                "allow_rebase_merge": rebase,
                "allow_squash_merge": True,
            },
            lane=LANE,
        )


def test_squash_and_merge_commits_are_supported():
    from axiom_encode.notary.protection import require_merge_methods

    for method in ("allow_squash_merge", "allow_merge_commit"):
        require_merge_methods(
            {"full_name": LANE, "allow_rebase_merge": False, method: True}, lane=LANE
        )
