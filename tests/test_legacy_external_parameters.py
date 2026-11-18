import copy

import pytest

from axiom_encode.legacy_external_parameters import (
    carry_forward_external_proved_rules,
    external_parameter_preservation_issues,
)


def _admitted_rules():
    return [
        {"name": "legacy_scalar", "kind": "parameter", "versions": [{"formula": "30"}]},
        {
            "name": "legacy_applies",
            "kind": "condition",
            "versions": [{"formula": "true"}],
        },
    ]


def test_carry_forward_restores_only_admitted_rules():
    admitted = _admitted_rules()
    payload = {
        "rules": [
            {"name": "new_rule", "kind": "condition"},
            {
                "name": "legacy_scalar",
                "kind": "parameter",
                "versions": [{"formula": "60"}],
            },
        ]
    }
    restored = carry_forward_external_proved_rules(
        payload, {"required_unchanged_rules": admitted}
    )
    assert restored == ["legacy_scalar", "legacy_applies"]
    assert payload["rules"][0] == {"name": "new_rule", "kind": "condition"}
    assert payload["rules"][1:] == admitted
    assert payload["rules"][1] is not admitted[0]
    assert (
        external_parameter_preservation_issues(
            payload, {"required_unchanged_rules": admitted}
        )
        == []
    )


def test_carry_forward_leaves_matching_rules_unchanged():
    admitted = _admitted_rules()
    payload = {"rules": copy.deepcopy(admitted)}
    original_rules = payload["rules"]
    assert (
        carry_forward_external_proved_rules(
            payload, {"required_unchanged_rules": admitted}
        )
        == []
    )
    assert payload["rules"] is original_rules


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"rules": [{"name": "same"}, {"name": "same"}]},
        {"rules": [{"kind": "condition"}]},
    ],
)
def test_carry_forward_rejects_malformed_candidates(payload):
    with pytest.raises(ValueError):
        carry_forward_external_proved_rules(
            payload, {"required_unchanged_rules": _admitted_rules()}
        )
