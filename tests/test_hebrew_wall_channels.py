"""Combined wall mutants keep the valid format needed to reach their channels."""

from __future__ import annotations

import copy

from tests.hebrew_wall_channels import KINDS, mutants, mutate_text


def test_combined_channel_mutants_preserve_format_and_reasons():
    payload = {
        "format": "rulespec/v1",
        "module": {
            "name": "sample",
            "deferred_outputs": [{"name": "result", "reason": "Missing section 2."}],
        },
    }
    test_cases = [{"name": "sample", "input": {"amount": "one"}}]
    original = copy.deepcopy((payload, test_cases))
    built = list(mutants(payload, test_cases))
    combined = [
        (label, body, cases) for label, body, cases in built if label.startswith("all:")
    ]
    assert [label for label, _, _ in combined] == [f"all:{kind}" for kind in KINDS]
    for (label, body, cases), kind in zip(combined, KINDS, strict=True):
        assert body["format"] == "rulespec/v1", label
        assert body["module"]["name"] == mutate_text("sample", kind), label
        assert body["module"]["deferred_outputs"][0] == {
            "name": mutate_text("result", kind),
            "reason": "Missing section 2.",
        }, label
        assert cases[0]["input"]["amount"] == mutate_text("one", kind), label
    individual_format = [
        body["format"]
        for label, body, _ in built
        if label.startswith('payload:["format"]:')
    ]
    assert individual_format == [mutate_text("rulespec/v1", KINDS[0])]
    assert (payload, test_cases) == original
