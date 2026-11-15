"""Synthetic 64-rule test inputs projected from immutable historical records.

The historical 66-rule files retain their original bytes and provenance. This
test-only recipe removes the two unsupported cross-table exports, their expected
outputs, and their recorded query/response entries. Projected compiler/process
records exercise validators; they are not fresh engine execution or capture
authority. Real-engine tests compile and execute the projected source again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

FIXTURES = Path(__file__).parent / "fixtures"
BASE_SHA256 = {
    "t1170-static.json": "d46a1ce4b7fef96942979662ed3ab26ffb7572beee2b5ba1631f260e51883b43",
    "t1170-execution.json": "26b5c90256b488a75068decd4c51df94502a32e50bc5add4766850e5ed24ff78",
}
REMOVED_RULES = frozenset(
    {
        "federal_2025_t1170_total_gain_eligible_zero_inclusion",
        "federal_2025_t1170_total_gain_subject_to_50_percent_inclusion",
    }
)
PREFIX = "ca:policies/cra/t1-2025/capital-gains-on-gifts#"
REMOVED_OUTPUTS = frozenset(PREFIX + name for name in REMOVED_RULES)


def historical(name):
    raw = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA256[name]
    return json.loads(raw)


def projected_static():
    data = historical("t1170-static.json")
    policy = yaml.safe_load(data["policy_yaml"])
    assert {r["name"] for r in policy["rules"]} >= REMOVED_RULES
    assert len(policy["rules"]) == 66
    policy["rules"] = [
        rule for rule in policy["rules"] if rule["name"] not in REMOVED_RULES
    ]
    assert len(policy["rules"]) == 64
    data["policy_yaml"] = yaml.safe_dump(policy, sort_keys=False)
    data["notice"] = __doc__
    return data


def projected_execution():
    data = historical("t1170-execution.json")
    artifact = data["artifact"]
    artifact["program"]["derived"] = [
        rule
        for rule in artifact["program"]["derived"]
        if rule["name"] not in REMOVED_RULES
    ]
    artifact["metadata"]["evaluation_order"] = [
        name
        for name in artifact["metadata"]["evaluation_order"]
        if name not in REMOVED_RULES
    ]
    cases = yaml.safe_load(data["companion_yaml"])
    cases = cases["cases"] if isinstance(cases, dict) else cases
    removed = 0
    for case, call in zip(cases, data["calls"], strict=True):
        outputs = case["output"]
        removed += len(REMOVED_OUTPUTS & outputs.keys())
        case["output"] = {
            name: value
            for name, value in outputs.items()
            if name not in REMOVED_OUTPUTS
        }
        assert case["output"]
        for query in call["request"]["queries"]:
            query["outputs"] = [
                name for name in query["outputs"] if name not in REMOVED_OUTPUTS
            ]
        response = json.loads(call["stdout"])
        for result in response["results"]:
            for field in ("outputs", "trace"):
                result[field] = {
                    name: value
                    for name, value in result[field].items()
                    if name not in REMOVED_OUTPUTS
                }
        call["stdout"] = json.dumps(response)
    assert removed == 8
    data["companion_yaml"] = yaml.safe_dump(cases, sort_keys=False)
    data["notice"] = __doc__
    return data
