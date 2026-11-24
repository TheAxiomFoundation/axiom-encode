"""Exact source-bound Part C refusal, separate from successful execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

from .bound_investment_adjustment_operands import FLAGS, INPUT_ROWS
from .investment_adjustment_execution import (
    PREFIX,
    _same,
    verify_investment_adjustment_compiled,
)
from .investment_adjustment_records import PERIOD, _number

TARGET = PREFIX + "schedule_n_other_investment_expenses_line_54"
ERROR = (
    f"no `match` arm in `{TARGET}` covers `schedule_n_part_c_input_consistent` = false "
    "(arms: true); add an arm for it or a final `_ =>` arm\n"
)


@dataclass(frozen=True)
class InvestmentAdjustmentRefusal:
    case_sha256: str
    root: str
    inputs: tuple
    output: str = TARGET


def investment_adjustment_refusal_case(case):
    """Validate the raw disjoint case, returning typed facts and exact request."""
    if type(case) is not dict or set(case) != {
        "name",
        "entity_id",
        "period",
        "input",
        "expected_refusal",
    }:
        raise ValueError("closed refusal case required")
    if any(type(case[n]) is not str or not case[n] for n in ("name", "entity_id")):
        raise ValueError("nonempty refusal identities required")
    if not _same(case["period"], PERIOD) or not _same(
        case["expected_refusal"], {"output": TARGET, "reason": "NoMatchingArm"}
    ):
        raise ValueError("exact2025 direct PartC refusal required")
    money = {"schedule_n_" + n for n in INPUT_ROWS.values()} | {
        "schedule_n_resource_deduction_recovery_base"
    }
    if type(case["input"]) is not dict or set(case["input"]) != {
        PREFIX + "input." + n for n in money | set(FLAGS)
    }:
        raise ValueError("complete explicit inputs required")
    root = case["entity_id"]
    inputs, values = [], {}
    for canonical, value in case["input"].items():
        name = canonical.removeprefix(PREFIX + "input.")
        if name in money:
            if type(value) not in (int, float):
                raise ValueError("numeric source input required")
            typed = _number(value)
            wire = (
                {"kind": "integer", "value": value}
                if type(value) is int
                else {"kind": "decimal", "value": str(value)}
            )
        else:
            if type(value) is not bool:
                raise ValueError("explicit Boolean presence required")
            typed = value
            wire = {"kind": "bool", "value": value}
        values[name] = typed
        inputs.append(
            {
                "name": canonical,
                "entity": "Person",
                "entity_id": root,
                "interval": {"start": PERIOD["start"], "end": PERIOD["end"]},
                "value": wire,
            }
        )
    expected_request = {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": [{"entity_id": root, "period": PERIOD, "outputs": [TARGET]}],
    }
    return values, expected_request


def observe_investment_adjustment_refusal(
    bound, content, parent, guide, artifact, engine_pin, case, call
):
    """No other nonzero process result can satisfy the Part C source condition."""
    verify_investment_adjustment_compiled(
        bound, content, parent, guide, artifact, engine_pin
    )
    values, request = investment_adjustment_refusal_case(case)
    # The independently source-bound compiled graph is precisely
    # applies = presence50 OR presence52; consistent = applies OR both zero;
    # principal54 has only the true arm. Recompute this before trusting stderr.
    if (
        values[FLAGS[0]]
        or values[FLAGS[1]]
        or (
            values["schedule_n_" + INPUT_ROWS[50]] == 0
            and values["schedule_n_" + INPUT_ROWS[52]] == 0
        )
    ):
        raise ValueError("independent PartC selector is not false")
    if (
        type(call) is not dict
        or set(call) != {"request", "returncode", "stdout", "stderr"}
        or type(call["returncode"]) is not int
        or call["returncode"] != 1
        or call["stdout"] != ""
        or call["stderr"] != ERROR
        or not _same(call["request"], request)
    ):
        raise ValueError("exact official PartC refusal record required")
    return InvestmentAdjustmentRefusal(
        hashlib.sha256(
            json.dumps(case, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        case["entity_id"],
        tuple(sorted(values.items())),
    )


def pair_investment_adjustment_refusal(positives, refusal):
    """Pair observations only after consumer revalidates their raw records."""
    negative = dict(refusal.inputs)
    pairs = []
    for positive in positives:
        outputs = dict(positive.outputs)
        if TARGET not in outputs or type(outputs[TARGET]) is not Decimal:
            continue
        values = {name: value for entity, name, value in positive.inputs}
        if set(values) != set(negative):
            continue
        changed = {
            name
            for name in values
            if type(values[name]) is not type(negative[name])
            or values[name] != negative[name]
        }
        if len(changed) != 1 or not changed <= set(FLAGS[:2]):
            continue
        flag = next(iter(changed))
        if values[flag] is not True or negative[flag] is not False:
            continue
        if (
            outputs[TARGET]
            != values["schedule_n_" + INPUT_ROWS[50]]
            + values["schedule_n_" + INPUT_ROWS[52]]
        ):
            continue
        pairs.append((positive.case_sha256, refusal.case_sha256, flag))
    if not pairs:
        raise ValueError("source-owned single-presence-fact positive pair required")
    return tuple(pairs)
