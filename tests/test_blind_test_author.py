"""The blind test author: shadow tool, fake model only.

No test here calls a model. The subscription CLI backend is exercised with
its subprocess launcher replaced. Tests that execute cases use the real
rules engine through the ``real_engine`` fixture (tests/real_engine.py).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import blind_test_author as bta
from axiom_encode.harness import validator_pipeline
from scripts import blind_test_author as cli

FIXTURES = Path(__file__).parent / "fixtures" / "blind_test_author"
SOURCE = (FIXTURES / "installation-fee.source.txt").read_text(encoding="utf-8")
MODULE_TEXT = (FIXTURES / "installation-fee.yaml").read_text(encoding="utf-8")
MODULE_RELATIVE = "policies/des/faa5/na-utility-expenses-and-allowances/block-13.yaml"
MODULE = "us-az:policies/des/faa5/na-utility-expenses-and-allowances/block-13"
OUTPUT = "snap_initial_utility_installation_component_allowable"
INPUTS = (
    "component_is_initial_utility_installation_fee",
    "installation_performed_by_utility_provider",
    "component_billed_in_evaluated_month",
    "component_is_one_time_deposit",
)
ALLOWABLE_SENTENCE = (
    "The initial utility installation fee is an allowable NA Utility Expense "
    "in the month the expense is billed when all of the following occur:"
)
DEPOSIT_SENTENCE = "One-time deposits are not an allowable NA Utility Expense."


def _case(name, expected, sentence, period="2026-08", **overrides):
    inputs = {
        INPUTS[0]: True,
        INPUTS[1]: True,
        INPUTS[2]: True,
        INPUTS[3]: False,
    }
    inputs.update(overrides)
    return {
        "name": name,
        "period": period,
        "inputs": inputs,
        "expected": {OUTPUT: expected},
        "source_sentence": sentence,
    }


# What a careful reader of the source text writes, without the formula.
CORRECT_CASES = [
    _case("all_conditions_met", "holds", ALLOWABLE_SENTENCE),
    _case(
        "not_an_initial_installation",
        "not_holds",
        "The utility installation fee is for an initial installation of a utility.",
        **{INPUTS[0]: False},
    ),
    _case(
        "installed_by_someone_else",
        "not_holds",
        "The utility provider performs the installation.",
        **{INPUTS[1]: False},
    ),
    _case("one_time_deposit", "not_holds", DEPOSIT_SENTENCE, **{INPUTS[3]: True}),
]


def _answer(cases, **top):
    return json.dumps({"cases": cases, **top})


def _interface() -> bta.RuleInterface:
    return bta.RuleInterface(
        module=MODULE,
        inputs=tuple(bta.InterfaceInput(name, "Boolean") for name in INPUTS),
        outputs=(bta.InterfaceOutput(OUTPUT, "derived", "Judgment"),),
    )


class _StubRunner:
    def __init__(self, outcome=None):
        self.outcome = outcome or bta.CaseOutcome(bta.STATUS_AGREE)
        self.cases = []

    def run(self, case):
        self.cases.append(case)
        return self.outcome


# ---------------------------------------------------------------------------
# Blindness
# ---------------------------------------------------------------------------

# Every token below is planted where the second model must not look.
PLANTED = {
    "formula": "ZZFORMULATOKEN",
    "formula_number": "73519",
    "parameter_value": "864213",
    "effective_from": "2031-03-17",
    "effective_to": "2032-11-29",
    "summary": "ZZSUMMARYTOKEN",
    "rule_source": "ZZRULESOURCETOKEN",
    "proof_excerpt": "ZZPROOFEXCERPTTOKEN",
    "label": "ZZLABELTOKEN",
    "description": "ZZDESCRIPTIONTOKEN",
    "comment": "ZZCOMMENTTOKEN",
    "test_name": "zz_existing_test_name_token",
    "test_value": "918273",
    "unknown_key": "ZZUNKNOWNKEYTOKEN",
}

PLANTED_MODULE = f"""
format: rulespec/v1
module:
  summary: {PLANTED["summary"]} restates the provision.
  source_verification:
    corpus_citation_path: us-xx/manual/planted/block-1
# {PLANTED["comment"]}
inputs:
  - name: household_gross_income
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    label: {PLANTED["label"]}
    description: {PLANTED["description"]}
    default: {PLANTED["unknown_key"]}
  - name: household_is_elderly
    entity: Household
    dtype: Boolean
    period: Month
rules:
  - name: income_limit
    kind: parameter
    dtype: Money
    period: Month
    unit: USD
    source: {PLANTED["rule_source"]}
    values: [{PLANTED["parameter_value"]}]
    versions:
      - effective_from: '{PLANTED["effective_from"]}'
        effective_to: '{PLANTED["effective_to"]}'
        formula: '{PLANTED["parameter_value"]}'
  - name: household_is_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              excerpt: {PLANTED["proof_excerpt"]}
    versions:
      - effective_from: '{PLANTED["effective_from"]}'
        formula: |-
          household_gross_income <= income_limit + {PLANTED["formula_number"]}
          and not {PLANTED["formula"]}
"""
PLANTED_TESTS = [
    {
        "name": PLANTED["test_name"],
        "period": "2031-04",
        "input": {"us-xx:x#input.household_gross_income": int(PLANTED["test_value"])},
        "output": {"us-xx:x#household_is_eligible": "holds"},
    }
]
PLANTED_SOURCE = (
    "A household is eligible when its gross monthly income does not exceed "
    "the income limit. Elderly households are exempt from the limit."
)


def test_interface_copies_declarations_only():
    interface = bta.extract_rule_interface(
        yaml.safe_load(PLANTED_MODULE), module="us-xx:x"
    )
    assert interface.to_json() == {
        "module": "us-xx:x",
        "inputs": [
            {
                "name": "household_gross_income",
                "dtype": "Money",
                "entity": "Household",
                "period": "Month",
                "unit": "USD",
                "declared": True,
            },
            {
                "name": "household_is_elderly",
                "dtype": "Boolean",
                "entity": "Household",
                "period": "Month",
                "declared": True,
            },
        ],
        "outputs": [
            {
                "name": "income_limit",
                "kind": "parameter",
                "dtype": "Money",
                "period": "Month",
                "unit": "USD",
            },
            {
                "name": "household_is_eligible",
                "kind": "derived",
                "dtype": "Judgment",
                "entity": "Household",
                "period": "Month",
            },
        ],
        "relations": [],
    }


def test_no_formula_value_or_existing_test_reaches_the_second_model():
    payload = yaml.safe_load(PLANTED_MODULE)
    backend = bta.FakeBlindAuthorBackend(_answer([]))
    report = bta.run_blind_test_author(
        rulespec_payload=payload,
        module="us-xx:x",
        source_text=PLANTED_SOURCE,
        backend=backend,
        runner=_StubRunner(),
        companion_cases=PLANTED_TESTS,
        citation_path="us-xx/manual/planted/block-1",
        source_as_of="2031-01-05",
    )
    assert len(backend.prompts) == 1
    prompt = backend.prompts[0]
    for where, token in PLANTED.items():
        assert token not in prompt, f"{where} leaked into the prompt"
    # The interface and the source text are there; that is all the model gets.
    for visible in ("household_gross_income", "income_limit", PLANTED_SOURCE):
        assert visible in prompt
    assert report["shadow"] is True and report["gates"] == "nothing"
    for token in PLANTED.values():
        assert token not in json.dumps(report["interface"])


def test_implicit_inputs_take_names_only_from_the_compiled_program():
    compiled = {
        "program": {
            "derived": [
                {
                    "entity": "Household",
                    "period": "Month",
                    "expr": {
                        "kind": "and",
                        "items": [
                            {
                                "kind": "comparison",
                                "op": "eq",
                                "left": {"kind": "input", "name": "fact_a"},
                                "right": {
                                    "kind": "literal",
                                    "value": {"kind": "bool", "value": True},
                                },
                            },
                            {
                                "kind": "comparison",
                                "op": "gt",
                                "left": {"kind": "input", "name": "amount_b"},
                                "right": {
                                    "kind": "literal",
                                    "value": {"kind": "decimal", "value": "424242"},
                                },
                            },
                        ],
                    },
                }
            ]
        }
    }
    interface = bta.extract_rule_interface(
        {"rules": [{"name": "out", "kind": "derived", "dtype": "Judgment"}]},
        module="us-xx:x",
        compiled_payload=compiled,
    )
    rendered = json.dumps(interface.to_json())
    assert [item.name for item in interface.inputs] == ["amount_b", "fact_a"]
    assert interface.input("fact_a").dtype == "Boolean"
    assert interface.input("fact_a").declared is False
    assert interface.input("amount_b").dtype is None
    for hidden in ("424242", "gt", "eq", "comparison"):
        assert hidden not in rendered


def test_guard_stops_a_leaking_prompt_before_any_model_call(monkeypatch):
    payload = yaml.safe_load(PLANTED_MODULE)
    real = bta.extract_rule_interface

    def leaky(rulespec_payload, **kwargs):
        interface = real(rulespec_payload, **kwargs)
        # A regression that copies a parameter value into the interface.
        return bta.RuleInterface(
            module=interface.module,
            inputs=interface.inputs,
            outputs=(
                *interface.outputs,
                bta.InterfaceOutput(
                    "income_limit_copy",
                    "parameter",
                    enumeration=(PLANTED["parameter_value"],),
                ),
            ),
        )

    monkeypatch.setattr(bta, "extract_rule_interface", leaky)
    backend = bta.FakeBlindAuthorBackend(_answer([]))
    with pytest.raises(bta.BlindnessViolation, match=PLANTED["parameter_value"]):
        bta.run_blind_test_author(
            rulespec_payload=payload,
            module="us-xx:x",
            source_text=PLANTED_SOURCE,
            backend=backend,
            runner=_StubRunner(),
        )
    assert backend.prompts == []


def _planted_prompt():
    payload = yaml.safe_load(PLANTED_MODULE)
    interface = bta.extract_rule_interface(payload, module="us-xx:x")
    prompt = bta.build_blind_prompt(source_text=PLANTED_SOURCE, interface=interface)
    return payload, interface, prompt


def _with_interface_text(prompt, extra):
    return bta.BlindPrompt(
        instructions=prompt.instructions,
        citation_line=prompt.citation_line,
        interface_block=prompt.interface_block + extra + "\n",
        source_text=prompt.source_text,
    )


@pytest.mark.parametrize(
    "leak",
    [
        f"household_gross_income <= income_limit + {PLANTED['formula_number']}",
        f"and not {PLANTED['formula']}",
        f"{PLANTED['summary']} restates the provision.",
    ],
)
def test_guard_finds_formula_lines_and_module_prose(leak):
    payload, interface, prompt = _planted_prompt()
    assert bta.find_blindness_violations(prompt, rulespec_payload=payload) == []
    assert bta.find_interface_leaks(prompt, rulespec_payload=payload) == []
    assert bta.find_blindness_violations(
        _with_interface_text(prompt, leak), rulespec_payload=payload
    )


def test_guard_finds_existing_test_names_but_not_interface_names():
    payload, interface, prompt = _planted_prompt()
    leaking = _with_interface_text(prompt, f'"x": "{PLANTED["test_name"]}"')
    violations = bta.find_blindness_violations(
        leaking,
        rulespec_payload=payload,
        companion_cases=PLANTED_TESTS,
        interface=interface,
    )
    assert violations and PLANTED["test_name"] in violations[0]
    # A test named after the output it asserts shows nothing new.
    named_after_output = [{**PLANTED_TESTS[0], "name": "household_is_eligible"}]
    assert (
        bta.find_blindness_violations(
            prompt,
            rulespec_payload=payload,
            companion_cases=named_after_output,
            interface=interface,
        )
        == []
    )
    report = bta.run_blind_test_author(
        rulespec_payload=payload,
        module="us-xx:x",
        source_text=PLANTED_SOURCE,
        backend=bta.FakeBlindAuthorBackend(_answer([])),
        runner=_StubRunner(),
        companion_cases=named_after_output,
    )
    assert report["counts"]["authored"] == 0


# The reviewer's doctored shape: values and dates spelled inside names, a
# unit, an enumeration member and the citation path.
DOCTORED_MODULE = """
format: rulespec/v1
inputs:
  - name: limit_is_1500_usd_and_rule_starts_2026_08_01
    entity: Household
    dtype: Money
    period: Month
    unit: USD_1500_per_month
  - name: tier
    entity: Household free text holds iff fee_ok
    dtype: String limit 1500
    period: Month effective 2026-08-01
    values: [threshold_1500, other]
rules:
  - name: fee_ok
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-08-01'
        formula: limit_is_1500_usd_and_rule_starts_2026_08_01 <= 1500
"""


def test_free_text_in_a_declaration_field_is_dropped_not_copied():
    interface = bta.extract_rule_interface(
        yaml.safe_load(DOCTORED_MODULE), module="us-xx:x"
    )
    money, tier = interface.inputs
    assert (money.unit, money.dtype, money.entity) == (None, "Money", "Household")
    assert (tier.entity, tier.dtype, tier.period) == (None, None, None)
    assert interface.dropped == (
        "inputs.limit_is_1500_usd_and_rule_starts_2026_08_01.unit",
        "inputs.tier.dtype",
        "inputs.tier.entity",
        "inputs.tier.period",
    )
    rendered = json.dumps(interface.to_json())
    for free_text in ("per_month", "iff", "effective", "limit 1500"):
        assert free_text not in rendered


@pytest.mark.parametrize(
    ("field_name", "accepted", "refused"),
    [
        ("dtype", ["Money", "Boolean", "Judgment"], ["Money 1500", "Enum[a,b]", "x"]),
        ("period", ["Month", "Year"], ["Month_from_2026_08", "2026-08", "Monthly"]),
        ("unit", ["USD", "CAD", "years"], ["USD_1500_per_month", "usd 5", "1500"]),
        ("entity", ["Household", "T2036CalculationUnit"], ["tax_unit_1500", "A b"]),
        ("kind", ["derived", "parameter"], ["derived 1500", "formula"]),
    ],
)
def test_declaration_words_are_a_closed_vocabulary(field_name, accepted, refused):
    for word in accepted:
        assert bta._type_word(word, field_name) == word
    for word in (*refused, 1500, None, ["USD"]):
        assert bta._type_word(word, field_name) is None


def test_parameter_values_are_never_read_as_an_enumeration():
    interface = bta.extract_rule_interface(
        {
            "rules": [
                {"name": "tiers", "kind": "parameter", "values": ["tier_1500", "b"]},
                {"name": "status", "kind": "derived", "values": ["open", "closed"]},
            ]
        },
        module="us-xx:x",
    )
    assert interface.output("tiers").enumeration == ()
    assert interface.output("status").enumeration == ("open", "closed")
    assert "tier_1500" not in json.dumps(interface.to_json())


def test_values_and_dates_inside_names_are_reported_as_named_leaks():
    payload = yaml.safe_load(DOCTORED_MODULE)
    backend = bta.FakeBlindAuthorBackend(_answer([]))
    kwargs = dict(
        rulespec_payload=payload,
        module="us-xx:x",
        source_text=PLANTED_SOURCE,
        backend=backend,
        runner=_StubRunner(),
    )
    with pytest.raises(bta.InterfaceLeak) as refused:
        bta.run_blind_test_author(**kwargs)
    assert backend.prompts == []
    for expected in (
        "formula_number 1500",
        "effective_date 2026-08-01",
        "threshold_1500",
    ):
        assert expected in str(refused.value)
    assert "--allow-interface-leaks" in str(refused.value)

    report = bta.run_blind_test_author(**kwargs, allow_interface_leaks=True)
    assert len(backend.prompts) == 1
    assert report["interface_leaks_allowed"] is True
    leaks = {
        (leak["kind"], leak["value"], leak["carrier"])
        for leak in report["interface_leaks"]
    }
    name = "limit_is_1500_usd_and_rule_starts_2026_08_01"
    assert ("formula_number", "1500", name) in leaks
    assert ("effective_date", "2026-08-01", name) in leaks
    assert ("formula_number", "1500", "threshold_1500") in leaks
    summary = bta.render_text_summary(report)
    assert "[interface leak, allowed by the caller]" in summary
    count = len(report["interface_leaks"])
    assert count >= 3
    assert report["blindness"] == f"NOT FULLY BLIND ({count} leaks allowed)"
    assert summary.startswith(f"Test author, NOT FULLY BLIND ({count} leaks allowed)")
    assert "Blind test author" not in summary
    assert bta.blindness_label([{"kind": "x"}]) == "NOT FULLY BLIND (1 leak allowed)"
    assert report["interface_dropped_fields"]


def test_a_number_in_the_module_or_citation_path_is_a_named_leak_too():
    payload = {
        "rules": [
            {
                "name": "allowance",
                "kind": "parameter",
                "versions": [{"effective_from": "2026-08-01", "formula": "273"}],
            }
        ]
    }
    interface = bta.extract_rule_interface(payload, module="us:policies/x/block-273")
    prompt = bta.build_blind_prompt(
        source_text=PLANTED_SOURCE,
        interface=interface,
        citation_path="us/regulation/7/273/9",
    )
    leaks = bta.find_interface_leaks(prompt, rulespec_payload=payload)
    assert {(leak["where"], leak["carrier"]) for leak in leaks} == {
        ("citation_path", "us/regulation/7/273/9"),
        ("interface", "us:policies/x/block-273"),
    }


def test_identifier_digits_in_a_formula_and_single_digits_are_not_values():
    payload = {
        "rules": [
            {
                "name": "line_32600",
                "kind": "derived",
                "versions": [{"formula": "line_32600_base * 2 + other_7"}],
            }
        ]
    }
    interface = bta.extract_rule_interface(payload, module="us:policies/x/block-2")
    prompt = bta.build_blind_prompt(source_text=PLANTED_SOURCE, interface=interface)
    assert bta.find_interface_leaks(prompt, rulespec_payload=payload) == []


@pytest.mark.parametrize(
    "citation",
    [
        "us-xx/manual/x NOTE holds iff fee_ok unless deposit; limit 1500",
        "holds iff fee_ok",
        "us-xx/manual/x\neffective 2026-08-01",
        "us-xx/not-a-document-class/x",
        "US-XX/manual/x/",
        # `statutes` is not a corpus document class; the real path is `statute`.
        "us-la/statutes/47:294",
        "us-xx/manual/x limit is fifteen hundred",
        "us-xx/manual/limit_1500",
    ],
)
def test_a_free_text_citation_path_is_refused(citation):
    payload, interface, _ = _planted_prompt()
    with pytest.raises(bta.BlindnessViolation, match="citation path"):
        bta.build_blind_prompt(
            source_text=PLANTED_SOURCE, interface=interface, citation_path=citation
        )


def test_no_model_call_when_the_candidate_does_not_compile():
    backend = bta.FakeBlindAuthorBackend(_answer(CORRECT_CASES))
    runner = _StubRunner()
    runner.compile_error = "compile failed: synthetic"
    with pytest.raises(bta.BlindTestAuthorError, match="does not compile"):
        bta.run_blind_test_author(
            rulespec_payload=yaml.safe_load(MODULE_TEXT),
            module=MODULE,
            source_text=SOURCE,
            backend=backend,
            runner=runner,
        )
    assert backend.prompts == [] and runner.cases == []


# ---------------------------------------------------------------------------
# Strict typed answer
# ---------------------------------------------------------------------------


def test_case_with_a_sentence_not_in_the_source_is_rejected_and_reported():
    invented = "Deposits are allowable when the provider requires them in writing."
    runner = _StubRunner()
    report = bta.run_blind_test_author(
        rulespec_payload=yaml.safe_load(MODULE_TEXT),
        module=MODULE,
        source_text=SOURCE,
        backend=bta.FakeBlindAuthorBackend(
            _answer(
                [
                    CORRECT_CASES[0],
                    _case("invented_rule", "holds", invented, **{INPUTS[3]: True}),
                ]
            )
        ),
        runner=runner,
        compiled_payload={
            "program": {
                "derived": [
                    {"expr": [{"kind": "input", "name": name} for name in INPUTS]}
                ]
            }
        },
    )
    assert [case.name for case in runner.cases] == ["all_conditions_met"]
    assert report["counts"] == {
        "agree": 1,
        "disagree": 0,
        "not_runnable": 0,
        "rejected": 1,
        "authored": 2,
    }
    assert report["rejected_cases"] == [
        {
            "index": 2,
            "name": "invented_rule",
            "reason": "source_sentence is not found verbatim in the source text",
            "source_sentence": invented,
        }
    ]
    assert "[rejected] #2 invented_rule" in bta.render_text_summary(report)


def test_verbatim_means_exact_characters_with_only_whitespace_folded():
    assert bta.sentence_in_source(
        "all of the following occur:\n\n\u25cf The utility installation fee", SOURCE
    )
    assert not bta.sentence_in_source(DEPOSIT_SENTENCE.lower(), SOURCE)
    assert not bta.sentence_in_source(
        "One-time deposits are an allowable NA Utility Expense.", SOURCE
    )


BOUNDARY_SOURCE = """\
(a) In general. A household is eligible when income is below the limit; and
the household applies in person. See the "exceptions." Then stop!

NOTE One-time deposits are not an allowable expense.

1. The first numbered item applies to elderly households.
\u25cf A bulleted item about shelter costs, which has two parts.
"""


@pytest.mark.parametrize(
    ("quote", "at_boundaries", "full_item"),
    [
        # whole items, with and without their marker
        ("NOTE One-time deposits are not an allowable expense.", True, True),
        ("One-time deposits are not an allowable expense.", True, True),
        ("The first numbered item applies to elderly households.", True, True),
        ("A bulleted item about shelter costs, which has two parts.", True, True),
        # whole sentences or list members inside an item
        # a semicolon is not an end: the clause after it can change the rule
        ("A household is eligible when income is below the limit;", False, False),
        (
            "A household is eligible when income is below the limit; and the "
            "household applies in person.",
            True,
            False,
        ),
        ('See the "exceptions."', True, False),
        # several items in a row
        (
            "One-time deposits are not an allowable expense. 1. The first "
            "numbered item applies to elderly households.",
            True,
            True,
        ),
        # cut inside a sentence: mid-word, conditions dropped, across items
        ("time deposits are not an allowable expense.", False, False),
        ("A household is eligible when income", False, False),
        ("One-time deposits are not an allowable", False, False),
        ("household is eligible when income is below the limit;", False, False),
        ("allowable expense. 1. The first numbered item applies", False, False),
        ("item about shelter costs, which has two parts.", False, False),
        ("A bulleted item about shelter costs,", False, False),
    ],
)
def test_a_quote_must_start_and_end_at_a_sentence_or_item_boundary(
    quote, at_boundaries, full_item
):
    match = bta.SourceQuotes(BOUNDARY_SOURCE).locate(quote)
    assert match.found
    assert (match.at_boundaries, match.covers_full_item) == (at_boundaries, full_item)


def test_cut_quotes_are_rejected_and_partial_item_quotes_are_counted():
    source = SOURCE + "\nThe fee is allowable once. A second fee is never allowable.\n"
    cut = [
        _case(
            "mid_word",
            "not_holds",
            "time deposits are not an allowable NA Utility Expense.",
        ),
        _case(
            "conditions_cut_off",
            "holds",
            "The initial utility installation fee is an allowable NA Utility Expense",
        ),
        _case(
            "across_bullets",
            "holds",
            "initial installation of a utility. \u25cf The utility provider performs",
            **{INPUTS[1]: False},
        ),
    ]
    partial = _case(
        "one_sentence_of_an_item",
        "not_holds",
        "A second fee is never allowable.",
        **{INPUTS[0]: False},
    )
    report = bta.run_blind_test_author(
        rulespec_payload={
            "rules": [{"name": OUTPUT, "kind": "derived", "dtype": "Judgment"}]
        },
        module=MODULE,
        source_text=source,
        backend=bta.FakeBlindAuthorBackend(_answer([*cut, CORRECT_CASES[0], partial])),
        runner=_StubRunner(),
        compiled_payload={
            "program": {
                "derived": [{"expr": [{"kind": "input", "name": n} for n in INPUTS]}]
            }
        },
    )
    assert [item["name"] for item in report["rejected_cases"]] == [
        "mid_word",
        "conditions_cut_off",
        "across_bullets",
    ]
    assert {item["reason"] for item in report["rejected_cases"]} == {
        "source_sentence does not start and end at a sentence or list-item "
        "boundary of the source text"
    }
    assert [(c["name"], c["quote_covers_full_item"]) for c in report["cases"]] == [
        ("all_conditions_met", True),
        ("one_sentence_of_an_item", False),
    ]
    assert report["quote_stats"] == {
        "accepted_cases": 2,
        "quote_shorter_than_its_source_item": 1,
        "share": 0.5,
        "list_item_without_its_chapeau": 0,
    }
    assert "1 of 2 accepted cases quote less" in bta.render_text_summary(report)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda c: c.update(source_sentence="not an"), "shorter than"),
        (lambda c: c["inputs"].update(undeclared_fact=True), "not in the interface"),
        (
            lambda c: c.update(expected={"other_output": "holds"}),
            "not in the interface",
        ),
        (lambda c: c["inputs"].update({INPUTS[0]: "yes"}), "must be true or false"),
        (lambda c: c.update(expected={OUTPUT: True}), "Judgment value must be"),
        (lambda c: c.update(period="2026-08-01"), "YYYY-MM"),
        (lambda c: c.update(rationale="because I think so"), "keys must be exactly"),
        (lambda c: c.pop("source_sentence"), "keys must be exactly"),
        (lambda c: c.update(expected={}), "non-empty"),
    ],
)
def test_malformed_cases_are_rejected(mutate, reason):
    case = json.loads(json.dumps(CORRECT_CASES[0]))
    mutate(case)
    parsed = bta.parse_blind_response(
        _answer([case]), interface=_interface(), source_text=SOURCE
    )
    assert parsed.cases == ()
    assert reason in parsed.rejected[0].reason


@pytest.mark.parametrize(
    "answer",
    [
        "Here are the cases:\n" + _answer(CORRECT_CASES),
        _answer(CORRECT_CASES) + "\nThe deposit case is the important one.",
        json.dumps(CORRECT_CASES),
        _answer(CORRECT_CASES, commentary="the rule looks right"),
        '{"cases": [], "stated_commencement": NaN}',
        '{"cases": [{"name": "a", "period": "2026-08", "inputs": {"x": Infinity}}]}',
        '{"cases": [{"name": "a", "period": "2026-08", "inputs": {"x": -Infinity}}]}',
        '{"cases": [], "cases": []}',
        "",
    ],
)
def test_free_prose_is_never_parsed(answer):
    parsed = bta.parse_blind_response(
        answer, interface=_interface(), source_text=SOURCE
    )
    assert parsed.cases == () and parsed.rejected == ()
    assert parsed.response_error


def test_one_enclosing_fence_and_duplicates_and_the_case_limit():
    answer = "```json\n" + _answer([CORRECT_CASES[0]] * 2 + CORRECT_CASES[1:]) + "\n```"
    parsed = bta.parse_blind_response(
        answer, interface=_interface(), source_text=SOURCE, max_cases=3
    )
    assert [case.name for case in parsed.cases] == [
        "all_conditions_met",
        "not_an_initial_installation",
    ]
    assert [item.reason for item in parsed.rejected] == [
        "duplicate case name",
        "more than 3 cases; not run",
        "more than 3 cases; not run",
    ]


ABBREVIATION_SOURCE = """\
The deduction is allowed; except that it is reduced for boarders.
Payments made under 7 U.S.C. 2014 are excluded from income.
Payments under Pub. L. No. 119-21 count as income. See 7 C.F.R. 273.9 et seq. for details.
The fee applies, e.g. to wiring, i.e. to work the provider does. Sec. 5 governs deposits.
Judge A. B. Smith decided the case. The rule follows.
"""


@pytest.mark.parametrize(
    ("quote", "at_boundaries"),
    [
        ("The deduction is allowed;", False),
        ("The deduction is allowed; except that it is reduced for boarders.", True),
        ("Payments made under 7 U.S.C.", False),
        ("Payments made under 7 U.S.C. 2014 are excluded from income.", True),
        ("Payments under Pub. L. No.", False),
        ("Payments under Pub. L.", False),
        ("Payments under Pub. L. No. 119-21 count as income.", True),
        ("See 7 C.F.R.", False),
        ("See 7 C.F.R. 273.9 et seq.", False),
        ("See 7 C.F.R. 273.9 et seq. for details.", True),
        ("The fee applies, e.g.", False),
        ("The fee applies, e.g. to wiring, i.e.", False),
        ("The fee applies, e.g. to wiring, i.e. to work the provider does.", True),
        ("Sec. 5 governs deposits.", True),
        ("5 governs deposits.", False),
        ("Judge A. B. Smith decided the case.", True),
        ("B. Smith decided the case.", False),
        ("The rule follows.", True),
    ],
)
def test_a_semicolon_or_an_abbreviation_does_not_end_a_sentence(quote, at_boundaries):
    match = bta.SourceQuotes(ABBREVIATION_SOURCE).locate(quote)
    assert match.found and match.at_boundaries is at_boundaries


def test_a_list_item_quoted_without_its_chapeau_is_counted():
    quotes = bta.SourceQuotes(SOURCE)
    item = "The utility provider performs the installation."
    assert quotes.locate(item).lacks_chapeau is True
    assert quotes.locate(item).covers_full_item is True
    assert quotes.locate(ALLOWABLE_SENTENCE).lacks_chapeau is False
    with_chapeau = (
        f"{ALLOWABLE_SENTENCE} \u25cf The utility installation fee is for an "
        "initial installation of a utility."
    )
    assert quotes.locate(with_chapeau).at_boundaries is True
    assert quotes.locate(with_chapeau).lacks_chapeau is False
    # The list ends at the next item that is not a list item.
    assert quotes.locate(DEPOSIT_SENTENCE).lacks_chapeau is False

    report = bta.run_blind_test_author(
        rulespec_payload=yaml.safe_load(MODULE_TEXT),
        module=MODULE,
        source_text=SOURCE,
        backend=bta.FakeBlindAuthorBackend(_answer(CORRECT_CASES)),
        runner=_StubRunner(),
        compiled_payload={
            "program": {
                "derived": [{"expr": [{"kind": "input", "name": n} for n in INPUTS]}]
            }
        },
    )
    assert [case["quote_lacks_chapeau"] for case in report["cases"]] == [
        False,
        True,
        True,
        False,
    ]
    assert report["quote_stats"] == {
        "accepted_cases": 4,
        "quote_shorter_than_its_source_item": 0,
        "share": 0.0,
        "list_item_without_its_chapeau": 2,
    }
    assert (
        "2 of 4 accepted cases quote a list item without its chapeau"
        in bta.render_text_summary(report)
    )
    assert report["blindness"] == "blind (the guard found no leak)"
    assert bta.render_text_summary(report).startswith("Blind test author (shadow")


def test_words_that_rulespec_us_main_uses_are_declaration_words():
    # From a read-only scan of rulespec-us main (4,886 modules, 2026-11-13).
    used = {
        "dtype": [
            "Judgment",
            "Money",
            "Rate",
            "Count",
            "Integer",
            "Boolean",
            "Text",
            "Decimal",
            "Bool",
            "String",
            "Number",
            "Float",
            "Date",
            "Enum[HcaGrantPaymentTier]",
        ],
        "period": ["Day", "Month", "Year", "Week", "Hour"],
        "unit": [
            "USD",
            "years",
            "calendar_year",
            "months",
            "day",
            "year",
            "calendar_day",
            "month",
        ],
        "kind": [
            "derived",
            "parameter",
            "source_relation",
            "data_relation",
            "derived_relation",
        ],
    }
    for field_name, words in used.items():
        for word in words:
            assert bta._type_word(word, field_name) == word
    for field_name, word in (
        ("dtype", "Enum[a, b]"),
        ("dtype", "Enum[limit_1500]"),
        ("dtype", "Numberish"),
        ("period", "Fortnight"),
        ("unit", "per_1500_usd"),
    ):
        assert bta._type_word(word, field_name) is None
    interface = bta.extract_rule_interface(
        {
            "inputs": [
                {"name": "hours", "dtype": "Number", "period": "Hour", "unit": "day"},
                {"name": "tier", "dtype": "Enum[HcaGrantPaymentTier]"},
                {"name": "odd", "dtype": "Numberish"},
            ],
            "rules": [{"name": "out", "kind": "derived", "dtype": "Judgment"}],
        },
        module="us-xx:x",
    )
    assert interface.dropped == ("inputs.odd.dtype",)
    hours, tier, _ = interface.inputs
    assert (hours.dtype, hours.period, hours.unit) == ("Number", "Hour", "day")
    # A known word is type-checked; before, a dropped word let any scalar pass.
    assert "JSON number" in bta._value_issue("many", hours.dtype, ())
    assert bta._value_issue(7.5, hours.dtype, ()) == ""
    assert "must be a string" in bta._value_issue(3, tier.dtype, ())
    assert bta._value_issue("tier_one", tier.dtype, ()) == ""


@pytest.mark.parametrize(
    "citation",
    [
        "us-la/statute/47:294",
        "us-la/statute/47:297.14",
        "us-ak/regulation/aac/title-7/chapter-45/7 AAC 45.280",
        "us/regulation/7/273/9",
        "us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-13",
        "us/guidance/irs/rev-proc-2025-32/page-15",
    ],
)
def test_real_canonical_citation_paths_are_accepted(citation):
    payload, interface, _ = _planted_prompt()
    prompt = bta.build_blind_prompt(
        source_text=PLANTED_SOURCE, interface=interface, citation_path=citation
    )
    assert prompt.citation_line == f"Source citation path: {citation}\n"


@pytest.mark.parametrize(
    ("month", "sentence", "issue"),
    [
        ("2027-03", "This section applies on or after March 1, 2027.", ""),
        ("2027-03", "This section takes effect in March 2027.", ""),
        ("2027-03", "This section takes effect on the 1st day of March, 2027.", ""),
        ("2027-03", "This section takes effect on 1 Mar. 2027.", ""),
        ("2027-03", "This section takes effect on 2027-03-01.", ""),
        ("2027-03", "This section takes effect on 3/1/2027.", ""),
        ("2026-10", "Allotments for FY 2026 are in the table below.", "year mentioned"),
        (
            "2026-10",
            "For fiscal year 2026 the standard deduction is $209.",
            "year mentioned",
        ),
        ("2026-10", "See 7 U.S.C. 2014(e) (2026) for the deduction.", "year mentioned"),
        (
            "2025-07",
            "As amended by Pub. L. 119-21, July 4, 2025, the deduction is limited.",
            "year mentioned",
        ),
        (
            "2025-10",
            "See the notice dated October 3, 2025 for the rule.",
            "year mentioned",
        ),
        (
            "2027-04",
            "This section applies on or after March 1, 2027.",
            "not in that month",
        ),
        ("2027-03", "The provider performs the installation.", "states no date"),
    ],
)
def test_a_stated_date_is_a_date_phrase_not_a_year_or_a_citation(
    month, sentence, issue
):
    result = bta._stated_date_issue(month, sentence)
    assert (result == "") if not issue else (issue in result)


def test_a_fiscal_year_label_raises_no_period_finding():
    label = "Allotments for FY 2026 are in the table below."
    report, runner = _probe_report(
        "2025-10",
        as_of=None,
        stated={"month": "2026-10", "source_sentence": label},
        source=SOURCE + f"\n{label}\n",
    )
    stated = report["period_probes"]["source_stated_date"]
    assert stated["finding"] == "not_probed" and stated["stated_month"] is None
    assert report["notes"] == [
        "stated_commencement ignored: year mentioned, not a stated date"
    ]
    assert "year mentioned, not a stated date" in bta.render_text_summary(report)
    assert runner.periods == ["2026-08"]


def test_the_same_case_under_two_names_counts_once():
    again = {**CORRECT_CASES[0], "name": "all_conditions_met_again"}
    reordered = {
        **CORRECT_CASES[1],
        "name": "reordered_inputs",
        "inputs": dict(reversed(CORRECT_CASES[1]["inputs"].items())),
        "source_sentence": ALLOWABLE_SENTENCE,
    }
    other_period = {**CORRECT_CASES[0], "name": "next_month", "period": "2026-09"}
    parsed = bta.parse_blind_response(
        _answer([*CORRECT_CASES[:2], again, reordered, other_period]),
        interface=_interface(),
        source_text=SOURCE,
    )
    assert [case.name for case in parsed.cases] == [
        "all_conditions_met",
        "not_an_initial_installation",
        "next_month",
    ]
    assert [(item.name, item.reason) for item in parsed.rejected] == [
        (
            "all_conditions_met_again",
            "same inputs, period and expected outputs as `all_conditions_met`",
        ),
        (
            "reordered_inputs",
            "same inputs, period and expected outputs as `not_an_initial_installation`",
        ),
    ]


def test_non_finite_numbers_are_rejected():
    interface = bta.RuleInterface(
        module=MODULE,
        inputs=(bta.InterfaceInput("income", "Money"), bta.InterfaceInput("untyped")),
        outputs=(bta.InterfaceOutput("benefit", "derived", "Money"),),
    )

    def answer(inputs, expected):
        return (
            '{"cases": [{"name": "a", "period": "2026-08", "inputs": %s, '
            '"expected": %s, "source_sentence": %s}]}'
            % (inputs, expected, json.dumps(DEPOSIT_SENTENCE))
        )

    for inputs, expected in (
        ('{"income": 1e999}', '{"benefit": 1}'),
        ('{"untyped": -1e999}', '{"benefit": 1}'),
        ('{"income": 1}', '{"benefit": 1e999}'),
    ):
        parsed = bta.parse_blind_response(
            answer(inputs, expected), interface=interface, source_text=SOURCE
        )
        assert parsed.cases == ()
        assert "number must be finite" in parsed.rejected[0].reason
    for constant in ("NaN", "Infinity", "-Infinity"):
        parsed = bta.parse_blind_response(
            answer('{"income": %s}' % constant, '{"benefit": 1}'),
            interface=interface,
            source_text=SOURCE,
        )
        assert parsed.cases == () and "not a JSON number" in parsed.response_error
    ok = bta.parse_blind_response(
        answer('{"income": 1500.5}', '{"benefit": 0}'),
        interface=interface,
        source_text=SOURCE,
    )
    assert len(ok.cases) == 1


DATED_SOURCE = (
    SOURCE + "\nThis section applies to expenses billed on or after March 1, 2027.\n"
)
DATED_SENTENCE = "This section applies to expenses billed on or after March 1, 2027."


@pytest.mark.parametrize(
    ("stated", "month", "note"),
    [
        ({"month": "2027-03", "source_sentence": DATED_SENTENCE}, "2027-03", None),
        (None, None, None),
        # A sentence that is real but states no such year proves nothing.
        ({"month": "2027-03", "source_sentence": DEPOSIT_SENTENCE}, None, "no date"),
        (
            {"month": "2027-03", "source_sentence": "Effective March 2027."},
            None,
            "verbatim",
        ),
        (
            {
                "month": "2027-03",
                "source_sentence": "applies to expenses billed on or after March 1, 2027.",
            },
            None,
            "boundary",
        ),
        ({"month": "March 2027", "source_sentence": DATED_SENTENCE}, None, "YYYY-MM"),
        # The model may not name the as-of floor: that month is computed in code.
        ({"month": "2027-01", "basis": "source_as_of"}, None, "keys must be"),
        ("2027-03", None, "keys must be"),
    ],
)
def test_a_stated_commencement_needs_the_verbatim_sentence_with_its_year(
    stated, month, note
):
    parsed = bta.parse_blind_response(
        _answer([], stated_commencement=stated),
        interface=_interface(),
        source_text=DATED_SOURCE,
    )
    assert parsed.response_error is None
    assert parsed.stated_commencement_month == month
    assert (parsed.stated_commencement_sentence is not None) == (month is not None)
    if note:
        assert note in parsed.notes[0]
    else:
        assert parsed.notes == ()


def test_the_old_model_chosen_floor_keys_are_refused():
    parsed = bta.parse_blind_response(
        _answer(
            [],
            earliest_supported_period="2027-01",
            earliest_period_basis="source_as_of",
        ),
        interface=_interface(),
        source_text=SOURCE,
    )
    assert "unexpected top-level key" in parsed.response_error


@pytest.mark.parametrize(
    ("as_of", "floor"),
    [
        ("2026-07-17", "2026-08"),
        ("2026-07-01", "2026-07"),
        ("2026-07-31", "2026-08"),
        ("2026-12-02", "2027-01"),
        ("2024-01-31", "2024-02"),
    ],
)
def test_the_assumed_evidence_floor_is_computed_in_code(as_of, floor):
    assert bta.assumed_evidence_floor(as_of) == floor


class _DatedRunner:
    """A candidate that computes from ``starts`` on. No engine."""

    def __init__(self, starts):
        self.starts = starts
        self.periods = []

    def run(self, case):
        self.periods.append(case.period)
        if case.period < self.starts:
            return bta.classify_outcome(
                [
                    f"Test case `{case.name}` execution failed: derived `o` has no formula version at {case.period}-01"
                ],
                [],
            )
        return bta.classify_outcome([], [{"output": "o", "mismatch": None}])


def _probe_report(
    starts, *, as_of="2026-07-17", stated=None, cases=CORRECT_CASES[:1], source=SOURCE
):
    runner = _DatedRunner(starts)
    report = bta.run_blind_test_author(
        rulespec_payload=yaml.safe_load(MODULE_TEXT),
        module=MODULE,
        source_text=source,
        backend=bta.FakeBlindAuthorBackend(_answer(cases, stated_commencement=stated)),
        runner=runner,
        compiled_payload={
            "program": {
                "derived": [{"expr": [{"kind": "input", "name": n} for n in INPUTS]}]
            }
        },
        source_as_of=as_of,
    )
    return report, runner


def test_floor_probe_is_marked_as_an_assumption_and_names_the_open_issue():
    report, runner = _probe_report("2025-10")
    floor = report["period_probes"]["assumed_evidence_floor"]
    assert floor["finding"] == "before_assumed_evidence_floor"
    assert floor["assumption"] is True
    assert floor["reference"] == "TheAxiomFoundation/axiom-encode#2212"
    assert (floor["floor_month"], floor["probe_period"]) == ("2026-08", "2026-07")
    assert (
        "depends on the decision in TheAxiomFoundation/axiom-encode#2212"
        in floor["meaning"]
    )
    assert runner.periods == ["2026-08", "2026-07"]
    summary = bta.render_text_summary(report)
    assert (
        "ASSUMPTION, not a defect finding; see TheAxiomFoundation/axiom-encode#2212"
        in summary
    )
    assert "is not decided" in summary
    assert report["period_probes"]["source_stated_date"]["finding"] == "not_probed"


def test_floor_probe_is_quiet_for_a_rule_that_starts_at_the_floor():
    report, _ = _probe_report("2026-08")
    floor = report["period_probes"]["assumed_evidence_floor"]
    assert floor["finding"] == "not_before_assumed_evidence_floor"
    assert floor["assumption"] is True
    assert "[period" not in bta.render_text_summary(report)


def test_the_model_cannot_move_the_floor():
    # Round-1 finding: an answer naming 2027-01 made an unchanged rule "start
    # too early". The floor now comes from the as-of date only.
    dated = {"month": "2027-03", "source_sentence": DATED_SENTENCE}
    report, runner = _probe_report("2026-08", stated=dated, source=DATED_SOURCE)
    probes = report["period_probes"]
    assert (
        probes["assumed_evidence_floor"]["finding"]
        == "not_before_assumed_evidence_floor"
    )
    assert probes["assumed_evidence_floor"]["floor_month"] == "2026-08"
    # The stated date is its own probe, with its sentence, and not an assumption.
    stated = probes["source_stated_date"]
    assert stated["finding"] == "before_source_stated_date"
    assert stated["assumption"] is False
    assert (stated["stated_month"], stated["probe_period"]) == ("2027-03", "2027-02")
    assert stated["source_sentence"] == DATED_SENTENCE
    summary = bta.render_text_summary(report)
    assert "[period, second model's reading]" in summary and DATED_SENTENCE in summary
    assert "ASSUMPTION" not in summary


def test_probes_do_not_run_without_a_reference_month_or_a_runnable_case():
    report, runner = _probe_report("2026-08", as_of=None)
    probes = report["period_probes"]
    assert probes["assumed_evidence_floor"]["finding"] == "not_probed"
    assert probes["source_stated_date"]["finding"] == "not_probed"
    assert runner.periods == ["2026-08"]
    early = _case("too_early", "holds", ALLOWABLE_SENTENCE, period="2026-01")
    report, _ = _probe_report("2026-08", cases=[early])
    assert report["period_probes"]["assumed_evidence_floor"]["finding"] == "not_probed"


def test_the_prompt_does_not_state_a_start_date_rule():
    interface = _interface()
    prompt = bta.build_blind_prompt(
        source_text=SOURCE, interface=interface, source_as_of="2026-07-17"
    )
    assert "captured on 2026-07-17" in prompt.text
    for rule_like in ("first full", "earliest", "source_as_of", "floor"):
        assert rule_like not in prompt.controlled
    with pytest.raises(bta.BlindTestAuthorError, match="YYYY-MM-DD"):
        bta.build_blind_prompt(
            source_text=SOURCE,
            interface=interface,
            source_as_of="July 2026; limit 1500",
        )


def test_a_mismatch_is_a_disagreement_even_beside_another_issue():
    mismatch = "Test case `a` output `o` expected not_holds, got holds."
    binding = "Test case `a` input `x` does not resolve to an input slot."
    compared = [{"output": "o", "mismatch": mismatch}]
    assert bta.classify_outcome([], [{"output": "o", "mismatch": None}]).status == (
        bta.STATUS_AGREE
    )
    assert bta.classify_outcome([mismatch], compared).status == bta.STATUS_DISAGREE
    row = bta.classify_outcome([f"{mismatch} (row #1)"], compared)
    assert (row.status, row.other_issues) == (bta.STATUS_DISAGREE, ())
    both = bta.classify_outcome([mismatch, binding], compared)
    assert both.status == bta.STATUS_DISAGREE
    assert both.issues == (mismatch,) and both.other_issues == (binding,)
    only_binding = bta.classify_outcome([binding], [{"output": "o", "mismatch": None}])
    assert only_binding.status == bta.STATUS_NOT_RUNNABLE
    assert only_binding.not_runnable_kind == "execution_or_binding"
    undated = bta.classify_outcome(
        [
            "Test case `a` execution failed: derived `o` has no formula version "
            "at 2026-01-01"
        ],
        [],
    )
    assert undated.not_runnable_kind == "no_formula_version_at_period"
    assert bta.classify_outcome([], []).status == bta.STATUS_NOT_RUNNABLE

    report = bta.run_blind_test_author(
        rulespec_payload=yaml.safe_load(MODULE_TEXT),
        module=MODULE,
        source_text=SOURCE,
        backend=bta.FakeBlindAuthorBackend(_answer(CORRECT_CASES[:1])),
        runner=_StubRunner(both),
        compiled_payload={
            "program": {
                "derived": [{"expr": [{"kind": "input", "name": n} for n in INPUTS]}]
            }
        },
    )
    assert report["counts"]["disagree"] == 1
    assert report["cases"][0]["other_issues"] == [binding]
    assert f"also: {binding}" in bta.render_text_summary(report)


def test_real_defect_listing_reads_the_parent_commit_from_each_case():
    corpus = Path(__file__).parents[1] / "benchmarks" / "verifier" / "real_defects_v0"
    index = json.loads((corpus / "index.json").read_text())
    # The round-1 plan read this key from the index, where it does not exist.
    assert all("parent_commit" not in case for case in index["cases"])
    rows = bta.real_defect_cases(corpus)
    assert len(rows) == 106
    for row in rows:
        assert row["id"].startswith("us-")
        assert len(row["fix_commit"]) == len(row["parent_commit"]) == 40
        assert row["fix_commit"] != row["parent_commit"]
        assert row["module_path"].endswith(".yaml")
        assert Path(row["provision"]).is_file()
    assert len(bta.real_defect_cases(corpus, judgeable_only=False)) == 158


def test_script_lists_the_defect_cases_without_other_arguments(capsys):
    corpus = Path(__file__).parents[1] / "benchmarks" / "verifier" / "real_defects_v0"
    assert cli.main(["--list-defect-cases", str(corpus)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 106 and all(len(line.split("\t")) == 5 for line in lines)
    assert cli.main([]) == 2
    assert "missing --rules-file" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Subscription CLI backend: subprocess replaced, no model call
# ---------------------------------------------------------------------------


@pytest.fixture
def claude_launcher(monkeypatch):
    calls = []
    state = SimpleNamespace(
        calls=calls, stdout='{"cases": []}', stderr="", returncode=0, cwd_listing=None
    )
    monkeypatch.setattr(
        validator_pipeline, "_protected_reviewer_context_active", lambda: False
    )
    monkeypatch.setattr(
        validator_pipeline,
        "_resolve_local_reviewer_executable",
        lambda name: f"/synthetic/bin/{name}",
    )
    monkeypatch.setattr(
        validator_pipeline,
        "_local_reviewer_environment",
        # Synthetic input only: never inspect the caller's credentials.
        lambda executables: {
            "ANTHROPIC_API_KEY": "synthetic-credit-key",
            "HOME": "/synthetic/home",
        },
    )

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        state.cwd_listing = sorted(Path(kwargs["cwd"]).iterdir())
        return SimpleNamespace(
            stdout=state.stdout, stderr=state.stderr, returncode=state.returncode
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("the shared launcher joins stderr to stdout")

    monkeypatch.setattr(bta.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", forbidden
    )
    return state


def test_cli_backend_reuses_the_reviewer_launch_rules_and_subscription_login(
    claude_launcher,
):
    backend = bta.ClaudeCliBlindAuthorBackend(model="opus", timeout=77)
    assert backend.complete("synthetic prompt") == '{"cases": []}'
    ((cmd, kwargs),) = claude_launcher.calls
    assert cmd[0] == "/synthetic/bin/claude"
    assert cmd[-2:] == ["-p", "synthetic prompt"]
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--print", "--safe-mode", "--no-session-persistence"):
        assert flag in cmd
    assert kwargs["env"] == {"HOME": "/synthetic/home"}
    assert kwargs["timeout"] == 77
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_cli_backend_runs_the_model_in_a_fresh_empty_directory(
    claude_launcher, tmp_path, monkeypatch
):
    (tmp_path / "candidate.yaml").write_text("formula: secret")
    monkeypatch.chdir(tmp_path)
    backend = bta.ClaudeCliBlindAuthorBackend()
    backend.complete("first")
    backend.complete("second")
    first, second = (Path(kwargs["cwd"]) for _, kwargs in claude_launcher.calls)
    assert claude_launcher.cwd_listing == []
    assert first != second
    for cwd in (first, second):
        assert cwd.resolve() != tmp_path.resolve()
        assert tmp_path.resolve() not in cwd.resolve().parents
        assert not cwd.exists()


def test_cli_backend_parses_stdout_only_and_keeps_stderr_for_the_log(claude_launcher):
    claude_launcher.stdout = _answer(CORRECT_CASES[:1])
    claude_launcher.stderr = "warning: synthetic CLI notice\n"
    backend = bta.ClaudeCliBlindAuthorBackend()
    report = bta.run_blind_test_author(
        rulespec_payload=yaml.safe_load(MODULE_TEXT),
        module=MODULE,
        source_text=SOURCE,
        backend=backend,
        runner=_StubRunner(),
        compiled_payload={
            "program": {
                "derived": [{"expr": [{"kind": "input", "name": n} for n in INPUTS]}]
            }
        },
    )
    assert report["response_error"] is None
    assert report["counts"]["agree"] == 1
    assert report["backend"]["stderr"] == "warning: synthetic CLI notice\n"
    assert "synthetic CLI notice" not in report["raw_response"]


def test_cli_backend_never_falls_back_to_another_model_family(
    claude_launcher, monkeypatch
):
    monkeypatch.setattr(
        validator_pipeline, "_resolve_local_reviewer_executable", lambda name: None
    )
    with pytest.raises(bta.BlindAuthorBackendError, match="not found"):
        bta.ClaudeCliBlindAuthorBackend().complete("synthetic prompt")
    assert claude_launcher.calls == []


def test_cli_backend_reports_a_failed_run_and_a_timeout(claude_launcher, monkeypatch):
    claude_launcher.returncode = 1
    claude_launcher.stderr = "synthetic failure"
    with pytest.raises(
        bta.BlindAuthorBackendError, match="exited 1: synthetic failure"
    ):
        bta.ClaudeCliBlindAuthorBackend().complete("synthetic prompt")

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(bta.subprocess, "run", timeout)
    with pytest.raises(bta.BlindAuthorBackendError, match="claude CLI failed"):
        bta.ClaudeCliBlindAuthorBackend().complete("synthetic prompt")


def test_cli_backend_refuses_protected_contexts(claude_launcher, monkeypatch):
    monkeypatch.setattr(
        validator_pipeline, "_protected_reviewer_context_active", lambda: True
    )
    with pytest.raises(bta.BlindAuthorBackendError, match="shadow tool"):
        bta.ClaudeCliBlindAuthorBackend().complete("synthetic prompt")
    assert claude_launcher.calls == []


def test_script_stops_before_the_model_when_the_candidate_does_not_compile(
    tmp_path, capsys
):
    policy_root = tmp_path / "rulespec-us" / "us-az"
    module = policy_root / MODULE_RELATIVE
    module.parent.mkdir(parents=True)
    module.write_text(MODULE_TEXT)
    answer = tmp_path / "answer.json"
    answer.write_text(_answer(CORRECT_CASES))
    base = [
        "--rules-file",
        str(module),
        "--policy-repo-path",
        str(policy_root),
        # Nothing here can compile: there is no engine at this path.
        "--axiom-rules-engine-path",
        str(tmp_path / "no-engine-here"),
        "--source-text-file",
        str(FIXTURES / "installation-fee.source.txt"),
        "--backend",
        "fake",
        "--fake-response-file",
        str(answer),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    for extra in ([], ["--print-prompt"]):
        assert cli.main([*base, *extra]) == 2
        assert "does not compile; no model call was made" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_script_refuses_provider_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    module = tmp_path / "rulespec-us" / "us-az" / MODULE_RELATIVE
    module.parent.mkdir(parents=True)
    module.write_text(MODULE_TEXT)
    code = cli.main(
        [
            "--rules-file",
            str(module),
            "--policy-repo-path",
            str(tmp_path / "rulespec-us" / "us-az"),
            "--axiom-rules-engine-path",
            str(tmp_path / "no-engine"),
            "--source-text-file",
            str(FIXTURES / "installation-fee.source.txt"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("inside", ["rulespec", "encoder"])
def test_script_keeps_results_out_of_git(tmp_path, capsys, inside):
    policy_root = tmp_path / "rulespec-us" / "us-az"
    module = policy_root / MODULE_RELATIVE
    module.parent.mkdir(parents=True)
    module.write_text(MODULE_TEXT)
    out = (
        tmp_path / "rulespec-us" / "results"
        if inside == "rulespec"
        else Path(cli.__file__).resolve().parents[1] / "results-never-created"
    )
    code = cli.main(
        [
            "--rules-file",
            str(module),
            "--policy-repo-path",
            str(policy_root),
            "--axiom-rules-engine-path",
            str(tmp_path),
            "--backend",
            "fake",
            "--out-dir",
            str(out),
        ]
    )
    assert code == 2
    assert "results stay out of git" in capsys.readouterr().err
    assert not out.exists()


# ---------------------------------------------------------------------------
# Real engine: agree, disagree, planted defect
# ---------------------------------------------------------------------------


def _checkout(tmp_path: Path, module_text: str = MODULE_TEXT) -> tuple[Path, Path]:
    root = tmp_path / "rulespec-us"
    module = root / "us-az" / MODULE_RELATIVE
    module.parent.mkdir(parents=True)
    module.write_text(module_text)
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root / "us-az", module


def _report(tmp_path, real_engine, answer, module_text=MODULE_TEXT, **kwargs):
    policy_root, module = _checkout(tmp_path, module_text)
    work = tmp_path / "work"
    work.mkdir()
    runner = bta.EngineCaseRunner(
        rules_file=module,
        policy_repo_path=policy_root,
        axiom_rules_path=real_engine.root,
        work_dir=work,
    )
    assert runner.compile_error is None
    assert runner.module == MODULE
    backend = bta.FakeBlindAuthorBackend(answer)
    report = bta.run_blind_test_author(
        rulespec_payload=yaml.safe_load(module_text),
        module=runner.module,
        source_text=SOURCE,
        backend=backend,
        runner=runner,
        compiled_payload=runner.compiled_payload,
        citation_path="us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-13",
        **kwargs,
    )
    return report, backend


def _statuses(report):
    return {case["name"]: case["status"] for case in report["cases"]}


def test_correct_blind_cases_agree_with_the_correct_rule(tmp_path, real_engine):
    report, backend = _report(tmp_path, real_engine, _answer(CORRECT_CASES))
    assert _statuses(report) == dict.fromkeys(
        (case["name"] for case in CORRECT_CASES), bta.STATUS_AGREE
    )
    assert report["counts"]["agree"] == 4 and report["counts"]["disagree"] == 0
    # The implicit boolean facts came from the compiled program, names only.
    assert sorted(item["name"] for item in report["interface"]["inputs"]) == sorted(
        INPUTS
    )
    assert {item["dtype"] for item in report["interface"]["inputs"]} == {"Boolean"}
    prompt = backend.prompts[0]
    for hidden in ("and not component_is_one_time_deposit", "2026-08-01", "block 13"):
        assert hidden not in prompt


def test_a_blind_case_that_misreads_the_source_is_reported_as_disagreement(
    tmp_path, real_engine
):
    misread = _case("deposit_allowed", "holds", DEPOSIT_SENTENCE, **{INPUTS[3]: True})
    report, _ = _report(tmp_path, real_engine, _answer([CORRECT_CASES[0], misread]))
    assert _statuses(report) == {
        "all_conditions_met": bta.STATUS_AGREE,
        "deposit_allowed": bta.STATUS_DISAGREE,
    }
    disagreement = report["cases"][1]
    assert disagreement["source_sentence"] == DEPOSIT_SENTENCE
    assert "expected holds, got not_holds" in disagreement["issues"][0]
    summary = bta.render_text_summary(report)
    assert "1 agree, 1 disagree, 0 not runnable, 0 rejected" in summary
    assert "[disagree] deposit_allowed (2026-08)" in summary
    assert DEPOSIT_SENTENCE in summary


def test_a_planted_negated_condition_is_caught_by_a_correct_blind_case(
    tmp_path, real_engine
):
    planted = MODULE_TEXT.replace(
        "and not component_is_one_time_deposit", "and component_is_one_time_deposit"
    )
    assert planted != MODULE_TEXT
    report, _ = _report(tmp_path, real_engine, _answer(CORRECT_CASES), planted)
    statuses = _statuses(report)
    # The rule now allows deposits and refuses everything else.
    assert statuses["one_time_deposit"] == bta.STATUS_DISAGREE
    assert statuses["all_conditions_met"] == bta.STATUS_DISAGREE
    caught = next(c for c in report["cases"] if c["name"] == "one_time_deposit")
    assert caught["source_sentence"] == DEPOSIT_SENTENCE
    assert "expected not_holds, got holds" in caught["issues"][0]
    # The defect does not change the interface, so the prompt is the same.
    clean, _ = _report(tmp_path / "clean", real_engine, _answer(CORRECT_CASES))
    assert report["prompt_sha256"] == clean["prompt_sha256"]


def test_a_case_outside_the_formula_versions_is_not_runnable_not_a_disagreement(
    tmp_path, real_engine
):
    early = _case("before_any_version", "holds", ALLOWABLE_SENTENCE, period="2026-01")
    report, _ = _report(tmp_path, real_engine, _answer([early]))
    (case,) = report["cases"]
    assert case["status"] == bta.STATUS_NOT_RUNNABLE
    assert case["not_runnable_kind"] == "no_formula_version_at_period"


@pytest.mark.parametrize(
    ("effective_from", "finding"),
    [
        ("2026-08-01", "not_before_assumed_evidence_floor"),
        # The withheld block-16 shape: a start before the ASSUMED floor.
        ("2025-10-01", "before_assumed_evidence_floor"),
    ],
)
def test_floor_probe_on_the_real_engine(tmp_path, real_engine, effective_from, finding):
    module_text = MODULE_TEXT.replace("'2026-08-01'", f"'{effective_from}'")
    report, _ = _report(
        tmp_path,
        real_engine,
        _answer(CORRECT_CASES[:1]),
        module_text,
        source_as_of="2026-07-17",
    )
    assert report["cases"][0]["status"] == bta.STATUS_AGREE
    floor = report["period_probes"]["assumed_evidence_floor"]
    assert (floor["floor_month"], floor["probe_period"]) == ("2026-08", "2026-07")
    assert floor["finding"] == finding
    assert floor["assumption"] is True


def test_script_refuses_then_reports_the_doctored_shape(tmp_path, real_engine, capsys):
    doctored = (
        MODULE_TEXT.replace(
            "    versions:",
            "    values: [threshold_1500, other]\n    versions:",
        ).replace(
            "and not component_is_one_time_deposit",
            "and not component_is_one_time_deposit\n"
            "          and limit_is_1500_usd_and_rule_starts_2026_08_01 <= 1500",
        )
        + """inputs:
  - name: limit_is_1500_usd_and_rule_starts_2026_08_01
    entity: UtilityExpenseComponent
    dtype: Money
    period: Month
    unit: USD_1500_per_month
"""
    )
    _, module = _checkout(tmp_path, doctored)
    out = tmp_path / "out"
    base = [
        "--rules-file",
        str(module),
        "--axiom-rules-engine-path",
        str(real_engine.root),
        "--source-text-file",
        str(FIXTURES / "installation-fee.source.txt"),
        "--out-dir",
        str(out),
        "--print-prompt",
    ]
    assert cli.main(base) == 2
    refusal = capsys.readouterr().err
    for expected in (
        "formula_number 1500",
        "effective_date 2026-08-01",
        "limit_is_1500_usd_and_rule_starts_2026_08_01",
        "threshold_1500",
        "--allow-interface-leaks",
    ):
        assert expected in refusal
    assert not out.exists()

    free_text = "us-az/manual/x NOTE holds iff fee_ok unless deposit; limit 1500"
    assert cli.main([*base, "--citation", free_text, "--allow-interface-leaks"]) == 2
    assert "citation path" in capsys.readouterr().err
    assert not out.exists()

    assert cli.main([*base, "--allow-interface-leaks"]) == 0
    captured = capsys.readouterr()
    assert "interface leak (allowed)" in captured.err
    assert "declaration dropped" in captured.out
    prompt = (out / "prompt.txt").read_text()
    assert "USD_1500_per_month" not in prompt and "<= 1500" not in prompt


def test_script_writes_the_report_and_exits_zero_on_disagreement(
    tmp_path, real_engine, capsys
):
    planted = MODULE_TEXT.replace(
        "and not component_is_one_time_deposit", "and component_is_one_time_deposit"
    )
    _, module = _checkout(tmp_path, planted)
    answer = tmp_path / "answer.json"
    answer.write_text(_answer(CORRECT_CASES))
    out = tmp_path / "out"
    base = [
        "--rules-file",
        str(module),
        "--axiom-rules-engine-path",
        str(real_engine.root),
        "--source-text-file",
        str(FIXTURES / "installation-fee.source.txt"),
        "--backend",
        "fake",
        "--fake-response-file",
        str(answer),
    ]
    assert cli.main([*base, "--out-dir", str(out), "--print-prompt"]) == 0
    assert "and component_is_one_time_deposit" not in (out / "prompt.txt").read_text()
    assert not (out / "report.json").exists()

    assert cli.main([*base, "--out-dir", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["schema"] == bta.REPORT_SCHEMA
    assert report["counts"]["disagree"] >= 1
    assert "disagree" in (out / "summary.txt").read_text()
    assert "disagree" in capsys.readouterr().out

    # The same stored answer replays against another candidate, with no model.
    _, fixed = _checkout(tmp_path / "fixed")
    replay = tmp_path / "replay"
    assert (
        cli.main(
            [
                "--rules-file",
                str(fixed),
                "--axiom-rules-engine-path",
                str(real_engine.root),
                "--source-text-file",
                str(FIXTURES / "installation-fee.source.txt"),
                "--replay-report",
                str(out / "report.json"),
                "--out-dir",
                str(replay),
            ]
        )
        == 0
    )
    replayed = json.loads((replay / "report.json").read_text())
    assert replayed["backend"]["replayed_response"] is True
    assert replayed["counts"] == {**report["counts"], "agree": 4, "disagree": 0}
