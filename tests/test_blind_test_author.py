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


@pytest.mark.parametrize(
    "leak",
    [
        f"household_gross_income <= income_limit + {PLANTED['formula_number']}",
        f"and not {PLANTED['formula']}",
        f"{PLANTED['summary']} restates the provision.",
    ],
)
def test_guard_finds_formula_lines_and_module_prose(leak):
    payload = yaml.safe_load(PLANTED_MODULE)
    prompt = bta.build_blind_prompt(
        source_text=PLANTED_SOURCE,
        interface=bta.extract_rule_interface(payload, module="us-xx:x"),
    )
    assert bta.find_blindness_violations(prompt, rulespec_payload=payload) == []
    leaking = bta.BlindPrompt(
        instructions=prompt.instructions,
        interface_block=prompt.interface_block + leak + "\n",
        source_text=prompt.source_text,
    )
    assert bta.find_blindness_violations(leaking, rulespec_payload=payload)


def test_guard_finds_dates_and_existing_test_names():
    payload = yaml.safe_load(PLANTED_MODULE)
    prompt = bta.build_blind_prompt(
        source_text=PLANTED_SOURCE,
        interface=bta.extract_rule_interface(payload, module="us-xx:x"),
    )
    for token in (PLANTED["effective_from"], PLANTED["test_name"]):
        leaking = bta.BlindPrompt(
            instructions=prompt.instructions,
            interface_block=prompt.interface_block + f'"x": "{token}"\n',
            source_text=prompt.source_text,
        )
        violations = bta.find_blindness_violations(
            leaking, rulespec_payload=payload, companion_cases=PLANTED_TESTS
        )
        assert violations and token in violations[0]


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
        "all of the following occur:\n\n● The utility installation fee", SOURCE
    )
    assert not bta.sentence_in_source(DEPOSIT_SENTENCE.lower(), SOURCE)
    assert not bta.sentence_in_source(
        "One-time deposits are an allowable NA Utility Expense.", SOURCE
    )


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


def test_earliest_period_needs_a_verbatim_basis():
    def parsed(basis, as_of=None):
        return bta.parse_blind_response(
            _answer(
                [], earliest_supported_period="2026-08", earliest_period_basis=basis
            ),
            interface=_interface(),
            source_text=SOURCE,
            source_as_of=as_of,
        )

    assert parsed(DEPOSIT_SENTENCE).earliest_supported_period == "2026-08"
    assert parsed("source_as_of", "2026-07-17").earliest_supported_period == "2026-08"
    assert parsed("source_as_of").earliest_supported_period is None
    assert parsed("Effective August 2026.").earliest_supported_period is None
    assert parsed("Effective August 2026.").notes


def test_only_comparison_mismatches_count_as_disagreement():
    mismatch = "Test case `a` output `o` expected not_holds, got holds."
    compared = [{"output": "o", "mismatch": mismatch}]
    assert bta.classify_outcome([], [{"output": "o", "mismatch": None}]).status == (
        bta.STATUS_AGREE
    )
    assert bta.classify_outcome([mismatch], compared).status == bta.STATUS_DISAGREE
    assert bta.classify_outcome([f"{mismatch} (row #1)"], compared).status == (
        bta.STATUS_DISAGREE
    )
    binding = bta.classify_outcome(
        [mismatch, "Test case `a` input `x` does not resolve to an input slot."],
        compared,
    )
    assert binding.status == bta.STATUS_NOT_RUNNABLE
    assert binding.not_runnable_kind == "execution_or_binding"
    undated = bta.classify_outcome(
        [
            "Test case `a` execution failed: derived `o` has no formula version at 2026-01-01"
        ],
        [],
    )
    assert undated.not_runnable_kind == "no_formula_version_at_period"
    assert bta.classify_outcome([], []).status == bta.STATUS_NOT_RUNNABLE


# ---------------------------------------------------------------------------
# Subscription CLI backend: subprocess replaced, no model call
# ---------------------------------------------------------------------------


@pytest.fixture
def claude_launcher(monkeypatch):
    calls = []
    state = SimpleNamespace(calls=calls, output='{"cases": []}', returncode=0)
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
        return SimpleNamespace(output=state.output, returncode=state.returncode)

    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", fake_run
    )
    return state


def test_cli_backend_reuses_the_reviewer_launcher_and_subscription_login(
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
    assert kwargs["env_is_final"] is True
    assert kwargs["timeout"] == 77


def test_cli_backend_never_falls_back_to_another_model_family(
    claude_launcher, monkeypatch
):
    monkeypatch.setattr(
        validator_pipeline, "_resolve_local_reviewer_executable", lambda name: None
    )
    with pytest.raises(bta.BlindAuthorBackendError, match="not found"):
        bta.ClaudeCliBlindAuthorBackend().complete("synthetic prompt")
    assert claude_launcher.calls == []


def test_cli_backend_reports_a_failed_run(claude_launcher):
    claude_launcher.returncode = 1
    claude_launcher.output = "synthetic failure"
    with pytest.raises(bta.BlindAuthorBackendError, match="exited 1"):
        bta.ClaudeCliBlindAuthorBackend().complete("synthetic prompt")


def test_cli_backend_refuses_protected_contexts(claude_launcher, monkeypatch):
    monkeypatch.setattr(
        validator_pipeline, "_protected_reviewer_context_active", lambda: True
    )
    with pytest.raises(bta.BlindAuthorBackendError, match="shadow tool"):
        bta.ClaudeCliBlindAuthorBackend().complete("synthetic prompt")
    assert claude_launcher.calls == []


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
        ("2026-08-01", "consistent"),
        # The withheld block-16 shape: a start date the source does not support.
        ("2025-10-01", "candidate_applies_before_blind_earliest_period"),
    ],
)
def test_period_probe_sees_a_rule_that_starts_before_the_source_supports(
    tmp_path, real_engine, effective_from, finding
):
    module_text = MODULE_TEXT.replace("'2026-08-01'", f"'{effective_from}'")
    report, _ = _report(
        tmp_path,
        real_engine,
        _answer(
            CORRECT_CASES[:1],
            earliest_supported_period="2026-08",
            earliest_period_basis="source_as_of",
        ),
        module_text,
        source_as_of="2026-07-17",
    )
    assert report["cases"][0]["status"] == bta.STATUS_AGREE
    assert report["period_probe"]["probe_period"] == "2026-07"
    assert report["period_probe"]["finding"] == finding


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
