"""The State-set amount exception is emitted only for federal CFR units.

Both prompt copies (the shared encoder prompt and the eval prompt) insert the
same rule directly after the cross-reference defer rule, and only when the
unit is a `us/regulation/<title>/<part>/<section>` provision. Every other
unit gets the prompt without it, byte for byte.
"""

import re
from pathlib import Path

import pytest

from axiom_encode.harness import evals
from axiom_encode.prompts import encoder as encoder_prompt

CFR_UNIT = "us/regulation/42/435/725"
NON_CFR_UNITS = [
    "us-az/manual/des/faa5/dependent-care-expense/block-4",
    "us/statute/42/1396a",
    "us/guidance/irs/pub-501/2025",
    "us-co/regulation/10-ccr-2506-1/4.804.1",
    "uk/statute/2020/1/section/1",
]

# Each sentence of the rule; removing any one of them fails a test.
SENTENCES = (
    "Exception for State-set amounts (federal CFR units only): it covers a "
    "value that this source applies from another provision when no copied "
    "context module exports that value and either (a) the citing phrase names "
    "the State or the agency as the actor that sets it, with a verb such as "
    '"establishes", "elects", "specifies", "sets", "selects", or "chooses" '
    '(for example household income "at or below the income standard '
    'established by the agency in its State plan"), or (b) the phrase cites a '
    "provision whose copied module takes that value as a local input and "
    'exports no rule for it (for example "the medically needy income standard '
    'established under § 435.811", whose module lists '
    "`medically_needy_income_standard_amount` as an input slot).",
    "Then encode the branch; do not defer it for that reason, also when the "
    "cited module is deferred.",
    "This exception takes precedence over the rule that defers when a cited "
    "sibling file is deferred.",
    "Encode literally every number, percentage of the Federal poverty level, "
    "or dollar amount that this source or a cited federal provision states; "
    "such a value is never an input.",
    'A federally stated floor or ceiling, such as "at least $30", stays a '
    "literal and is applied with max or min to the input.",
    "Federal text that limits what the State may set (a floor or ceiling on "
    "the State's own standard) is encoded as a rule about the State plan, not "
    "as an input that replaces that limit.",
    "Declare the State-set value in the document-root `inputs` list with "
    "`entity`, `dtype`, `period`, `unit` for money, and a `description` that "
    "names the provision under which the State or agency sets it.",
    "For case (b) use the cited module's input identifier; otherwise use a "
    "descriptive name.",
    "Give the rule that uses the State-set value a proof atom whose excerpt is "
    "the citing phrase from this unit's source text.",
    "Encode the branch's own comparisons, max/min, and conditions on that "
    "value, and set the input in every companion test.",
)
RULE_TAIL = (
    "import a real export, defer the affected executable surface, or encode a "
    "source-grounded overriding branch that avoids it."
)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _eval_prompt(tmp_path: Path, citation: str, items=(), backend="codex") -> str:
    work = tmp_path / f"prompt-{abs(hash((citation, backend)))}"
    work.mkdir(exist_ok=True)
    source = work / "source.txt"
    source.write_text("The annual allowance is $100.\n")
    workspace = evals.EvalWorkspace(work, source, work / "manifest.json")
    return evals._build_eval_prompt(
        citation,
        "cold",
        workspace,
        list(items),
        "target.yaml",
        include_tests=True,
        runner_backend=backend,
    )


def _encoder_prompt(citation: str) -> str:
    return encoder_prompt.get_encoder_prompt(citation, "target.yaml")


@pytest.fixture
def cfr_copies(tmp_path):
    return {
        "encoder": _flat(_encoder_prompt(CFR_UNIT)),
        "evals": _flat(_eval_prompt(tmp_path, CFR_UNIT)),
    }


@pytest.mark.parametrize("copy", ["encoder", "evals"])
@pytest.mark.parametrize("sentence", SENTENCES)
def test_each_sentence_is_in_both_cfr_prompt_copies(cfr_copies, copy, sentence):
    assert cfr_copies[copy].count(sentence) == 1


@pytest.mark.parametrize("copy", ["encoder", "evals"])
def test_rule_follows_the_cross_reference_defer_rule(cfr_copies, copy):
    text = cfr_copies[copy]
    end = text.index(RULE_TAIL) + len(RULE_TAIL)
    assert text[end:].startswith(" - " + SENTENCES[0])
    positions = [text.index(sentence) for sentence in SENTENCES]
    assert positions == sorted(positions)


@pytest.mark.parametrize("citation", NON_CFR_UNITS)
def test_non_cfr_encoder_prompt_is_unchanged(citation):
    prompt = _encoder_prompt(citation)
    assert "State-set" not in prompt
    assert prompt.startswith(encoder_prompt.ENCODER_PROMPT + "\n")
    assert encoder_prompt.with_cfr_state_set_amount_rule(prompt, citation) is prompt


@pytest.mark.parametrize("citation", NON_CFR_UNITS)
def test_non_cfr_eval_prompt_has_no_state_set_rule(tmp_path, citation):
    assert "State-set" not in _eval_prompt(tmp_path, citation)


def test_shared_encoder_prompt_constant_does_not_carry_the_rule():
    assert "State-set" not in encoder_prompt.ENCODER_PROMPT


def test_shared_sibling_carve_out_rule_is_unchanged_for_other_units():
    sibling_rule = (
        "If the cited sibling file is deferred, empty, unsupported, or missing a "
        "usable exported rule and the carve-out changes the result, defer the "
        "affected executable output"
    )
    assert sibling_rule in _flat(encoder_prompt.ENCODER_PROMPT)
    assert "takes precedence" not in encoder_prompt.ENCODER_PROMPT


def test_rule_does_not_use_the_state_plan_constraint_example():
    assert "§ 435.110(c)" not in encoder_prompt.CFR_STATE_SET_AMOUNT_RULE


def test_cfr_prompt_is_the_non_cfr_prompt_plus_the_rule():
    plain = encoder_prompt.ENCODER_PROMPT
    gated = encoder_prompt.with_cfr_state_set_amount_rule(plain, CFR_UNIT)
    tail = encoder_prompt.CROSS_REFERENCE_DEFER_RULE_TAIL
    assert gated == plain.replace(tail, tail + encoder_prompt.CFR_STATE_SET_AMOUNT_RULE)


def test_rule_lands_after_the_last_anchor_when_context_repeats_it(tmp_path):
    context = tmp_path / "context.txt"
    context.write_text(
        "head\n" + encoder_prompt.CROSS_REFERENCE_DEFER_RULE_TAIL + "tail\n"
    )
    item = evals.EvalContextFile(
        source_path=str(context),
        workspace_path="context.txt",
        import_path="us:context/0",
        kind="precedent",
    )
    work = tmp_path / "ws"
    work.mkdir()
    (work / "context.txt").write_text(context.read_text())
    source = work / "source.txt"
    source.write_text("Text.\n")
    prompt = evals._build_eval_prompt(
        CFR_UNIT,
        "cold",
        evals.EvalWorkspace(work, source, work / "manifest.json"),
        [item],
        "target.yaml",
        include_tests=True,
        runner_backend="openai",
    )
    tail = encoder_prompt.CROSS_REFERENCE_DEFER_RULE_TAIL
    assert prompt.count(tail) == 2
    first = prompt.index(tail) + len(tail)
    assert not prompt[first:].startswith("- Exception for State-set amounts")
    last = prompt.rindex(tail) + len(tail)
    assert prompt[last:].startswith("- Exception for State-set amounts")


def test_missing_anchor_fails_closed():
    with pytest.raises(ValueError, match="cross-reference defer rule"):
        encoder_prompt.with_cfr_state_set_amount_rule("no rules here", CFR_UNIT)


def test_gate_is_the_shared_unit_key():
    assert evals._us_cfr_unit_section_key(CFR_UNIT) == ("42", "435", "725")
    assert encoder_prompt.us_cfr_unit_section_key(CFR_UNIT) == ("42", "435", "725")
    for citation in NON_CFR_UNITS:
        assert encoder_prompt.us_cfr_unit_section_key(citation) is None


def test_cited_regulation_guidance_points_to_the_exception(tmp_path):
    root = tmp_path / "repos" / "rulespec-us" / "us"
    module = root / "regulations" / "42-cfr" / "435" / "811.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "format: rulespec/v1\nrules:\n  - name: complies\n    kind: derived\n"
        "    entity: Person\n    dtype: Judgment\n    period: Month\n"
        "    versions:\n      - effective_from: '2014-01-01'\n"
        "        formula: medically_needy_income_standard_amount > 0\n"
    )
    item = evals.EvalContextFile(
        source_path=str(module),
        workspace_path="context/regulations/42-cfr/435/811.yaml",
        import_path="us:regulations/42-cfr/435/811",
        kind="implementation_precedent",
    )
    guidance = _flat(
        evals._format_cited_regulation_context_guidance(
            CFR_UNIT, "the standard the agency establishes under § 435.811", [item]
        )
    )
    assert guidance.endswith(
        "When the source applies a State-set amount that a cited module only "
        "takes as a local input, follow the exception for State-set amounts. "
    ) or guidance.endswith(
        "When the source applies a State-set amount that a cited module only "
        "takes as a local input, follow the exception for State-set amounts."
    )


def test_encoder_prompt_version_is_bumped_for_the_gated_rule():
    assert encoder_prompt.__version__ == 7
