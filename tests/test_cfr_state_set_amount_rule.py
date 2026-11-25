"""The State-set amount exception is emitted only for allow-listed CFR parts.

Both prompt copies (the shared encoder prompt and the eval prompt) insert the
same rule directly after the cross-reference defer rule, and only when the
unit is a federal unit of a CFR part in `CFR_STATE_SET_AMOUNT_PARTS`
(42 CFR part 435). Every other unit, also every other CFR unit, gets the
prompt without it, byte for byte.
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
# Federal CFR units outside the allow-list: the rule was not measured there.
OTHER_CFR_UNITS = [
    "us/regulation/7/273/9",
    "us/regulation/7/273/10",
    "us/regulation/42/436/831",
    "us/regulation/42/457/608",
    "us/regulation/20/416/1100",
]

# Each sentence of the rule; removing any one of them fails a test.
FORMS_SENTENCE = (
    "Exception for State-set amounts (federal units of 42 CFR part 435 only): "
    "it covers an amount that this source applies when no copied context "
    "module exports that amount and this unit's own source text attributes "
    "the amount to the State in one of exactly two forms: (a) the text names "
    "the State, the State agency, or the agency (in 42 CFR part 435 the agency "
    "is the State Medicaid agency) as the actor of one of these verbs, with "
    'the amount as the thing acted on: "establishes", "sets", "elects", '
    '"specifies", "selects", "chooses", or "determines ... in its State plan" '
    '(for example household income "at or below the income standard '
    'established by the agency in its State plan"); or (b) the text names a '
    "level in effect on a stated calendar date under the State's own plan: "
    "the State's AFDC payment standard or AFDC income standard, or a level "
    "under the Medicaid State plan or a waiver of the State plan (for example "
    '"the State\'s AFDC payment standard in effect as of July 16, 1996").'
)
CROSS_REFERENCE_SENTENCE = (
    "For an amount that meets form (a) or (b), and for no other value, these "
    "rules of this prompt do not apply, wherever they appear, before or after "
    "this exception: the rule that a cited context file that is deferred, "
    "unsupported, or empty gets no local cross-reference input; the rule "
    "against local `section_...` or `subsection_...` inputs for a cited "
    "source; the rule against a local cross-reference fact when the context "
    "lacks the needed export; and the rule that defers when a cited sibling "
    "file is deferred."
)
BINDING_SENTENCES = (
    "Each State-set input needs its own source phrase.",
    "In that phrase the amount itself is the object of the verb in form (a), "
    "or is the level named in form (b), and the input's name uses the words "
    "of that amount.",
    "One phrase supports one input; a phrase for one amount does not support "
    "a second amount of the same rule.",
    "When the phrase defines the amount from another quantity (a percentage "
    "of a benefit amount or of the Federal poverty level, an adjustment by a "
    "price index), only the parameter that the State chooses, such as the "
    "percentage that the State elects, can be the input.",
    "The other quantity is imported, encoded literally, or deferred.",
)
# The rules that the exception narrows or cites, as each prompt copy words
# them. The eval prompt states some of them after the insertion point.
REFERENCED_RULES = {
    "encoder": (
        "Do not preserve, rename, or recreate a local cross-reference input "
        "for that cited source.",
        "do not create a local `section_...` or `subsection_...` placeholder "
        "input for that cited source.",
        "do not preserve, rename, or recreate the local cross-reference fact;",
        "If the cited sibling file is deferred, empty, unsupported, or missing "
        "a usable exported rule",
        "Federal provisions that authorize state agencies to set a value create "
        "the delegated slot; encode those source graph records with "
        "`source_relation.type: delegates`.",
    ),
    "evals": (
        "Do not preserve, rename, or recreate a local cross-reference input "
        "for that cited source.",
        "do not create local `section_...` or `subsection_...` inputs for "
        "those cited sources.",
        "do not preserve, rename, or recreate the local cross-reference fact;",
        "If the cited sibling file is deferred, empty, unsupported, or missing "
        "a usable exported rule",
        "Federal provisions that authorize state agencies to set a value create "
        "the delegated slot; encode those source graph records with "
        "`source_relation.type: delegates`.",
    ),
}
FEDERAL_SENTENCE = (
    "That holds for every amount that a federal statute, a federal regulation, "
    "a federal agency, or a federal formula sets, also when the amount differs "
    "by State and also when this unit states no number: import it when a "
    "copied context module exports it; otherwise defer that branch with a "
    "reason that names that federal provision."
)
DELEGATES_SENTENCE = (
    "Emit the `kind: source_relation` record with `source_relation.type: "
    "delegates` that the source-relation rules of this prompt require for it."
)
SENTENCES = (
    FORMS_SENTENCE,
    "No other wording qualifies.",
    "A phrase that only calls the amount the State's, or that only says the "
    "amount is established under a cited section, does not name the State as "
    "the actor that sets it.",
    "Then encode the branch; do not defer it for that reason, also when the "
    "cited module is deferred.",
    *BINDING_SENTENCES,
    "This exception is narrow.",
    CROSS_REFERENCE_SENTENCE,
    "Every other value stays under those rules.",
    "An amount that the source text does not attribute to the State in form "
    "(a) or (b) is not a State-set input.",
    FEDERAL_SENTENCE,
    "Encode literally every number, percentage of the Federal poverty level, "
    "or dollar amount that this source or a cited federal provision states; "
    "such a value is never an input.",
    'A federally stated floor or ceiling, such as "at least $30", stays a '
    "literal and is applied with max or min to the input.",
    "Federal text that limits what the State may set (a floor or ceiling on "
    "the State's own standard) is encoded as a rule about the State plan, not "
    "as an input that replaces that limit.",
    "Record the delegation: the federal text that lets the State set the "
    "amount creates the delegated slot.",
    DELEGATES_SENTENCE,
    "Do not use `sets` or `implements` here; those belong to the State "
    "authority that fills the slot.",
    "Declare the State-set value in the document-root `inputs` list with "
    "`entity`, `dtype`, `period`, `unit` for money, and a `description` that "
    "names the provision under which the State or agency sets it.",
    "When the source cites a section of this CFR part whose copied module "
    "lists the value as a local input slot under a name made of those words, "
    "use that module's input identifier; otherwise use a name made of the "
    "words of that amount.",
    "Give the rule that uses the State-set value one proof atom per State-set "
    "input whose excerpt is, word for word, that input's phrase of this "
    "unit's source text.",
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
    assert text.count("Exception for State-set amounts") == 1
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


@pytest.mark.parametrize("citation", OTHER_CFR_UNITS)
def test_cfr_unit_outside_the_allow_list_gets_no_rule(tmp_path, citation):
    """7 CFR 273 ("the maximum allotment established under § 273.10(e)(4)"),
    42 CFR 436/457 and 20 CFR 416 are federal CFR units, and get no rule."""
    assert encoder_prompt.us_cfr_unit_section_key(citation) is not None
    assert encoder_prompt.cfr_state_set_amount_rule_applies(citation) is False
    prompt = _encoder_prompt(citation)
    assert "State-set" not in prompt
    assert prompt.startswith(encoder_prompt.ENCODER_PROMPT + "\n")
    assert encoder_prompt.with_cfr_state_set_amount_rule(prompt, citation) is prompt
    assert "State-set" not in _eval_prompt(tmp_path, citation)


def test_allow_list_is_the_measured_part_only():
    assert encoder_prompt.CFR_STATE_SET_AMOUNT_PARTS == frozenset({("42", "435")})
    assert (
        encoder_prompt.CFR_PARTS_WHERE_AGENCY_IS_THE_STATE
        <= encoder_prompt.CFR_STATE_SET_AMOUNT_PARTS
    )
    assert encoder_prompt.cfr_state_set_amount_rule_applies(CFR_UNIT) is True
    assert encoder_prompt.cfr_state_set_amount_rule_applies("") is False
    for citation in NON_CFR_UNITS:
        assert encoder_prompt.cfr_state_set_amount_rule_applies(citation) is False


def test_gate_follows_the_allow_list(monkeypatch):
    """The gate is the allow-list: adding a part emits the rule for it."""
    plain = encoder_prompt.ENCODER_PROMPT
    unit = "us/regulation/42/436/831"
    assert encoder_prompt.with_cfr_state_set_amount_rule(plain, unit) is plain
    monkeypatch.setattr(
        encoder_prompt,
        "CFR_STATE_SET_AMOUNT_PARTS",
        frozenset({("42", "435"), ("42", "436")}),
    )
    assert encoder_prompt.CFR_STATE_SET_AMOUNT_RULE in (
        encoder_prompt.with_cfr_state_set_amount_rule(plain, unit)
    )
    assert (
        encoder_prompt.with_cfr_state_set_amount_rule(plain, "us/regulation/7/273/9")
        is plain
    )


def test_acceptance_is_a_closed_positive_condition():
    """A closed verb list, no possessive test, no list of federal names."""
    rule = _flat(encoder_prompt.CFR_STATE_SET_AMOUNT_RULE)
    assert "such as" not in FORMS_SENTENCE
    assert "one of exactly two forms" in rule
    assert "names the State's own standard or level" not in rule
    assert "established under a section of this CFR part whose" not in rule
    for federal_name in ("SSI", "Supplemental Security Income", "benefit rate"):
        assert federal_name not in rule


@pytest.mark.parametrize("copy", ["encoder", "evals"])
def test_every_rule_the_exception_refers_to_is_in_that_prompt_copy(cfr_copies, copy):
    """The cross-reference is true in both copies: each rule it names or cites
    is in the same prompt, exactly once, in that copy's own wording."""
    text = cfr_copies[copy]
    assert text.count(CROSS_REFERENCE_SENTENCE) == 1
    assert text.count(DELEGATES_SENTENCE) == 1
    for rule in REFERENCED_RULES[copy]:
        assert text.count(rule) == 1, rule


def test_cross_reference_does_not_depend_on_the_position_of_the_rule(cfr_copies):
    """The eval prompt states the delegates, `section_...` and sibling-defer
    rules after the exception, the encoder prompt before it. The exception
    therefore names no position."""
    rule = _flat(encoder_prompt.CFR_STATE_SET_AMOUNT_RULE)
    for positional in ("rules above", "rule above", "directly above", "rules below"):
        assert positional not in rule
    assert "wherever they appear, before or after this exception" in rule
    sibling, delegates = REFERENCED_RULES["evals"][3], REFERENCED_RULES["evals"][4]
    for copy, after in (("encoder", False), ("evals", True)):
        text = cfr_copies[copy]
        start = text.index("Exception for State-set amounts")
        assert (text.index(sibling) > start) is after
        assert (text.index(delegates) > start) is after


def test_existing_rules_are_not_reworded_for_any_unit():
    plain = _flat(encoder_prompt.ENCODER_PROMPT)
    gated = _flat(
        encoder_prompt.with_cfr_state_set_amount_rule(
            encoder_prompt.ENCODER_PROMPT, CFR_UNIT
        )
    )
    assert gated.replace(_flat(encoder_prompt.CFR_STATE_SET_AMOUNT_RULE), "") == plain


def test_eval_prompt_differs_from_another_cfr_unit_only_by_the_rule(tmp_path):
    """Removing the rule from the 435 eval prompt leaves no State-set text."""
    prompt = _eval_prompt(tmp_path, CFR_UNIT)
    assert prompt.count(encoder_prompt.CFR_STATE_SET_AMOUNT_RULE) == 1
    assert "State-set" not in prompt.replace(
        encoder_prompt.CFR_STATE_SET_AMOUNT_RULE, ""
    )


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


def test_cited_regulation_guidance_has_no_pointer_outside_the_allow_list(tmp_path):
    root = tmp_path / "repos" / "rulespec-us" / "us"
    module = root / "regulations" / "7-cfr" / "273" / "10.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "format: rulespec/v1\nrules:\n  - name: allotment\n    kind: derived\n"
        "    entity: Household\n    dtype: Money\n    period: Month\n"
        "    versions:\n      - effective_from: '2014-01-01'\n"
        "        formula: maximum_allotment_amount * 1\n"
    )
    item = evals.EvalContextFile(
        source_path=str(module),
        workspace_path="context/regulations/7-cfr/273/10.yaml",
        import_path="us:regulations/7-cfr/273/10",
        kind="implementation_precedent",
    )
    guidance = _flat(
        evals._format_cited_regulation_context_guidance(
            "us/regulation/7/273/9",
            "the maximum allotment established under § 273.10(e)(4)",
            [item],
        )
    )
    assert "Source cites" in guidance
    assert "State-set" not in guidance
    assert guidance.rstrip().endswith("use its bare name in the formula.")


def test_encoder_prompt_version_is_bumped_for_the_gated_rule():
    assert encoder_prompt.__version__ == 7
