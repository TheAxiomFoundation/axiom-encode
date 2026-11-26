"""The encoder names range and threshold bounds instead of folding them into comparisons."""

import re
from pathlib import Path

import pytest

from axiom_encode.prompts import encoder
from tests.test_claude_api_context import _prompt, _prompt_workspace

RULE_SENTENCES = (
    "Name every number that the source states as a rule's range or threshold "
    'bound (both ends of a `9-17` row, the `18` of `18+`, the age in "60 or '
    'older") as a scalar `kind: parameter` with a proof atom, or as an indexed '
    "bound column for an interval table.",
    'Keep the source operator: `9-17` gives `>= 9`; "more than 8" gives `> 8`.',
    'For "a member aged 60 or older" use `count_where(member_of_household, '
    "<Person rule: age >= that parameter>) >= 1`, not an input name, a household "
    "proxy age, or an unused parameter.",
    'Do not name labels ("48 States"), dates, citations or page numbers as parameters.',
)

AGE_SENTENCE = (
    'If the source says someone is "aged 18 or over", "under 25", or similar, '
    "model the legal age predicate as a Person rule that compares age with a named "
    "threshold parameter (`person_age >= <threshold>`), not as a documentary age "
    "input or an unnamed age constant."
)

RULE_BLOCK = """- Name every number that the source states as a rule's range or threshold
  bound (both ends of a `9-17` row, the `18` of `18+`, the age in "60 or
  older") as a scalar `kind: parameter` with a proof atom, or as an indexed
  bound column for an interval table. Keep the source operator: `9-17` gives
  `>= 9`; "more than 8" gives `> 8`. For "a member aged 60 or older" use
  `count_where(member_of_household, <Person rule: age >= that parameter>) >= 1`,
  not an input name, a household proxy age, or an unused parameter. Do not
  name labels ("48 States"), dates, citations or page numbers as parameters.
"""

AGE_BLOCK = """- If the source says someone is "aged 18 or over", "under 25", or similar,
  model the legal age predicate as a Person rule that compares age with a named
  threshold parameter (`person_age >= <threshold>`), not as a documentary age
  input or an unnamed age constant.
"""

ENCODER_ANCHOR = "  identities should remain as formula literals.\n"
EVAL_ANCHOR = (
    "consuming formulas should reference names rather than embedding the policy "
    "literal.\n"
)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _eval_prompt(tmp_path: Path, backend: str = "openai") -> str:
    workspace, items = _prompt_workspace(tmp_path, ["context\n"])
    return _prompt(workspace, items, backend)


def _copies(tmp_path: Path) -> dict[str, str]:
    return {
        "encoder": encoder.get_encoder_prompt(
            "FNS SNAP FY2027 COLA page 4",
            "policies/usda/fns/snap-fy2027-cola/page-4.yaml",
            corpus_citation_path="us/guidance/usda/fns/snap-fy2027-cola/page-4",
            require_complete_source_unit=True,
        ),
        "eval-openai": _eval_prompt(tmp_path / "openai"),
        "eval-claude-api": _eval_prompt(tmp_path / "claude", "claude-api"),
    }


@pytest.mark.parametrize("sentence", RULE_SENTENCES)
def test_each_rule_sentence_appears_once_in_every_prompt_copy(tmp_path, sentence):
    for name, prompt in _copies(tmp_path).items():
        assert _flat(prompt).count(sentence) == 1, name


def test_both_copies_carry_the_identical_rule_block(tmp_path):
    encoder_source = Path(encoder.__file__).read_text()
    from axiom_encode.harness import evals

    evals_source = Path(evals.__file__).read_text()
    assert encoder_source.count(RULE_BLOCK) == 1
    assert encoder_source.count(AGE_BLOCK) == 1
    assert evals_source.count(AGE_BLOCK) == 1
    assert evals_source.count(RULE_BLOCK) == 1
    assert encoder.ENCODER_PROMPT.count(RULE_BLOCK) == 1
    assert _eval_prompt(tmp_path).count(RULE_BLOCK) == 1


def test_rule_sits_with_the_numeric_concept_guidance(tmp_path):
    assert ENCODER_ANCHOR + AGE_BLOCK + RULE_BLOCK in encoder.ENCODER_PROMPT
    assert encoder._NUMERIC_GROUNDING.count(RULE_BLOCK) == 1
    assert EVAL_ANCHOR + RULE_BLOCK in _eval_prompt(tmp_path)


@pytest.mark.parametrize(
    "retired",
    [
        "instead of inventing documentary age constants",
        "Name every number that the source states as a range or threshold bound: both",
        "An input named like `*_aged_60_or_older` does not name 60",
    ],
)
def test_retired_wording_is_gone_from_every_prompt_copy(tmp_path, retired):
    for name, prompt in _copies(tmp_path).items():
        assert _flat(retired) not in _flat(prompt), name


def test_age_predicate_guidance_appears_once_in_every_prompt_copy(tmp_path):
    for name, prompt in _copies(tmp_path).items():
        assert _flat(prompt).count(AGE_SENTENCE) == 1, name


def test_rule_sentences_are_the_rule_block():
    assert _flat(RULE_BLOCK).strip("- ").replace(" - ", " ") == " ".join(RULE_SENTENCES)


def test_prompt_version_is_bumped_for_the_new_rule():
    assert encoder.__version__ == 7
