"""One encoder prompt bullet: never import from a composition module.

The rules engine refuses a module that declares ``module.kind`` on its atomic
surface, and the import targets of a file the encoder writes load on that
surface. No test here can show the refusal without the engine; the tests pin
the text, its position, and that both prompt copies (``prompts/encoder.py``
and ``harness/evals.py``) carry it.

Two other bullets were proposed with this one and withdrawn after review: one
on which text a list-condition proof excerpt quotes, and one on a
``dtype: Count`` input for a count threshold. The last tests pin that neither
prompt carries them, so adding either again is a deliberate change.
"""

from __future__ import annotations

from axiom_encode.harness import evals
from axiom_encode.prompts.encoder import _NAMING_PROTOCOL, ENCODER_PROMPT

COMPOSITION_IMPORT = (
    "- Never import from a module that declares `module.kind: composition`. "
    "In a file you write, the rules engine loads each import target as an "
    "atomic module, and an atomic module must not declare `module.kind`, so "
    "the file fails to compile. Treat the exports of such a module as not "
    "available for import; no other import rule changes."
)
BEFORE = "- Never introduce an import cycle."
AFTER = "- Never create a derived rule whose formula references that same rule's name."


def _flat(text: str) -> str:
    return " ".join(text.split())


def _bullet(flat_text: str, start: str) -> str:
    begin = flat_text.index(start)
    return flat_text[begin : flat_text.index(" - ", begin + 2)]


def _eval_prompt(tmp_path) -> str:
    source = tmp_path / "source.txt"
    source.write_text("The annual allowance is 100.\n")
    workspace = evals.EvalWorkspace(tmp_path, source, tmp_path / "manifest.json")
    return _flat(
        evals._build_eval_prompt(
            "us/guidance/example/allowance",
            "cold",
            workspace,
            [],
            "policies/example/allowance.yaml",
            include_tests=True,
        )
    )


def test_encoder_prompt_carries_the_composition_import_bullet():
    block = _flat(_NAMING_PROTOCOL)
    assert _bullet(block, COMPOSITION_IMPORT[:60]) == COMPOSITION_IMPORT
    assert block.index(BEFORE) < block.index(COMPOSITION_IMPORT) < block.index(AFTER)
    # The bullet directly follows its neighbor: no other bullet lies between.
    assert _bullet(block, BEFORE) + " " + COMPOSITION_IMPORT in block
    prompt = _flat(ENCODER_PROMPT)
    assert block in prompt
    assert prompt.count(COMPOSITION_IMPORT) == 1


def test_eval_prompt_carries_the_same_composition_import_bullet(tmp_path):
    prompt = _eval_prompt(tmp_path)
    assert prompt.count(COMPOSITION_IMPORT) == 1
    assert prompt.index(BEFORE) < prompt.index(COMPOSITION_IMPORT) < prompt.index(AFTER)
    assert _bullet(prompt, BEFORE) + " " + COMPOSITION_IMPORT in prompt


def test_composition_bullet_names_only_the_import_of_a_composition_module():
    # One prohibition and its reason; nothing about boundary inputs, deferral
    # or proof atoms, which their own rules keep.
    assert COMPOSITION_IMPORT.endswith("no other import rule changes.")
    # The engine claim covers a file the encoder writes, not every load.
    assert "In a file you write, the rules engine loads each import target" in (
        COMPOSITION_IMPORT
    )
    lowered = COMPOSITION_IMPORT.lower()
    for word in ("boundary", "defer", "proof", "cycle", "prefix", "hash", "every"):
        assert word not in lowered, word


def test_neither_prompt_tells_which_text_a_list_condition_excerpt_quotes(tmp_path):
    # Withdrawn: with item-only excerpts the gates-collapsed check does not
    # see a dropped item of a conjunctive list.
    for prompt in (_flat(ENCODER_PROMPT), _eval_prompt(tmp_path)):
        assert "the line that introduces the list" not in prompt
        assert "not the chapeau" not in prompt


def test_neither_prompt_carries_the_withdrawn_count_threshold_guidance(tmp_path):
    # Withdrawn: whether a `dtype: Count` input passes the boundary and pair
    # checks depends on the source wording, negation, sentence shape and the
    # input name, which a prompt line cannot control.
    for prompt in (_flat(ENCODER_PROMPT), _eval_prompt(tmp_path)):
        assert "compares a count of persons or things" not in prompt
        assert "whole-number count" not in prompt
        assert "`dtype: Count` input when no import or relation" not in prompt
        assert "`dtype: Boolean` input that restates the comparison" not in prompt
        assert "one comparison of the count with N" not in prompt
        assert "a count of listed conditions" not in prompt
