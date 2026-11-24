"""CFR cited-section context and the State-established amount prompt rule.

Part 1: a CFR unit's source text is read with one closed citation grammar
(`N CFR P.S`, `§ P.S`), each hit is re-parsed by the corpus resolver, and the
cited sections' RuleSpec modules join the cross-section context (bounded, in
source order, live modules preferred over a deferred section file).

Part 2: both encoder prompt copies tell the encoder to encode a branch that
applies a State-established amount as an input instead of deferring it.
"""

import re
from pathlib import Path

import pytest

from axiom_encode import corpus_resolver
from axiom_encode.harness import evals
from axiom_encode.prompts import encoder as encoder_prompt
from tests.test_evals import _canonical_rulespec_content_root

UNIT = "us/regulation/42/435/725"


def _live(name: str, *, inputs: tuple[str, ...] = ()) -> str:
    formula = " and ".join(inputs) if inputs else "true"
    return (
        "format: rulespec/v1\n"
        "rules:\n"
        f"  - name: {name}\n"
        "    kind: derived\n"
        "    entity: Person\n"
        "    dtype: Judgment\n"
        "    period: Month\n"
        "    versions:\n"
        "      - effective_from: '2014-01-01'\n"
        f"        formula: {formula}\n"
    )


_DEFERRED = (
    "format: rulespec/v1\n"
    "module:\n"
    "  status: deferred\n"
    "  summary: Deferred until a later encoding.\n"
    "rules: []\n"
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def tree(tmp_path):
    root = _canonical_rulespec_content_root(tmp_path / "repos", "us")
    part = root / "regulations" / "42-cfr" / "435"
    files = {
        "811": _write(
            part / "811.yaml",
            _live(
                "medically_needy_income_standard_complies",
                inputs=("medically_needy_income_standard_amount",),
            ),
        ),
        "120": _write(part / "120.yaml", _DEFERRED),
        "120/ssi": _write(
            part / "120" / "ssi-mandatory-group.yaml", _live("ssi_group_eligible")
        ),
        "230": _write(part / "230.yaml", _live("optional_group_eligible")),
        "230/a": _write(part / "230" / "a.yaml", _live("optional_group_a")),
        "230/b/1": _write(part / "230" / "b" / "1.yaml", _live("optional_group_b1")),
        "230/c": _write(part / "230" / "c.yaml", _DEFERRED),
        "236": _write(part / "236.yaml", _DEFERRED),
        "123": _write(part / "123.yaml", _live("section_123_rule")),
        "123/b": _write(part / "123" / "b.yaml", _live("section_123_b_rule")),
        "725": _write(part / "725.yaml", _live("own_section_rule")),
        "416-1100": _write(
            root / "regulations" / "20-cfr" / "416" / "1100.yaml",
            _live("ssi_income_rule"),
        ),
    }
    _write(part / "811.test.yaml", "[]\n")
    return root, files


def _labels(source_text: str, title: str = "42") -> list[str]:
    return [cited.label for cited in evals._iter_cited_cfr_sections(source_text, title)]


class TestCfrCitationGrammar:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("under § 435.811.", ["42 CFR 435.811"]),
            ("under §435.110(c)(1) of this part", ["42 CFR 435.110(c)(1)"]),
            ("see 42 CFR 435.811 and", ["42 CFR 435.811"]),
            ("see 42 C.F.R. 435.811 and", ["42 CFR 435.811"]),
            ("see 42 CFR § 435.4(b)", ["42 CFR 435.4(b)"]),
            ("see 20 CFR 416.1100", ["20 CFR 416.1100"]),
            ("standard (§ 435.120).", ["42 CFR 435.120"]),
            ("§ 435.960 applies", ["42 CFR 435.960"]),
        ],
    )
    def test_reads_cfr_display_and_section_sign_forms(self, text, expected):
        assert _labels(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "§§ 435.130 through 435.134",
            "under § 435 of this chapter",
            "42 CFR part 435",
            "section 435.811",
            "§ 435.811a",
            "26 CFR 1.414(v)-1",
            "42CFR435.811",
            "435.811",
            "§ 435.811.5",
        ],
    )
    def test_does_not_read_forms_outside_the_grammar(self, text):
        assert _labels(text) == []

    def test_section_sign_takes_the_unit_title(self):
        assert _labels("under § 273.9", title="7") == ["7 CFR 273.9"]

    def test_drops_a_hit_the_resolver_parses_differently(self, monkeypatch):
        monkeypatch.setattr(
            corpus_resolver,
            "normalize_corpus_identifier",
            lambda identifier: "us/regulation/42/999/1",
        )
        assert _labels("under § 435.811") == []

    def test_drops_a_hit_the_resolver_rejects(self, monkeypatch):
        def reject(identifier):
            raise corpus_resolver.InvalidCorpusCitationError(identifier)

        monkeypatch.setattr(corpus_resolver, "normalize_corpus_identifier", reject)
        assert _labels("under § 435.811") == []

    def test_cited_sections_exclude_own_section_and_repeat_citations(self):
        cited = evals._cited_cfr_sections_for_unit(
            UNIT,
            "§ 435.725(b) and § 435.811, § 435.811(a) and § 435.120",
        )
        assert [item.label for item in cited] == ["42 CFR 435.811", "42 CFR 435.120"]

    @pytest.mark.parametrize(
        "unit",
        [
            "us/statute/42/1396a",
            "us-az/manual/des/faa5/dependent-care-expense/block-4",
            "us-co/regulation/10-ccr-2506-1/4.804.1",
            "us/statute/26/32/1",
        ],
    )
    def test_non_cfr_units_read_no_cfr_citations(self, unit):
        assert evals._cited_cfr_sections_for_unit(unit, "under § 435.811") == []

    def test_paragraph_unit_excludes_its_own_section(self):
        cited = evals._cited_cfr_sections_for_unit(
            "us/regulation/42/435/601/b", "§ 435.601(d) and § 435.811"
        )
        assert [item.label for item in cited] == ["42 CFR 435.811"]

    def test_section_bound_keeps_first_cited_sections(self):
        text = " ".join(f"§ 435.{number}" for number in range(100, 120))
        cited = evals._cited_cfr_sections_for_unit(UNIT, text)
        assert len(cited) == evals._MAX_CITED_CFR_CONTEXT_SECTIONS == 8
        assert [item.section for item in cited] == [str(n) for n in range(100, 108)]


class TestCitedCfrContextSelection:
    def _select(self, root, text, unit=UNIT):
        return evals._select_cited_cfr_section_context_files(unit, text, root)

    def test_maps_section_sign_to_the_part_directory(self, tree):
        root, files = tree
        assert self._select(root, "under § 435.811") == [files["811"]]

    def test_prefers_live_subdirectory_module_over_deferred_section_file(self, tree):
        root, files = tree
        assert self._select(root, "eligible under § 435.120") == [files["120/ssi"]]

    def test_keeps_deferred_section_file_when_nothing_live_exists(self, tree):
        root, files = tree
        assert self._select(root, "under § 435.236") == [files["236"]]

    def test_live_section_file_then_live_children_by_depth_and_path(self, tree):
        root, files = tree
        assert self._select(root, "under § 435.230") == [
            files["230"],
            files["230/a"],
            files["230/b/1"],
        ]

    def test_children_sort_by_depth_before_path(self, tree):
        root, files = tree
        deep_last = _write(
            root / "regulations" / "42-cfr" / "435" / "230" / "z.yaml",
            _live("optional_group_z"),
        )
        assert self._select(root, "under § 435.230") == [
            files["230"],
            files["230/a"],
            deep_last,
            files["230/b/1"],
        ]

    def test_cited_fragment_module_wins_over_section(self, tree):
        root, files = tree
        assert self._select(root, "requirements in § 435.123(b)") == [files["123/b"]]

    def test_missing_fragment_falls_back_to_section(self, tree):
        root, files = tree
        assert self._select(root, "requirements in § 435.811(z)") == [files["811"]]

    def test_own_section_is_excluded(self, tree):
        root, _files = tree
        assert self._select(root, "under § 435.725 and § 435.725(a)") == []

    def test_other_title_maps_to_its_own_directory(self, tree):
        root, files = tree
        assert self._select(root, "under 20 CFR 416.1100") == [files["416-1100"]]

    def test_section_sign_in_another_title_unit_does_not_reach_42_cfr(self, tree):
        root, _files = tree
        assert self._select(root, "under § 435.811", unit="us/regulation/7/273/9") == []

    def test_non_cfr_unit_selects_nothing(self, tree):
        root, _files = tree
        assert self._select(root, "under § 435.811", unit="us/statute/42/1396a") == []

    def test_order_follows_first_citation_and_is_deterministic(self, tree):
        root, files = tree
        text = "under § 435.230, § 435.811, § 435.120 and again § 435.230"
        first = self._select(root, text)
        assert first == [
            files["230"],
            files["230/a"],
            files["230/b/1"],
            files["811"],
            files["120/ssi"],
        ]
        assert self._select(root, text) == first

    def test_per_section_module_bound(self, tree):
        root, _files = tree
        part = root / "regulations" / "42-cfr" / "435" / "300"
        for letter in "abcdefg":
            _write(part / f"{letter}.yaml", _live(f"group_{letter}"))
        selected = self._select(root, "under § 435.300")
        assert len(selected) == evals._MAX_CITED_CFR_CONTEXT_MODULES_PER_SECTION == 4
        assert [path.stem for path in selected] == ["a", "b", "c", "d"]

    def test_total_module_bound(self, tree):
        root, _files = tree
        part = root / "regulations" / "42-cfr" / "435"
        for number in range(400, 408):
            for letter in "ab":
                _write(
                    part / str(number) / f"{letter}.yaml", _live(f"g{number}{letter}")
                )
        text = " ".join(f"§ 435.{number}" for number in range(400, 408))
        selected = self._select(root, text)
        assert len(selected) == evals._MAX_CITED_CFR_CONTEXT_MODULES == 12
        assert selected[-1].parent.name == "405"

    def test_cross_section_selection_includes_cited_cfr_modules(self, tree):
        root, files = tree
        selected = evals._select_cross_section_context_files(
            UNIT, "standard established under § 435.811", root
        )
        assert selected == [files["811"]]


class TestCitedRegulationGuidance:
    def _items(self, root, *paths):
        items = []
        for path in paths:
            relative = evals._context_import_relative_target(path, root)
            items.append(
                evals.EvalContextFile(
                    source_path=str(path),
                    workspace_path=str(Path("context") / relative),
                    import_path=evals._context_import_target(path, relative),
                    kind="implementation_precedent",
                )
            )
        return items

    def test_lists_exports_and_input_slot_names_of_cited_modules(self, tree):
        root, files = tree
        guidance = evals._format_cited_regulation_context_guidance(
            UNIT,
            "the standard established under § 435.811 and § 435.236",
            self._items(root, files["811"], files["236"], files["230"]),
        )
        assert "Cited regulation RuleSpec context detected from source text:" in (
            guidance
        )
        assert (
            "- Source cites `42 CFR 435.811`; copied context "
            "`us:regulations/42-cfr/435/811` exports "
            "`us:regulations/42-cfr/435/811#medically_needy_income_standard_complies`;"
            " local input slot names `medically_needy_income_standard_amount`."
        ) in guidance
        assert (
            "- Source cites `42 CFR 435.236`; copied context "
            "`us:regulations/42-cfr/435/236` has `module.status: deferred` and no "
            "executable exports."
        ) in guidance
        assert "435/230" not in guidance
        flat = " ".join(guidance.split())
        assert (
            "When this source computes with a value that one of these modules "
            "exports, import that export and use its bare name in the formula."
        ) in flat
        assert (
            "When the source applies a State-established amount that the cited "
            "module only takes as a local input slot, follow the exception for "
            "State-established amounts and reuse that slot name."
        ) in flat

    def test_lists_every_input_slot_name_up_to_the_bound(self, tree):
        root, _files = tree
        slots = tuple(f"state_amount_slot_{index:02d}" for index in range(12))
        module = _write(
            root / "regulations" / "42-cfr" / "435" / "500.yaml",
            _live("standard_complies", inputs=slots),
        )
        guidance = evals._format_cited_regulation_context_guidance(
            UNIT, "the standard under § 435.500", self._items(root, module)
        )
        assert evals._MAX_CITED_CFR_CONTEXT_INPUT_SLOTS == 32
        assert "local input slot names " + ", ".join(f"`{s}`" for s in slots) + "." in (
            guidance
        )

    def test_empty_for_non_cfr_unit_and_uncited_modules(self, tree):
        root, files = tree
        items = self._items(root, files["811"])
        assert (
            evals._format_cited_regulation_context_guidance(
                "us/statute/42/1396a", "under § 435.811", items
            )
            == ""
        )
        assert (
            evals._format_cited_regulation_context_guidance(UNIT, "no citation", items)
            == ""
        )

    def test_skips_test_context_and_own_section(self, tree):
        root, files = tree
        items = self._items(root, files["725"], files["811"])
        items[1].kind = "implementation_test_context"
        assert (
            evals._format_cited_regulation_context_guidance(
                UNIT, "§ 435.725 and § 435.811", items
            )
            == ""
        )

    def test_eval_prompt_carries_the_guidance(self, tree, tmp_path):
        root, files = tree
        work = tmp_path / "ws"
        work.mkdir()
        source = work / "source.txt"
        source.write_text("must not exceed the standard established under § 435.811.\n")
        workspace = evals.EvalWorkspace(work, source, work / "manifest.json")
        prompt = evals._build_eval_prompt(
            UNIT,
            "repo-augmented",
            workspace,
            self._items(root, files["811"]),
            "regulations/42-cfr/435/725.yaml",
            include_tests=True,
            runner_backend="codex",
        )
        assert "Cited regulation RuleSpec context detected from source text:" in (
            prompt
        )
        assert "local input slot names `medically_needy_income_standard_amount`" in (
            prompt
        )


# Each sentence of the new prompt text, asserted in both prompt copies so that
# removing any one of them fails a test.
STATE_AMOUNT_SENTENCES = (
    "Exception for State-established amounts: when a branch applies an amount, "
    "standard, rate, or limit that the source says the State or agency "
    'establishes under another provision (such as "the income standard '
    'established under § 435.811" or "the standard in the State plan") and no '
    "copied context module exports that value, encode the branch; do not defer "
    "it for that reason.",
    "Declare the value in the document-root `inputs` list with `entity`, "
    "`dtype`, `period`, and `unit` for money.",
    "When the cited provision's copied module lists a local input slot for that "
    "value, use that exact slot name; otherwise use a descriptive name for the "
    "amount.",
    "Give the rule that uses it a proof atom whose excerpt is the citing phrase "
    "from this unit's source text.",
    "Encode the branch's own comparisons, max/min, and conditions on that "
    "input, and set the input in every companion test.",
    "A State-established amount is not a carve-out; follow the exception for "
    "State-established amounts.",
)
CROSS_REFERENCE_RULE_TAIL = (
    "import a real export, defer the affected executable surface, or encode a "
    "source-grounded overriding branch that avoids it."
)
CARVE_OUT_RULE_TAIL = (
    "Do not emit a formula that ignores the carve-out, and do not invent a "
    "local boolean for the cited sibling source."
)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _eval_prompt(tmp_path) -> str:
    work = tmp_path / "prompt"
    work.mkdir()
    source = work / "source.txt"
    source.write_text("The annual allowance is $100.\n")
    workspace = evals.EvalWorkspace(work, source, work / "manifest.json")
    return evals._build_eval_prompt(
        UNIT,
        "cold",
        workspace,
        [],
        "regulations/42-cfr/435/725.yaml",
        include_tests=True,
        runner_backend="codex",
    )


@pytest.fixture
def prompt_copies(tmp_path):
    return {
        "encoder": _flat(encoder_prompt.ENCODER_PROMPT),
        "evals": _flat(_eval_prompt(tmp_path)),
    }


@pytest.mark.parametrize("copy", ["encoder", "evals"])
@pytest.mark.parametrize("sentence", STATE_AMOUNT_SENTENCES)
def test_state_amount_sentence_in_both_prompt_copies(prompt_copies, copy, sentence):
    assert prompt_copies[copy].count(sentence) == 1


@pytest.mark.parametrize("copy", ["encoder", "evals"])
def test_state_amount_bullet_follows_cross_reference_defer_rule(prompt_copies, copy):
    text = prompt_copies[copy]
    tail = text.index(CROSS_REFERENCE_RULE_TAIL) + len(CROSS_REFERENCE_RULE_TAIL)
    assert text[tail:].startswith(" - " + STATE_AMOUNT_SENTENCES[0])
    sentences = [text.index(sentence) for sentence in STATE_AMOUNT_SENTENCES[:5]]
    assert sentences == sorted(sentences)


@pytest.mark.parametrize("copy", ["encoder", "evals"])
def test_carve_out_rule_points_to_the_state_amount_exception(prompt_copies, copy):
    text = prompt_copies[copy]
    assert CARVE_OUT_RULE_TAIL + " " + STATE_AMOUNT_SENTENCES[5] in text


def test_encoder_prompt_version_is_bumped_for_the_state_amount_rule():
    assert encoder_prompt.__version__ == 7
