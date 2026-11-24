"""Cited-section context for CFR units.

A CFR unit's source text is read with one closed citation grammar
(`N CFR P.S`, `§ P.S`); each hit is re-parsed by the corpus resolver; the
cited sections' RuleSpec modules join the context in a bounded, deterministic
order (cited paragraphs first, live modules preferred over a deferred section
file), and the context they add after transitive expansion is bounded.
"""

import json
from pathlib import Path

import pytest

from axiom_encode import corpus_resolver
from axiom_encode.harness import evals
from tests.test_evals import _canonical_rulespec_content_root

UNIT = "us/regulation/42/435/725"


def _live(name: str, *, inputs: tuple[str, ...] = (), imports: tuple[str, ...] = ()):
    formula = " and ".join(inputs) if inputs else "true"
    import_block = "".join(f"  - {target}\n" for target in imports)
    return (
        "format: rulespec/v1\n"
        + (f"imports:\n{import_block}" if imports else "")
        + "rules:\n"
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
    for letter in "abcdef":
        files[f"603/{letter}"] = _write(
            part / "603" / f"{letter}.yaml", _live(f"magi_{letter}")
        )
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

    def test_section_sign_after_another_title_in_the_sentence_is_dropped(self):
        assert _labels("see 45 CFR 155.305 and § 155.310 for the rule") == [
            "45 CFR 155.305"
        ]
        assert _labels("see 45 C.F.R. 155.305 and § 155.310(a).") == ["45 CFR 155.305"]

    def test_section_sign_after_the_unit_title_in_the_sentence_is_kept(self):
        assert _labels("see 42 CFR 435.4 and § 435.811") == [
            "42 CFR 435.4",
            "42 CFR 435.811",
        ]

    def test_section_sign_in_the_next_sentence_takes_the_unit_title(self):
        assert _labels("See 45 CFR 155.305. The agency applies § 435.811.") == [
            "45 CFR 155.305",
            "42 CFR 435.811",
        ]
        assert _labels("See 45 CFR 155.305;\n(b) under § 435.811") == [
            "45 CFR 155.305",
            "42 CFR 435.811",
        ]

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


class TestCitedSections:
    def test_excludes_own_section_and_groups_cited_paragraphs(self):
        groups = evals._cited_cfr_sections_for_unit(
            UNIT,
            "§ 435.725(b), § 435.120, § 435.811(b), § 435.811(a), § 435.811(b)",
        )
        assert [group.label for group in groups] == ["42 CFR 435.811", "42 CFR 435.120"]
        assert groups[0].fragment_paths == (("b",), ("a",))
        assert groups[1].fragment_paths == ()

    def test_orders_by_citation_count_then_first_citation(self):
        groups = evals._cited_cfr_sections_for_unit(
            UNIT, "§ 435.120, § 435.230, § 435.811, § 435.811, § 435.230"
        )
        assert [group.section for group in groups] == ["230", "811", "120"]

    @pytest.mark.parametrize(
        "unit",
        [
            "us/statute/42/1396a",
            "us-az/manual/des/faa5/dependent-care-expense/block-4",
            "us-co/regulation/10-ccr-2506-1/4.804.1",
            "us/statute/26/32/1",
            "us/regulation/7/273/a",
            "us/regulation/42/a/1",
            "us/regulation/x/435/1",
        ],
    )
    def test_non_cfr_units_read_no_cfr_citations(self, unit):
        assert evals._us_cfr_unit_section_key(unit) is None
        assert evals._cited_cfr_sections_for_unit(unit, "under § 435.811") == []

    def test_paragraph_unit_excludes_its_own_section(self):
        groups = evals._cited_cfr_sections_for_unit(
            "us/regulation/42/435/601/b", "§ 435.601(d) and § 435.811"
        )
        assert [group.label for group in groups] == ["42 CFR 435.811"]

    def test_section_bound_keeps_first_cited_sections(self):
        text = " ".join(f"§ 435.{number}" for number in range(100, 120))
        groups = evals._cited_cfr_sections_for_unit(UNIT, text)
        assert len(groups) == evals._MAX_CITED_CFR_CONTEXT_SECTIONS == 8
        assert [group.section for group in groups] == [str(n) for n in range(100, 108)]


class TestCitedCfrCandidates:
    def _candidates(self, root, text, unit=UNIT):
        return evals._cited_cfr_context_candidates(unit, text, root)

    def test_maps_section_sign_to_the_part_directory(self, tree):
        root, files = tree
        assert self._candidates(root, "under § 435.811") == [files["811"]]

    def test_prefers_live_subdirectory_module_over_deferred_section_file(self, tree):
        root, files = tree
        assert self._candidates(root, "eligible under § 435.120") == [files["120/ssi"]]

    def test_keeps_deferred_section_file_when_nothing_live_exists(self, tree):
        root, files = tree
        assert self._candidates(root, "under § 435.236") == [files["236"]]

    def test_live_section_file_then_live_children_by_depth_and_path(self, tree):
        root, files = tree
        assert self._candidates(root, "under § 435.230") == [
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
        assert self._candidates(root, "under § 435.230") == [
            files["230"],
            files["230/a"],
            deep_last,
            files["230/b/1"],
        ]

    def test_cited_paragraph_module_comes_first_then_section(self, tree):
        root, files = tree
        assert self._candidates(root, "requirements in § 435.123(b)") == [
            files["123/b"],
            files["123"],
        ]

    def test_cited_paragraphs_win_over_name_order(self, tree):
        root, files = tree
        assert self._candidates(root, "methods in § 435.603(f) and § 435.603") == [
            files["603/f"],
            files["603/a"],
            files["603/b"],
            files["603/c"],
        ]
        assert self._candidates(root, "under § 435.603(e) and § 435.603(f)(2)") == [
            files["603/e"],
            files["603/f"],
            files["603/a"],
            files["603/b"],
        ]

    def test_missing_paragraph_falls_back_to_section(self, tree):
        root, files = tree
        assert self._candidates(root, "requirements in § 435.811(z)") == [files["811"]]

    def test_own_section_is_excluded(self, tree):
        root, _files = tree
        assert self._candidates(root, "under § 435.725 and § 435.725(a)") == []

    def test_other_title_maps_to_its_own_directory(self, tree):
        root, files = tree
        assert self._candidates(root, "under 20 CFR 416.1100") == [files["416-1100"]]

    def test_section_sign_in_another_title_unit_does_not_reach_42_cfr(self, tree):
        root, _files = tree
        assert (
            self._candidates(root, "under § 435.811", unit="us/regulation/7/273/9")
            == []
        )

    def test_order_is_deterministic(self, tree):
        root, files = tree
        text = "under § 435.230, § 435.811, § 435.120 and again § 435.230"
        first = self._candidates(root, text)
        assert first == [
            files["230"],
            files["230/a"],
            files["230/b/1"],
            files["811"],
            files["120/ssi"],
        ]
        assert self._candidates(root, text) == first

    def test_per_section_module_bound(self, tree):
        root, _files = tree
        part = root / "regulations" / "42-cfr" / "435" / "300"
        for letter in "abcdefg":
            _write(part / f"{letter}.yaml", _live(f"group_{letter}"))
        selected = self._candidates(root, "under § 435.300")
        assert len(selected) == evals._MAX_CITED_CFR_CONTEXT_MODULES_PER_SECTION == 4
        assert [path.stem for path in selected] == ["a", "b", "c", "d"]

    def test_total_candidate_bound(self, tree):
        root, _files = tree
        part = root / "regulations" / "42-cfr" / "435"
        for number in range(400, 408):
            for letter in "ab":
                _write(
                    part / str(number) / f"{letter}.yaml", _live(f"g{number}{letter}")
                )
        text = " ".join(f"§ 435.{number}" for number in range(400, 408))
        selected = self._candidates(root, text)
        assert len(selected) == evals._MAX_CITED_CFR_CONTEXT_MODULES == 12
        assert selected[-1].parent.name == "405"


class TestExpandedContextBound:
    def _select(self, root, text, base=()):
        return evals._select_cited_cfr_section_context_files(
            UNIT,
            text,
            root,
            base_selected=list(base),
            target_rel=Path("regulations/42-cfr/435/725.yaml"),
        )

    def test_bounds_are_stated(self):
        assert evals._MAX_CITED_CFR_CONTEXT_ADDED_FILES == 16
        assert evals._MAX_CITED_CFR_CONTEXT_ADDED_CHARS == 32_000

    def test_non_cfr_unit_selects_nothing(self, tree):
        root, _files = tree
        assert (
            evals._select_cited_cfr_section_context_files(
                "us/statute/42/1396a",
                "under § 435.811",
                root,
                base_selected=[],
                target_rel=None,
            )
            == []
        )

    def test_counts_transitive_imports_and_tests_against_the_file_bound(self, tree):
        root, files = tree
        part = root / "regulations" / "42-cfr" / "435"
        deps = [
            _write(part / "900" / f"d{index:02d}.yaml", _live(f"dep_{index}"))
            for index in range(16)
        ]
        heavy = _write(
            part / "500.yaml",
            _live(
                "heavy_rule",
                imports=tuple(
                    f"us:regulations/42-cfr/435/900/d{index:02d}#dep_{index}"
                    for index in range(16)
                ),
            ),
        )
        assert len(evals._expand_context_files([heavy], root, None)) == 17
        assert len(deps) == 16
        # 500 expands to 17 added files and is skipped; 811 still fits.
        assert self._select(root, "§ 435.500 and § 435.811") == [files["811"]]

    def test_skips_a_candidate_over_the_character_bound(self, tree):
        root, files = tree
        big = _write(
            root / "regulations" / "42-cfr" / "435" / "501.yaml",
            _live("big_rule") + "# " + "x" * 32_000 + "\n",
        )
        assert big.stat().st_size > 32_000
        assert self._select(root, "§ 435.501 and § 435.811") == [files["811"]]

    def test_files_already_in_context_do_not_count(self, tree):
        root, files = tree
        filler = [
            _write(
                root / "regulations" / "42-cfr" / "435" / "950" / f"f{i:02d}.yaml",
                _live(f"filler_{i}"),
            )
            for i in range(16)
        ]
        importer = _write(
            root / "regulations" / "42-cfr" / "435" / "502.yaml",
            _live(
                "importer_rule",
                imports=tuple(
                    f"us:regulations/42-cfr/435/950/f{i:02d}#filler_{i}"
                    for i in range(16)
                ),
            ),
        )
        assert self._select(root, "§ 435.502", base=filler) == [importer]

    def test_added_total_stays_within_both_bounds(self, tree):
        root, _files = tree
        part = root / "regulations" / "42-cfr" / "435"
        for number in range(700, 708):
            module = _write(part / f"{number}.yaml", _live(f"g{number}"))
            _write(module.with_suffix(".test.yaml"), "# " + "t" * 5_000 + "\n")
        text = " ".join(f"§ 435.{number}" for number in range(700, 708))
        selected = self._select(root, text)
        added = evals._expand_context_files(selected, root, None)
        assert len(added) <= evals._MAX_CITED_CFR_CONTEXT_ADDED_FILES
        assert sum(len(p.read_text()) for p, _ in added) <= 32_000
        assert [path.stem for path in selected] == [str(n) for n in range(700, 706)]

    def test_workspace_selection_includes_cited_cfr_modules(self, tree):
        root, files = tree
        assert (
            evals._select_cross_section_context_files(
                UNIT, "standard established under § 435.811", root
            )
            == []
        )
        assert self._select(root, "standard established under § 435.811") == [
            files["811"]
        ]


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
        assert (
            "When this source computes with a value that one of these modules "
            "exports, import that export and use its bare name in the formula."
        ) in " ".join(guidance.split())

    def test_lists_eight_exports_then_an_ellipsis(self, tree):
        root, _files = tree
        names = [f"export_{index}" for index in range(9)]
        module = _write(
            root / "regulations" / "42-cfr" / "435" / "503.yaml",
            "format: rulespec/v1\nrules:\n"
            + "".join(
                f"  - name: {name}\n    kind: derived\n    entity: Person\n"
                "    dtype: Judgment\n    period: Month\n    versions:\n"
                "      - effective_from: '2014-01-01'\n        formula: 'true'\n"
                for name in names
            ),
        )
        guidance = evals._format_cited_regulation_context_guidance(
            UNIT, "under § 435.503", self._items(root, module)
        )
        refs = ", ".join(f"`us:regulations/42-cfr/435/503#{n}`" for n in names[:8])
        assert f"exports {refs}, ...." in guidance
        assert "export_8" not in guidance

    def test_lists_every_input_slot_name_up_to_the_bound(self, tree):
        root, _files = tree
        slots = tuple(f"state_amount_slot_{index:02d}" for index in range(34))
        module = _write(
            root / "regulations" / "42-cfr" / "435" / "500.yaml",
            _live("standard_complies", inputs=slots),
        )
        guidance = evals._format_cited_regulation_context_guidance(
            UNIT, "the standard under § 435.500", self._items(root, module)
        )
        assert evals._MAX_CITED_CFR_CONTEXT_INPUT_SLOTS == 32
        listed = ", ".join(f"`{s}`" for s in slots[:32])
        assert f"local input slot names {listed}, ...." in guidance
        assert "state_amount_slot_32" not in guidance

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


def test_prepare_eval_workspace_copies_cited_cfr_modules(tree, tmp_path):
    root, files = tree
    # A unit in another part, so no sibling selection copies 435.811.
    citation = "us/regulation/42/436/10"
    workspace = evals.prepare_eval_workspace(
        citation=citation,
        runner=evals.parse_runner_spec("openai:gpt-5.5"),
        output_root=tmp_path / "out",
        source_text="The agency uses the standard established under 42 CFR 435.811.",
        axiom_rules_path=root,
        mode="repo-augmented",
        source_metadata_payload={
            "source_attestation": {"requested_corpus_citation_path": citation}
        },
        extra_context_paths=[],
    )
    manifest = json.loads(workspace.manifest_file.read_text())
    copied = {item["source_path"]: item for item in manifest["context_files"]}
    assert copied[str(files["811"])]["import_path"] == "us:regulations/42-cfr/435/811"
    assert copied[str(files["811"])]["kind"] == "implementation_precedent"
