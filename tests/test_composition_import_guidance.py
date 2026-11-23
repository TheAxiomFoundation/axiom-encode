"""Composition imports: one prompt bullet, and the import lists that obey it.

Validation compiles a file that does not declare ``module.kind: composition``
on the atomic surface of the rules engine. That surface refuses an import
target whose ``module`` mapping has a ``kind`` key. A file that declares
``module.kind: composition`` compiles on the composed surface, which accepts a
composition as an import target. The prompt allows such an import only when
the file being replaced already is a composition: a fact on disk, not a
property of the model's own output. No test here runs the engine.

The first tests pin the complete bullet in both prompt copies
(``prompts/encoder.py`` and ``harness/evals.py``). The rendered-prompt tests
pin that the eval prompt never offers or requires an export of a copied module
the target cannot import, and never drops an export of a module it can import.

Two other bullets were proposed with this one and withdrawn after review: one
on which text a list-condition proof excerpt quotes, and one on a
``dtype: Count`` input for a count threshold. The last tests pin that neither
prompt carries them, so adding either again is a deliberate change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from axiom_encode.harness import evals
from axiom_encode.prompts.encoder import _NAMING_PROTOCOL, ENCODER_PROMPT

COMPOSITION_IMPORT = (
    "- Unless the file you are replacing already declares "
    "`module.kind: composition`, never import from a module that declares it. "
    "Validation compiles a file without that declaration on the atomic "
    "surface of the rules engine, where no imported module may declare "
    "`module.kind`, so the file fails to compile. Treat the exports of such a "
    "module as not available for import; no other import rule changes."
)
BULLET_START = "- Unless the file you are replacing already declares"
BEFORE = "- Never introduce an import cycle."
AFTER = "- Never create a derived rule whose formula references that same rule's name."

MANDATORY = "Mandatory cited RuleSpec imports detected from source text:"
UNAVAILABLE = "Unavailable cited RuleSpec context detected:"
COMPOSITION_REASON = "`module.kind: composition`, which this target cannot import"
MODULE_ID_REASON = "a `module.id` declaration, which no file can import"
OTHER_KIND_REASON = (
    "a `module.kind` value other than `composition`, which no file can import"
)
NOT_A_MAPPING_REASON = (
    "a `module` block that is not a mapping, which no file can import"
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _complete_bullet(text: str) -> tuple[str, str, str]:
    """Return the bullet before, the composition bullet, and the line after.

    The composition bullet is the complete text from its `- ` line to the next
    `- ` line, so a sentence appended to it is part of what is compared.
    """
    assert text.count(BULLET_START) == 1
    start = text.index(BULLET_START)
    assert text[start - 1] == "\n"
    end = text.index("\n- ", start) + 1
    before = text.rindex("\n- ", 0, start) + 1
    after_end = text.index("\n", end)
    return _flat(text[before:start]), _flat(text[start:end]), text[end:after_end]


def _assert_complete_bullet(text: str) -> None:
    before, bullet, after = _complete_bullet(text)
    assert bullet == COMPOSITION_IMPORT
    assert before.startswith(BEFORE)
    assert AFTER.startswith(after)


def _eval_prompt(tmp_path) -> str:
    source = tmp_path / "source.txt"
    source.write_text("The annual allowance is 100.\n")
    workspace = evals.EvalWorkspace(tmp_path, source, tmp_path / "manifest.json")
    return evals._build_eval_prompt(
        "us/guidance/example/allowance",
        "cold",
        workspace,
        [],
        "policies/example/allowance.yaml",
        include_tests=True,
    )


def test_encoder_prompt_carries_the_complete_composition_import_bullet():
    _assert_complete_bullet(_NAMING_PROTOCOL)
    _assert_complete_bullet(ENCODER_PROMPT)
    assert ENCODER_PROMPT.count("module.kind") == COMPOSITION_IMPORT.count(
        "module.kind"
    )


def test_eval_prompt_carries_the_same_complete_composition_import_bullet(tmp_path):
    prompt = _eval_prompt(tmp_path)
    _assert_complete_bullet(prompt)
    assert prompt.count("module.kind") == COMPOSITION_IMPORT.count("module.kind")


@pytest.mark.parametrize(
    "appended",
    [
        " Use a boundary input instead.",
        "\n  Use a boundary input instead.",
        "\nUse a boundary input instead.",
    ],
)
def test_a_sentence_appended_to_either_rendered_bullet_fails(tmp_path, appended):
    for prompt in (ENCODER_PROMPT, _eval_prompt(tmp_path)):
        end = prompt.index("\n- ", prompt.index(BULLET_START))
        mutated = prompt[:end] + appended + prompt[end:]
        with pytest.raises(AssertionError):
            _assert_complete_bullet(mutated)


def test_the_rendered_bullet_opens_with_a_fact_the_model_does_not_control(tmp_path):
    # Read from the prompts, not from the expected constant.
    for prompt in (ENCODER_PROMPT, _eval_prompt(tmp_path)):
        _before, bullet, _after = _complete_bullet(prompt)
        prohibition, reason, consequence = bullet.split(". ")
        assert prohibition.startswith(
            "- Unless the file you are replacing already declares "
            "`module.kind: composition`, never import"
        )
        assert reason.startswith("Validation compiles a file without that")
        assert consequence.startswith("Treat the exports of such a module")
        lowered = bullet.lower()
        for word in ("boundary", "defer", "proof", "cycle", "prefix", "hash", "every"):
            assert word not in lowered, word


# --- Rendered eval prompts with copied modules -----------------------------

TARGET = "us:statutes/26/151"
ATOMIC = ""
SUMMARY_ONLY = "module:\n  summary: An atomic module.\n"
COMPOSITION = "module:\n  kind: composition\n  summary: A composition.\n"
OTHER_KIND = "module:\n  kind: program\n"
NULL_KIND = "module:\n  kind:\n"
NOT_A_MAPPING = "module: composition\n"


@dataclass(frozen=True)
class _Module:
    """One copied module: its section, `module` block, export and imports."""

    section: str
    module: str
    export: str | None = "dependent"
    imports: tuple[str, ...] = ()

    @property
    def target(self) -> str:
        return f"us:statutes/26/{self.section}"

    def text(self) -> str:
        imports = ""
        if self.imports:
            imports = "imports:\n" + "".join(f"  - {item}\n" for item in self.imports)
        if self.export is None:
            return f"format: rulespec/v1\n{self.module}{imports}rules: []\n"
        return (
            f"format: rulespec/v1\n{self.module}{imports}"
            "rules:\n"
            f"  - name: {self.export}\n"
            "    kind: derived\n"
            "    entity: Person\n"
            "    dtype: Judgment\n"
            "    period: Year\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            f"        formula: fact_for_{self.export}\n"
        )


def _render(
    tmp_path: Path,
    modules: tuple[_Module, ...],
    *,
    existing_target: str | None = None,
    existing_target_rules: str = "rules: []\n",
    retry_candidate: str | None = None,
    replacement_baseline: str | None = None,
    required_import_targets: tuple[str, ...] = (),
    policyengine_rule_hint: str | None = None,
    source_text: str | None = None,
    canonical_concepts: tuple[str, ...] = (),
) -> str:
    root = tmp_path.resolve() / "rulespec-us" / "us"
    paths = []
    for module in modules:
        path = root / "statutes" / "26" / f"{module.section}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(module.text())
        paths.append(path)
    target_relative = Path("statutes/26/151.yaml")
    if existing_target is not None:
        target_path = root / target_relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            f"format: rulespec/v1\n{existing_target}{existing_target_rules}"
        )
        paths.append(target_path)
    cited = sorted({module.section.split("/")[0] for module in modules} - {"151"})
    if source_text is None:
        source_text = " ".join(
            f"A term is defined in section {section}." for section in cited
        )
    workspace = evals.prepare_eval_workspace(
        citation="26 USC 151",
        runner=evals.parse_runner_spec("codex:gpt-5.4"),
        output_root=tmp_path.resolve() / "out",
        source_text=source_text or "The allowance is 100.",
        axiom_rules_path=root,
        mode="repo-augmented",
        extra_context_paths=paths,
        target_relative_output=target_relative,
    )
    candidate = None
    if retry_candidate is not None:
        candidate = evals.ValidationRetryCandidate(
            rulespec=f"format: rulespec/v1\n{retry_candidate}rules: []\n"
        )
    if replacement_baseline is not None:
        candidate = evals.ValidationRetryCandidate(
            rulespec=f"format: rulespec/v1\n{replacement_baseline}rules: []\n",
            context_role="replacement_baseline",
        )
    # A resolved canonical concept is a copied repository module with its own
    # context kind (`_materialize_resolved_canonical_concept`).
    context_files = [
        replace(item, kind="canonical_concept", label=f"concept {item.import_path}")
        if item.import_path
        in {f"us:statutes/26/{section}" for section in canonical_concepts}
        else item
        for item in workspace.context_files
    ]
    return evals._build_eval_prompt(
        "26 USC 151",
        "repo-augmented",
        workspace,
        context_files,
        target_file_name="151.yaml",
        target_ref_prefix=TARGET,
        include_tests=True,
        validation_retry_candidate=candidate,
        required_import_targets=required_import_targets,
        policyengine_rule_hint=policyengine_rule_hint,
    )


def _section(prompt: str, heading: str) -> str:
    if heading not in prompt:
        return ""
    start = prompt.index(heading)
    end = prompt.find("\n\n", start)
    return prompt[start : end if end != -1 else len(prompt)]


def _listing(prompt: str, module: _Module) -> str:
    marker = f"- inspect `context/statutes/26/{module.section}.yaml`"
    lines = [line for line in prompt.splitlines() if line.startswith(marker)]
    assert len(lines) == 1, module.section
    return lines[0]


def _assert_importable(prompt: str, module: _Module, *, cited: bool = True) -> None:
    """The module is offered as an import, and required when the source cites it."""
    reference = f"`{module.target}#{module.export}`"
    listing = _listing(prompt, module)
    assert f"import target `{module.target}`" in listing
    assert f"exports {reference}" in listing
    assert "not an import target" not in listing
    assert f"copied context target `{module.target}` has" not in prompt
    if cited:
        assert (
            f"copied context target `{module.target}` exports {reference}."
            in _section(prompt, MANDATORY)
        )


def _assert_not_importable(prompt: str, module: _Module, reason: str) -> None:
    """No section offers or requires an export of the module as an import."""
    assert f"{module.target}#" not in prompt
    assert f"import target `{module.target}`" not in prompt
    listing = _listing(prompt, module)
    assert f"not an import target: `{module.target}` has {reason}" in listing
    assert f"`{module.target}`" not in _section(prompt, MANDATORY)
    if module.section.split("/")[0] != "151":
        assert f"copied context target `{module.target}` has {reason}." in _section(
            prompt, UNAVAILABLE
        )


def test_a_cited_composition_module_is_not_a_mandatory_import(tmp_path):
    # The probe of the review of 503fb27a6: before the fix the rendered prompt
    # required `us:statutes/26/152#dependent` and prohibited it at once.
    module = _Module("152", COMPOSITION)

    prompt = _render(tmp_path, (module,))

    assert MANDATORY not in prompt
    assert "omit or defer only the affected executable surface" in _section(
        prompt, UNAVAILABLE
    )
    _assert_not_importable(prompt, module, COMPOSITION_REASON)
    assert _complete_bullet(prompt)[1] == COMPOSITION_IMPORT


def test_a_cited_atomic_module_exporting_the_same_name_stays_mandatory(tmp_path):
    module = _Module("152", ATOMIC)

    prompt = _render(tmp_path, (module,))

    assert UNAVAILABLE not in prompt
    _assert_importable(prompt, module)
    assert "not an import target" not in prompt


@dataclass(frozen=True)
class _Case:
    modules: tuple[_Module, ...]
    importable: tuple[str, ...] = ()
    blocked: tuple[tuple[str, str], ...] = ()
    existing_target: str | None = None
    retry_candidate: str | None = None
    replacement_baseline: str | None = None


NESTED_REASON_2 = "an import of `us:statutes/26/2`, which is not an import target"

CASES = {
    "01_composition_alone": _Case(
        (_Module("152", COMPOSITION),),
        blocked=(("152", COMPOSITION_REASON),),
    ),
    "02_atomic_with_summary_metadata": _Case(
        (_Module("152", SUMMARY_ONLY),), importable=("152",)
    ),
    "03_same_name_from_atomic_and_composition": _Case(
        (_Module("152", ATOMIC), _Module("2", COMPOSITION)),
        importable=("152",),
        blocked=(("2", COMPOSITION_REASON),),
    ),
    "04_different_names_from_atomic_and_composition": _Case(
        (_Module("152", ATOMIC), _Module("63", COMPOSITION, "taxable_income")),
        importable=("152",),
        blocked=(("63", COMPOSITION_REASON),),
    ),
    "05_two_compositions_and_no_atomic_module": _Case(
        (_Module("152", COMPOSITION), _Module("2", COMPOSITION, "surviving_spouse")),
        blocked=(("152", COMPOSITION_REASON), ("2", COMPOSITION_REASON)),
    ),
    "06_another_kind_value": _Case(
        (_Module("152", OTHER_KIND), _Module("2", ATOMIC, "surviving_spouse")),
        importable=("2",),
        blocked=(("152", OTHER_KIND_REASON),),
    ),
    "07_kind_key_without_a_value": _Case(
        (_Module("152", NULL_KIND),),
        blocked=(("152", OTHER_KIND_REASON),),
    ),
    "08_atomic_module_importing_a_composition": _Case(
        (
            _Module("152", ATOMIC, imports=("us:statutes/26/2#surviving_spouse",)),
            _Module("2", COMPOSITION, "surviving_spouse"),
        ),
        blocked=(("152", NESTED_REASON_2), ("2", COMPOSITION_REASON)),
    ),
    "09_atomic_module_importing_an_atomic_module": _Case(
        (
            _Module("152", ATOMIC, imports=("us:statutes/26/2#surviving_spouse",)),
            _Module("2", ATOMIC, "surviving_spouse"),
        ),
        importable=("152", "2"),
    ),
    "10_composed_target_importing_a_composition": _Case(
        (
            _Module("152", COMPOSITION),
            _Module("2", ATOMIC, "surviving_spouse"),
            _Module("63", OTHER_KIND, "taxable_income"),
        ),
        importable=("152", "2"),
        blocked=(("63", OTHER_KIND_REASON),),
        existing_target=COMPOSITION,
    ),
    "11_replacement_baseline_composition_with_nested_composition_imports": _Case(
        (
            _Module("152", COMPOSITION, imports=("us:statutes/26/2#surviving_spouse",)),
            _Module("2", COMPOSITION, "surviving_spouse"),
        ),
        importable=("152", "2"),
        replacement_baseline=COMPOSITION,
    ),
    "12_atomic_existing_target_and_atomic_retry_candidate": _Case(
        (_Module("152", COMPOSITION), _Module("2", ATOMIC, "surviving_spouse")),
        importable=("2",),
        blocked=(("152", COMPOSITION_REASON),),
        existing_target=SUMMARY_ONLY,
        retry_candidate=ATOMIC,
    ),
    "13_rejected_composition_candidate_for_a_new_target": _Case(
        (_Module("152", COMPOSITION), _Module("2", ATOMIC, "surviving_spouse")),
        importable=("2",),
        blocked=(("152", COMPOSITION_REASON),),
        retry_candidate=COMPOSITION,
    ),
    "14_rejected_composition_candidate_for_an_atomic_existing_target": _Case(
        (
            _Module("152", COMPOSITION, imports=("us:statutes/26/2#surviving_spouse",)),
            _Module("2", COMPOSITION, "surviving_spouse"),
        ),
        blocked=(("152", COMPOSITION_REASON), ("2", COMPOSITION_REASON)),
        existing_target=ATOMIC,
        retry_candidate=COMPOSITION,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_rendered_prompt_requires_and_forbids_one_import(tmp_path, name):
    case = CASES[name]
    by_section = {module.section: module for module in case.modules}
    assert set(case.importable) | {s for s, _ in case.blocked} == set(by_section)

    prompt = _render(
        tmp_path,
        case.modules,
        existing_target=case.existing_target,
        retry_candidate=case.retry_candidate,
        replacement_baseline=case.replacement_baseline,
    )

    for section in case.importable:
        _assert_importable(prompt, by_section[section])
    for section, reason in case.blocked:
        _assert_not_importable(prompt, by_section[section], reason)
    assert (MANDATORY in prompt) == bool(case.importable)
    assert (UNAVAILABLE in prompt) == bool(case.blocked)
    assert _complete_bullet(prompt)[1] == COMPOSITION_IMPORT


def test_an_empty_composition_module_is_unavailable_and_never_mandatory(tmp_path):
    module = _Module("152", COMPOSITION, export=None)

    prompt = _render(tmp_path, (module,))

    assert MANDATORY not in prompt
    _assert_not_importable(prompt, module, COMPOSITION_REASON)


def test_an_empty_atomic_module_keeps_its_earlier_unavailable_reason(tmp_path):
    module = _Module("152", ATOMIC, export=None)

    prompt = _render(tmp_path, (module,))

    assert MANDATORY not in prompt
    assert (
        "copied context target `us:statutes/26/152` has no executable exports."
        in _section(prompt, UNAVAILABLE)
    )
    assert "not an import target" not in prompt


def test_a_composition_child_is_not_offered_as_a_parent_child_import(tmp_path):
    composition_child = _Module("151/a", COMPOSITION, "exemption_amount")
    atomic_child = _Module("151/b", ATOMIC, "additional_exemption_amount")

    prompt = _render(tmp_path, (composition_child, atomic_child))

    children = _section(prompt, "Aggregate parent child outputs detected:")
    assert "`us:statutes/26/151/b#additional_exemption_amount`" in children
    assert "us:statutes/26/151/a" not in children
    _assert_not_importable(prompt, composition_child, COMPOSITION_REASON)
    _assert_importable(prompt, atomic_child, cited=False)


def test_a_composition_export_is_not_a_preferred_policyengine_hint_import(tmp_path):
    composition = _Module("152", COMPOSITION)
    atomic = _Module("2", ATOMIC, "surviving_spouse")

    prompt = _render(
        tmp_path, (composition, atomic), policyengine_rule_hint="is_tax_unit_dependent"
    )

    hinted = _section(prompt, "Executable Judgment exports visible in copied context")
    assert "`us:statutes/26/2#surviving_spouse`" in hinted
    _assert_not_importable(prompt, composition, COMPOSITION_REASON)


def test_a_required_import_the_target_cannot_import_is_refused(tmp_path):
    module = _Module("152", COMPOSITION)

    with pytest.raises(ValueError, match="required import `us:statutes/26/152` has"):
        _render(tmp_path, (module,), required_import_targets=(module.target,))


def test_a_required_atomic_import_stays_required(tmp_path):
    module = _Module("152", ATOMIC)

    prompt = _render(tmp_path, (module,), required_import_targets=(module.target,))

    assert "Protected atomic-composition requirements:\n- `us:statutes/26/152`" in (
        prompt
    )
    _assert_importable(prompt, module)


@pytest.mark.parametrize(
    ("module", "atomic_target", "composed_target"),
    [
        (ATOMIC, None, None),
        (SUMMARY_ONLY, None, None),
        ("module:\n", None, None),
        (COMPOSITION, COMPOSITION_REASON, None),
        (OTHER_KIND, OTHER_KIND_REASON, OTHER_KIND_REASON),
        (NULL_KIND, OTHER_KIND_REASON, OTHER_KIND_REASON),
        ("module:\n  kind: [composition]\n", OTHER_KIND_REASON, OTHER_KIND_REASON),
        ("module:\n  kind: Composition\n", OTHER_KIND_REASON, OTHER_KIND_REASON),
        ("module:\n  id: us.example\n", MODULE_ID_REASON, MODULE_ID_REASON),
        (
            "module:\n  kind: composition\n  id: us.example\n",
            MODULE_ID_REASON,
            MODULE_ID_REASON,
        ),
        ("module:\n  kind: program\n  id: x\n", OTHER_KIND_REASON, OTHER_KIND_REASON),
        (NOT_A_MAPPING, NOT_A_MAPPING_REASON, NOT_A_MAPPING_REASON),
        ("module: [kind, composition]\n", NOT_A_MAPPING_REASON, NOT_A_MAPPING_REASON),
    ],
)
def test_module_metadata_decides_whether_a_copied_module_is_importable(
    tmp_path, module, atomic_target, composed_target
):
    path = tmp_path / "module.yaml"
    path.write_text(_Module("152", module).text())

    for composed, expected in ((False, atomic_target), (True, composed_target)):
        assert (
            evals._context_file_module_import_blocker(
                str(path), composed_target=composed
            )
            == expected
        )


@pytest.mark.parametrize(
    "content",
    [
        "format: rulespec/v1\nmodule: {kind: composition\n",
        "module:\n  kind: composition\nrules:\n  - name: dependent\n",
        "- kind: composition\n",
        "",
    ],
)
def test_a_file_that_is_not_readable_rulespec_has_no_exports_to_offer(
    tmp_path, content
):
    # Fail safe: such a file returns no blocker, and it also returns no
    # exports, so no section can require an import from it.
    path = tmp_path / "module.yaml"
    path.write_text(content)

    assert (
        evals._context_file_module_import_blocker(str(path), composed_target=False)
        is None
    )
    assert evals._context_file_exports(str(path)) == []
    assert evals._context_file_exports(str(tmp_path / "absent.yaml")) == []


def test_a_rejected_composition_candidate_does_not_switch_the_filter_off(tmp_path):
    # The probe of the review of b5764cc65: the rejected candidate is model
    # output, so its `module.kind` must not make a composition importable.
    module = _Module("63", COMPOSITION, "taxable_income")

    without = _render(tmp_path / "a", (module,), source_text="See section 63.")
    with_candidate = _render(
        tmp_path / "b",
        (module,),
        source_text="See section 63.",
        retry_candidate=COMPOSITION,
    )

    for prompt in (without, with_candidate):
        assert MANDATORY not in prompt
        _assert_not_importable(prompt, module, COMPOSITION_REASON)
    with pytest.raises(ValueError, match="required import `us:statutes/26/63` has"):
        _render(
            tmp_path / "c",
            (module,),
            source_text="See section 63.",
            retry_candidate=COMPOSITION,
            required_import_targets=(module.target,),
        )


def test_a_new_composition_target_cannot_require_a_composition_import(tmp_path):
    # Fail closed: with no file on disk to replace, the target counts as atomic.
    module = _Module("152", COMPOSITION)

    with pytest.raises(ValueError, match="required import `us:statutes/26/152` has"):
        _render(tmp_path, (module,), required_import_targets=(module.target,))


def test_a_composition_child_is_not_an_included_child_of_an_excluded_branch(tmp_path):
    modules = (
        _Module("152", ATOMIC, "qualifying_amount"),
        _Module("152/a", COMPOSITION, "amount_a"),
        _Module("152/b", ATOMIC, "amount_b"),
        _Module("152/c", ATOMIC, "amount_c"),
    )

    prompt = _render(
        tmp_path,
        modules,
        source_text=(
            "The amount determined under section 152(a) and section 152(c), "
            "other than section 152(b), is allowed."
        ),
    )

    excluded = _section(prompt, "Excluded cited child branch guidance:")
    assert "`us:statutes/26/152/c#amount_c`" in excluded
    assert "us:statutes/26/152/a" not in excluded
    assert "us:statutes/26/152/a#" not in prompt


def test_a_composition_child_exception_is_not_a_required_exception_import(tmp_path):
    modules = (
        _Module("151/a", COMPOSITION, "first_exception"),
        _Module("151/b", ATOMIC, "second_exception"),
    )

    prompt = _render(
        tmp_path,
        modules,
        source_text="The term shall not include the following:",
    )

    exceptions = _section(prompt, "Parent exception-list child fragments detected:")
    assert "`us:statutes/26/151/b#second_exception`" in exceptions
    assert "us:statutes/26/151/a" not in exceptions
    assert "us:statutes/26/151/a#" not in prompt


EXISTING_TARGET_WITH_SECTION_63_INPUT = (
    "rules:\n"
    "  - name: exemption_amount\n"
    "    kind: derived\n"
    "    entity: TaxUnit\n"
    "    dtype: Money\n"
    "    period: Year\n"
    "    versions:\n"
    "      - effective_from: '2026-01-01'\n"
    "        formula: taxable_income_under_section_63 + filer_count\n"
)
SECTION_63_INPUT = "`us:statutes/26/151#input.taxable_income_under_section_63`"


@pytest.mark.parametrize("module", [COMPOSITION, ATOMIC])
def test_an_input_that_restates_a_copied_module_is_invalid_in_both_variants(
    tmp_path, module
):
    copied = _Module("63", module, "taxable_income")

    prompt = _render(
        tmp_path,
        (copied,),
        existing_target=ATOMIC,
        existing_target_rules=EXISTING_TARGET_WITH_SECTION_63_INPUT,
    )

    invalid = _section(prompt, "Invalid copied local input names:")
    valid = _section(prompt, "Existing target local factual inputs:")
    assert f"- {SECTION_63_INPUT}: encoded cross-reference placeholder" in invalid
    assert SECTION_63_INPUT not in valid
    assert "`us:statutes/26/151#input.filer_count`" in valid
    if module == COMPOSITION:
        # The Unavailable section forbids the `_under_section_63` suffix, so
        # no section may present the input as valid or ask for an import.
        assert "`us:statutes/26/63` is not an import target" in invalid
        assert "unless the generated file imports" not in invalid
        assert "`_under_section_63`" in _section(prompt, UNAVAILABLE)
        _assert_not_importable(prompt, copied, COMPOSITION_REASON)
    else:
        assert "a context file for `us:statutes/26/63` is available" in invalid


def test_a_composition_module_is_not_a_resolved_canonical_concept_import(tmp_path):
    composition = _Module("152", COMPOSITION)
    atomic = _Module("2", ATOMIC, "surviving_spouse")

    prompt = _render(tmp_path, (composition, atomic), canonical_concepts=("152", "2"))

    resolved = _section(prompt, "Resolved canonical concept files from this corpus")
    assert "- concept us:statutes/26/2\n" in resolved
    assert "us:statutes/26/152" not in resolved
    assert "import or re-export that exact canonical concept" in resolved


def test_only_composition_concepts_give_no_canonical_import_instruction(tmp_path):
    composition = _Module("152", COMPOSITION)

    prompt = _render(tmp_path, (composition,), canonical_concepts=("152",))

    assert "Resolved canonical concept files from this corpus" not in prompt
    assert "import or re-export that exact canonical concept" not in prompt


def test_the_blocker_pass_reads_each_copied_file_once(tmp_path, monkeypatch):
    # A chain: module 0 is a composition and module n imports module n - 1,
    # listed in the order that needs the most passes.
    count = 40
    items = []
    for index in range(count):
        path = tmp_path / f"{index}.yaml"
        imports = (f"us:statutes/26/{index - 1}#x",) if index else ()
        path.write_text(
            _Module(
                str(index), COMPOSITION if index == 0 else ATOMIC, "x", imports
            ).text()
        )
        items.append(
            evals.EvalContextFile(
                source_path=str(path),
                workspace_path=f"context/statutes/26/{index}.yaml",
                import_path=f"us:statutes/26/{index}",
                kind="implementation_precedent",
            )
        )
    reads: list[str] = []
    read = evals._read_context_rulespec_payload

    def counting_read(source_path: str) -> object:
        reads.append(source_path)
        return read(source_path)

    monkeypatch.setattr(evals, "_read_context_rulespec_payload", counting_read)

    blockers = evals._context_import_blockers(
        list(reversed(items)), composed_target=False
    )

    assert sorted(reads) == sorted(item.source_path for item in items)
    assert set(blockers) == {item.source_path for item in items}
    assert blockers[items[0].source_path] == COMPOSITION_REASON
    assert blockers[items[5].source_path] == (
        "an import of `us:statutes/26/4`, which is not an import target"
    )
    assert evals._context_import_blockers(items[1:], composed_target=False) == {}


def test_neither_prompt_tells_which_text_a_list_condition_excerpt_quotes(tmp_path):
    # Withdrawn: with item-only excerpts the gates-collapsed check does not
    # see a dropped item of a conjunctive list.
    for prompt in (_flat(ENCODER_PROMPT), _flat(_eval_prompt(tmp_path))):
        assert "the line that introduces the list" not in prompt
        assert "not the chapeau" not in prompt


def test_neither_prompt_carries_the_withdrawn_count_threshold_guidance(tmp_path):
    # Withdrawn: whether a `dtype: Count` input passes the boundary and pair
    # checks depends on the source wording, negation, sentence shape and the
    # input name, which a prompt line cannot control.
    for prompt in (_flat(ENCODER_PROMPT), _flat(_eval_prompt(tmp_path))):
        assert "compares a count of persons or things" not in prompt
        assert "whole-number count" not in prompt
        assert "`dtype: Count` input when no import or relation" not in prompt
        assert "`dtype: Boolean` input that restates the comparison" not in prompt
        assert "one comparison of the count with N" not in prompt
        assert "a count of listed conditions" not in prompt
