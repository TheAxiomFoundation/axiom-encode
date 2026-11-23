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
    # A repository path outside title 26 of the statutes, without `.yaml`.
    relative: str | None = None

    @property
    def path(self) -> str:
        return self.relative or f"statutes/26/{self.section}"

    @property
    def target(self) -> str:
        return f"us:{self.path}"

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
    runner_backend: str = "codex",
    reverse_context: bool = False,
) -> str:
    root = tmp_path.resolve() / "rulespec-us" / "us"
    paths = []
    for module in modules:
        path = root / f"{module.path}.yaml"
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
    if reverse_context:
        context_files.reverse()
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
        runner_backend=runner_backend,
    )


def _section(prompt: str, heading: str) -> str:
    if heading not in prompt:
        return ""
    start = prompt.index(heading)
    end = prompt.find("\n\n", start)
    return prompt[start : end if end != -1 else len(prompt)]


def _listing(prompt: str, module: _Module) -> str:
    marker = f"- inspect `context/{module.path}.yaml`"
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


# --- The generic import rules and the provider of a cited provision --------
#
# Review of e4e556254. Finding 1: the generic import rules still asked for an
# import of a copied module the target cannot import. Finding 2: a blocked
# parent caused deferral guidance although the exact cited provider was
# importable.

NOT_IMPORT_TARGETS = "Copied context that is not an import target:"
BACKENDS = ("codex", "claude", "openai", "claude-api")
# Each generic import rule: the text before the place where the eligibility
# qualifier goes, the qualifier, and the text after it.
GENERIC_IMPORT_RULES = (
    (
        "- If a copied context file",
        " listed with an import target",
        " already defines the exact symbol you need, import that exact symbol",
    ),
    (
        "a copied context file for that citation is listed",
        " with an import target",
        ", import and use the listed exported symbol",
    ),
    (
        "aggregate parent provision and copied child-fragment files",
        " listed with an import target",
        " already encode subparagraphs, import those child outputs",
    ),
    (
        "a more specific child file under the current target path",
        ", listed with an import target,",
        " that exports the exact scalar needed by this source",
    ),
    (
        "- When a copied context file",
        " listed with an import target",
        " already exports the operative legal condition that the requested "
        "source consumes, import and use that canonical output",
    ),
    (
        "- If a copied child-fragment file",
        " listed with an import target",
        " encodes a limitation, branch, amount, or predicate needed by the "
        "requested parent provision, import the child output",
    ),
    (
        "If a copied sibling/context file",
        " listed with an import target",
        " already encodes the limitation, import it and compose with it",
    ),
    (
        "If an exact same-path child scalar is available in context",
        " in a file listed with an import target",
        ", import it instead of duplicating it locally",
    ),
)
# The complete text of the section after its list of blocked files.
NOT_IMPORT_TARGETS_RULES = (
    "Every import rule in this prompt applies only to a copied file that is listed\n"
    "with an import target. A file in this list is context to read, also where an\n"
    "inline copy shows its rules: never import from it, even when it is the only\n"
    "copied file that defines the symbol, cited provision, child output, or child\n"
    "scalar that a rule in this prompt tells you to import. For every rule in this\n"
    "prompt, an import from a file in this list is an import that is not available.\n"
    "When a calculation needs something that only a file in this list provides,\n"
    "follow the first of these that applies:\n"
    "1. If `Unavailable cited RuleSpec context detected` lists the file, follow\n"
    "   that section for what the source cites from it.\n"
    "2. If another rule in this prompt says what to do when an import or a cited\n"
    "   definition is not available, for example a source-named local boundary\n"
    "   input, follow that rule.\n"
    "3. Otherwise omit or defer only the affected executable surface.\n"
    "In every case keep encoding the independent rules from this source."
)
# Rules on main for an import or a cited definition that is not available.
# The section must not contradict them: step 2 points at them.
FALLBACK_RULES = (
    "keep the result executable by exposing a local boundary input for that "
    "base instead of deferring",
    "If they are not available, paragraph-specific local inputs are acceptable",
    "if no canonical import is available for the cited override, model a "
    "local override/displacement boolean",
    "defer only that branch or expose a source-named boundary input for that branch",
)
CHILD_SCALAR_RULE = "Import the exact child export and use it"
RATE_SOURCE = "Multiply the qualifying amount by the exemption rate."


def _assert_generic_import_rules(prompt: str, *, scoped: bool) -> None:
    flat = _flat(prompt)
    for before, qualifier, after in GENERIC_IMPORT_RULES:
        assert (before + qualifier + after in flat) == scoped, before
        assert (before + after in flat) == (not scoped), before
    assert flat.count(CHILD_SCALAR_RULE) == 1


def _blocked_lines(prompt: str) -> list[str]:
    """Return the list of blocked files; the rest of the section is compared whole."""
    section = _section(prompt, NOT_IMPORT_TARGETS)
    head, rules = section.split("\nEvery import rule", 1)
    assert "Every import rule" + rules == NOT_IMPORT_TARGETS_RULES
    heading, *lines = head.splitlines()
    assert heading == NOT_IMPORT_TARGETS
    # The scope statement comes before the rules it scopes.
    assert prompt.index(NOT_IMPORT_TARGETS) < prompt.index("Import and context rules:")
    return lines


def _assert_defers_to(prompt: str, module: _Module, reason: str) -> None:
    """The prompt names the module as no import target, with the complete rules."""
    line = f"- `context/{module.path}.yaml`: `{module.target}` has {reason}."
    assert _blocked_lines(prompt).count(line) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_blocked_child_scalar_is_not_what_the_generic_rules_import(tmp_path, backend):
    # The reviewer's probe for finding 1: before the fix the prompt marked the
    # child as no import target and still said, without a condition, "Import
    # the exact child export and use it"; nothing asked for deferral.
    child = _Module("151/rate", COMPOSITION, "exemption_rate")

    prompt = _render(
        tmp_path, (child,), source_text=RATE_SOURCE, runner_backend=backend
    )

    _assert_generic_import_rules(prompt, scoped=True)
    _assert_defers_to(prompt, child, COMPOSITION_REASON)
    # The source cites no section, so deferral cannot come from this section.
    assert UNAVAILABLE not in prompt
    assert MANDATORY not in prompt
    assert f"{child.target}#" not in prompt
    assert f"import target `{child.target}`" not in prompt
    assert "not an import target: `us:statutes/26/151/rate`" in _listing(prompt, child)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_atomic_child_scalar_keeps_the_generic_rules_word_for_word(
    tmp_path, backend
):
    child = _Module("151/rate", ATOMIC, "exemption_rate")

    prompt = _render(
        tmp_path, (child,), source_text=RATE_SOURCE, runner_backend=backend
    )

    _assert_generic_import_rules(prompt, scoped=False)
    assert NOT_IMPORT_TARGETS not in prompt
    assert "listed with an import target" not in _flat(prompt)
    assert "not an import target" not in prompt
    assert f"exports `{child.target}#exemption_rate`" in _listing(prompt, child)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_blocked_policies_dependency_is_deferred_on_every_backend(tmp_path, backend):
    # A policies-root module has no citation in the source text, so the
    # "Unavailable cited RuleSpec context" section never names it.
    dependency = _Module(
        "plan",
        COMPOSITION,
        "plan_rate",
        relative="policies/usda/snap/state-plan-composition",
    )

    prompt = _render(
        tmp_path,
        (dependency,),
        source_text="Multiply the allotment by the plan rate.",
        runner_backend=backend,
    )

    _assert_generic_import_rules(prompt, scoped=True)
    _assert_defers_to(prompt, dependency, COMPOSITION_REASON)
    assert UNAVAILABLE not in prompt
    assert f"{dependency.target}#" not in prompt
    inline = "=== FILE: context/policies/usda/snap/state-plan-composition.yaml ==="
    if backend in {"openai", "claude-api"}:
        # The inline copy shows the raw rule; the scope statement covers it.
        copy = prompt[prompt.index(inline) :]
        assert "- name: plan_rate" in copy
        assert "also where an inline copy shows its rules" in _flat(
            _section(prompt, NOT_IMPORT_TARGETS)
        )
    else:
        assert inline not in prompt


def test_each_blocked_module_is_named_once_and_no_importable_one(tmp_path):
    atomic = _Module("152", ATOMIC)
    composition = _Module("2", COMPOSITION)
    nested = _Module("63", ATOMIC, imports=("us:statutes/26/2#dependent",))

    prompt = _render(tmp_path, (atomic, composition, nested))

    section = _section(prompt, NOT_IMPORT_TARGETS)
    assert len(_blocked_lines(prompt)) == 2
    _assert_defers_to(prompt, composition, COMPOSITION_REASON)
    _assert_defers_to(prompt, nested, NESTED_REASON_2)
    assert atomic.target not in section
    _assert_importable(prompt, atomic)


def test_a_composition_target_scopes_nothing_for_a_composition_import(tmp_path):
    composition = _Module("152", COMPOSITION)

    prompt = _render(tmp_path, (composition,), existing_target=COMPOSITION)

    assert NOT_IMPORT_TARGETS not in prompt
    _assert_generic_import_rules(prompt, scoped=False)
    _assert_importable(prompt, composition)


EXISTING_TARGET_WITH_SECTION_63_A_INPUT = EXISTING_TARGET_WITH_SECTION_63_INPUT.replace(
    "taxable_income_under_section_63", "taxable_income_under_section_63_a"
)
SECTION_63_A_INPUT = "`us:statutes/26/151#input.taxable_income_under_section_63_a`"
SECTION_63_A_SOURCE = "An amount is defined in section 63(a)."
DEFER_INPUT = "omit or defer the surface that depends on it"


def _render_section_63(tmp_path, parent, child, *, reverse, source=SECTION_63_A_SOURCE):
    modules = (
        _Module("63", parent, "taxable_income_total"),
        _Module("63/a", child, "taxable_income"),
    )
    prompt = _render(
        tmp_path,
        modules,
        existing_target=ATOMIC,
        existing_target_rules=EXISTING_TARGET_WITH_SECTION_63_A_INPUT,
        source_text=source,
        reverse_context=reverse,
    )
    # The two orders are different prompts: the listings swap places.
    first, second = (_listing(prompt, module) for module in modules)
    assert (prompt.index(first) < prompt.index(second)) != reverse
    return modules, prompt


@pytest.mark.parametrize("reverse", [False, True])
def test_a_blocked_parent_does_not_defer_what_the_exact_cited_file_provides(
    tmp_path, reverse
):
    # The reviewer's probe for finding 2: before the fix the prompt required
    # `us:statutes/26/63/a#taxable_income` and also asked to omit or defer the
    # surface, because the first copied file that covered the input was the
    # blocked parent.
    (parent, child), prompt = _render_section_63(
        tmp_path, COMPOSITION, ATOMIC, reverse=reverse
    )

    assert (
        "copied context target `us:statutes/26/63/a` exports "
        "`us:statutes/26/63/a#taxable_income`." in _section(prompt, MANDATORY)
    )
    invalid = _section(prompt, "Invalid copied local input names:")
    assert (
        f"- {SECTION_63_A_INPUT}: encoded cross-reference placeholder; a "
        "context file for `us:statutes/26/63/a` is available" in invalid
    )
    assert DEFER_INPUT not in prompt
    assert "`us:statutes/26/63` is not an import target" not in invalid
    # No citation-based deferral: the cited provision has its provider.
    assert UNAVAILABLE not in prompt
    assert "copied context target `us:statutes/26/63` has" not in prompt
    # The parent itself stays blocked, and is named as such.
    assert "not an import target: `us:statutes/26/63` has" in _listing(prompt, parent)
    assert f"{parent.target}#" not in prompt
    _assert_defers_to(prompt, parent, COMPOSITION_REASON)
    assert child.target not in _section(prompt, NOT_IMPORT_TARGETS)


@pytest.mark.parametrize("reverse", [False, True])
def test_a_blocked_exact_cited_file_defers_beside_an_importable_parent(
    tmp_path, reverse
):
    (parent, child), prompt = _render_section_63(
        tmp_path, ATOMIC, COMPOSITION, reverse=reverse
    )

    invalid = _section(prompt, "Invalid copied local input names:")
    assert (
        f"- {SECTION_63_A_INPUT}: encoded cross-reference placeholder; the "
        "copied context file for `us:statutes/26/63/a` is not an import "
        f"target, so do not preserve this local input and {DEFER_INPUT}" in invalid
    )
    assert "a context file for `us:statutes/26/63` is available" not in invalid
    assert (
        f"copied context target `us:statutes/26/63/a` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )
    assert f"{child.target}#" not in prompt
    _assert_importable(prompt, parent)


@pytest.mark.parametrize("reverse", [False, True])
def test_two_blocked_providers_defer_and_name_the_exact_cited_file(tmp_path, reverse):
    _modules, prompt = _render_section_63(
        tmp_path, COMPOSITION, COMPOSITION, reverse=reverse
    )

    invalid = _section(prompt, "Invalid copied local input names:")
    assert (
        "the copied context file for `us:statutes/26/63/a` is not an import "
        f"target, so do not preserve this local input and {DEFER_INPUT}" in invalid
    )
    unavailable = _section(prompt, UNAVAILABLE)
    for target in ("us:statutes/26/63", "us:statutes/26/63/a"):
        assert f"copied context target `{target}` has {COMPOSITION_REASON}." in (
            unavailable
        )
    assert MANDATORY not in prompt


@pytest.mark.parametrize("reverse", [False, True])
def test_a_blocked_parent_stays_unavailable_when_the_source_also_cites_it(
    tmp_path, reverse
):
    _modules, prompt = _render_section_63(
        tmp_path,
        COMPOSITION,
        ATOMIC,
        reverse=reverse,
        source="An amount is defined in section 63(a). See also section 63.",
    )

    assert (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )
    assert "`us:statutes/26/63/a#taxable_income`" in _section(prompt, MANDATORY)
    # The input names section 63(a), whose exact file is importable.
    assert DEFER_INPUT not in prompt


@pytest.mark.parametrize("reverse", [False, True])
def test_a_blocked_parent_stays_unavailable_without_an_exact_cited_file(
    tmp_path, reverse
):
    # The importable file is 63(a)(1); the source cites 63(a), which no
    # importable copied file encodes exactly.
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("63/a/1", ATOMIC, "taxable_income"),
    )

    prompt = _render(
        tmp_path, modules, source_text=SECTION_63_A_SOURCE, reverse_context=reverse
    )

    assert (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_an_importable_relative_decides_when_no_exact_file_is_copied(tmp_path, reverse):
    # Neither copied file is section 63(a) itself. The input is deferred only
    # when every covering file is blocked, whichever file is listed first.
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("63/a/1", ATOMIC, "taxable_income"),
    )

    prompt = _render(
        tmp_path,
        modules,
        existing_target=ATOMIC,
        existing_target_rules=EXISTING_TARGET_WITH_SECTION_63_A_INPUT,
        source_text=SECTION_63_A_SOURCE,
        reverse_context=reverse,
    )

    invalid = _section(prompt, "Invalid copied local input names:")
    assert (
        f"- {SECTION_63_A_INPUT}: encoded cross-reference placeholder; a "
        "context file for `us:statutes/26/63/a/1` is available" in invalid
    )
    assert DEFER_INPUT not in prompt


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_blocked_import_is_an_unavailable_import_for_the_fallback_rules(
    tmp_path, backend
):
    # The prompt on main has rules for an import or a cited definition that
    # is not available. The section does not forbid what they allow: it names
    # the order. The cited-context section decides first, as on main.
    cited = _Module("152", COMPOSITION)
    uncited = _Module(
        "plan", COMPOSITION, "plan_rate", relative="policies/example/plan-composition"
    )

    prompt = _render(tmp_path, (cited, uncited), runner_backend=backend)

    flat = _flat(prompt)
    for rule in FALLBACK_RULES:
        assert flat.count(rule) == 1, rule
    assert len(_blocked_lines(prompt)) == 2
    rules = _flat(NOT_IMPORT_TARGETS_RULES)
    first = rules.index("1. If `Unavailable cited RuleSpec context detected` lists")
    second = rules.index("2. If another rule in this prompt says what to do when")
    third = rules.index("3. Otherwise omit or defer only the affected executable")
    assert rules.index("follow the first of these that applies:") < first < second
    assert second < third
    lowered = rules.lower()
    for forbidden in ("do not add a local", "stands in", "never add", "must defer"):
        assert forbidden not in lowered
    # Step 1 names a section of this prompt, which lists only the cited file.
    unavailable = _section(prompt, UNAVAILABLE)
    assert f"copied context target `{cited.target}` has" in unavailable
    assert uncited.target not in unavailable
    assert "Do not synthesize local facts to stand in for unavailable cited" in _flat(
        unavailable
    )


def test_a_file_copied_twice_is_named_once_as_no_import_target(tmp_path):
    path = tmp_path / "152.yaml"
    path.write_text(_Module("152", COMPOSITION).text())
    items = [
        evals.EvalContextFile(
            source_path=str(path),
            workspace_path="context/statutes/26/152.yaml",
            import_path="us:statutes/26/152",
            kind=kind,
        )
        for kind in ("implementation_precedent", "canonical_concept")
    ]
    blockers = evals._context_import_blockers(items, composed_target=False)

    section = evals._format_blocked_context_import_guidance(
        items, import_blockers=blockers
    )

    assert section.count("- `context/statutes/26/152.yaml`: `us:statutes/26/152`") == 1


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("unrelated_blocker", [False, True])
def test_no_blocked_covering_file_keeps_the_input_reason_of_the_earlier_prompt(
    tmp_path, reverse, unrelated_blocker
):
    # Atomic 63 and atomic 63(a): the earlier prompt named the first covering
    # file in context order. The exact-file preference applies only when a
    # covering file is blocked, so this sentence does not change.
    modules = (
        _Module("63", ATOMIC, "taxable_income_total"),
        _Module("63/a", ATOMIC, "taxable_income"),
    )
    if unrelated_blocker:
        modules += (_Module("152", COMPOSITION),)

    prompt = _render(
        tmp_path,
        modules,
        existing_target=ATOMIC,
        existing_target_rules=EXISTING_TARGET_WITH_SECTION_63_A_INPUT,
        source_text=SECTION_63_A_SOURCE,
        reverse_context=reverse,
    )

    first = "us:statutes/26/63/a" if reverse else "us:statutes/26/63"
    assert (
        f"- {SECTION_63_A_INPUT}: encoded cross-reference placeholder; a context "
        f"file for `{first}` is available, so do not preserve this local input "
        "unless the generated file imports an actual exported output for that "
        "same referenced legal concept\n"
        in _section(prompt, "Invalid copied local input names:")
        + "\n"
    )
    assert (NOT_IMPORT_TARGETS in prompt) == unrelated_blocker


def _render_input_63_a(tmp_path, modules, *, reverse):
    return _render(
        tmp_path,
        modules,
        existing_target=ATOMIC,
        existing_target_rules=EXISTING_TARGET_WITH_SECTION_63_A_INPUT,
        source_text=SECTION_63_A_SOURCE,
        reverse_context=reverse,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_two_blocked_relatives_name_the_nearest_ancestor_in_either_order(
    tmp_path, reverse
):
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("63/a/1", COMPOSITION, "taxable_income"),
    )

    prompt = _render_input_63_a(tmp_path, modules, reverse=reverse)

    invalid = _section(prompt, "Invalid copied local input names:")
    assert (
        "the copied context file for `us:statutes/26/63` is not an import "
        f"target, so do not preserve this local input and {DEFER_INPUT}" in invalid
    )
    assert "`us:statutes/26/63/a/1` is not an import target" not in invalid


@pytest.mark.parametrize("reverse", [False, True])
def test_a_blocked_child_of_the_cited_provision_is_not_its_exact_file(
    tmp_path, reverse
):
    # 63(a)(1) is blocked and is not the file of section 63(a), so it does not
    # decide alone: the importable parent is a covering file as well.
    modules = (
        _Module("63", ATOMIC, "taxable_income_total"),
        _Module("63/a/1", COMPOSITION, "taxable_income"),
    )

    prompt = _render_input_63_a(tmp_path, modules, reverse=reverse)

    assert "a context file for `us:statutes/26/63` is available" in _section(
        prompt, "Invalid copied local input names:"
    )
    assert DEFER_INPUT not in prompt


@pytest.mark.parametrize("reverse", [False, True])
def test_two_importable_children_name_the_same_one_in_either_order(tmp_path, reverse):
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("63/a/2", ATOMIC, "second_amount"),
        _Module("63/a/1", ATOMIC, "taxable_income"),
    )

    prompt = _render_input_63_a(tmp_path, modules, reverse=reverse)

    assert "a context file for `us:statutes/26/63/a/1` is available" in _section(
        prompt, "Invalid copied local input names:"
    )


# How a source can name section 63(a), and whether the text also reads as a
# citation of section 63 itself. The first matcher of the prompt builder
# decides both; the provider is resolved from the same matches.
SECTION_63_A_PHRASINGS = {
    "An amount is defined in section 63(a).": False,
    "An amount is defined in section 63(a) of the Social Security Act.": False,
    "An amount is defined in section 63(a) of this title.": False,
    "An amount is defined in section 63(a) of the Internal Revenue Code.": False,
    "An amount is defined in section 63 (a).": False,
    "An amount is defined in Section 63(a), as amended.": False,
    "An amount is defined in sections 63(a) and 152.": False,
    "An amount is defined in § 63(a).": False,
    "An amount is defined in subsection (a) of section 63.": True,
    "An amount is defined in section 63(a)(1).": False,
    "An amount is defined in 26 U.S.C. 63(a).": False,
    "An amount is defined in section 63(a) or (b).": False,
    "An amount is defined in section 63(a) of such Code.": False,
    "An amount is defined in section 63(a) of the Act.": False,
}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("source", sorted(SECTION_63_A_PHRASINGS))
def test_no_phrasing_of_the_citation_requires_the_child_and_defers_the_parent(
    tmp_path, source, reverse
):
    _modules, prompt = _render_section_63(
        tmp_path, COMPOSITION, ATOMIC, reverse=reverse, source=source
    )

    # Every phrasing reads as a citation of the importable file of 63(a).
    assert "`us:statutes/26/63/a#taxable_income`" in _section(prompt, MANDATORY)
    assert DEFER_INPUT not in prompt
    parent_unavailable = (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )
    # The blocked parent is listed only where the words name section 63 alone.
    assert parent_unavailable == SECTION_63_A_PHRASINGS[source]


@pytest.mark.parametrize(
    "source",
    [
        "Amounts under section 63(a) and section 63(b).",
        "Amounts under section 63(a), and under section 63 (b).",
    ],
)
def test_one_cited_provision_without_a_provider_keeps_the_parent_unavailable(
    tmp_path, source
):
    _modules, prompt = _render_section_63(
        tmp_path, COMPOSITION, ATOMIC, reverse=False, source=source
    )

    assert (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )


def test_a_file_of_another_section_does_not_provide_for_a_blocked_file(tmp_path):
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("64/a", ATOMIC, "taxable_income"),
    )

    prompt = _render(
        tmp_path,
        modules,
        source_text="Amounts are defined in section 63(a) and section 64(a).",
    )

    assert "`us:statutes/26/64/a#taxable_income`" in _section(prompt, MANDATORY)
    assert (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )


def test_a_file_the_prompt_does_not_read_as_cited_is_no_provider(tmp_path):
    # The words name another act. The prompt builder on main still reads them
    # as a citation of the copied file of section 63, and does not read them
    # as a citation of the file of 63(a). The provider follows the same
    # reading: no importable file is required, so the parent stays listed.
    _modules, prompt = _render_section_63(
        tmp_path,
        COMPOSITION,
        ATOMIC,
        reverse=False,
        source="An amount is defined in section 63(a)(1) of the Social Security Act.",
    )

    assert MANDATORY not in prompt
    assert (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )


@pytest.mark.parametrize(
    ("sibling_or_parent", "listed"), [("63/b", True), ("63", False)]
)
def test_a_blocked_child_of_the_cited_section_needs_the_file_of_that_section(
    tmp_path, sibling_or_parent, listed
):
    # The source cites section 63. Its blocked child 63(a) stays unavailable
    # beside an importable sibling, and not beside the file of section 63.
    modules = (
        _Module("63/a", COMPOSITION, "taxable_income"),
        _Module(sibling_or_parent, ATOMIC, "other_amount"),
    )

    prompt = _render(tmp_path, modules, source_text="See section 63 for the amount.")

    assert (
        f"copied context target `us:statutes/26/63/a` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    ) == listed
    assert f"`us:statutes/26/{sibling_or_parent}#other_amount`" in _section(
        prompt, MANDATORY
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_two_blocked_ancestors_name_the_nearer_one_in_either_order(tmp_path, reverse):
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("63/a", COMPOSITION, "taxable_income"),
    )

    prompt = _render(
        tmp_path,
        modules,
        existing_target=ATOMIC,
        existing_target_rules=EXISTING_TARGET_WITH_SECTION_63_INPUT.replace(
            "taxable_income_under_section_63", "taxable_income_under_section_63_a_1"
        ),
        source_text="An amount is defined in section 63(a)(1).",
        reverse_context=reverse,
    )

    assert (
        "the copied context file for `us:statutes/26/63/a` is not an import target"
        in _section(prompt, "Invalid copied local input names:")
    )


def test_a_file_of_another_title_does_not_provide_for_a_blocked_file(tmp_path):
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("other", ATOMIC, "taxable_income", relative="statutes/42/63/a"),
    )

    prompt = _render(tmp_path, modules, source_text=SECTION_63_A_SOURCE)

    assert (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )


def test_the_copied_target_does_not_provide_for_its_blocked_child(tmp_path):
    # The target is not an import of itself, so a citation of its own section
    # leaves the blocked child listed.
    child = _Module("151/a", COMPOSITION, "exemption_part")

    prompt = _render(
        tmp_path,
        (child,),
        existing_target=ATOMIC,
        existing_target_rules=EXISTING_TARGET_WITH_SECTION_63_INPUT,
        source_text="The amount under section 151 applies.",
    )

    assert (
        f"copied context target `{child.target}` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )


def test_a_resolved_blocked_parent_without_exports_is_not_listed_at_all(tmp_path):
    modules = (
        _Module("63", COMPOSITION, export=None),
        _Module("63/a", ATOMIC, "taxable_income"),
    )

    prompt = _render(tmp_path, modules, source_text=SECTION_63_A_SOURCE)

    assert UNAVAILABLE not in prompt
    _assert_defers_to(prompt, modules[0], COMPOSITION_REASON)


@pytest.mark.parametrize(
    ("source", "target", "cited"),
    [
        ("See section 63 (a)(1).", "us:statutes/26/63", [("63", ("a", "1"))] * 2),
        ("See section 63(a).", "us:statutes/26/63/a", [("63", ("a",))] * 3),
        ("The amount in 63(a) (2).", "us:statutes/26/63/a", [("63", ("a", "2"))]),
        ("See sections 63, 64(b) and 65.", "us:statutes/26/64/b", [("64", ("b",))] * 2),
        ("See section 64 of the Other Act.", "us:statutes/26/64", [("64", ())]),
        ("See section 631.", "us:statutes/26/63", []),
    ],
)
def test_each_citation_match_yields_the_provision_the_source_names(
    source, target, cited
):
    citation = evals._import_target_to_statute_citation(target)

    matches = list(evals._iter_source_statute_citation_matches(source, citation))

    assert matches == cited
    # The matcher that lists a copied file as cited is this iterator.
    assert evals._source_text_cites_statute(source, citation) == bool(cited)


def test_an_exact_cited_file_without_exports_is_no_provider(tmp_path):
    modules = (
        _Module("63", COMPOSITION, "taxable_income_total"),
        _Module("63/a", ATOMIC, export=None),
    )

    prompt = _render(tmp_path, modules, source_text=SECTION_63_A_SOURCE)

    unavailable = _section(prompt, UNAVAILABLE)
    assert (
        f"copied context target `us:statutes/26/63` has {COMPOSITION_REASON}."
        in unavailable
    )
    assert (
        "copied context target `us:statutes/26/63/a` has no executable exports."
        in unavailable
    )


def test_a_blocked_file_cited_without_the_word_section_stays_unavailable(tmp_path):
    # No parsed citation reaches the blocked file, so nothing resolves it.
    module = _Module("63/a", COMPOSITION, "taxable_income")

    prompt = _render(tmp_path, (module,), source_text="The amount in 63(a) applies.")

    assert (
        f"copied context target `us:statutes/26/63/a` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )


def test_an_empty_parent_keeps_its_unavailable_line_beside_an_exact_file(tmp_path):
    # Only a module the target cannot import is resolved against the exact
    # cited file; the earlier reasons of this section are as on main.
    modules = (
        _Module("63", ATOMIC, export=None),
        _Module("63/a", ATOMIC, "taxable_income"),
    )

    prompt = _render(tmp_path, modules, source_text=SECTION_63_A_SOURCE)

    assert (
        "copied context target `us:statutes/26/63` has no executable exports."
        in _section(prompt, UNAVAILABLE)
    )
    assert NOT_IMPORT_TARGETS not in prompt


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("regulations/7-cfr/273/9", "Income is counted under section 273.9."),
        ("regulations/10-ccr-2506-1/4.401", "See section 4.401 for the standard."),
    ],
)
def test_a_cited_regulation_composition_is_listed_as_unavailable(
    tmp_path, relative, source
):
    # The citation-based section reaches CFR and supported CCR paths as well
    # as statutes; it does not reach the policies root (test above).
    regulation = _Module("regulation", COMPOSITION, "countable", relative=relative)

    prompt = _render(tmp_path, (regulation,), source_text=source)

    assert (
        f"copied context target `{regulation.target}` has {COMPOSITION_REASON}."
        in _section(prompt, UNAVAILABLE)
    )
    _assert_defers_to(prompt, regulation, COMPOSITION_REASON)
    assert f"{regulation.target}#" not in prompt


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
