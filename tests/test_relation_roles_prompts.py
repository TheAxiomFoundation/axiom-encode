"""Keep relation-role instructions in both encoder prompts and retry paths."""

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness.evals import (
    EvalWorkspace,
    _build_empty_artifact_retry_prompt,
    _build_rulespec_eval_prompt,
)
from axiom_encode.prompts.encoder import (
    RELATION_ROLES_FORMULA_PROTOCOL,
    RELATION_ROLES_SELF_CHECK,
    RELATION_ROLES_SHAPE,
    RELATION_ROLES_STRUCTURE_PROTOCOL,
    RELATION_ROLES_TESTS_PROTOCOL,
    get_encoder_prompt,
)

ROLE_BLOCKS = (
    RELATION_ROLES_STRUCTURE_PROTOCOL,
    RELATION_ROLES_FORMULA_PROTOCOL,
    RELATION_ROLES_TESTS_PROTOCOL,
    RELATION_ROLES_SELF_CHECK,
    RELATION_ROLES_SHAPE,
)


@settings(
    max_examples=36,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    include_tests=st.booleans(),
    complete_source_unit=st.booleans(),
    backend=st.sampled_from(["codex", "openai", "claude"]),
    retry=st.sampled_from(["none", "cached", "uncached"]),
    source=st.sampled_from(
        [
            "The household contains people who live together.",
            "The applicant and the applicant's spouse form a household.",
            "The source requires the stated group and excludes a participant.",
        ]
    ),
)
def test_relation_role_rules_match_across_prompt_paths(
    tmp_path: Path,
    include_tests: bool,
    complete_source_unit: bool,
    backend: str,
    retry: str,
    source: str,
):
    source_file = tmp_path / "source.txt"
    source_file.write_text(source)
    manifest_file = tmp_path / "context-manifest.json"
    manifest_file.write_text("{}")
    workspace = EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=manifest_file,
    )
    encoder_prompt = get_encoder_prompt(
        citation="us/regulation/7/273/1",
        output_path="regulations/7/273/1.yaml",
        corpus_citation_path="us/regulation/7/273/1",
        require_complete_source_unit=complete_source_unit,
    )
    eval_prompt = _build_rulespec_eval_prompt(
        citation="us/regulation/7/273/1",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="1.yaml",
        target_ref_prefix="us:regulations/7/273/1",
        include_tests=include_tests,
        runner_backend=backend,
        policyengine_rule_hint=None,
        require_complete_source_unit=complete_source_unit,
    )
    if retry != "none":
        # A plain string exercises the older source-scope trimming path; the
        # prompt subclass exercises the path that preserves its cache boundary.
        eval_prompt = _build_empty_artifact_retry_prompt(
            str(eval_prompt) if retry == "uncached" else eval_prompt,
            target_file_name="1.yaml",
            include_tests=include_tests,
        )

    for block in ROLE_BLOCKS:
        assert encoder_prompt.count(block.strip()) == 1
        assert eval_prompt.count(block.strip()) == 1


@pytest.mark.parametrize(
    "instruction",
    [
        "A contested legal reading gets no default",
        "a provision or cited State policy requires that closure",
        "composition or participation",
        "including false assertions",
        "Never approximate a closure or silently truncate a chain",
        "Never give an election a default",
        "does not yet declare completeness",
    ],
)
def test_relation_role_protocol_keeps_legal_and_runtime_limits(instruction: str):
    normalized = " ".join("\n".join(ROLE_BLOCKS).split())
    assert instruction in normalized


def test_relation_role_shape_preserves_filtered_entity_route():
    prompt = get_encoder_prompt("example", "example.yaml")
    assert "entity: SnapUnit" in prompt
    assert "member_relation: members" in prompt
    assert "source_relation: member_of_household" in prompt
    assert "source_path:" in prompt
    assert "candidate.snap_household_unit == anchor.snap_household_unit" in prompt
    assert "It does not encode any provision in full" in " ".join(prompt.split())


# Independent prose contract: a shared-constant deletion must fail rendering.
PINNED_ROLE_SENTENCES = (
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Relation-role structure: - When the source defines a legal unit's membership, derive a relation from the applicant or other source-stated anchor to persons.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "The engine advertises this surface as `relation_roles`; it adds no `kind: unit` shorthand.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "- Declare `arity: 2`, `slot_entities`, `current_slot` and `related_slot` for an anchored membership relation.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "`anchor` names the current-slot id and `candidate` names the related-slot id.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Declare both slot kinds explicitly when both ends are `Person`; do not infer a direction from their names.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "- Use exactly one candidate source: `source_relation`, or a nonempty `source_path`.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "A path requires explicit binary arity, two slot kinds, and distinct current/related slots in {0, 1}.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Each step has `relation`, `from_slot` and `to_slot`, and traverses a binary relation between distinct slots.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Match kinds at every junction and at the anchor/candidate endpoints.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Traverse a derived step only from its current slot to its related slot.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Paths produce distinct sorted final candidates; an anchor with no roster record reaches nobody.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Path sources cannot declare `entity` or `member_relation`.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "- Keep the existing single-source filtered-entity form when the law filters a structural entity's members and its current id identifies the filtered entity: `source_relation`, `entity`, `member_relation`, `slot_entities`.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "Use an anchored relation when membership depends on the chosen applicant, either endpoint's facts, pair facts, bounded joins or a source path.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "- Preserve the worked guard's caller contract: every co-resident, evaluated or not, has exactly one roster record covering the entire query period; every supplied pair fact covers that period; the pair facts are complete; and caller-supplied household ids are correct.",
    ),
    (
        "RELATION_ROLES_STRUCTURE_PROTOCOL",
        "The guard does not establish these factual preconditions.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Relation-role formulas: - Only derived-relation predicates bind `anchor`, `candidate` and variables introduced by `exists`.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Read facts or rules with `anchor.x`, `candidate.x` or `p.x`; a qualified rule must have the term's entity kind.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Read scalar parameters and indexed parameter tables unqualified, as in a rule formula.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Do not qualify a parameter, relation, `period_start` or `period_end`.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Qualified reads, atoms, term comparisons and `exists` have no scope in ordinary rule bodies, aggregate values or aggregation `where` clauses.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- A term is an entity id, not a scalar input: use it only before `.`, as an atom argument, or in a term comparison.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Compare same-kind terms only with `==` or `!=`, such as `candidate != anchor`; never compare a term to a value.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Reserve `anchor` and `candidate` from declaration names.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- Write relation atoms as `R(t1, ..., tn)` in the relation's declared slot order.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Every argument is a term in scope with the corresponding slot kind; the atom has exactly the declared arity and produces a Judgment.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "A data relation carries tuple membership, not a value on a pair.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "A derived atom requires arity 2 and tests its related-slot term against the members for its current-slot term.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Do not use a built-in function name for an atom.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- Write bounded quantification as `exists(R(anchor, p), body)`: its generator is an atom with exactly one new lowercase `[a-z][a-z0-9_]*` variable and all other arguments already bound.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "A data generator can bind any slot; a derived generator binds only its related slot.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "The body is a Judgment and may nest `exists`; the variable takes its slot kind and exists only inside its body.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Do not hide a visible declaration or rebind an enclosing variable.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "A bound spelling used outside its body never becomes an input.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Domains contain distinct matching ids in sorted order; an empty domain makes `exists` false.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Express a universal test as `not exists(G, not body)`.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- Put pair atoms before the value reads they guard.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Ground numbers in role predicates exactly as in other formulas: `candidate.age < 22` needs the same source support for 22, and a source-stated age threshold should be a named numeric concept consumed by the predicate.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- Give no input read by a relation predicate a default.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "A contested legal reading gets no default: encode it with cited authority or keep it an explicit input.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Do not turn pairwise purchasing or other pair facts into a group closure unless a provision or cited State policy requires that closure, including the State-policy question under 7 CFR 273.1(c).",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- When cited authority requires a closure beyond the bounded language, use the guarded form from engine `docs/relation-roles.md` section 8: compute the set within two ties exactly, prove no tie leaves it, and use that set only when the guard holds and no unresolved exception applies.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Otherwise compare caller-supplied household ids, with no default.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Never approximate a closure or silently truncate a chain.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Keep composition separate from participation; every aggregation names in its rule and proof whether it reads composition or participation, and reads the corresponding relation.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- Purchases that form no single group, such as a buying with b and b with c while a does not buy with c, have no true `meal_group`.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Treat this as a 7 CFR 273.1(c) case: set the exception input and supply household ids.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Supply any `meal_group` value because the fixture still reads it and fails when it is missing.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "With the exception flag set, the ids alone decide the household.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- Declare each election in top-level `inputs:` with `election: {actor, provision}` and exactly those two election keys.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "`actor` is the choosing entity kind in UpperCamelCase, and `provision` is the granting corpus citation path.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "The same module must read the declared input.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Never give an election a default or use `input_or_else` for it; the runtime evaluates the supplied choice and the simulation layer chooses scenarios.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "- This surface provides no recursion, inverse derived traversal, pair-valued facts or dense execution of relation roles.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Atoms, generator relations, qualified rules and path steps remain dependencies; do not create a cycle.",
    ),
    (
        "RELATION_ROLES_FORMULA_PROTOCOL",
        "Supply tuples only for data relations, never for a derived relation, and use strict tuple binding rather than `relation_binding: lenient`.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Relation-role companion tests: - Each case explicitly assigns every base pair relation, including a known false relation as `[]`, and every supported completeness assertion, including false assertions.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Do not treat an omitted pair fact as a supplied false fact.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "The section 8 fixture requires complete pair facts as a caller precondition; this engine surface does not yet declare completeness.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Do not invent completeness keys or imply that the engine checks that precondition.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "- Supply shared person ids and factual reads in `tables.Person` rows, for example `{id: a, <jurisdiction>:<repo-path>#input.meal_group: 1}`.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Put pair tuples under top-level `input:` as canonical `#relation.<name>` keys with rows such as `{tuple: [a, b]}`.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "A tuple row contains only `tuple`, whose nonempty string ids follow the relation's declared slot order.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "These tuples cover the whole case period and connect the named table rows.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "- Assert membership through a derived output that reads the relation, such as `count_where(household_member, test_target)`, with `test_target` true only on the person being tested.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Assert composition and participation separately.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Named table cases use row-ordered scalar output lists.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Cover every derived relation through an asserted output that reads it; relation views themselves are not executable scalar outputs.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "- Exercise both the exact guarded branch and the supplied-id branch, including a tie leaving the two-tie set and an exception that requires supplied ids.",
    ),
    (
        "RELATION_ROLES_TESTS_PROTOCOL",
        "Set every reached endpoint/quantifier fact explicitly, including false facts, and keep the full-period roster and complete pair facts explicit.",
    ),
    (
        "RELATION_ROLES_SELF_CHECK",
        "Relation-role inventory: name both slot roles, check every atom and path-step dependency/import, keep bound variables inside their bodies, and ground every predicate number.",
    ),
    ("RELATION_ROLES_SELF_CHECK", "Give predicate inputs and elections no defaults."),
    (
        "RELATION_ROLES_SELF_CHECK",
        "Cite authority for any grouping closure and contested reading.",
    ),
    (
        "RELATION_ROLES_SELF_CHECK",
        "Identify composition or participation for every aggregation.",
    ),
    (
        "RELATION_ROLES_SELF_CHECK",
        "Check that every pair relation and supported completeness assertion is explicit in each companion case, including `[]` or false.",
    ),
    (
        "RELATION_ROLES_SELF_CHECK",
        "Cover relation membership through asserted derived outputs.",
    ),
    (
        "RELATION_ROLES_SELF_CHECK",
        "Check the exact closure guard and the caller-id fallback against all section 8 factual preconditions.",
    ),
    (
        "RELATION_ROLES_SHAPE",
        "Guarded anchored membership shape: This synthetic encoder example adapts engine `docs/relation-roles.md` section 8.",
    ),
    ("RELATION_ROLES_SHAPE", "It does not encode any provision in full."),
    (
        "RELATION_ROLES_SHAPE",
        "Use it only when cited authority requires the symmetric ties to form a household closure.",
    ),
    (
        "RELATION_ROLES_SHAPE",
        "Resolve the legal tie predicate and exceptions from source; preserve all section 8 preconditions.",
    ),
    ("RELATION_ROLES_SHAPE", "Every input below has no default."),
)


def assert_role_sentences(prompt: str):
    normalized = " ".join(prompt.split())
    for _block_name, sentence in PINNED_ROLE_SENTENCES:
        assert sentence in normalized, sentence


def test_rendered_prompt_pins_every_relation_role_sentence():
    assert_role_sentences(get_encoder_prompt("example", "example.yaml"))


@pytest.mark.parametrize("block_name,sentence", PINNED_ROLE_SENTENCES)
def test_shared_sentence_deletion_is_detected(monkeypatch, block_name, sentence):
    from axiom_encode.harness import evals
    from axiom_encode.prompts import encoder

    block = getattr(encoder, block_name)
    normalized = " ".join(block.split())
    assert sentence in normalized
    shortened = normalized.replace(sentence, "", 1)
    # Protocol strings assembled at import time contain the shared block too.
    # Rebuild their value as a source deletion would when the module imports.
    for module in (encoder, evals):
        for attribute, value in tuple(vars(module).items()):
            if isinstance(value, str) and block.strip() in value:
                monkeypatch.setattr(
                    module, attribute, value.replace(block.strip(), shortened)
                )
    with pytest.raises(AssertionError):
        assert_role_sentences(get_encoder_prompt("example", "example.yaml"))


def test_uncached_scope_hint_keeps_person_anchored_aggregates_on_person(tmp_path):
    from axiom_encode.harness.evals import _MINIMAL_SCOPE_HINT

    source = tmp_path / "source.txt"
    source.write_text("The household contains people who live together.")
    manifest = tmp_path / "context-manifest.json"
    manifest.write_text("{}")
    prompt = _build_rulespec_eval_prompt(
        citation="us/regulation/7/273/1",
        mode="cold",
        workspace=EvalWorkspace(
            root=tmp_path, source_text_file=source, manifest_file=manifest
        ),
        context_files=[],
        target_file_name="1.yaml",
        target_ref_prefix="us:regulations/7/273/1",
        include_tests=True,
        runner_backend="codex",
        policyengine_rule_hint=None,
    )
    retry = _build_empty_artifact_retry_prompt(
        str(prompt), target_file_name="1.yaml", include_tests=True
    )
    assert (
        "For household-level rules without a relation anchored on `Person`"
        in _MINIMAL_SCOPE_HINT
    )
    assert "keep composition and participation aggregates on `entity: Person`" in retry
    assert_role_sentences(retry)
