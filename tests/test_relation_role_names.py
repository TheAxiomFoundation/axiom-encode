"""Keep predicate terms scoped and preserve every real declaration dependency."""

from __future__ import annotations

import ast
import hashlib
import inspect
import subprocess
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.formula_references import (
    FORMULA_BUILTINS,
    analyze_relation_predicate,
    formula_identifiers,
    relation_source_names,
)
from axiom_encode.harness import evals, same_case_execution, source_completeness
from axiom_encode.harness import validator_pipeline as validation

FIXTURES = Path(__file__).parent / "fixtures" / "relation_roles"

# Context-dependent import, source-placement, companion and corpus-authenticity
# checks receive their required contexts in dedicated tests. These checks accept
# the synthetic reference content and its explicitly supplied source evidence.
CONTENT_STATIC_VALIDATORS = (
    "find_aggregate_exception_predicate_issues",
    "find_anaphoric_scope_omission_issues",
    "find_broad_application_passthrough_issues",
    "find_current_purpose_placeholder_issues",
    "find_deferred_output_issues",
    "find_deferred_purpose_specific_limitation_issues",
    "find_delegated_policy_setting_issues",
    "find_deprecated_source_url_issues",
    "find_employer_scoped_entity_issues",
    "find_empty_rules_module_issues",
    "find_entity_limited_aggregation_order_issues",
    "find_filtered_entity_dependency_issues",
    "find_formula_absolute_reference_issues",
    "find_formula_date_literal_issues",
    "find_helper_only_definition_issues",
    "find_import_shape_issues",
    "find_interval_table_reencoding_candidates",
    "find_interval_table_reencoding_issues",
    "find_judgment_conditional_formula_issues",
    "find_local_dependency_temporal_coverage_issues",
    "find_mixed_case_rule_name_token_issues",
    "find_nonnegative_amount_reduction_issues",
    "find_partial_extent_zeroing_issues",
    "find_person_scoped_definition_unit_issues",
    "find_person_scoped_rate_base_unit_issues",
    "find_proof_import_reference_issues",
    "find_relation_aggregate_syntax_issues",
    "find_relation_role_issues",
    "find_role_limited_relation_scope_issues",
    "find_rule_source_metadata_issues",
    "find_scoped_exception_category_gate_issues",
    "find_shared_statutory_rate_entity_suffix_name_issues",
    "find_source_claim_reference_issues",
    "find_source_condition_coverage_issues",
    "find_source_limitation_application_issues",
    "find_source_relation_issues",
    "find_source_scope_consistency_issues",
    "find_source_subparagraph_coverage_issues",
    "find_source_table_row_scalar_parameter_issues",
    "find_source_verification_issues",
    "find_structured_scale_parameter_issue_records",
    "find_structured_scale_parameter_issues",
    "find_synthetic_source_authorized_input_issues",
    "find_tax_filing_status_enum_representation_issues",
    "find_tax_filing_status_local_input_issues",
    "find_tax_filing_status_surviving_spouse_issues",
    "find_tax_status_component_local_input_issues",
    "find_temporal_value_fact_name_issues",
    "find_unconsumed_local_exception_output_issues",
    "find_ungrounded_numeric_issues",
    "find_unused_import_issues",
    "find_unused_modifier_parameter_issues",
    "find_upstream_placement_issues",
    "find_versioned_derived_formula_issues",
)


def declaration(formula):
    """Build a synthetic encoder test input, never a corpus module."""
    return {
        "format": "rulespec/v1",
        "inputs": {"age": {"dtype": "Integer"}, "income": {"dtype": "Integer"}},
        "rules": [
            {
                "name": name,
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
            }
            for name in ("people", "pair")
        ]
        + [
            {
                "name": "person_value",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "period": "Month",
                "versions": [{"effective_from": "2026-01-01", "formula": "age"}],
            },
            {
                "name": "limit",
                "kind": "parameter",
                "dtype": "Integer",
                "versions": [{"effective_from": "2026-01-01", "formula": 22}],
            },
            {
                "name": "selected",
                "kind": "derived_relation",
                "derived_relation": {
                    "arity": 2,
                    "slot_entities": ["Person", "Person"],
                    "current_slot": 0,
                    "related_slot": 1,
                    "source_relation": "people",
                },
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            },
        ],
    }


@st.composite
def predicates(draw, depth=0, scope=("anchor", "candidate")):
    """Generate only the engine's finite relation-predicate grammar."""
    choices = ("read", "atom", "same")
    if depth < 3:
        choices += ("exists", "not", "and", "or")
    shape = draw(st.sampled_from(choices))
    left = draw(st.sampled_from(scope))
    right = draw(st.sampled_from(scope))
    if shape == "read":
        field = draw(st.sampled_from(("age", "income", "person_value")))
        return f"{left}.{field} < limit", {field, "limit"}, set()
    if shape == "atom":
        return f"pair({left}, {right})", {"pair"}, set()
    if shape == "same":
        operator = draw(st.sampled_from(("==", "!=")))
        return f"{left} {operator} {right}", set(), set()
    if shape == "exists":
        variable = f"v{depth}"
        body, names, bound = draw(predicates(depth + 1, (*scope, variable)))
        return (
            f"exists(pair({left}, {variable}), {body})",
            names | {"pair"},
            bound | {variable},
        )
    body, names, bound = draw(predicates(depth + 1, scope))
    if shape == "not":
        return f"not ({body})", names, bound
    other, other_names, other_bound = draw(predicates(depth + 1, scope))
    return f"({body}) {shape} ({other})", names | other_names, bound | other_bound


@settings(max_examples=120, deadline=None)
@given(predicates())
def test_generated_predicate_names_equal_declared_names_used(generated):
    text, expected, bound = generated
    actual = formula_identifiers(text, relation_predicate=True) - FORMULA_BUILTINS
    analyzed = analyze_relation_predicate(text)
    assert actual == expected
    assert not actual & ({"anchor", "candidate"} | bound)
    assert analyzed.bound_variables == bound
    assert analyzed.out_of_scope_variables == frozenset()
    assert analyzed.unbound_atom_terms == frozenset()
    assert validation.find_relation_role_issues(yaml.safe_dump(declaration(text))) == []
    missing = (
        f"exists(pair(anchor, witness), ({text}) and missing_pair(witness, candidate))"
    )
    assert formula_identifiers(missing, relation_predicate=True) - FORMULA_BUILTINS == (
        expected | {"pair", "missing_pair"}
    )
    assert validation.find_relation_role_issues(
        yaml.safe_dump(declaration(missing))
    ) == [
        "[relation-predicate-unknown-relation] Derived relation `selected` calls "
        "undeclared relation `missing_pair`. Declare or import the relation."
    ]


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(predicates(), st.booleans())
def test_generated_unknown_atom_verdict_agrees_with_engine(
    tmp_path, real_engine, generated, undeclared
):
    text, _names, _bound = generated
    if undeclared:
        text = f"exists(pair(anchor, witness), ({text}) and missing_pair(witness, candidate))"
    content = yaml.safe_dump(declaration(text))
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/predicate.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n" + content
    )
    result = subprocess.run(
        [
            str(real_engine.binary),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(tmp_path / "artifact.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    issues = validation.find_relation_role_issues(content)
    assert bool(issues) is undeclared
    assert (result.returncode != 0) is undeclared, result.stderr + result.stdout
    if undeclared:
        assert "unknown function `missing_pair`" in result.stderr + result.stdout


@pytest.mark.parametrize(
    "formula,refusal",
    [
        ("candidate.limit > 0", "relation_role_entity_mismatch"),
        ("exists(pair(anchor, age), age.age < 22)", "relation_term_used_as_value"),
        ("exists(pair(anchor, p), p.missing_fact > 22)", None),
    ],
)
def test_minimized_engine_name_contract_cases(tmp_path, real_engine, formula, refusal):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/predicate.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(declaration(formula))
    )
    result = subprocess.run(
        [
            str(real_engine.binary),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(tmp_path / "artifact.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if refusal is None:
        assert result.returncode == 0, result.stderr + result.stdout
    else:
        assert result.returncode != 0
        assert f"[{refusal}]" in result.stderr + result.stdout


def reference():
    return yaml.safe_load((FIXTURES / "guarded_household.yaml").read_text())


def replace_formula(payload, text, name="snap_within_two_ties"):
    rule = next(rule for rule in payload["rules"] if rule["name"] == name)
    rule["versions"][0]["formula"] = text


def test_guarded_fixture_passes_static_relation_and_numeric_validators():
    content = (FIXTURES / "guarded_household.yaml").read_text()
    assert validation.find_relation_role_issues(content) == []
    assert (
        validation.find_ungrounded_numeric_issues(
            content, source_text=(FIXTURES / "source.txt").read_text()
        )
        == []
    )
    assert validation.find_unused_import_issues(content) == []
    assert validation.find_filtered_entity_dependency_issues(content) == []


def test_guarded_fixture_passes_content_static_validator_inventory(monkeypatch):
    content = (FIXTURES / "guarded_household.yaml").read_text()
    validation._safe_load_unique_keys(content)
    source = (FIXTURES / "source.txt").read_text()

    def synthetic_source(citation):
        assert citation == "us/guidance/axiom/relation-roles"
        return source

    # Supply the fixture's own synthetic evidence through the source reader;
    # the fixture is not a corpus release and makes no corpus-authenticity claim.
    monkeypatch.setattr(validation, "_fetch_corpus_source_text", synthetic_source)
    checked = []
    for name, function in vars(validation).items():
        if not (
            name.startswith("find_")
            and inspect.isfunction(function)
            and function.__module__ == validation.__name__
        ):
            continue
        parameters = list(inspect.signature(function).parameters.values())
        if not parameters or parameters[0].name not in {"content", "rulespec_content"}:
            continue
        if any(
            parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for parameter in parameters[1:]
        ):
            continue
        checked.append(name)
    assert set(checked) == set(CONTENT_STATIC_VALIDATORS)
    for name in CONTENT_STATIC_VALIDATORS:
        assert getattr(validation, name)(content) == [], name


def test_role_coverage_reaches_a_legacy_filtered_member_alias(tmp_path):
    payload = declaration("true")
    payload["rules"].extend(
        [
            {
                "name": "legacy_unit",
                "kind": "derived_relation",
                "derived_relation": {
                    "arity": 2,
                    "source_relation": "people",
                    "entity": "SnapUnit",
                    "member_relation": "members",
                    "slot_entities": ["Person", "Person"],
                    "current_slot": 0,
                    "related_slot": 1,
                },
                "versions": [{"effective_from": "2026-01-01", "formula": True}],
            },
            {
                "name": "legacy_count",
                "kind": "derived",
                "entity": "SnapUnit",
                "dtype": "Integer",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "len(members)"}
                ],
            },
            {
                "name": "selected_count",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "len(selected)"}
                ],
            },
        ]
    )
    root = tmp_path / "rulespec-us/us"
    module = root / "policies/example/legacy.yaml"
    module.parent.mkdir(parents=True)
    content = yaml.safe_dump(payload)
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n" + content
    )
    target = "us:policies/example/legacy"
    cases = [
        {
            "output": {
                f"{target}#{name}": 1
                for name in ("person_value", "legacy_count", "selected_count")
            }
        }
    ]
    assert (
        validation.find_missing_derived_companion_output_issues(
            content, cases, rules_file=module, policy_repo_path=root
        )
        == []
    )


@pytest.mark.parametrize(
    "formula,expected",
    [
        (
            "exists(snap_tie(anchor, p), missing_pair(p, candidate))",
            "[relation-predicate-unknown-relation] Derived relation "
            "`snap_within_two_ties` calls undeclared relation `missing_pair`. "
            "Declare or import the relation.",
        ),
        (
            "exists(snap_tie(anchor, p), snap_tie(p, ghost))",
            "[relation-predicate-unbound-term] Derived relation "
            "`snap_within_two_ties` uses unbound term `ghost` in a relation atom.",
        ),
        (
            "exists(snap_tie(anchor, p), p.age < 22) or p.age < 22",
            "[relation-predicate-variable-scope] Derived relation "
            "`snap_within_two_ties` reads bound variable `p` outside its `exists` body.",
        ),
    ],
)
def test_guarded_fixture_reports_exact_name_findings(formula, expected):
    payload = reference()
    replace_formula(payload, formula)
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [expected]


def test_guarded_fixture_reports_exact_default_finding():
    payload = reference()
    payload["inputs"]["snap_household_unit"]["default"] = 0
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [
        "[relation-input-default] Input `snap_household_unit` read by derived "
        "relation `snap_household_member` must not declare a default. Supply the fact explicitly."
    ]


def test_role_numeric_literal_uses_the_existing_grounding_check():
    payload = declaration("exists(pair(anchor, p), p.age < 22)")
    payload["rules"] = [rule for rule in payload["rules"] if rule["name"] != "limit"]
    assert validation.find_ungrounded_numeric_issues(
        yaml.safe_dump(payload), source_text="The limit is 18."
    ) == [
        "Ungrounded generated numeric literal: 22 does not appear as a substantive "
        "numeric value in the source text."
    ]


@pytest.mark.parametrize(
    "change,expected",
    [
        (
            {"source_relation": "spouse_of"},
            "declares both `source_relation` and `source_path`; declare exactly one.",
        ),
        (
            {"entity": "SnapUnit"},
            "uses `source_path` with `entity` or `member_relation`; path sources "
            "cannot create a filtered-entity alias.",
        ),
        (
            {"current_slot": None},
            "must declare `arity: 2`, two `slot_entities`, and distinct explicit "
            "`current_slot`/`related_slot` values 0 and 1 for `source_path`.",
        ),
        ({"source_path": []}, "must declare a nonempty `source_path` list."),
        (
            {
                "source_path": [
                    {"relation": "member_of_household", "from_slot": 0, "to_slot": 0}
                ]
            },
            "step 1 must name a binary relation and distinct `from_slot`/`to_slot` values 0 and 1.",
        ),
    ],
)
def test_guarded_fixture_reports_exact_path_findings(change, expected):
    payload = reference()
    lives_with = next(rule for rule in payload["rules"] if rule["name"] == "lives_with")
    lives_with["derived_relation"].update(change)
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [
        f"[relation-source-path] Derived relation `lives_with` {expected}"
    ]


def test_default_check_follows_scalar_helpers():
    payload = declaration("candidate.person_value < limit")
    payload["inputs"]["age"]["default"] = 0
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [
        "[relation-input-default] Input `age` read by derived relation `selected` "
        "must not declare a default. Supply the fact explicitly."
    ]


@pytest.mark.parametrize("mapping", [False, True])
def test_election_actor_and_no_default_checks(mapping):
    payload = declaration("candidate.requested")
    entry = {
        "name": "requested",
        "election": {"actor": "Household", "provision": "us/regulation/7/273/1/b/3"},
    }
    payload["inputs"] = {"requested": entry} if mapping else [entry]
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == []
    entry["default"] = False
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [
        "[relation-input-default] Input `requested` read by derived relation "
        "`selected` must not declare a default. Supply the fact explicitly.",
        "[election-default] Election input `requested` must not declare a default. "
        "Supply the actor's choice explicitly.",
    ]
    del entry["default"]
    del entry["election"]["actor"]
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [
        "[election-declaration] Input `requested` must declare exactly "
        "`election.actor` as an UpperCamelCase entity kind and "
        "`election.provision` as a canonical corpus citation path."
    ]


def test_unknown_scalar_inside_exists_remains_an_inferred_input():
    payload = declaration("exists(pair(anchor, p), p.missing_fact)")
    payload["module"] = {"proof_validation": {"required": True}}
    cases = [
        {"name": "explicit_false_facts", "input": {}, "output": {"selected": True}}
    ]
    assert "missing_fact" in validation._rulespec_formula_identifiers(payload)
    assert validation.find_test_input_assignment_issues(
        yaml.safe_dump(payload), cases
    ) == [
        "Test input assignment missing: `explicit_false_facts` does not assign "
        "#input.missing_fact. Every test for a proof-required RuleSpec module must "
        "set all local factual inputs, including false facts, so tests cannot "
        "pass through implicit defaults."
    ]


def test_bare_function_names_stay_inputs_and_calls_stay_builtins():
    text = "candidate.age < max(exists, min)"
    assert formula_identifiers(text, relation_predicate=True) == {
        "age",
        "exists",
        "min",
    }
    assert validation._rulespec_formula_identifiers(declaration(text)) >= {
        "age",
        "exists",
        "min",
    }
    assert formula_identifiers("anchor + candidate + exists") == {
        "anchor",
        "candidate",
        "exists",
    }


@pytest.mark.parametrize(
    "formula",
    ["not (anchor.age < limit)", "(anchor == candidate) or (candidate.age < limit)"],
)
def test_boolean_keywords_before_groups_are_not_relation_atoms(formula):
    assert analyze_relation_predicate(formula).atom_relations == frozenset()
    assert (
        validation.find_relation_role_issues(yaml.safe_dump(declaration(formula))) == []
    )


def test_relation_predicate_consumes_a_helper_named_after_a_builtin(tmp_path):
    payload = declaration("candidate.min < limit")
    payload["rules"][2]["name"] = "min"
    module = tmp_path / "context.yaml"
    module.write_text(yaml.safe_dump(payload))
    assert "min" not in evals._context_file_terminal_exports(str(module))
    assert "min" not in validation._rulespec_terminal_executable_names_from_payload(
        payload
    )


def test_atom_and_path_sources_count_as_import_dependencies():
    payload = declaration("exists(pair(anchor, p), external_pair(p, candidate))")
    payload["imports"] = ["us:policies/example/pairs#relation.external_pair"]
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == []
    selected = payload["rules"][-1]
    selected["versions"][0]["formula"] = True
    selected["derived_relation"].pop("source_relation")
    selected["derived_relation"]["source_path"] = [
        {"relation": "external_pair", "from_slot": 0, "to_slot": 1}
    ]
    assert relation_source_names(selected["derived_relation"]) == {"external_pair"}
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []


def test_relation_atom_analysis_follows_the_import_closure(tmp_path):
    root = tmp_path / "rulespec-us/us"
    directory = root / "policies/example"
    directory.mkdir(parents=True)
    (directory / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [
                    {
                        "name": "external_pair",
                        "kind": "data_relation",
                        "data_relation": {
                            "arity": 2,
                            "arguments": ["Person", "Person"],
                        },
                    }
                ],
            }
        )
    )
    (directory / "pair_export.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [],
                "imports": ["us:policies/example/base#relation.external_pair"],
            }
        )
    )
    payload = declaration("exists(pair(anchor, p), external_pair(p, candidate))")
    payload["imports"] = ["us:policies/example/pair_export"]
    content = yaml.safe_dump(payload)
    module = directory / "predicate.yaml"
    module.write_text(content)
    assert (
        validation.find_relation_role_issues(
            content, rules_file=module, policy_repo_path=root
        )
        == []
    )


def test_relation_atoms_and_path_sources_keep_temporal_dependency_checks():
    payload = declaration("pair(anchor, candidate)")
    payload["rules"][1]["kind"] = "derived_relation"
    payload["rules"][1]["derived_relation"] = {
        "arity": 2,
        "slot_entities": ["Person", "Person"],
        "current_slot": 0,
        "related_slot": 1,
        "source_relation": "people",
    }
    payload["rules"][1]["versions"] = [
        {"effective_from": "2026-02-01", "formula": True}
    ]
    expected = (
        "[temporal-dependency-coverage] Derived rule `selected` version 1 covers "
        "2026-01-01 through unbounded but directly references local rule `pair`, "
        "whose version intervals (2026-02-01 through unbounded) do not cover that "
        "entire period. Align the derived interval, extend the dependency with "
        "authoritative coverage, or add a later derived version."
    )
    assert validation.find_local_dependency_temporal_coverage_issues(
        yaml.safe_dump(payload)
    ) == [expected]
    selected = payload["rules"][-1]
    selected["versions"][0]["formula"] = True
    selected["derived_relation"].pop("source_relation")
    selected["derived_relation"]["source_path"] = [
        {"relation": "pair", "from_slot": 0, "to_slot": 1}
    ]
    assert validation.find_local_dependency_temporal_coverage_issues(
        yaml.safe_dump(payload)
    ) == [expected]


def test_proof_import_atom_can_name_a_relation_used_by_a_path():
    payload = declaration("pair(anchor, candidate)")
    selected = payload["rules"][-1]
    selected["metadata"] = {
        "proof": {
            "atoms": [
                {
                    "import": {
                        "target": "us:policies/example/pairs#relation.pair",
                        "output": "pair",
                    }
                }
            ]
        }
    }
    assert validation.find_proof_import_reference_issues(yaml.safe_dump(payload)) == []
    selected["versions"][0]["formula"] = True
    selected["derived_relation"]["source_relation"] = "pair"
    assert validation.find_proof_import_reference_issues(yaml.safe_dump(payload)) == []


def test_role_reads_reach_context_and_source_dependency_walks(tmp_path):
    payload = declaration(
        "exists(pair(anchor, p), p.age < limit and candidate.income > 0)"
    )
    module = tmp_path / "context.yaml"
    module.write_text(yaml.safe_dump(payload))
    assert evals._context_file_local_inputs(str(module)) == {"age", "income"}
    selected = payload["rules"][-1]
    assert source_completeness._rule_formula_identifiers(
        selected
    ) - FORMULA_BUILTINS == {"age", "income", "limit", "pair", "people"}
    assert same_case_execution._formula_names(
        "exists(pair(anchor, p), p.imported_rule)", relation_predicate=True
    ) == {"pair", "imported_rule"}


def legacy_filtered_fixtures():
    """Read every tracked constant fixture instead of copying selected examples."""
    root = Path(__file__).parents[1]
    filenames = subprocess.check_output(
        ["git", "ls-files", "tests/test_*.py"], cwd=root, text=True
    ).splitlines()
    fixtures = []
    for filename in filenames:
        if filename == "tests/test_relation_role_names.py":
            continue
        source = (root / filename).read_text()
        if "member_relation" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if (
                "format: rulespec/v1" not in node.value
                or "member_relation" not in node.value
            ):
                continue
            try:
                payload = yaml.safe_load(node.value)
            except yaml.YAMLError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
                fixtures.append((f"{filename}:{node.lineno}", node.value))
    assert fixtures
    return fixtures


LEGACY_FILTERED_FIXTURES = legacy_filtered_fixtures()
BASELINE = yaml.safe_load((FIXTURES / "legacy_filtered_findings.yaml").read_text())


def fixture_identity(label, content):
    return label.rsplit(":", 1)[0], hashlib.sha256(content.encode()).hexdigest()


BASELINE_FINDINGS = {
    (row["fixture"].rsplit(":", 1)[0], row["sha256"]): row
    for row in BASELINE["fixtures"]
}


def test_pinned_baseline_covers_every_existing_filtered_entity_fixture():
    assert BASELINE["baseline_commit"] == "5d40209bc0e1fea5f740581ad5bb1b114ceeff07"
    assert {fixture_identity(*fixture) for fixture in LEGACY_FILTERED_FIXTURES} == set(
        BASELINE_FINDINGS
    )
    assert len(LEGACY_FILTERED_FIXTURES) == len(BASELINE["fixtures"]) == 6


def assert_legacy_filtered_findings(label, content, tmp_path):
    baseline = BASELINE_FINDINGS[fixture_identity(label, content)]
    payload = yaml.safe_load(content)
    assert validation._rulespec_formula_identifiers(payload) == set(
        baseline["identifiers"]
    ), label
    assert validation.find_relation_role_issues(content) == [], label
    for name, expected in baseline["findings"].items():
        args = (content,)
        kwargs = {}
        if name == "find_test_input_assignment_issues":
            args += ([{"name": "empty_assignments", "input": {}, "output": None}],)
        elif name == "find_missing_derived_companion_output_issues":
            root = tmp_path / "rulespec-us/us"
            module = root / "policies/example/legacy.yaml"
            module.parent.mkdir(parents=True, exist_ok=True)
            args += ([],)
            kwargs = {"rules_file": module, "policy_repo_path": root}
        assert getattr(validation, name)(*args, **kwargs) == expected, (label, name)


@pytest.mark.parametrize("label,content", LEGACY_FILTERED_FIXTURES)
def test_every_existing_filtered_entity_fixture_keeps_its_name_findings(
    label, content, tmp_path
):
    assert_legacy_filtered_findings(label, content, tmp_path)


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.sampled_from(LEGACY_FILTERED_FIXTURES))
def test_filtered_entity_regression_property(tmp_path, fixture):
    assert_legacy_filtered_findings(*fixture, tmp_path)


@pytest.mark.parametrize(
    "formula,predicate,expected",
    [
        ("len(candidate)", False, {"candidate"}),
        ("len(period_start)", False, {"period_start"}),
        ("len((members))", False, {"members"}),
        ("sum((members).age)", False, {"members"}),
        ("only(members.age)", False, {"members"}),
        ("min(members.age)", False, {"members"}),
        ("max(members.age)", False, {"members"}),
        ("count_where(members, eligible)", False, {"members"}),
        ("sum_where(members, amount, eligible)", False, {"members"}),
        ("min_where(members, amount, eligible)", False, {"members"}),
        ("max_where(members, amount, eligible)", False, {"members"}),
        ("only_related(members, eligible)", False, {"members"}),
        ("min(candidate.age)", True, set()),
        ("min(age, 22)", False, set()),
        ("members", False, set()),
        ("candidate", False, set()),
        ("sum_over_periods(amount, period_start, period_end)", False, set()),
    ],
)
def test_relation_arguments_follow_native_function_positions(formula, predicate, expected):
    assert analyze_relation_predicate(
        formula, predicate_terms=predicate
    ).relation_arguments == expected


def test_scalar_formula_scans_keep_ordinary_role_named_inputs():
    rule = {"kind": "derived", "versions": [{"formula": "anchor + candidate"}]}
    assert analyze_relation_predicate(
        "anchor + candidate", predicate_terms=False
    ).identifiers == {"anchor", "candidate"}
    assert source_completeness._rule_formula_identifiers(rule) == {"anchor", "candidate"}
    assert same_case_execution._formula_names("anchor + candidate") == {"anchor", "candidate"}
    assert formula_identifiers("anchor + candidate") == {"anchor", "candidate"}


def _owned_default_program(tmp_path, *, provider_default, local_default):
    """Build synthetic encoder test inputs, never corpus modules."""
    root = tmp_path / "rulespec-us/us"
    module = root / "policies/example/predicate.yaml"
    module.parent.mkdir(parents=True)
    provider = {
        "format": "rulespec/v1",
        "inputs": {
            "raw_flag": {"entity": "Person", "dtype": "Judgment"},
            "unused_fact": {"entity": "Person", "dtype": "Judgment", "default": False},
        },
        "rules": [{
            "name": "eligible", "kind": "derived", "entity": "Person",
            "dtype": "Judgment", "period": "Month",
            "versions": [{"effective_from": "2026-01-01", "formula": "raw_flag"}],
        }],
    }
    if provider_default:
        provider["inputs"]["raw_flag"]["default"] = False
    payload = {
        "format": "rulespec/v1",
        "imports": ["us:policies/example/provider#eligible"],
        "inputs": {"raw_flag": {"entity": "Person", "dtype": "Judgment"}},
        "rules": [
            {"name": "people", "kind": "data_relation", "data_relation": {
                "arity": 2, "arguments": ["Person", "Person"],
            }},
            {"name": "selected", "kind": "derived_relation", "derived_relation": {
                "arity": 2, "slot_entities": ["Person", "Person"],
                "current_slot": 0, "related_slot": 1, "source_relation": "people",
            }, "versions": [{"effective_from": "2026-01-01", "formula": "candidate.eligible"}]},
            {"name": "selected_count", "kind": "derived", "entity": "Person",
             "dtype": "Integer", "period": "Month", "versions": [{
                 "effective_from": "2026-01-01", "formula": "len(selected)",
             }]},
        ],
    }
    if local_default:
        payload["inputs"]["raw_flag"]["default"] = False
    for path, document in [(module, payload), (module.with_name("provider.yaml"), provider)]:
        path.write_text("# Synthetic encoder test input; not a RuleSpec module.\n" + yaml.safe_dump(document))
    return root, module


@pytest.mark.parametrize("provider_default,local_default", [(True, False), (False, True), (True, True), (False, False)])
def test_imported_predicate_helper_defaults_keep_input_ownership(tmp_path, provider_default, local_default):
    root, module = _owned_default_program(
        tmp_path, provider_default=provider_default, local_default=local_default
    )
    expected = ([
        "[relation-input-default] Input `us:policies/example/provider#input.raw_flag` "
        "read by derived relation `selected` must not declare a default. "
        "Supply the fact explicitly."
    ] if provider_default else [])
    assert validation.find_relation_role_issues(
        module.read_text(), rules_file=module, policy_repo_path=root
    ) == expected


@pytest.mark.parametrize("provider_default,local_default", [(True, False), (False, True)])
@pytest.mark.parametrize("omit_optional", [False, True])
def test_imported_predicate_helper_default_counterexamples_compile(tmp_path, real_engine, provider_default, local_default, omit_optional):
    root, module = _owned_default_program(
        tmp_path, provider_default=provider_default, local_default=local_default
    )
    result = subprocess.run([
        str(real_engine.binary), "compile", "--program", str(module),
        "--rulespec-root", str(root.parent), "--output", str(tmp_path / "artifact.json"),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    compiled = yaml.safe_load((tmp_path / "artifact.json").read_text())
    if omit_optional:
        helper = next(rule for rule in compiled["program"]["derived"] if rule["name"] == "eligible")
        helper.pop("id", None)
        helper.pop("versions", None)
    expected = ([
        "[relation-input-default] Input `us:policies/example/provider#input.raw_flag` "
        "read by derived relation `selected` must not declare a default. "
        "Supply the fact explicitly."
    ] if provider_default else [])
    assert validation.find_relation_role_issues(
        module.read_text(), rules_file=module, policy_repo_path=root,
        compiled_payload=compiled,
    ) == expected
    # Unrelated scalar operators do not widen relation predicate validation.
    compiled["program"]["derived"].append({
        "id": "us:policies/example/predicate#unrelated",
        "name": "unrelated", "expr": {"kind": "future_operator"},
    })
    assert validation.find_relation_role_issues(
        module.read_text(), rules_file=module, policy_repo_path=root,
        compiled_payload=compiled,
    ) == expected


def _mixed_alias_payload(alias, formula):
    """Build a synthetic mixed-form encoder input, never a corpus module."""
    payload = {'format': 'rulespec/v1',
     'rules': [{'data_relation': {'arguments': ['Person', 'Household'], 'arity': 2},
                'kind': 'data_relation',
                'name': 'roster'},
               {'dtype': 'Integer',
                'entity': 'Person',
                'kind': 'derived',
                'name': 'person_marker',
                'period': 'Month',
                'versions': [{'effective_from': '2026-01-01', 'formula': 'age'}]},
               {'dtype': 'Integer',
                'entity': 'Household',
                'kind': 'derived',
                'name': 'household_marker',
                'period': 'Month',
                'versions': [{'effective_from': '2026-01-01', 'formula': 1}]},
               {'derived_relation': {'arity': 2,
                                     'current_slot': 1,
                                     'entity': 'SnapUnit',
                                     'member_relation': 'members',
                                     'related_slot': 0,
                                     'slot_entities': ['Person', 'Household'],
                                     'source_relation': 'roster'},
                'kind': 'derived_relation',
                'name': 'legacy_unit',
                'versions': [{'effective_from': '2026-01-01', 'formula': True}]},
               {'dtype': 'Integer',
                'entity': 'SnapUnit',
                'kind': 'derived',
                'name': 'legacy_count',
                'period': 'Month',
                'versions': [{'effective_from': '2026-01-01', 'formula': 'legacy_unit'}]},
               {'derived_relation': {'arity': 2,
                                     'current_slot': 1,
                                     'related_slot': 0,
                                     'slot_entities': ['Person', 'Household'],
                                     'source_relation': 'roster'},
                'kind': 'derived_relation',
                'name': 'role_members',
                'versions': [{'effective_from': '2026-01-01',
                              'formula': 'candidate.age >= 0'}]},
               {'dtype': 'Integer',
                'entity': 'Household',
                'kind': 'derived',
                'name': 'role_count',
                'period': 'Month',
                'versions': [{'effective_from': '2026-01-01',
                              'formula': 'len(role_members)'}]}]}
    for rule in payload["rules"]:
        if rule["name"] == "legacy_unit":
            rule["derived_relation"]["member_relation"] = alias
        elif rule["name"] == "legacy_count":
            rule["versions"][0]["formula"] = formula
    return payload


def _write_mixed_alias_program(tmp_path, payload):
    root = tmp_path / "rulespec-us/us"
    module = root / "policies/example/alias.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("# Synthetic encoder test input; not a RuleSpec module.\n" + yaml.safe_dump(payload))
    cases = [{"output": {
        f"us:policies/example/alias#{rule['name']}": 1
        for rule in payload["rules"] if rule["kind"] == "derived"
    }}]
    return root, module, cases


@pytest.mark.parametrize("alias", ["members", "anchor", "candidate", "period_start", "min"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_relation_coverage_requires_actual_aggregate_alias_arguments(tmp_path, alias, aggregate):
    formula = f"len({alias})" if aggregate else alias
    root, module, cases = _write_mixed_alias_program(tmp_path, _mixed_alias_payload(alias, formula))
    expected = [] if aggregate else [
        "Derived relation missing companion output coverage: "
        "`us:policies/example/alias#legacy_unit` is not reached by an asserted "
        "scalar output in the companion `.test.yaml` file."
    ]
    assert validation.find_missing_derived_companion_output_issues(
        module.read_text(), cases, rules_file=module, policy_repo_path=root
    ) == expected


@pytest.mark.parametrize("judgment", [False, True])
def test_scalar_bare_relation_name_does_not_cover_membership(tmp_path, judgment):
    payload = _mixed_alias_payload("members", "legacy_unit")
    if judgment:
        next(rule for rule in payload["rules"] if rule["name"] == "legacy_count")["dtype"] = "Judgment"
    root, module, cases = _write_mixed_alias_program(tmp_path, payload)
    assert validation.find_missing_derived_companion_output_issues(
        module.read_text(), cases, rules_file=module, policy_repo_path=root
    ) == [
        "Derived relation missing companion output coverage: "
        "`us:policies/example/alias#legacy_unit` is not reached by an asserted "
        "scalar output in the companion `.test.yaml` file."
    ]


def test_bare_relation_judgment_in_predicate_keeps_membership_dependency(tmp_path):
    payload = _mixed_alias_payload("members", "len(members)")
    next(rule for rule in payload["rules"] if rule["name"] == "role_members")["versions"][0]["formula"] = "legacy_unit"
    root, module, cases = _write_mixed_alias_program(tmp_path, payload)
    # Assert the wrapper for role_members only: its predicate reads legacy_unit.
    cases[0]["output"].pop("us:policies/example/alias#legacy_count")
    findings = validation.find_missing_derived_companion_output_issues(
        module.read_text(), cases, rules_file=module, policy_repo_path=root
    )
    assert not any("#legacy_unit`" in finding for finding in findings)


@pytest.mark.parametrize("alias,formula,kind", [
    ("candidate", "candidate", "input"),
    ("candidate", "len(candidate)", "count_related"),
    ("period_start", "len(period_start)", "count_related"),
    ("min", "min", "input"),
    ("min", "len(min)", "count_related"),
    ("members", "members", "input"),
    ("members", "len(members)", "count_related"),
    ("members", "legacy_unit", "input"),
])
def test_native_alias_counterexamples_keep_scalar_and_relation_positions_distinct(tmp_path, real_engine, alias, formula, kind):
    root, module, _cases = _write_mixed_alias_program(tmp_path, _mixed_alias_payload(alias, formula))
    artifact = tmp_path / "artifact.json"
    result = subprocess.run([
        str(real_engine.binary), "compile", "--program", str(module),
        "--rulespec-root", str(root.parent), "--output", str(artifact),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    compiled = yaml.safe_load(artifact.read_text())
    rule = next(rule for rule in compiled["program"]["derived"] if rule["name"] == "legacy_count")
    assert rule["expr"]["kind"] == kind


def test_native_predicate_bare_membership_counterexample_compiles(tmp_path, real_engine):
    payload = _mixed_alias_payload("members", "len(members)")
    next(rule for rule in payload["rules"] if rule["name"] == "role_members")["versions"][0]["formula"] = "legacy_unit"
    root, module, _cases = _write_mixed_alias_program(tmp_path, payload)
    artifact = tmp_path / "artifact.json"
    result = subprocess.run([
        str(real_engine.binary), "compile", "--program", str(module),
        "--rulespec-root", str(root.parent), "--output", str(artifact),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    compiled = yaml.safe_load(artifact.read_text())
    relation = next(relation for relation in compiled["program"]["relations"] if relation["name"].endswith("#relation.role_members"))
    assert relation["derivation"]["predicate"]["kind"] == "relation_member"


@pytest.mark.parametrize("collision", ["scalar_pair_input", "aggregate_alias_metadata"])
def test_native_default_inventory_distinguishes_value_reads_and_alias_arguments(tmp_path, real_engine, collision):
    payload = _mixed_alias_payload("members", "len(members)")
    relation = next(rule for rule in payload["rules"] if rule["name"] == "role_members")
    if collision == "scalar_pair_input":
        payload["inputs"] = {"roster": {"entity": "Person", "dtype": "Judgment", "default": False}}
        relation["versions"][0]["formula"] = "roster == true"
    else:
        payload["inputs"] = {"members": {"entity": "SnapUnit", "dtype": "Integer", "default": 0}}
        payload["rules"].append({"name": "groups", "kind": "data_relation", "data_relation": {"arity": 2, "arguments": ["Household", "Household"]}})
        relation["derived_relation"]["slot_entities"] = ["Household", "Household"]
        relation["derived_relation"]["source_relation"] = "groups"
        relation["versions"][0]["formula"] = "candidate.legacy_count >= 0"
    root, module, _cases = _write_mixed_alias_program(tmp_path, payload)
    artifact = tmp_path / "artifact.json"
    result = subprocess.run([
        str(real_engine.binary), "compile", "--program", str(module),
        "--rulespec-root", str(root.parent), "--output", str(artifact),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    compiled = yaml.safe_load(artifact.read_text())
    expected = ([
        "[relation-input-default] Input `roster` read by derived relation `role_members` "
        "must not declare a default. Supply the fact explicitly."
    ] if collision == "scalar_pair_input" else [])
    assert validation.find_relation_role_issues(
        module.read_text(), rules_file=module, policy_repo_path=root, compiled_payload=compiled
    ) == expected
    # An unknown native node must fail the inventory rather than erase checks.
    next(relation for relation in compiled["program"]["relations"] if relation.get("derivation"))["derivation"]["predicate"] = {"kind": "future_operator"}
    assert any(finding.startswith("[relation-input-analysis]") for finding in validation.find_relation_role_issues(
        module.read_text(), rules_file=module, policy_repo_path=root, compiled_payload=compiled
    ))


def terminal_relation_payload(formula, *, relation_name="selected", member_alias=None):
    payload = declaration("true")
    relation = payload["rules"][-1]
    relation["name"] = relation_name
    entity = "Person"
    if member_alias is not None:
        entity = "SnapUnit"
        relation["derived_relation"].update(entity=entity, member_relation=member_alias)
    payload["rules"].append(
        {
            "name": "relation_count",
            "kind": "derived",
            "entity": entity,
            "dtype": "Integer",
            "period": "Month",
            "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        }
    )
    return payload


def assert_terminal_relation_use(tmp_path, payload, *, relation_name, consumed):
    module = tmp_path / "terminal.yaml"
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(payload)
    )
    assert (
        relation_name in evals._context_file_terminal_exports(str(module))
    ) is not consumed
    assert (
        relation_name
        in validation._rulespec_terminal_executable_names_from_payload(payload)
    ) is not consumed


@pytest.mark.parametrize(
    "relation_name,formula,consumed",
    [
        ("min", "len(min)", True),
        ("min", "min(age, 22)", False),
        ("min", "min", False),
        ("members", "len(members)", True),
        ("members", "members", False),
    ],
)
def test_terminal_relation_exports_distinguish_aggregate_call_and_input(
    tmp_path, relation_name, formula, consumed
):
    assert_terminal_relation_use(
        tmp_path,
        terminal_relation_payload(formula, relation_name=relation_name),
        relation_name=relation_name,
        consumed=consumed,
    )


@pytest.mark.parametrize("member_alias", ["members", "anchor", "candidate"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_terminal_relation_exports_follow_only_aggregated_filtered_aliases(
    tmp_path, member_alias, aggregate
):
    formula = f"len({member_alias})" if aggregate else member_alias
    assert_terminal_relation_use(
        tmp_path,
        terminal_relation_payload(
            formula, relation_name="legal_unit", member_alias=member_alias
        ),
        relation_name="legal_unit",
        consumed=aggregate,
    )
