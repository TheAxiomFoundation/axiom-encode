"""Compare formula syntax and read positions with the engine's grammar."""

from __future__ import annotations

import json
import subprocess

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.formula_grammar import FormulaSyntaxError, parse_formula, tokenise
from axiom_encode.formula_references import (
    analyze_relation_predicate,
    formula_identifiers,
    rule_uses_relation_predicate,
)
from tests.test_companion_declared_entities import _compile
from tests.test_relation_role_round2 import LABEL, collision_payload


@pytest.mark.parametrize(
    "formula,predicate",
    [
        ("member_entry_date >= 2025-01-01", False),
        ("age < 22", False),
        ('code == "candidate.age exists(pair(anchor, p), true)"', False),
        ('""" candidate.age """ age < 22', False),
        ("candidate.age < 22", True),
        ("anchor == candidate", True),
        ("exists((pair(anchor, p)), p.age < 22)", True),
        ("pair(anchor, candidate)", True),
        ("candidate.age >= 2025-01-01", True),
        ("'unterminated", True),
    ],
)
def test_legacy_dispatch_preserves_scalar_findings_and_keeps_role_syntax(
    formula, predicate
):
    rule = {
        "kind": "derived_relation",
        "derived_relation": {"entity": "Unit", "member_relation": "members"},
        "versions": [{"formula": formula}],
    }
    assert rule_uses_relation_predicate(rule) is predicate
    if not predicate:
        assert analyze_relation_predicate(
            formula, predicate_terms=False
        ).identifiers == formula_identifiers(formula) - {"true", "false"}
    rule["derived_relation"]["source_path"] = []
    assert rule_uses_relation_predicate(rule)
    rule["derived_relation"] = {"entity": "Unit"}
    assert rule_uses_relation_predicate(rule)


@pytest.mark.parametrize(
    "formula,inputs,relations",
    [
        pytest.param(
            "exists((pair(anchor, p)), true)",
            set(),
            {"pair"},
            id="b1-parenthesized-generator",
        ),
        pytest.param(
            "exists(pair(anchor, (p)), true)",
            set(),
            {"pair"},
            id="b1-parenthesized-binder",
        ),
        pytest.param(
            "(len)(people_forward) > 0",
            set(),
            {"people_forward"},
            id="b1-parenthesized-function",
        ),
        pytest.param("1_0 > 0", set(), set(), id="b2-number-underscores"),
        pytest.param("(match 1: _ => 1) > 0", set(), set(), id="b2-match-wildcard"),
        pytest.param(
            '""" " ghost(anchor, candidate) " """ true', set(), set(), id="b2-docstring"
        ),
        pytest.param("x/y > 0", {"x/y"}, set(), id="b2-slash-path"),
        pytest.param(
            "len(people_forward.ignored) > 0",
            set(),
            {"people_forward"},
            id="b3-len-ignores-projection",
        ),
        pytest.param(
            "sum(people_forward.income) > 0",
            {"income"},
            {"people_forward"},
            id="b3-sum-reads-projection",
        ),
        pytest.param(
            "people_forward > 0",
            {"people_forward"},
            set(),
            id="a1-relation-name-in-value-position",
        ),
        pytest.param(
            "people_forward",
            set(),
            {"people_forward"},
            id="a1-relation-name-in-judgment-position",
        ),
        pytest.param(
            "(match phantom: _ => 1) > 0",
            set(),
            set(),
            id="wildcard-only-match-discards-subject",
        ),
        pytest.param(
            "len(people_forward, phantom) > 0",
            set(),
            {"people_forward"},
            id="len-ignores-trailing-arguments",
        ),
    ],
)
def test_review_reference_regressions(formula, inputs, relations):
    actual = analyze_relation_predicate(
        formula, relation_names=frozenset({"pair", "people_forward"})
    )
    assert actual.parse_error is None
    assert actual.analysis_errors == ()
    assert actual.input_reads == inputs
    assert actual.atom_relations | actual.relation_arguments == relations
    assert actual.identifiers == inputs | relations
    assert not actual.out_of_scope_variables
    assert not actual.unbound_atom_terms


@pytest.mark.parametrize("name", ["max", "count", "min", "exists", "len"])
def test_c1_bare_builtin_spelling_survives_as_a_scalar_read(name):
    assert analyze_relation_predicate(
        f"{name} + 1", predicate_terms=False
    ).identifiers == {name}
    if name != "count":
        assert (
            name
            not in analyze_relation_predicate(
                f"{name}(a, b)", predicate_terms=False
            ).identifiers
        )


def test_lexer_preserves_engine_tokens_and_byte_positions():
    tokens = tokenise(
        '"é\\🙂" # ignored\n  2026-01-01 1_0 1.2_5 x/y true True false False => -> <= >= == != : , . = + - * / < > ( ) [ ]'
    )
    assert [(token.kind, token.value) for token in tokens[:9]] == [
        ("string", "é🙂"),
        ("date", "2026-01-01"),
        ("int", "1_0"),
        ("float", "1.2_5"),
        ("path", "x/y"),
        ("true", "true"),
        ("true", "True"),
        ("false", "false"),
        ("false", "False"),
    ]
    assert (tokens[1].line, tokens[1].col) == (2, 3)
    assert [token.kind for token in tokens[9:-1]] == [
        "=>",
        "->",
        "<=",
        ">=",
        "==",
        "!=",
        ":",
        ",",
        ".",
        "=",
        "+",
        "-",
        "*",
        "/",
        "<",
        ">",
        "(",
        ")",
        "[",
        "]",
    ]
    assert tokenise('"é" x')[1].col == 6


@pytest.mark.parametrize("quote", ['"', "\x27"])
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=30))
def test_string_escapes_follow_whole_unicode_characters(quote, value):
    encoded = value.replace("\\", "\\\\").replace(quote, "\\" + quote)
    assert tokenise(quote + encoded + quote)[0].value == value


@pytest.mark.parametrize(
    "formula,position",
    [
        ("x = 1 x", (1, 1)),
        ("entity Person: true", (1, 1)),
        ("true\n false", (2, 2)),
        ("anchor.foo/bar > 0", (1, 8)),
        ("(match 1: _ => 1 2 => 2) > 0", (1, 18)),
        ("1e3 > 0", (1, 2)),
        (".5 > 0", (1, 1)),
        ("'unterminated", (1, 1)),
        ("true { false }", (1, 6)),
        ("true\n \ud800", (2, 2)),
    ],
)
def test_predicate_parse_failure_closes_analysis_and_scalar_fallback_is_visible(
    formula, position
):
    with pytest.raises(FormulaSyntaxError) as caught:
        parse_formula(formula)
    assert (caught.value.line, caught.value.col) == position
    predicate = analyze_relation_predicate(formula)
    assert predicate.parse_error
    assert predicate.parse_error_position == position
    assert predicate.identifiers == frozenset()
    assert not predicate.fallback_used
    scalar = analyze_relation_predicate(formula, predicate_terms=False)
    assert scalar.fallback_used
    assert scalar.parse_error_position == position
    assert scalar.identifiers == formula_identifiers(formula)
    assert not scalar.atom_relations | scalar.relation_arguments


def test_large_integer_refuses_with_a_position_and_zero_prefixes_still_parse():
    assert analyze_relation_predicate("9" * 5000 + " > 0").parse_error_position == (
        1,
        1,
    )
    assert analyze_relation_predicate("0" * 5000 + "1 > 0").parse_error is None


@st.composite
def scoped_formulas(draw):
    depth = draw(st.integers(min_value=1, max_value=5))
    variables = tuple(f"w{i}" for i in range(depth))
    formula = f"{variables[-1]}.age > 0 and pair({variables[-1]}, candidate)"
    for i in reversed(range(depth)):
        parent = "anchor" if i == 0 else variables[i - 1]
        # Parentheses disappear in the parser, including around names and calls.
        formula = f"(exists)((pair(({parent}), ({variables[i]}))), ({formula}))"
    return formula, variables


@settings(max_examples=50, deadline=None)
@given(scoped_formulas())
def test_analysis_is_deterministic_and_bound_names_stay_in_their_body(generated):
    formula, variables = generated
    actual = analyze_relation_predicate(formula)
    assert actual == analyze_relation_predicate(formula)
    assert actual.parse_error is None
    assert actual.analysis_errors == ()
    assert actual.bound_variables == frozenset(variables)
    assert actual.identifiers == {"age", "pair"}
    assert not actual.out_of_scope_variables | actual.unbound_atom_terms
    outside = analyze_relation_predicate(f"({formula}) and {variables[-1]}.age > 0")
    assert outside.out_of_scope_variables == {variables[-1]}
    assert not outside.identifiers.intersection(variables)


def differential_module(formula):
    """Build a synthetic encoder test input, never a corpus module."""

    return {
        "format": "rulespec/v1",
        "inputs": [
            {"name": name, "entity": "Person", "dtype": dtype, "period": "Month"}
            for name, dtype in (
                ("age", "Integer"),
                ("flag", "Boolean"),
                ("code", "Text"),
                ("income", "Integer"),
                ("scalar_input", "Integer"),
                ("ignored", "Integer"),
            )
        ],
        "rules": [
            {
                "name": name,
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
            }
            for name in ("people", "pair", "other_pair")
        ]
        + [
            {
                "name": "people_forward",
                "kind": "derived_relation",
                "derived_relation": {
                    "arity": 2,
                    "slot_entities": ["Person", "Person"],
                    "current_slot": 0,
                    "related_slot": 1,
                    "source_relation": "people",
                },
                "versions": [{"effective_from": "2026-01-01", "formula": True}],
            },
            {
                "name": "person_scalar",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "period": "Month",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "scalar_input"}
                ],
            },
            {
                "name": "person_flag",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Judgment",
                "period": "Month",
                "versions": [{"effective_from": "2026-01-01", "formula": "flag"}],
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
            {
                "name": "selected_size",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "len(selected)"}
                ],
            },
        ],
    }


def short_name(name):
    return name.rsplit("#", 1)[-1].removeprefix("relation.").removeprefix("input.")


def compiled_reads(node):
    """Read actual IR keys independently of the source analyzer's walk."""

    reads = {"inputs": set(), "rules": set(), "parameters": set(), "relations": set()}

    pending = [node]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            kind = value.get("type", value.get("kind"))
            if kind in {"input", "role_input", "input_for_year"}:
                reads["inputs"].add(short_name(value.get("name", value.get("input"))))
            if kind in {"derived", "role_derived"}:
                reads["rules"].add(short_name(value["name"]))
            if kind in {"parameter", "parameter_lookup"}:
                reads["parameters"].add(
                    short_name(value.get("name", value.get("parameter")))
                )
            if isinstance(value.get("relation"), str):
                reads["relations"].add(short_name(value["relation"]))
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)

    return reads


# Expand the independent 145-case round-one review grammar differential.
def grammar_cases():
    leaves = [
        "anchor.age > 0",
        "candidate.age > limit",
        "anchor.person_scalar > candidate.person_scalar",
        "anchor.person_flag",
        "pair(anchor, candidate)",
        "pair(candidate, anchor)",
        "anchor == candidate",
        "anchor != candidate",
        "count_where(people, flag) > 0",
        "sum_where(people, income, flag) > 0",
        "only_related(people, flag) > 0",
        "len(people) > 0",
        "len(people.ignored) > 0",
        "sum(people.income) > 0",
        "min(people.income) > 0",
        "max(people.income) > 0",
        "(if anchor.flag: candidate.income else: anchor.income) > 0",
        '(match anchor.code: "A" => candidate.income _ => anchor.income) > 0',
        'candidate.code == "pair(anchor, p) p.age missing_fact"',
        "anchor.age > 1_000",
        "len > 0",
        "exists > 0",
        "count_where > 0",
        "anchor.age > 0 # missing_pair(anchor, q) q.ghost\n and candidate.income > 0",
    ]
    for i, leaf in enumerate(leaves):
        yield f"leaf_{i}", leaf
        yield f"paren_leaf_{i}", f"(({leaf}))"
    for outer in ("anchor", "candidate"):
        for var in ("p", "ifx", "len", "exists", "period_start"):
            for rev in (False, True):
                args = f"{var}, {outer}" if rev else f"{outer}, {var}"
                yield (
                    f"exists_{outer}_{var}_{rev}",
                    f"exists(pair({args}), {var}.age > 0)",
                )
    # Parentheses at each syntactically meaningful position, independently.
    patterns = [
        "exists((pair(anchor, p)), p.age > 0)",
        "exists(pair((anchor), p), p.age > 0)",
        "exists(pair(anchor, (p)), p.age > 0)",
        "exists(pair(anchor, p), (p).age > 0)",
        "(exists)(pair(anchor, p), p.age > 0)",
        "exists((pair)(anchor, p), p.age > 0)",
        "(pair)(anchor, candidate)",
        "exists(pair(anchor,p), (other_pair)(p,candidate))",
        "exists(pair(anchor,p), exists(other_pair(p,q), q.age > p.age and pair(q,candidate)))",
        "exists(pair(p,anchor), exists(other_pair(q,p), q.age > p.age and pair(candidate,q)))",
        "exists((pair(anchor,p)), exists(other_pair(p,q), q.age > p.age))",
        "exists(pair(anchor,p), exists((other_pair(p,q)), q.age > p.age))",
        "exists(pair(anchor,p), exists(other_pair(p,p), p.age > 0))",
        "exists(pair(anchor,p), exists(other_pair(p,anchor), p.age > 0))",
        "exists(pair(anchor,p), exists(other_pair(p,candidate), p.age > 0))",
        "exists(pair(anchor,p), true) and p.age > 0",
        "exists(pair(anchor,p), p.age > 0) or exists(pair(candidate,p), p.income > 0)",
        "exists(pair(anchor,p), (if p.flag: candidate.income else: anchor.income) > 0)",
        'exists(pair(anchor,p), (match p.code: "A" => p.income _ => anchor.income) > 0)',
        "(anchor).age > 0",
        "((anchor)).person_scalar > 0",
        "(len)(people) > 0",
        "(len)(people.ignored) > 0",
        "(count_where)(people, flag) > 0",
        "(sum_where)(people, income, flag) > 0",
        "(only_related)(people, flag) > 0",
        "len((people).ignored) > 0",
        '""" ignore pair(anchor, p) and p.age """ anchor.age > 0',
        '""" ignore " pair(anchor, p) and p.age " """ anchor.age > 0',
        "(missing_pair)(anchor,candidate)",
        "exists(pair(anchor,p), (missing_pair)(p,candidate))",
        "count_where(people, anchor.flag) > 0",
        "sum_where(people, candidate.income, flag) > 0",
        "only_related(people, anchor.income) > 0",
        "pair(anchor, candidate) == true",
        "anchor.foo/bar > 0",
        "foo/bar > 0",
    ]
    for i, p in enumerate(patterns):
        yield f"structural_{i}", p
    # Small compositional grammar, no Hypothesis/PR generation code.
    base = [
        "anchor.age > 0",
        "candidate.person_scalar > 0",
        "pair(anchor,candidate)",
        "anchor != candidate",
    ]
    for i, a in enumerate(base):
        for j, b in enumerate(base):
            for op in ("and", "or"):
                yield f"binary_{i}_{j}_{op}", f"({a}) {op} ({b})"
        yield (
            f"nested_{i}",
            f"exists(pair(anchor,p), exists(other_pair(p,q), ({a}) and q.income > p.age))",
        )
        yield f"not_{i}", f"not ({a})"


def differential_cases():
    for label, formula in grammar_cases():
        for function in (
            "count_where",
            "sum_where",
            "only_related",
            "len",
            "sum",
            "min",
            "max",
        ):
            formula = formula.replace(
                function + "(people", function + "(people_forward"
            )
            formula = formula.replace(
                "(" + function + ")(people", "(" + function + ")(people_forward"
            )
        yield label, formula.replace("(people).ignored", "(people_forward).ignored")
    yield from [
        ("b1-parenthesized-constant-binder", "exists((pair(anchor, p)), true)"),
        ("b2-number-leading-underscore", "1_0 > 0"),
        ("b2-match-wildcard", "(match 1: _ => 1) > 0"),
        ("b2-double-quote-docstring", '""" " ghost(anchor, candidate) " """ true'),
        ("b2-one-slash-name", "x/y > 0"),
        ("a1-relation-value-read", "people_forward > 0"),
        ("only-reduction", "only(people_forward.income) > 0"),
        ("min-where-reduction", "min_where(people_forward, income, flag) > 0"),
        ("max-where-reduction", "max_where(people_forward, income, flag) > 0"),
        ("match-wildcard-discards-read", "(match unvisited: _ => 1) > 0"),
        ("len-discards-trailing-read", "len(people_forward, unvisited) > 0"),
        ("sum-discards-trailing-read", "sum(people_forward.income, unvisited) > 0"),
        ("elif-chain", "(if candidate.flag: 1 elif anchor.flag: 2 else: 3) > 0"),
        (
            "single-quoted-formula-text",
            "candidate.code == 'ghost(anchor, p) p.unknown'",
        ),
        ("escaped-string", 'candidate.code == "a\\" ghost(anchor, p)"'),
        ("keyword-like-identifiers", "ifx > 0 and matchx > 0"),
        ("unary-and-arithmetic", "-candidate.age + 2 * anchor.income / 3 > 0"),
        ("empty-exists-refuses", "exists(pair(anchor, candidate), true)"),
        ("shadowing-refuses", "exists(pair(anchor, p), exists(pair(p, anchor), true))"),
        ("atom-in-scalar-argument-refuses", "max(pair(anchor, candidate), 1) > 0"),
        ("atom-in-match-arm-refuses", "(match 1: _ => pair(anchor, candidate)) > 0"),
        ("term-in-aggregation-refuses", "sum(candidate.income) > 0"),
        ("leading-statement-refuses", "x = 1 x > 0"),
        ("leading-entity-refuses", "entity Person: true"),
        (
            "only-related-derived-refuses",
            "only_related(people_forward, person_scalar) > 0",
        ),
        ("year-input-derived-refuses", "input_for_year(person_scalar, 2026) > 0"),
        ("year-input-parameter-refuses", "input_for_year(limit, 2026) > 0"),
        ("year-zero-refuses", "input_for_year(age, 0) > 0"),
        ("period-start-judgment-refuses", "period_start"),
        ("period-end-judgment-refuses", "period_end"),
        ("discarded-match-atom-refuses", "(match pair(anchor, candidate): _ => 1) > 0"),
        (
            "discarded-match-unknown-call-refuses",
            "(match unknown_function(): _ => 1) > 0",
        ),
        ("discarded-match-period-reads-nothing", "(match period_start: _ => 1) > 0"),
        ("unknown-relation-reduction-refuses", "len(inferred_relation) > 0"),
        ("decimal-max-rounds-down", "79228162514264337593543950335.1 > 0"),
        ("decimal-max-rounds-up-refuses", "79228162514264337593543950335.5 > 0"),
        ("deep-parentheses", "(" * 500 + "anchor.age > 0" + ")" * 500),
        ("long-arithmetic-chain", "age" + " + 1" * 1100 + " > 0"),
        ("long-boolean-chain", " and ".join(["candidate.flag"] * 1100)),
        ("deep-not-chain", "not " * 300 + "candidate.flag"),
    ]


@pytest.mark.parametrize(
    "label,formula", list(differential_cases()), ids=lambda value: value
)
def test_generated_predicate_reads_equal_compiled_artifact(
    tmp_path, real_engine, label, formula
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/reference.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + json.dumps(differential_module(formula))
    )
    artifact_path = tmp_path / "artifact.json"
    compiled = subprocess.run(
        [
            str(real_engine.binary),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(artifact_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    analyzed = analyze_relation_predicate(
        formula,
        relation_names=frozenset(
            {"people", "people_forward", "pair", "other_pair", "selected"}
        ),
        rule_names=frozenset({"person_scalar", "person_flag", "selected_size"}),
        parameter_names=frozenset({"limit"}),
    )
    if compiled.returncode:
        assert (
            analyzed.parse_error
            or analyzed.analysis_errors
            or analyzed.out_of_scope_variables
            or analyzed.unbound_atom_terms
        ), (label, formula, compiled.stderr)
        return
    assert analyzed.parse_error is None, (formula, analyzed.parse_error)
    assert analyzed.analysis_errors == (), (formula, analyzed.analysis_errors)
    # The C-backed YAML reader also reads JSON and accepts the engine's deep
    # expression artifacts without adding Python's JSON recursion limit.
    artifact = yaml.load(artifact_path.read_text(), Loader=yaml.CSafeLoader)
    relation = next(
        row
        for row in artifact["program"]["relations"]
        if short_name(row["name"]) == "selected"
    )
    expected = compiled_reads(relation["derivation"]["predicate"])
    actual = {
        "inputs": set(analyzed.input_reads),
        "rules": set(analyzed.rule_reads),
        "parameters": set(analyzed.parameter_reads),
        "relations": set(analyzed.atom_relations | analyzed.relation_arguments),
    }
    assert actual == expected, (label, formula, actual, expected)
    assert analyzed.identifiers == set().union(*expected.values())


@settings(max_examples=12, deadline=None)
@given(
    formula=st.sampled_from(
        ["view + 1", "if view: 1 else: 0", "len(view)", "if view: len(view) else: 0"]
    ),
    legacy=st.booleans(),
)
def test_generated_ordinary_collision_reads_equal_compiled_artifact(
    real_engine, formula, legacy
):
    """Ordinary rule lowering keeps a relation-spelled input in its namespace."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        temp = Path(scratch)
        root = temp / "rulespec-us"
        module = root / "us/policies/example/collision.yaml"
        module.parent.mkdir(parents=True)
        module.write_text(LABEL + yaml.safe_dump(collision_payload(formula, legacy)))
        artifact = _compile(
            real_engine.binary, real_engine.root, root, module, temp / "artifact.json"
        )
        row = next(
            row
            for row in artifact["program"]["derived"]
            if short_name(row["name"]) == "result"
        )
        expected = compiled_reads(row)
        actual = analyze_relation_predicate(
            formula,
            predicate_terms=False,
            lowering_entry="rule",
            relation_names=frozenset({"view", "pairs"}),
        )
        assert actual.input_reads == expected["inputs"]
        assert (
            actual.relation_arguments | actual.atom_relations == expected["relations"]
        )
