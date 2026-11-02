"""Pin the import and proof forms that the round 11 and 12 line edits left untested."""

from __future__ import annotations

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as validation
from tests.test_relation_role_round6 import PAIRS, VIEW, rule
from tests.test_relation_role_round7 import program, view
from tests.test_relation_role_round8 import EVERYONE, certified
from tests.test_relation_role_round10 import engine_reads, predicate

SCALAR_VIEW = "us:policies/example/scalars#view"
EXTRA = "us:policies/example/extra#unrelated"
SPELLINGS = {
    "valid": EXTRA,
    "whole-module": "us:policies/example/extra",
    "padded-leading": f" {EXTRA}",
    "padded-trailing": f"{EXTRA} ",
    "tab": f"{EXTRA}\t",
    "quoted": f"'{EXTRA}'",
    "no-prefix": "bad#unrelated",
    "upper-prefix": EXTRA.replace("us:", "US:"),
    "three-letter-prefix": EXTRA.replace("us:", "usa:"),
    "two-hashes": f"{EXTRA}#x",
    "empty-fragment": "us:policies/example/extra#",
    "fragment-space": "us:policies/example/extra#unre lated",
    "fragment-slash": "us:policies/example/extra#a/b",
    "yaml-suffix": "us:policies/example/extra.yaml#unrelated",
    "yml-suffix": "us:policies/example/extra.yml#unrelated",
    "test-suffix": "us:policies/example/extra.test#unrelated",
    "upper-suffix": "us:policies/example/extra.YAML#unrelated",
    "one-segment": "us:extra#unrelated",
    "root-only": "us:policies#unrelated",
    "non-atomic-root": "us:tests/example/extra#unrelated",
    "empty-segment": "us:policies//example/extra#unrelated",
    "dot-segment": "us:policies/./example/extra#unrelated",
    "dot-dot-segment": "us:policies/../example/extra#unrelated",
    "colon-in-path": "us:policies/example:extra#unrelated",
    "non-ascii-segment": "us:policies/example/exträ#unrelated",
    "non-ascii-fragment": "us:policies/example/extra#unrelatëd",
    "region-trailing-dash": EXTRA.replace("us:", "us-:"),
    "leading-slash": "us:/policies/example/extra#unrelated",
    "trailing-slash": "us:policies/example/extra/#unrelated",
    "map-entry": {"target": EXTRA},
    "number-entry": 7,
}


def test_an_import_of_another_kind_with_the_same_name_keeps_mains_finding(
    tmp_path, real_engine
):
    """The engine refuses a short name that two declarations share.

    A relation import and a derived rule import of one short name make the
    engine refuse the module, so the relation import earns no short-name
    credit and keeps main's finding. Main's text match still credits the rule.
    """
    imports = [VIEW, SCALAR_VIEW]
    reads, content = engine_reads(
        tmp_path,
        real_engine,
        imports,
        predicate("view(anchor, candidate)"),
        scalars=program(rule("view", "Person", "1")),
    )
    assert reads is None
    issues = validation.find_unused_import_issues(content)
    assert [item for item in imports if any(f"`{item}`" in i for i in issues)] == [VIEW]


@pytest.mark.parametrize(
    "source,target,expected",
    [("pairs", "pairs", False), ("roster", "roster", False), (VIEW, VIEW, True)],
    ids=["local-bare-target", "undeclared-bare-target", "full-target"],
)
def test_only_a_full_proof_target_earns_source_credit(source, target, expected):
    """A bare target names no import, so it keeps main's text match."""
    selected = view("selected", source, "true", slots=EVERYONE)
    output = target.rsplit("#relation.", 1)[-1]
    selected["metadata"] = {
        "proof": {"atoms": [{"import": {"target": target, "output": output}}]}
    }
    content = yaml.safe_dump(program(PAIRS, selected, imports=[VIEW]))
    assert certified(content) is expected


@pytest.mark.parametrize("second", list(SPELLINGS.values()), ids=list(SPELLINGS))
def test_one_import_the_engine_cannot_load_keeps_every_finding(
    tmp_path, real_engine, second
):
    """The engine refuses a whole module over one import it cannot resolve.

    Each entry sits beside a relation import that an atom reads, so the scan
    credits that import exactly when the engine loads the module.
    """
    reads, content = engine_reads(
        tmp_path,
        real_engine,
        [VIEW, second],
        predicate("view(anchor, candidate)"),
        extra=program(rule("unrelated", "Person", "1")),
    )
    loads = isinstance(second, str) and validation._engine_accepts_import_spelling(
        second
    )
    assert (reads is not None) is loads
    issues = validation.find_unused_import_issues(content)
    assert any(f"`{VIEW}`" in issue for issue in issues) is not loads


@pytest.mark.parametrize(
    "spelling",
    [f" {VIEW}", f"{VIEW} ", "bad#relation.view", VIEW.replace("us:", "US:")],
    ids=["padded-leading", "padded-trailing", "no-prefix", "upper-prefix"],
)
def test_a_relation_import_the_engine_cannot_resolve_keeps_mains_finding(
    tmp_path, real_engine, spelling
):
    """An atom reads no import whose spelling the engine refuses."""
    reads, content = engine_reads(
        tmp_path, real_engine, [spelling], predicate("view(anchor, candidate)")
    )
    assert reads is None
    assert validation.find_unused_import_issues(content) == [
        f"Unused import `{spelling.strip()}`: imported symbol `relation.view` is "
        "not referenced by any formula or proof import."
    ]


@pytest.mark.parametrize(
    "imports,expected",
    [
        ([VIEW], True),
        ([VIEW, f" {EXTRA}"], False),
        ([VIEW, {"target": EXTRA}], False),
        (None, False),
    ],
    ids=["loads", "padded-import", "map-entry", "null-imports"],
)
def test_only_a_module_the_engine_loads_earns_source_credit(imports, expected):
    """A module whose imports the engine refuses reads no source."""
    selected = view("selected", VIEW, "true", slots=EVERYONE)
    selected["metadata"] = {
        "proof": {"atoms": [{"import": {"target": VIEW, "output": "view"}}]}
    }
    content = yaml.safe_dump(program(PAIRS, selected, imports=imports))
    assert certified(content) is expected


@st.composite
def import_spellings(draw):
    """Spellings near the engine's grammar, valid and not."""
    prefix = draw(st.sampled_from(["us", "us-ca", "uk", "US", "usa", "u", "", "us-"]))
    root = draw(st.sampled_from(["policies", "statutes", "legislation", "tests", ""]))
    segments = draw(st.lists(st.text(alphabet="az09_.~- '/#:", max_size=6), max_size=3))
    spelling = f"{prefix}:{'/'.join([root, *segments])}"
    fragment = draw(st.none() | st.text(alphabet="az09_.- #/'", max_size=8))
    if fragment is not None:
        spelling = f"{spelling}#{fragment}"
    padding = st.sampled_from(["", " ", "\t", "\n"])
    return draw(padding) + spelling + draw(padding)


@settings(deadline=None, derandomize=True, max_examples=400)
@given(import_spellings())
def test_every_import_the_engine_accepts_passes_the_shape_check(spelling):
    """The engine's spellings form a subset of what the shape check accepts."""
    accepted = validation._engine_accepts_import_spelling(spelling)
    if spelling != spelling.strip():
        assert not accepted
    if accepted:
        content = yaml.safe_dump({"imports": [spelling]})
        assert validation.find_import_shape_issues(content) == []
