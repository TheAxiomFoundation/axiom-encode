"""Pin the import and proof forms that round 11's line edits left untested."""

from __future__ import annotations

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as validation
from tests.test_relation_role_round6 import PAIRS, VIEW, rule
from tests.test_relation_role_round7 import program, view
from tests.test_relation_role_round8 import EVERYONE, certified
from tests.test_relation_role_round10 import engine_reads, predicate

SCALAR_VIEW = "us:policies/example/scalars#view"


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
