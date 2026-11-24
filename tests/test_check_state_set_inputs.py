"""The A/B post-run checks for new inputs in an encoded CFR module."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.test_evals import _canonical_rulespec_content_root

_SPEC = importlib.util.spec_from_file_location(
    "check_state_set_inputs",
    Path(__file__).resolve().parents[1] / "scripts/check_state_set_inputs.py",
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

UNIT = "us/regulation/42/435/733"
SOURCE = (
    "(c)(2) The allowance must be at least $30 a month and must not exceed the "
    "medically needy income standard established under § 435.811, or 133 "
    "percent of the Federal poverty level.\n\n"
    "(d) Other text with $999 that no excerpt reaches.\n"
)
EXCERPT = "medically needy income standard established under § 435.811"


@pytest.fixture
def root(tmp_path):
    root = _canonical_rulespec_content_root(tmp_path / "repos", "us")
    module = root / "regulations" / "42-cfr" / "435" / "811.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "format: rulespec/v1\nrules:\n"
        "  - name: separate_floor\n    kind: derived\n    entity: StateAgency\n"
        "    dtype: Money\n    period: Month\n    versions:\n"
        "      - effective_from: '2014-01-01'\n"
        "        formula: medically_needy_income_standard_amount * 1\n"
    )
    return root


def _module(
    tmp_path: Path,
    *,
    name: str = "medically_needy_income_standard_amount",
    description: str | None = "Standard the State sets under 42 CFR 435.811.",
    root_input: bool = True,
    formula: str | None = None,
    excerpt: str = EXCERPT,
    summary: str = "Summary.",
) -> Path:
    formula = formula or f"max(30, min(need, {name}, 1.33 * fpl))"
    entry = ""
    if root_input:
        entry = f"  - name: {name}\n    entity: Person\n    dtype: Money\n    unit: USD\n    period: Month\n"
        if description is not None:
            entry += f"    description: '{description}'\n"
    path = tmp_path / "candidate.yaml"
    path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        f"  summary: '{summary}'\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {UNIT}\n"
        + (f"inputs:\n{entry}" if entry else "")
        + "rules:\n"
        "  - name: allowance\n    kind: derived\n    entity: Person\n"
        "    dtype: Money\n    unit: USD\n    period: Month\n"
        "    metadata:\n      proof:\n        atoms:\n"
        "          - path: versions[0].formula\n            kind: formula\n"
        "            source:\n"
        f"              corpus_citation_path: {UNIT}\n"
        f"              excerpt: '{excerpt}'\n"
        "    versions:\n      - effective_from: '2014-01-01'\n"
        f"        formula: '{formula}'\n"
    )
    return path


def _baseline(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.yaml"
    path.write_text(
        "format: rulespec/v1\nrules:\n  - name: base\n    kind: derived\n"
        "    entity: Person\n    dtype: Money\n    period: Month\n    versions:\n"
        "      - effective_from: '2014-01-01'\n        formula: need + fpl\n"
    )
    return path


def test_typed_trigger_without_actor_passes(root, tmp_path):
    rows, failures = script.check(root, SOURCE, _module(tmp_path), _baseline(tmp_path))
    assert failures == []
    assert len(rows) == 1
    assert (
        "| medically_needy_income_standard_amount | root | Person/Money/Month |"
        in (rows[0])
    )


def test_baseline_inputs_are_not_new(root, tmp_path):
    rows, _failures = script.check(root, SOURCE, _module(tmp_path), None)
    assert {row.split(" | ")[1] for row in rows} == {
        "medically_needy_income_standard_amount",
        "need",
        "fpl",
    }


def test_literal_only_in_summary_or_excerpt_fails(root, tmp_path):
    module = _module(
        tmp_path,
        formula="min(need, medically_needy_income_standard_amount)",
        summary="At least $30 and 133 percent of the Federal poverty level.",
        excerpt=EXCERPT,
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert sorted(f.split(" in the")[0] for f in failures) == [
        "literal: $30",
        "literal: 133 percent FPL",
    ]


def test_literal_outside_the_cited_paragraph_is_not_required(root, tmp_path):
    _rows, failures = script.check(root, SOURCE, _module(tmp_path), _baseline(tmp_path))
    assert not any("$999" in failure for failure in failures)


def test_input_without_description_fails(root, tmp_path):
    _rows, failures = script.check(
        root, SOURCE, _module(tmp_path, description=None), _baseline(tmp_path)
    )
    assert failures == [
        "description: `medically_needy_income_standard_amount` has no description "
        "naming a provision"
    ]


def test_description_without_provision_fails(root, tmp_path):
    _rows, failures = script.check(
        root,
        SOURCE,
        _module(tmp_path, description="The State standard."),
        _baseline(tmp_path),
    )
    assert len(failures) == 1 and failures[0].startswith("description:")


def test_implicit_identifier_is_an_input_and_fails_description(root, tmp_path):
    rows, failures = script.check(
        root, SOURCE, _module(tmp_path, root_input=False), _baseline(tmp_path)
    )
    assert "| medically_needy_income_standard_amount | implicit | -/-/- |" in rows[0]
    assert failures == [
        "description: `medically_needy_income_standard_amount` has no description "
        "naming a provision"
    ]


def test_exported_name_fails(root, tmp_path):
    module = _module(
        tmp_path,
        name="separate_floor",
        excerpt="the standard established by the agency under § 435.811",
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert failures == [
        "exported: `separate_floor` is a rule name under the RuleSpec root"
    ]


def test_no_trigger_fails(root, tmp_path):
    module = _module(
        tmp_path,
        name="unrelated_amount",
        excerpt="must not exceed the medically needy income standard",
    )
    _rows, failures = script.check(root, SOURCE, module, _baseline(tmp_path))
    assert any(failure.startswith("trigger:") for failure in failures)


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        ("the income standard established by the agency in its State plan", True),
        ("the standard set by the State agency", True),
        ("if the State elects to cover", True),
        ("the agency may specify a standard", True),
        ("the State plan specifies the standard", False),
        ("the standard established by the State plan", False),
        ("the amount set by the agency plan", False),
        ("as specified under section 1902(a)(10) of the Act", False),
        ("the agency must furnish Medicaid as specified in paragraph (b)", False),
        ("established under § 435.811", False),
    ],
)
def test_actor_phrase(excerpt, expected):
    assert script.actor_sets_value(excerpt) is expected


def test_main_exit_code_and_table(root, tmp_path, capsys):
    source = tmp_path / "source.txt"
    source.write_text(SOURCE)
    module = _module(tmp_path)
    code = script.main(
        [
            "--rulespec-root",
            str(root),
            "--source",
            str(source),
            "--module",
            str(module),
            "--baseline",
            str(_baseline(tmp_path)),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "new inputs: 1; failures: 0" in out
    assert (
        script.main(
            [
                "--rulespec-root",
                str(root),
                "--source",
                str(source),
                "--module",
                str(module),
            ]
        )
        == 1
    )
