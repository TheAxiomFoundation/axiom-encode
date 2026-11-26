"""Typed numeric recall credits only the engine's top-level import list."""

from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import ValidatorPipeline

CITATION = "us-az/manual/example/block-1"
URI = "us-az:policies/example/provider#expense_disregard"
SOURCE_TEXT = "A $35 disregard applies to the combined expenses.\n"
MODULE = f"""module:
  source_verification:
    corpus_citation_path: {CITATION}
"""
RULES = """rules:
  - name: claimed_expense
    kind: input
    entity: Household
    dtype: Money
    unit: USD
"""
PROVIDER = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-az/manual/example/provider
rules:
  - name: expense_disregard
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-01-01'
        formula: '35'
"""
BASELINE = "format: rulespec/v1\n" + MODULE + RULES
REAL_IMPORT = f"format: rulespec/v1\nimports:\n  - {URI}\n" + MODULE + RULES

FAKE_IMPORT_BLOCKS = {
    "module_summary_block_scalar": (
        "format: rulespec/v1\n"
        + MODULE
        + f"  summary: |\n    imports:\n      - {URI}\n"
        + RULES
    ),
    "nested_under_module": (
        "format: rulespec/v1\n" + MODULE + f"  imports:\n    - {URI}\n" + RULES
    ),
    "rule_description_block_scalar": (
        "format: rulespec/v1\n"
        + MODULE
        + RULES
        + f"    description: |\n      imports:\n        - {URI}\n"
    ),
    "top_level_block_scalar": (
        "format: rulespec/v1\n"
        + f"notes: |\n  imports:\n    - {URI}\n"
        + MODULE
        + RULES
    ),
}

NO_CREDIT_FILES = {
    "unparsable_yaml": REAL_IMPORT + "broken: [unclosed\n",
    "imports_is_mapping": (
        f"format: rulespec/v1\nimports:\n  provider: {URI}\n" + MODULE + RULES
    ),
    "imports_is_string": f"format: rulespec/v1\nimports: {URI}\n" + MODULE + RULES,
    "non_string_item": (
        f"format: rulespec/v1\nimports:\n  - {URI}\n  - 35\n" + MODULE + RULES
    ),
    "nested_list_item": (
        f"format: rulespec/v1\nimports:\n  - - {URI}\n" + MODULE + RULES
    ),
    "mapping_item": (
        f"format: rulespec/v1\nimports:\n  - path: {URI}\n" + MODULE + RULES
    ),
    "duplicate_imports_key": REAL_IMPORT + "imports: []\n",
    "second_document": REAL_IMPORT + "---\nformat: rulespec/v1\n",
    "merge_key_imports": (
        f"format: rulespec/v1\nbase: &base\n  imports:\n    - {URI}\n<<: *base\n"
        + MODULE
        + RULES
    ),
    "inline_merge_key_beside_real_import": (REAL_IMPORT + "<<: {notes: text}\n"),
    "quoted_inline_merge_key_beside_real_import": (
        REAL_IMPORT + '"<<": {notes: text}\n'
    ),
    "aliased_item": (
        f"format: rulespec/v1\nnotes: &uri {URI}\nimports:\n  - *uri\n" + MODULE + RULES
    ),
    "tagged_item": (
        f"format: rulespec/v1\nimports:\n  - !!str {URI}\n" + MODULE + RULES
    ),
    "literal_item_with_line_end": (
        f"format: rulespec/v1\nimports:\n  - |\n    {URI}\n" + MODULE + RULES
    ),
    "quoted_item_with_space": (
        f'format: rulespec/v1\nimports:\n  - "{URI} "\n' + MODULE + RULES
    ),
    "symbol_in_yaml_comment": (
        "format: rulespec/v1\nimports:\n"
        "  - us-az:policies/example/provider #expense_disregard\n" + MODULE + RULES
    ),
}

CREDITED_FILES = {
    "block_sequence": REAL_IMPORT,
    "quoted_imports_key": (
        f'format: rulespec/v1\n"imports":\n  - {URI}\n' + MODULE + RULES
    ),
    "crlf_line_ends": REAL_IMPORT.replace("\n", "\r\n"),
    "document_markers": "---\n" + REAL_IMPORT + "...\n",
    "folded_item": (
        f"format: rulespec/v1\nimports:\n  - >-\n    {URI}\n" + MODULE + RULES
    ),
    "quoted_item": f'format: rulespec/v1\nimports:\n  - "{URI}"\n' + MODULE + RULES,
    "flow_sequence": f"format: rulespec/v1\nimports: [{URI}]\n" + MODULE + RULES,
    "imports_after_rules": "format: rulespec/v1\n"
    + MODULE
    + RULES
    + f"imports:\n  - {URI}\n",
}


def _world(tmp_path: Path, content: str) -> tuple[ValidatorPipeline, Path]:
    # Resolve first: the import resolver refuses a path through a symlink.
    root = tmp_path.resolve() / "rulespec-us/us-az"
    provider = root / "policies/example/provider.yaml"
    provider.parent.mkdir(parents=True)
    provider.write_text(PROVIDER)
    rules_file = root / "policies/example/block-1.yaml"
    rules_file.write_text(content)
    pipeline = ValidatorPipeline(
        policy_repo_path=root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
        source_citation_path=CITATION,
    )
    return pipeline, rules_file


def _unnamed_35_issues(pipeline: ValidatorPipeline, rules_file: Path) -> list[str]:
    issues = pipeline._complete_source_unit_issues(
        rules_file.read_text(),
        validation_source_texts={CITATION: SOURCE_TEXT},
        test_cases=[],
        rules_file=rules_file,
    )
    return [
        issue
        for issue in issues
        if issue.startswith("[complete-source-unit:numeric-recall]")
        and "numeric value 35 has no named scalar representation" in issue
    ]


def test_module_without_import_reports_unnamed_source_number(tmp_path):
    pipeline, rules_file = _world(tmp_path, BASELINE)

    assert pipeline._complete_source_unit_import_bindings(rules_file) == ()
    assert len(_unnamed_35_issues(pipeline, rules_file)) == 1


@pytest.mark.parametrize("shape", sorted(FAKE_IMPORT_BLOCKS))
def test_fake_imports_block_earns_no_typed_numeric_credit(tmp_path, shape):
    content = FAKE_IMPORT_BLOCKS[shape]
    pipeline, rules_file = _world(tmp_path, content)

    # The line reader still sees the fake block; the typed check must not.
    assert pipeline._extract_import_items(content) == [URI]
    assert pipeline._complete_source_unit_import_bindings(rules_file) == ()
    assert pipeline._complete_source_unit_import_symbol_contents(rules_file) == ()
    assert len(_unnamed_35_issues(pipeline, rules_file)) == 1


@pytest.mark.parametrize("shape", sorted(CREDITED_FILES))
def test_real_top_level_import_keeps_typed_numeric_credit(tmp_path, shape):
    pipeline, rules_file = _world(tmp_path, CREDITED_FILES[shape])

    bindings = pipeline._complete_source_unit_import_bindings(rules_file)

    assert [binding.import_uri for binding in bindings] == [URI]
    assert bindings[0].export_name == "expense_disregard"
    assert bindings[0].consumer_imports == (URI,)
    assert bindings[0].content == PROVIDER
    assert pipeline._complete_source_unit_import_symbol_contents(rules_file) == (
        ("expense_disregard", PROVIDER),
    )
    assert _unnamed_35_issues(pipeline, rules_file) == []


@pytest.mark.parametrize("shape", sorted(NO_CREDIT_FILES))
def test_import_list_the_engine_cannot_read_earns_no_binding(tmp_path, shape):
    pipeline, rules_file = _world(tmp_path, NO_CREDIT_FILES[shape])

    assert pipeline._complete_source_unit_import_bindings(rules_file) == ()
    assert pipeline._complete_source_unit_import_symbol_contents(rules_file) == ()


def test_file_level_import_without_symbol_earns_no_binding(tmp_path):
    content = (
        "format: rulespec/v1\nimports:\n  - us-az:policies/example/provider\n"
        + MODULE
        + RULES
    )
    pipeline, rules_file = _world(tmp_path, content)

    assert pipeline._complete_source_unit_import_bindings(rules_file) == ()
    assert len(_unnamed_35_issues(pipeline, rules_file)) == 1
