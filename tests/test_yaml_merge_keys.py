"""YAML merge keys the engine does not apply are rejected outside rule metadata.

``yaml.safe_load`` applies a merge key (`<<`, or a key tagged `!!merge`) by
copying the merged mapping's keys into the mapping that holds it. Engine
``1aabc8383`` (serde_yaml 0.9.34) never applies merges, so `<<` is an ordinary
key to it: ignored in a rule or version, rejected in
`module.source_verification`. Before this guard a rule written ``- <<: *base``
passed the strict proof gate while the engine rejected it or compiled it
without the merged fields, and a load-and-dump rewrite (apply stamping, eval
repairs) wrote the merged fields out, so the stamped file compiled differently
from the raw one. The engine never reads a rule's `metadata`, and real modules
merge shared proof sources there, so merges stay allowed inside it. Companion
tests are read only by PyYAML, so they may use merges anywhere.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.cli import (
    _can_attempt_apply,
    _generated_rulespec_yaml_nonparser_issue,
    _stamp_generated_source_attestation_for_apply,
    _validate_generated_encoding_in_policy_overlay,
)
from axiom_encode.harness.evals import (
    _UNLOADABLE_YAML_ISSUE,
    ValidationRetryCandidate,
    _apply_generated_eval_repairs,
    _materialize_eval_artifact,
    _normalize_main_eval_content,
    _overlay_validation_retry_candidate,
    _rewrite_merge_key_issues,
)
from axiom_encode.harness.proof_validator import (
    _safe_load_with_raw_constraints,
    find_unsupported_yaml_tag_issues,
    find_yaml_merge_key_issues,
    validate_rulespec_proofs,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.yaml_serialization import safe_dump_yaml
from tests.test_cli import (
    _complete_source_attestation,
    _write_test_context_manifest,
)
from tests.test_module_interpretations import MERGE_KEY_REASON, _merge_key_issue
from tests.test_unsupported_yaml_tags import _engine_compile

CITATION = "us/guidance/example/sua"
SOURCE = "The standard utility allowance is $451 per month."
SOURCES = {CITATION: SOURCE}

FORMAT = "format: rulespec/v1\n"
MODULE_BLOCK = """module:
  summary: The standard utility allowance is $451.
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/guidance/example/sua
"""
SOURCE_BLOCK = """            source:
              corpus_citation_path: us/guidance/example/sua
              excerpt: The standard utility allowance is $451 per month.
"""
PROOF = (
    """    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
"""
    + SOURCE_BLOCK
)
VERSIONS = """    versions:
      - effective_from: '2024-01-01'
        formula: '451'
"""
RULE = (
    "  - name: sua_value\n    kind: parameter\n    dtype: Money\n    unit: USD\n"
    + PROOF
    + VERSIONS
)
MODULE = FORMAT + MODULE_BLOCK + "rules:\n" + RULE


def _module(anchors="", replacements=(), *, rules=RULE, head=MODULE_BLOCK):
    """The passing module with ``anchors`` after `format`.

    ``replacements`` apply to the module and rules only, never to the anchors.
    """

    body = head + "rules:\n" + rules
    for old, new in replacements:
        assert old in body, old
        body = body.replace(old, new, 1)
    return FORMAT + anchors + body


def _gate(content):
    return validate_rulespec_proofs(content, source_texts=SOURCES)


def _flattened(content):
    """What a load-and-dump rewrite writes: PyYAML's view, merges applied."""

    return safe_dump_yaml(yaml.safe_load(content), sort_keys=False, allow_unicode=True)


# Each case: the module, and the merge keys the scan reports as
# (location, written text at the key, occurrence of that text, tagged).
REJECTED = {
    "rule-supplies-kind": (
        _module(
            "x-defaults: &defaults\n  kind: parameter\n",
            [("    kind: parameter\n", "    <<: *defaults\n")],
        ),
        [("rules[0] key `<<`", "<<: *defaults", 0, False)],
    ),
    "rule-supplies-versions": (
        _module(
            "x-defaults: &defaults\n  versions:\n"
            "    - effective_from: '2024-01-01'\n      formula: '451'\n",
            [(VERSIONS, "    <<: *defaults\n")],
        ),
        [("rules[0] key `<<`", "<<: *defaults", 0, False)],
    ),
    "rule-copies-another-rule": (
        _module(
            rules=RULE.replace(
                "  - name: sua_value\n", "  - &first\n    name: sua_value\n"
            )
            + "  - <<: *first\n    name: sua_copy\n"
        ),
        [("rules[1] key `<<`", "<<: *first", 0, False)],
    ),
    "rule-supplies-an-optional-field": (
        _module(replacements=[("    unit: USD\n", "    <<: {unit: USD}\n")]),
        [("rules[0] key `<<`", "<<: {unit", 0, False)],
    ),
    "rule-sequence-merge": (
        _module(
            "x-unit: &unit\n  unit: USD\nx-dtype: &dtype\n  dtype: Money\n",
            [("    dtype: Money\n    unit: USD\n", "    <<: [*unit, *dtype]\n")],
        ),
        [("rules[0] key `<<`", "<<: [*unit", 0, False)],
    ),
    "rule-explicit-merge-key": (
        _module(replacements=[("    unit: USD\n", "    ? <<\n    : {unit: USD}\n")]),
        [("rules[0] key `<<`", "<<\n    :", 0, False)],
    ),
    "rule-flow-mapping": (
        _module(
            "x-defaults: &defaults {kind: parameter, dtype: Money, unit: USD}\n",
            rules="  - {<<: *defaults, name: sua_value, metadata: {proof: {atoms: "
            "[{path: 'versions[0].formula', kind: formula, source: "
            "{corpus_citation_path: us/guidance/example/sua, excerpt: The standard "
            "utility allowance is $451 per month.}}]}}, versions: "
            "[{effective_from: '2024-01-01', formula: '451'}]}\n",
        ),
        [("rules[0] key `<<`", "<<: *defaults,", 0, False)],
    ),
    "version-copies-another-version": (
        _module(
            replacements=[
                (
                    VERSIONS,
                    "    versions:\n      - &v\n        effective_from: '2024-01-01'\n"
                    "        formula: '451'\n      - <<: *v\n"
                    "        effective_from: '2025-01-01'\n",
                )
            ]
        ),
        [("rules[0].versions[1] key `<<`", "<<: *v", 0, False)],
    ),
    "version-supplies-effective-from": (
        _module(
            "x-start: &start\n  effective_from: '2024-01-01'\n",
            [("      - effective_from: '2024-01-01'\n", "      - <<: *start\n")],
        ),
        [("rules[0].versions[0] key `<<`", "<<: *start", 0, False)],
    ),
    "version-supplies-formula": (
        _module(
            "x-formula: &formula\n  formula: '451'\n",
            [("        formula: '451'\n", "        <<: *formula\n")],
        ),
        [("rules[0].versions[0] key `<<`", "<<: *formula", 0, False)],
    ),
    "sources-merge": (
        _module(
            "x-source: &source\n  citation: us/guidance/example/sua\n",
            [(VERSIONS, "    sources:\n      - <<: *source\n" + VERSIONS)],
        ),
        [("rules[0].sources[0] key `<<`", "<<: *source", 0, False)],
    ),
    "module-supplies-title": (
        _module(
            "x-meta: &meta\n  title: Standard utility allowance\n",
            [("module:\n", "module:\n  <<: *meta\n")],
        ),
        [("module key `<<`", "<<: *meta", 0, False)],
    ),
    "module-supplies-source-verification": (
        _module(
            "x-meta: &meta\n  source_verification:\n"
            "    corpus_citation_path: us/guidance/example/sua\n",
            [
                (
                    "  source_verification:\n"
                    "    corpus_citation_path: us/guidance/example/sua\n",
                    "  <<: *meta\n",
                )
            ],
        ),
        [("module key `<<`", "<<: *meta", 0, False)],
    ),
    "module-supplies-kind": (
        _module(
            "x-meta: &meta\n  kind: composition\n",
            [("module:\n", "module:\n  <<: *meta\n")],
        ),
        [("module key `<<`", "<<: *meta", 0, False)],
    ),
    "source-verification-merge": (
        _module(
            "x-verification: &verification\n"
            "  corpus_citation_path: us/guidance/example/sua\n",
            [
                (
                    "  source_verification:\n"
                    "    corpus_citation_path: us/guidance/example/sua\n",
                    "  source_verification:\n    <<: *verification\n",
                )
            ],
        ),
        [("module.source_verification key `<<`", "<<: *verification", 0, False)],
    ),
    "root-supplies-rules": (
        FORMAT
        + "x-base: &base\n  rules:\n"
        + "".join("  " + line for line in RULE.splitlines(True))
        + MODULE_BLOCK
        + "<<: *base\n",
        [("document root key `<<`", "<<: *base", 0, False)],
    ),
    "root-supplies-format": (
        "x-base: &base\n  format: rulespec/v1\n<<: *base\n"
        + MODULE_BLOCK
        + "rules:\n"
        + RULE,
        [("document root key `<<`", "<<: *base", 0, False)],
    ),
    "metadata-mapping-aliased-into-versions": (
        _module(
            replacements=[
                (
                    "    metadata:\n",
                    "    metadata:\n      x-version: &v\n"
                    "        <<: {effective_from: '2024-01-01'}\n"
                    "        formula: '451'\n",
                ),
                (VERSIONS, "    versions:\n      - *v\n"),
            ]
        ),
        [("rules[0].versions[0] key `<<`", "<<: {effective", 0, False)],
    ),
    "rule-key-tagged-as-merge": (
        _module(
            replacements=[("    unit: USD\n", "    !!merge defaults: {unit: USD}\n")]
        ),
        [("rules[0] key `defaults`", "!!merge defaults", 0, True)],
    ),
}

# Merges the engine cannot see a difference in, rejected because the rule is by
# location: only a rule's `metadata` may merge.
REJECTED_ALTHOUGH_INERT = {
    "rule-merge-overridden-by-every-key": (
        _module(
            "x-defaults: &defaults\n  kind: parameter\n  unit: USD\n",
            [("    kind: parameter\n", "    <<: *defaults\n    kind: parameter\n")],
        ),
        [("rules[0] key `<<`", "<<: *defaults", 0, False)],
    ),
    "root-supplies-unknown-key": (
        _module("x-base: &base\n  x-note: hello\n<<: *base\n"),
        [("document root key `<<`", "<<: *base", 0, False)],
    ),
    "proof-validation-merge": (
        _module(
            "x-strict: &strict\n  required: true\n",
            [("    required: true\n", "    <<: *strict\n")],
        ),
        [("module.proof_validation key `<<`", "<<: *strict", 0, False)],
    ),
    "rule-supplies-only-metadata": (
        _module(
            "x-meta: &meta\n" + "".join(line[2:] for line in PROOF.splitlines(True)),
            [(PROOF, "    <<: *meta\n")],
        ),
        [("rules[0] key `<<`", "<<: *meta", 0, False)],
    ),
    "rule-empty-merge": (
        _module(
            replacements=[
                ("    kind: parameter\n", "    <<: {}\n    kind: parameter\n")
            ]
        ),
        [("rules[0] key `<<`", "<<: {}", 0, False)],
    ),
}


def _copy_rule(source_block):
    """A second rule, `sua_copy`, whose proof source is ``source_block``."""

    return RULE.replace("sua_value", "sua_copy").replace(SOURCE_BLOCK, source_block)


ANCHORED_RULE = RULE.replace("            source:\n", "            source: &src\n", 1)
# Merges inside a rule's `metadata`, as real modules write them.
ACCEPTED = {
    "proof-source-merge": _module(
        rules=ANCHORED_RULE
        + _copy_rule("            source:\n              <<: *src\n")
    ),
    "proof-source-merge-with-override": _module(
        rules=ANCHORED_RULE
        + _copy_rule(
            "            source:\n              <<: *src\n"
            "              excerpt: The standard utility allowance is $451 per month.\n"
        )
    ),
    "proof-source-explicit-merge-key": _module(
        rules=ANCHORED_RULE
        + _copy_rule("            source:\n              ? <<\n              : *src\n")
    ),
    "metadata-merge": _module(
        rules=RULE.replace("    metadata:\n", "    metadata: &meta\n", 1)
        + RULE.replace("sua_value", "sua_copy").replace(
            PROOF, "    metadata:\n      <<: *meta\n"
        )
    ),
    "proof-atom-merge": _module(
        rules=RULE.replace(
            "          - path: versions[0].formula\n",
            "          - &atom\n            path: versions[0].formula\n",
            1,
        )
        + RULE.replace("sua_value", "sua_copy").replace(
            "          - path: versions[0].formula\n            kind: formula\n"
            + SOURCE_BLOCK,
            "          - <<: *atom\n",
        )
    ),
    "nested-merge-in-metadata": _module(
        rules=RULE.replace(
            "    metadata:\n",
            "    metadata:\n      x-base: &base\n        note: shared\n"
            "      x-more: &more\n        <<: *base\n        extra: yes\n"
            "      notes:\n        <<: [*more, *base]\n",
            1,
        )
    ),
    # A quoted or string-tagged `<<` is an ordinary key to PyYAML and the engine.
    "quoted-merge-spelling-in-a-rule": _module(
        "x-defaults: &defaults\n  unit: USD\n",
        [("    unit: USD\n", "    unit: USD\n    '<<': *defaults\n")],
    ),
    "string-tagged-merge-spelling-in-a-rule": _module(
        "x-defaults: &defaults\n  unit: USD\n",
        [("    unit: USD\n", "    unit: USD\n    !!str <<: *defaults\n")],
    ),
}


def _issues(content, holders):
    return [
        _merge_key_issue(content, location, written, occurrence=index, tagged=tagged)
        for location, written, index, tagged in holders
    ]


def test_the_unmerged_module_passes_the_strict_gate():
    result = _gate(MODULE)

    assert result.passed, result.issues
    assert result.proof_required
    assert find_yaml_merge_key_issues(MODULE) == []


@pytest.mark.parametrize(
    "case", sorted(REJECTED) + sorted(REJECTED_ALTHOUGH_INERT), ids=str
)
def test_merge_keys_outside_rule_metadata_are_reported(case):
    content, holders = {**REJECTED, **REJECTED_ALTHOUGH_INERT}[case]
    issues = _issues(content, holders)

    assert find_yaml_merge_key_issues(content) == issues
    result = _gate(content)
    assert not result.passed
    if case == "rule-key-tagged-as-merge":
        # The tag scan rejects every tag on a scalar but `!!str` first.
        assert result.issues == find_unsupported_yaml_tag_issues(content)
    else:
        assert result.issues == issues


@pytest.mark.parametrize(
    "case", sorted(set(REJECTED) - {"rule-key-tagged-as-merge"}), ids=str
)
def test_pyyaml_view_of_each_rejected_module_passes_the_gate(case):
    """The defect: the gate checked PyYAML's merged view, which passes."""

    content, _holders = REJECTED[case]
    flattened = _flattened(content)

    assert yaml.safe_load(flattened) == yaml.safe_load(content)
    assert "<<" not in flattened
    result = _gate(flattened)
    assert result.passed, result.issues


@pytest.mark.parametrize("case", sorted(ACCEPTED), ids=str)
def test_merge_keys_inside_rule_metadata_pass(case):
    content = ACCEPTED[case]

    assert find_yaml_merge_key_issues(content) == []
    result = _gate(content)
    assert result.passed, result.issues


def test_the_gate_validates_proofs_merged_inside_metadata():
    """PyYAML's merged proof is what the gate checks, and what a rewrite keeps."""

    content = _module(
        "x-src: &src\n  corpus_citation_path: us/guidance/example/sua\n"
        "  excerpt: Not in the source.\n",
        [(SOURCE_BLOCK, "            source:\n              <<: *src\n")],
    )

    assert find_yaml_merge_key_issues(content) == []
    assert _gate(content).issues == [
        "Proof source evidence not found: rule `sua_value` proof atom 0 "
        "`source.excerpt` does not appear in `us/guidance/example/sua`."
    ]


def test_merge_tags_inside_metadata_are_left_to_the_tag_scan():
    content = _module(
        rules=ANCHORED_RULE
        + _copy_rule("            source:\n              !!merge <<: *src\n")
    )

    assert find_yaml_merge_key_issues(content) == []
    assert _gate(content).issues == find_unsupported_yaml_tag_issues(content) != []


def test_a_merge_reached_both_inside_and_outside_metadata_is_reported_once():
    content, holders = REJECTED["metadata-mapping-aliased-into-versions"]

    assert find_yaml_merge_key_issues(content) == _issues(content, holders)
    assert len(find_yaml_merge_key_issues(content)) == 1


def test_a_rule_reached_only_through_an_unknown_root_key_is_not_a_rule():
    """Merges count as metadata only under the root `rules` sequence."""

    content = (
        FORMAT
        + "x-rules: &rules\n"
        + "".join("  " + line for line in ANCHORED_RULE.splitlines(True))
        + "".join(
            "  " + line
            for line in _copy_rule(
                "            source:\n              <<: *src\n"
            ).splitlines(True)
        )
        + MODULE_BLOCK
        + "rules: *rules\n"
    )

    # The rules are also reached under `rules`, but `x-rules` is read as data.
    assert find_yaml_merge_key_issues(content) == [
        _merge_key_issue(
            content, "x-rules[1].metadata.proof.atoms[0].source key `<<`", "<<: *src"
        )
    ]


@pytest.mark.parametrize(
    ("anchors", "replacements", "location"),
    [
        pytest.param(
            "x-nested:\n  rules:\n    - name: shadow\n      metadata:\n"
            "        <<: {note: shared}\n",
            (),
            "x-nested.rules[0].metadata key `<<`",
            id="under-an-unknown-root-key",
        ),
        pytest.param(
            "",
            [
                (
                    "  summary: The standard utility allowance is $451.\n",
                    "  summary: The standard utility allowance is $451.\n"
                    "  rules:\n    - metadata:\n        <<: {note: shared}\n",
                )
            ],
            "module.rules[0].metadata key `<<`",
            id="under-module",
        ),
    ],
)
def test_only_the_root_rules_sequence_holds_rules(anchors, replacements, location):
    """A `rules` key anywhere else is data, so its `metadata` is not exempt."""

    content = _module(anchors, replacements)

    assert find_yaml_merge_key_issues(content) == [
        _merge_key_issue(content, location, "<<: {note")
    ]


def test_a_merge_reached_in_two_roles_keeps_its_first_location():
    """A rule anchored under an unknown root key is walked as data and as a rule.

    Its merge key is reported once, where the text is written.
    """

    content = (
        FORMAT + "x-rule: &rule\n  <<: {kind: parameter}\n  name: sua_value\n"
        "  dtype: Money\n  versions:\n    - effective_from: '2024-01-01'\n"
        "      formula: '451'\n"
        + MODULE_BLOCK.replace("    required: true\n", "    required: false\n")
        + "rules:\n  - *rule\n"
    )

    assert find_yaml_merge_key_issues(content) == [
        _merge_key_issue(content, "x-rule key `<<`", "<<: {kind")
    ]


def test_merge_keys_are_reported_in_position_order():
    """An alias can reach an earlier key after a later one; the report is sorted."""

    content = _module(
        replacements=[
            (
                "    metadata:\n",
                "    metadata:\n      x-version: &v\n"
                "        <<: {effective_from: '2024-01-01'}\n"
                "        formula: '451'\n",
            ),
            (
                VERSIONS,
                "    sources:\n      - <<: {citation: us/guidance/example/sua}\n"
                "    versions:\n      - *v\n",
            ),
        ]
    )

    assert find_yaml_merge_key_issues(content) == [
        _merge_key_issue(content, "rules[0].versions[0] key `<<`", "<<: {effective"),
        _merge_key_issue(content, "rules[0].sources[0] key `<<`", "<<: {citation"),
    ]


def test_a_merge_tagged_metadata_key_does_not_exempt_what_it_merges():
    """`!!merge metadata:` merges into the rule itself; it names no `metadata`."""

    content = _module(
        replacements=[("    unit: USD\n", "    !!merge metadata: {<<: {unit: USD}}\n")]
    )

    assert find_yaml_merge_key_issues(content) == [
        _merge_key_issue(content, "rules[0] key `metadata`", "!!merge", tagged=True),
        _merge_key_issue(content, "rules[0].metadata key `<<`", "<<: {unit"),
    ]


def test_a_mapping_key_is_never_a_rule():
    """Keys are data even when they are mappings holding a `rules` sequence."""

    content = "? {rules: [{metadata: {<<: {a: 1}}}]}\n: x\n"

    assert find_yaml_merge_key_issues(content) == [
        _merge_key_issue(
            content, "document root key `?`.rules[0].metadata key `<<`", "<<"
        )
    ]


def test_merge_keys_in_documents_that_are_not_modules_do_not_fail_the_gate():
    """Companion tests and worklists are read only by PyYAML."""

    tests = (
        "- &base\n  name: sua_applies\n  period: 2024-01\n  input: {}\n"
        "  output:\n    sua_value: 451\n- <<: *base\n  name: sua_applies_again\n"
    )
    worklist = "x: &x {a: 1}\nentries:\n  - <<: *x\n    b: 2\n"

    for content in (tests, worklist):
        assert find_yaml_merge_key_issues(content) != []
        assert _gate(content).passed


def test_the_merge_report_is_bounded():
    rules = "".join(
        f"  - <<: *defaults\n    name: rate_{index}\n" for index in range(12)
    )
    content = _module("x-defaults: &defaults\n  kind: parameter\n", rules=rules)

    issues = find_yaml_merge_key_issues(content)

    assert len(issues) == 11
    assert issues[-1] == "2 more YAML merge key(s) not shown."
    assert issues[0].startswith("rules[0] key `<<` is a YAML merge key at line ")


def test_the_merge_scan_keeps_parser_errors():
    with pytest.raises(yaml.YAMLError):
        find_yaml_merge_key_issues("rules: [\n")
    assert find_yaml_merge_key_issues("") == []
    assert find_yaml_merge_key_issues("# only a comment\n") == []
    assert find_yaml_merge_key_issues("- <<: {a: 1}\n") == [
        f"document root[0] key `<<` is a YAML merge key at line 1, column 3: "
        f"{MERGE_KEY_REASON}"
    ]


def test_a_scalar_merge_value_still_fails_the_gate():
    """PyYAML cannot construct `<<: 1`; the engine ignores the key."""

    content = _module(replacements=[("    unit: USD\n", "    unit: USD\n    <<: 1\n")])

    assert find_yaml_merge_key_issues(content) == [
        _merge_key_issue(content, "rules[0] key `<<`", "<<: 1")
    ]
    assert not _gate(content).passed


def test_single_pass_load_reports_the_same_merge_keys():
    for content in [MODULE, *ACCEPTED.values()] + [
        case[0] for case in (*REJECTED.values(), *REJECTED_ALTHOUGH_INERT.values())
    ]:
        merges = _safe_load_with_raw_constraints(content)[4]
        assert merges == find_yaml_merge_key_issues(content)


# --- property: the scan passes exactly when PyYAML and the engine agree -----
#
# The engine reads `<<` as an ordinary key. A loader that does the same, with
# rule metadata stripped from both views, must agree with `yaml.safe_load`
# exactly when the scan reports nothing.


class _MergeBlindLoader(yaml.SafeLoader):
    """Load `<<` as an ordinary string key, as serde_yaml 0.9 does."""

    def flatten_mapping(self, node):
        return None


_MergeBlindLoader.add_constructor(
    "tag:yaml.org,2002:merge", lambda loader, node: loader.construct_scalar(node)
)


def _without_rule_metadata(value):
    if not isinstance(value, dict):
        return value
    rules = value.get("rules")
    if not isinstance(rules, list):
        return value
    stripped = [
        {key: item for key, item in rule.items() if key != "metadata"}
        if isinstance(rule, dict)
        else rule
        for rule in rules
    ]
    return {**value, "rules": stripped}


_ANCHORS = ("a0", "a1", "a2")
_MERGE = st.sampled_from(
    [
        None,
        "<<: *a0",
        "<<: *a1",
        "<<: [*a1, *a2]",
        "<<: {k9: merged}",
        "? <<\n{pad}: *a2",
    ]
)


def _block(lines, indent, merge):
    pad = " " * indent
    text = ""
    if merge is not None:
        text += pad + merge.replace("{pad}", pad) + "\n"
    for line in lines:
        text += pad + line + "\n"
    return text


@st.composite
def _documents(draw):
    anchors = "".join(
        f"x-{name}: &{name}\n  k{index}: v{index}\n  shared: {name}\n"
        for index, name in enumerate(_ANCHORS)
    )
    text = FORMAT + anchors
    text += "module:\n" + _block(["summary: s"], 2, draw(_MERGE))
    text += "  source_verification:\n" + _block(
        ["corpus_citation_path: us/guidance/example/sua"], 4, draw(_MERGE)
    )
    text += "rules:\n"
    aliased_version = None
    for index in range(draw(st.integers(1, 3))):
        rule = _block([f"name: r{index}", "kind: parameter"], 4, draw(_MERGE))
        text += "  -" + rule[3:]
        if draw(st.booleans()):
            text += "    metadata:\n" + _block(["note: m"], 6, draw(_MERGE))
            text += "      proof:\n" + _block(["required: true"], 8, draw(_MERGE))
            text += (
                "      atoms:\n        -" + _block(["path: p"], 10, draw(_MERGE))[9:]
            )
            if aliased_version is None and draw(st.booleans()):
                aliased_version = f"v{index}"
                text += f"      x-version: &{aliased_version}\n" + _block(
                    ["formula: '1'"], 8, draw(_MERGE)
                )
        text += (
            "    versions:\n      -"
            + _block(["effective_from: '2024-01-01'", "formula: '1'"], 8, draw(_MERGE))[
                7:
            ]
        )
        if aliased_version is not None and draw(st.booleans()):
            text += f"      - *{aliased_version}\n"
    return text


@settings(max_examples=400, deadline=None)
@given(content=_documents())
def test_the_scan_passes_exactly_when_both_loaders_agree_outside_metadata(content):
    pyyaml_view = _without_rule_metadata(yaml.safe_load(content))
    engine_view = _without_rule_metadata(yaml.load(content, Loader=_MergeBlindLoader))

    assert (find_yaml_merge_key_issues(content) == []) == (pyyaml_view == engine_view)


def _merge_key_positions(content):
    """Independent oracle: every plain `<<` key scalar in the parser's events."""

    return sorted(
        (event.start_mark.line + 1, event.start_mark.column + 1)
        for event in yaml.parse(content, Loader=yaml.SafeLoader)
        if isinstance(event, yaml.events.ScalarEvent)
        and event.value == "<<"
        and event.style is None
        and event.tag is None
    )


def _reported_position(issue):
    line, column = issue.split(" at line ", 1)[1].split(":", 1)[0].split(", column ")
    return int(line), int(column)


@settings(max_examples=300, deadline=None)
@given(content=_documents())
def test_outside_metadata_every_merge_key_is_reported_at_its_position(content):
    """With metadata removed from the text, every `<<` must be reported."""

    payload = yaml.safe_load(content)
    if any("metadata" in rule for rule in payload["rules"]):
        # Metadata merges are allowed; only documents without metadata (and
        # so without aliases into it) can be checked against every `<<`.
        return
    issues = find_yaml_merge_key_issues(content)
    positions = _merge_key_positions(content)

    if len(positions) > 10:
        assert issues[-1] == f"{len(positions) - 10} more YAML merge key(s) not shown."
        issues, positions = issues[:-1], positions[:10]
    assert [_reported_position(issue) for issue in issues] == positions


# --- the engine: every rejected merge changes what it compiles ------------------


@pytest.mark.parametrize("case", sorted(REJECTED), ids=str)
def test_engine_compiles_rejected_merges_apart_from_their_rewrite(
    real_engine, tmp_path, case
):
    """The engine rejects the raw module, or compiles it unlike its rewrite."""

    content, _holders = REJECTED[case]
    raw, raw_program = _engine_compile(real_engine.binary, tmp_path / "raw", content)
    rewritten, rewritten_program = _engine_compile(
        real_engine.binary, tmp_path / "rewritten", _flattened(content)
    )

    if raw.returncode == 0:
        assert rewritten.returncode != 0 or raw_program != rewritten_program
    else:
        assert raw.stderr


@pytest.mark.parametrize(
    "case",
    [
        "rule-supplies-kind",
        "rule-supplies-versions",
        "rule-copies-another-rule",
        "version-copies-another-version",
        "version-supplies-effective-from",
        "version-supplies-formula",
        "source-verification-merge",
        "root-supplies-format",
        "metadata-mapping-aliased-into-versions",
    ],
)
def test_engine_rejects_modules_the_old_gate_passed(real_engine, tmp_path, case):
    content, _holders = REJECTED[case]

    raw, _program = _engine_compile(real_engine.binary, tmp_path / "raw", content)
    rewritten, _program = _engine_compile(
        real_engine.binary, tmp_path / "rewritten", _flattened(content)
    )

    assert raw.returncode != 0
    assert rewritten.returncode == 0, rewritten.stderr


def test_engine_rejects_a_merge_key_in_source_verification(real_engine, tmp_path):
    content, _holders = REJECTED["source-verification-merge"]

    raw, _program = _engine_compile(real_engine.binary, tmp_path, content)

    assert "unknown field `<<`" in raw.stderr


@pytest.mark.parametrize(
    "case", ["rule-supplies-an-optional-field", "module-supplies-title"]
)
def test_engine_compiles_a_partial_module_where_pyyaml_sees_the_merged_field(
    real_engine, tmp_path, case
):
    content, _holders = REJECTED[case]

    raw, raw_program = _engine_compile(real_engine.binary, tmp_path / "raw", content)
    rewritten, rewritten_program = _engine_compile(
        real_engine.binary, tmp_path / "rewritten", _flattened(content)
    )

    assert raw.returncode == 0, raw.stderr
    assert rewritten.returncode == 0, rewritten.stderr
    assert raw_program != rewritten_program


@pytest.mark.parametrize("case", sorted(ACCEPTED), ids=str)
def test_engine_compiles_accepted_merges_like_their_rewrite(
    real_engine, tmp_path, case
):
    """A rewrite of an accepted module does not change what the engine compiles."""

    content = ACCEPTED[case]
    raw, raw_program = _engine_compile(real_engine.binary, tmp_path / "raw", content)
    rewritten, rewritten_program = _engine_compile(
        real_engine.binary, tmp_path / "rewritten", _flattened(content)
    )

    assert raw.returncode == 0, raw.stderr
    assert rewritten.returncode == 0, rewritten.stderr
    assert raw_program == rewritten_program


@pytest.mark.parametrize("case", sorted(REJECTED_ALTHOUGH_INERT), ids=str)
def test_engine_compiles_the_inert_rejected_merges_like_their_rewrite(
    real_engine, tmp_path, case
):
    """These are rejected by location only; the engine sees no difference."""

    content, _holders = REJECTED_ALTHOUGH_INERT[case]
    raw, raw_program = _engine_compile(real_engine.binary, tmp_path / "raw", content)
    rewritten, rewritten_program = _engine_compile(
        real_engine.binary, tmp_path / "rewritten", _flattened(content)
    )

    assert raw.returncode == 0, raw.stderr
    assert raw_program == rewritten_program


# --- the rewrite paths -------------------------------------------------------------

# A module the apply and eval paths accept: no strict proofs, so the repairs and
# stamping run on it.
APPLY_CLEAN = """format: rulespec/v1
module:
  summary: Fictional lantern review rates.
rules:
  - name: first_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.1'
"""
APPLY_CITATION = "us/statute/26/1"


def _apply_case(anchors, old, new, location, written):
    content = APPLY_CLEAN.replace(FORMAT, FORMAT + anchors, 1).replace(old, new, 1)
    return pytest.param(
        content,
        _merge_key_issue(content, location, written),
        id=location.split(" ")[0],
    )


APPLY_CASES = [
    _apply_case(
        "x-defaults: &defaults\n  kind: parameter\n",
        "    kind: parameter\n",
        "    <<: *defaults\n",
        "rules[0] key `<<`",
        "<<: *defaults",
    ),
    _apply_case(
        "",
        "      - effective_from: '2024-01-01'\n",
        "      - <<: {effective_from: '2024-01-01'}\n",
        "rules[0].versions[0] key `<<`",
        "<<: {effective_from",
    ),
    _apply_case(
        "x-meta: &meta\n  title: Lantern review rates\n",
        "module:\n",
        "module:\n  <<: *meta\n",
        "module key `<<`",
        "<<: *meta",
    ),
]
# Shared metadata a rule merges in; rewriting it changes nothing the engine reads.
APPLY_METADATA_MERGE = APPLY_CLEAN.replace(
    "    dtype: Float\n",
    "    dtype: Float\n    metadata:\n      x-shared: &shared\n        note: lantern\n"
    "      notes:\n        <<: *shared\n",
    1,
)
MERGED_TESTS = """- &base
  name: first_rate_applies
  period: 2024-01-01
  input: {}
  output:
    first_rate: 0.1
- <<: *base
  name: first_rate_applies_again
"""


def _candidate(tmp_path, content, test_content=None):
    output_root = tmp_path / "out"
    generated = output_root / "codex-test-model/statutes/26/1.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text(content)
    if test_content is not None:
        generated.with_name("1.test.yaml").write_text(test_content)
    result = SimpleNamespace(
        output_file=str(generated),
        runner="codex-test-model",
        backend="codex",
        source_attestation=_complete_source_attestation(APPLY_CITATION),
        context_manifest_file=str(
            _write_test_context_manifest(
                tmp_path / "context", citation_path=APPLY_CITATION
            )
        ),
    )
    return result, generated, output_root


APPLY_PREFIX = "Cannot apply RuleSpec YAML with merge keys the engine does not apply: "


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
def test_real_stamping_rejects_merge_keys_before_rewrite(tmp_path, content, diagnostic):
    result, generated, _ = _candidate(tmp_path, content)
    original = generated.read_bytes()

    with pytest.raises(RuntimeError) as exc_info:
        _stamp_generated_source_attestation_for_apply(result, generated)

    assert str(exc_info.value) == APPLY_PREFIX + diagnostic
    assert generated.read_bytes() == original


def test_real_stamping_still_rewrites_metadata_merges(tmp_path):
    result, generated, _ = _candidate(tmp_path, APPLY_METADATA_MERGE)

    _stamp_generated_source_attestation_for_apply(result, generated)

    stamped = yaml.safe_load(generated.read_text())
    assert "<<" not in generated.read_text()
    assert stamped["rules"][0]["metadata"]["notes"] == {"note": "lantern"}
    assert stamped["module"]["source_verification"]["corpus_citation_path"] == (
        APPLY_CITATION
    )


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
@pytest.mark.parametrize(
    "error",
    [
        "Generated RuleSpec failed compile validation",
        "Generated RuleSpec failed CI validation",
    ],
)
def test_apply_preflight_blocks_merge_keys_before_repairs(
    tmp_path, content, diagnostic, error
):
    result, generated, output_root = _candidate(tmp_path, content)
    result.success = False
    result.error = error
    assert _can_attempt_apply(result)
    original = generated.read_bytes()

    issue = _generated_rulespec_yaml_nonparser_issue(
        result, generated, output_root=output_root
    )

    assert issue == APPLY_PREFIX + diagnostic
    assert generated.read_bytes() == original


def test_apply_preflight_passes_metadata_merges_and_merged_tests(tmp_path):
    result, generated, output_root = _candidate(
        tmp_path, APPLY_METADATA_MERGE, MERGED_TESTS
    )
    result.success = False
    result.error = "Generated RuleSpec failed CI validation"

    assert (
        _generated_rulespec_yaml_nonparser_issue(
            result, generated, output_root=output_root
        )
        is None
    )


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
def test_apply_overlay_blocks_merge_keys_with_real_stamping(
    tmp_path, content, diagnostic
):
    result, generated, output_root = _candidate(tmp_path, content)
    original = generated.read_bytes()

    ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
        result,
        output_root=output_root,
        policy_repo_path=tmp_path / "rulespec-us/us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=MagicMock(),
        validate_dependents=False,
    )

    assert not ok
    assert any(diagnostic in issue for issue in issues), issues
    assert supplemental == {}
    assert generated.read_bytes() == original


# Generated `module.inputs` makes materialization lift them through a load and
# dump, so these documents exercise the rewrite the guard must skip.
MISPLACED_INPUTS = """  inputs:
    - name: supplied_review_count
      entity: Person
      dtype: Count
      period: Month
"""


def _materialize(tmp_path, content, delivery, test_content=None):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    target = artifact_root / "section.yaml"
    workspace_root = None
    files = {"section.yaml": content}
    if test_content is not None:
        files["section.test.yaml"] = test_content
    if delivery == "response":
        response = content
    elif delivery == "bundle":
        response = "".join(
            f"=== FILE: {name} ===\n{text}" for name, text in files.items()
        )
    else:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        for name, text in files.items():
            (workspace_root / name).write_text(text, encoding="utf-8")
        response = "Wrote the requested files."
    assert _materialize_eval_artifact(
        response, target, workspace_root=workspace_root, artifact_root=artifact_root
    )
    test_target = target.with_name("section.test.yaml")
    return (
        target.read_text(encoding="utf-8"),
        test_target.read_text(encoding="utf-8") if test_target.exists() else None,
    )


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_keeps_raw_merge_keys(tmp_path, content, diagnostic, delivery):
    content = content.replace("module:\n", "module:\n" + MISPLACED_INPUTS, 1)

    materialized, _tests = _materialize(tmp_path, content, delivery)

    assert materialized == content
    assert find_yaml_merge_key_issues(materialized)


@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_still_repairs_modules_with_metadata_merges(tmp_path, delivery):
    content = APPLY_METADATA_MERGE.replace(
        "module:\n", "module:\n" + MISPLACED_INPUTS, 1
    )

    materialized, _tests = _materialize(tmp_path, content, delivery)

    payload = yaml.safe_load(materialized)
    assert "inputs" not in payload["module"]
    assert payload["inputs"][0]["name"] == "supplied_review_count"
    assert payload["rules"][0]["metadata"]["notes"] == {"note": "lantern"}


@pytest.mark.parametrize("delivery", ["bundle", "workspace"])
def test_materialization_still_normalizes_merged_tests(tmp_path, delivery):
    _main, materialized_tests = _materialize(
        tmp_path, APPLY_CLEAN, delivery, MERGED_TESTS
    )

    cases = yaml.safe_load(materialized_tests)
    assert materialized_tests != MERGED_TESTS
    assert [case["name"] for case in cases] == [
        "first_rate_applies",
        "first_rate_applies_again",
    ]
    assert cases[1]["output"] == {"first_rate": 0.1}


def test_main_normalization_keeps_raw_merge_keys():
    content = APPLY_CASES[0].values[0]

    assert (
        _normalize_main_eval_content(
            content,
            target_path=Path("section.yaml"),
            single_amount_table_slice=False,
        )
        == content
    )


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
@pytest.mark.parametrize("merged_side", ["generated", "preserved"])
def test_retry_overlay_rejects_merge_keys_before_rewriting(
    tmp_path, content, diagnostic, merged_side
):
    generated = content if merged_side == "generated" else APPLY_CLEAN
    preserved = content if merged_side == "preserved" else APPLY_CLEAN
    target = tmp_path / "section.yaml"
    target.write_text(generated, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target,
            artifact_root=tmp_path,
            candidate=ValidationRetryCandidate(rulespec=preserved),
        )

    assert str(exc_info.value) == (
        f"{merged_side} repair overlay has YAML merge keys the engine does not "
        f"apply: {diagnostic}"
    )
    assert target.read_text(encoding="utf-8") == generated


def test_retry_overlay_still_merges_metadata_merges_and_merged_tests(tmp_path):
    target = tmp_path / "section.yaml"
    target.write_text(APPLY_METADATA_MERGE, encoding="utf-8")
    target.with_name("section.test.yaml").write_text(MERGED_TESTS, encoding="utf-8")
    preserved = APPLY_METADATA_MERGE.replace(
        "module:\n", "imports:\n  - us:statutes/26/2\nmodule:\n"
    )

    repairs = _overlay_validation_retry_candidate(
        target,
        artifact_root=tmp_path,
        candidate=ValidationRetryCandidate(rulespec=preserved, tests=MERGED_TESTS),
    )

    assert "import:us:statutes/26/2" in repairs
    assert yaml.safe_load(target.read_text())["imports"] == ["us:statutes/26/2"]


UNUSED_IMPORT = "us:statutes/26/2#lantern_base_rate"


def _with_unused_import(content):
    return content.replace("module:\n", f"imports:\n  - {UNUSED_IMPORT}\nmodule:\n", 1)


def _eval_repairs(tmp_path, content, test_content=None):
    rules_file = tmp_path / "section.yaml"
    rules_file.write_text(content)
    if test_content is not None:
        rules_file.with_name("section.test.yaml").write_text(test_content)
    repairs = _apply_generated_eval_repairs(
        rulespec_file=rules_file,
        policy_repo_root=tmp_path,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        issues=[f"Unused import `{UNUSED_IMPORT}`"],
        local_corpus_release=MagicMock(),
    )
    return repairs, rules_file.read_text()


def test_eval_repairs_skip_modules_with_merge_keys(tmp_path):
    content = _with_unused_import(APPLY_CASES[0].values[0])

    repairs, repaired = _eval_repairs(tmp_path, content)

    assert repairs == []
    assert repaired == content


def test_eval_repairs_still_run_with_metadata_merges_and_merged_tests(tmp_path):
    content = _with_unused_import(APPLY_METADATA_MERGE)

    repairs, repaired = _eval_repairs(tmp_path, content, MERGED_TESTS)

    assert repairs == [f"unused_import:{UNUSED_IMPORT}"]
    assert UNUSED_IMPORT not in repaired


def _pipeline(tmp_path):
    return ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
        local_corpus_release=None,
    )


def _pipeline_module(tmp_path, content, tests=None):
    rules_file = tmp_path / "rulespec-us/statutes/26/1.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(content)
    if tests is not None:
        rules_file.with_name("1.test.yaml").write_text(tests)
    return rules_file


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
def test_pipeline_compile_check_reports_merge_keys_before_compiling(
    tmp_path, content, diagnostic
):
    """Validators read the loaded payload, so the preflight stops first."""

    rules_file = _pipeline_module(tmp_path, content)

    result = _pipeline(tmp_path)._run_compile_check(rules_file)

    assert not result.passed
    assert result.issues == [
        "1.yaml has YAML merge keys the engine does not apply: " + diagnostic
    ]


def test_pipeline_ci_accepts_merged_companion_tests(tmp_path, monkeypatch):
    rules_file = _pipeline_module(tmp_path, APPLY_METADATA_MERGE, MERGED_TESTS)
    pipeline = _pipeline(tmp_path)

    def fake_compile(rules_file, output_path):
        return (
            subprocess.CompletedProcess(["axiom-rules-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        )

    monkeypatch.setattr(pipeline, "_compile_rulespec_to_artifact", fake_compile)

    result = pipeline._run_rulespec_ci(rules_file)

    assert not any("merge key" in issue for issue in result.issues), result.issues


def test_the_rewrite_merge_check_reports_text_pyyaml_cannot_load():
    """Like its sibling checks, it reports a plain ValueError instead of raising."""

    assert _rewrite_merge_key_issues('summary: "\\U00110000"\n') == [
        _UNLOADABLE_YAML_ISSUE
    ]
    assert _rewrite_merge_key_issues("rules: [\n") == []
    assert _rewrite_merge_key_issues(APPLY_METADATA_MERGE) == []
