"""Explicit YAML tags that PyYAML and the engine read differently are rejected.

``yaml.safe_load`` turns a `!!set` mapping into a set of its keys and an
`!!omap` or `!!pairs` sequence into a list of key-value tuples, while the
engine ignores those tags and reads the mapping or sequence as written. The
proof gate and the other validators skip a rule, version or proof that does not
load as a mapping, so before this guard a policy-bearing rule written
`- !!set` passed strict proof validation while the engine compiled it. Scalar
tags diverge too, and `!!bool` or `!!timestamp` on other text made the gate
raise instead of reporting.
"""

import json
import subprocess
from datetime import date
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
from axiom_encode.harness.amendment_timing import amendment_timing_issues
from axiom_encode.harness.evals import (
    ValidationRetryCandidate,
    _apply_generated_eval_repairs,
    _materialize_eval_artifact,
    _normalize_main_eval_content,
    _overlay_validation_retry_candidate,
    _rewrite_unsupported_tag_issues,
)
from axiom_encode.harness.proof_validator import (
    _safe_load_with_raw_constraints,
    find_interpretation_yaml_issues,
    find_unsupported_yaml_tag_issues,
    validate_rulespec_proofs,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.yaml_serialization import safe_dump_yaml
from tests.test_cli import (
    _complete_source_attestation,
    _write_test_context_manifest,
)

CITATION = "us/statute/26/1"
SOURCE = "The first rate is 0.1. The second rate is 0.2."
SOURCES = {CITATION: SOURCE}

# Each `{slot}` is a tag position; filling every slot with "" gives the
# untagged module, whose only proof issue is the missing proof of `second_rate`.
SLOTTED = """format: rulespec/v1
module:
  summary: {summary}Fictional lantern review rates.
  proof_validation: {proof_validation}
    required: {required}true
  source_verification: {source_verification}
    corpus_citation_path: us/statute/26/1
rules: {rules}
  - {rule}
    name: {name}first_rate
    kind: {kind}parameter
    dtype: Float
    metadata: {metadata}
      proof: {proof}
        atoms: {atoms}
          - {atom}
            path: versions[0].formula
            kind: formula
            source: {atom_source}
              corpus_citation_path: us/statute/26/1
              excerpt: {excerpt}The first rate is 0.1.
    versions: {versions}
      - {version}
        effective_from: {effective_from}'2024-01-01'
        formula: {formula}'0.1'
  - {second_rule}
    name: second_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
"""
SLOTS = (
    "summary",
    "proof_validation",
    "required",
    "source_verification",
    "rules",
    "rule",
    "name",
    "kind",
    "metadata",
    "proof",
    "atoms",
    "atom",
    "atom_source",
    "excerpt",
    "versions",
    "version",
    "effective_from",
    "formula",
    "second_rule",
)
LOCATIONS = {
    "summary": "module.summary",
    "proof_validation": "module.proof_validation",
    "required": "module.proof_validation.required",
    "source_verification": "module.source_verification",
    "rules": "rules",
    "rule": "rules[0]",
    "name": "rules[0].name",
    "kind": "rules[0].kind",
    "metadata": "rules[0].metadata",
    "proof": "rules[0].metadata.proof",
    "atoms": "rules[0].metadata.proof.atoms",
    "atom": "rules[0].metadata.proof.atoms[0]",
    "atom_source": "rules[0].metadata.proof.atoms[0].source",
    "excerpt": "rules[0].metadata.proof.atoms[0].source.excerpt",
    "versions": "rules[0].versions",
    "version": "rules[0].versions[0]",
    "effective_from": "rules[0].versions[0].effective_from",
    "formula": "rules[0].versions[0].formula",
    "second_rule": "rules[1]",
}
MAPPING_SLOTS = (
    "proof_validation",
    "source_verification",
    "rule",
    "metadata",
    "proof",
    "atom",
    "atom_source",
    "version",
    "second_rule",
)
SEQUENCE_SLOTS = ("rules", "atoms", "versions")
SCALAR_SLOTS = (
    "summary",
    "required",
    "name",
    "kind",
    "excerpt",
    "effective_from",
    "formula",
)
UNTAGGED_ISSUES = [
    "Proof missing: rule `second_rate` is policy-bearing and must declare "
    "`metadata.proof.atoms`."
]


def _fill(slot=None, tag=""):
    values = dict.fromkeys(SLOTS, "")
    if slot is not None:
        values[slot] = f"{tag} "
    return SLOTTED.format(**values)


UNTAGGED = _fill()


def _gate(content):
    return validate_rulespec_proofs(content, source_texts=SOURCES)


def _tag_position(content, slot, tag):
    """1-based line and column where ``tag`` was written into ``slot``."""

    marker = "{" + slot + "}"
    before = SLOTTED[: SLOTTED.index(marker)].format(**dict.fromkeys(SLOTS, ""))
    assert content.startswith(before + tag)
    line = before.count("\n") + 1
    return line, len(before) - (before.rfind("\n") + 1) + 1


SET_REASON = (
    "PyYAML loads this mapping as a set of its keys and drops their values, "
    "while the engine ignores the tag and reads the mapping. Remove the tag."
)
PAIRS_REASON = (
    "PyYAML loads this sequence as a list of key-value pairs instead of "
    "mappings, while the engine ignores the tag and reads the mappings. Remove "
    "the tag."
)
SCALAR_REASON = (
    "only `!!str` may tag a scalar. Other scalar tags can make PyYAML and the "
    "engine read the same text differently: `!!binary` loads as bytes and "
    "`!!null` as null whatever the text, and PyYAML resolves `! '0.1'` as the "
    "number 0.1, while the engine reads the text or rejects it."
)
MAPPING_REASON = "only `!!map` or the non-specific `!` may tag a mapping."
SEQUENCE_REASON = "only `!!seq` or the non-specific `!` may tag a sequence."


def _issue(slot, tag, short, reason):
    line, column = _tag_position(_fill(slot, tag), slot, tag)
    return (
        f"{LOCATIONS[slot]} has unsupported YAML tag `{short}` at line {line}, "
        f"column {column}: {reason}"
    )


SLOTTED_CASES = [
    *(
        pytest.param(slot, tag, short, SET_REASON, id=f"{slot}-{short}-{spelling}")
        for slot in MAPPING_SLOTS
        for tag, short, spelling in (
            ("!!set", "!!set", "shorthand"),
            ("!<tag:yaml.org,2002:set>", "!!set", "verbatim"),
        )
    ),
    *(
        pytest.param(slot, tag, short, PAIRS_REASON, id=f"{slot}-{short}")
        for slot in SEQUENCE_SLOTS
        for tag, short in (("!!omap", "!!omap"), ("!!pairs", "!!pairs"))
    ),
    *(
        pytest.param(slot, tag, tag, SCALAR_REASON, id=f"{slot}-{tag}")
        for slot in SCALAR_SLOTS
        for tag in (
            "!!binary",
            "!!null",
            "!!bool",
            "!!timestamp",
            "!!int",
            "!!float",
            "!",
        )
    ),
    pytest.param("kind", "!local", "!local", SCALAR_REASON, id="kind-local-tag"),
    pytest.param("rule", "!rule", "!rule", MAPPING_REASON, id="rule-local-tag"),
    pytest.param("rule", "!!seq", "!!seq", MAPPING_REASON, id="rule-seq-tag"),
    pytest.param("versions", "!!map", "!!map", SEQUENCE_REASON, id="versions-map"),
    pytest.param("versions", "!!set", "!!set", SEQUENCE_REASON, id="versions-set"),
]


@pytest.mark.parametrize(("slot", "tag", "short", "reason"), SLOTTED_CASES)
def test_raw_guard_reports_type_changing_tags_anywhere(slot, tag, short, reason):
    content = _fill(slot, tag)
    issue = _issue(slot, tag, short, reason)

    assert find_unsupported_yaml_tag_issues(content) == [issue]
    result = _gate(content)
    assert result.issues == [issue]
    assert not result.passed


@pytest.mark.parametrize(
    ("slot", "tag"),
    [
        *((slot, tag) for slot in MAPPING_SLOTS for tag in ("!!map", "!")),
        ("rule", "!<tag:yaml.org,2002:map>"),
        *((slot, tag) for slot in SEQUENCE_SLOTS for tag in ("!!seq", "!")),
        *((slot, "!!str") for slot in SCALAR_SLOTS),
        ("name", "!<tag:yaml.org,2002:str>"),
    ],
)
def test_raw_guard_allows_tags_that_both_loaders_read_as_written(slot, tag):
    content = _fill(slot, tag)

    assert find_unsupported_yaml_tag_issues(content) == []
    assert _gate(content).issues == _expected_accepted_issues(slot, tag)


def _expected_accepted_issues(slot, tag):
    # Both loaders read `!!str true` as the string "true", which does not turn
    # strict proof validation on.
    return [] if (slot, tag) == ("required", "!!str") else UNTAGGED_ISSUES


def test_untagged_module_fails_only_on_the_missing_proof():
    result = _gate(UNTAGGED)

    assert result.issues == UNTAGGED_ISSUES
    assert result.proof_required


@pytest.mark.parametrize("slot", ["second_rule", "proof_validation"])
def test_set_tags_no_longer_pass_the_strict_proof_gate(slot):
    """The defect: these loaded as sets, so the gate skipped them and passed."""

    content = _fill(slot, "!!set")
    payload = yaml.safe_load(content)
    hidden = payload["rules"][1] if slot == "second_rule" else payload["module"][slot]
    assert isinstance(hidden, set)

    result = _gate(content)

    assert not result.passed
    assert result.issues == [_issue(slot, "!!set", "!!set", SET_REASON)]


def _swap(*pairs):
    content = UNTAGGED
    for old, new in pairs:
        assert content.count(old) == 1, old
        content = content.replace(old, new)
    return content


SECOND_RULE = (
    "  - \n    name: second_rate\n    kind: parameter\n    dtype: Float\n"
    "    versions:\n      - effective_from: '2024-01-01'\n        formula: '0.2'\n"
)
EXPLICIT = {
    "flow-set-rule": (
        _swap(
            (
                SECOND_RULE,
                "  - !!set {name: second_rate, kind: parameter, dtype: Float, "
                "versions: [{effective_from: '2024-01-01', formula: '0.2'}]}\n",
            )
        ),
        "rules[1] has unsupported YAML tag `!!set` at line 26, column 5",
    ),
    "explicit-key-set-rule": (
        _swap(
            (
                SECOND_RULE,
                "  - !!set\n    ? name\n    : second_rate\n    ? kind\n"
                "    : parameter\n    ? dtype\n    : Float\n    ? versions\n"
                "    : - effective_from: '2024-01-01'\n        formula: '0.2'\n",
            )
        ),
        "rules[1] has unsupported YAML tag `!!set` at line 26, column 5",
    ),
    "aliased-set-rule": (
        _swap(
            (
                "rules: \n",
                "x-rule: &second !!set\n  name: second_rate\n  kind: parameter\n"
                "  dtype: Float\n  versions:\n    - effective_from: '2024-01-01'\n"
                "      formula: '0.2'\nrules: \n",
            ),
            (SECOND_RULE, "  - *second\n"),
        ),
        "x-rule has unsupported YAML tag `!!set` at line 8, column 9",
    ),
    "single-quoted-key-set-value": (
        _swap(("    metadata: \n", "    'metadata': !!set\n")),
        "rules[0].metadata has unsupported YAML tag `!!set` at line 13, column 17",
    ),
    "double-quoted-key-omap-value": (
        _swap(("rules: \n", '"rules": !!omap\n')),
        "rules has unsupported YAML tag `!!omap` at line 8, column 10",
    ),
    "loadable-pairs-sources": (
        _swap(
            (
                "    versions: \n",
                "    sources: !!pairs\n      - citation: us/statute/26/1\n"
                "    versions: \n",
            )
        ),
        "rules[0].sources has unsupported YAML tag `!!pairs` at line 22, column 14",
    ),
    "loadable-omap-imports": (
        _swap(("module:\n", "imports: !!omap\n  - us: statutes/26/2\nmodule:\n")),
        "imports has unsupported YAML tag `!!omap` at line 2, column 10",
    ),
    "empty-pairs": (
        _swap(("rules: \n", "x-empty: !!pairs []\nrules: \n")),
        "x-empty has unsupported YAML tag `!!pairs` at line 8, column 10",
    ),
    "binary-key": (
        _swap(("    name: first_rate\n", "    !!binary bmFtZQ==: first_rate\n")),
        "rules[0] key `bmFtZQ==` has unsupported YAML tag `!!binary` at line 10, "
        "column 5",
    ),
    "decodable-binary": (
        _swap(
            (
                "  summary: Fictional lantern review rates.\n",
                "  summary: !!binary TGFudGVybg==\n",
            )
        ),
        "module.summary has unsupported YAML tag `!!binary` at line 3, column 12",
    ),
    "nested-flow-set": (
        _swap(
            (
                "        formula: '0.1'\n",
                "        formula: '0.1'\n        x-deep: [{a: [{b: !!set {c: d}}]}]\n",
            )
        ),
        "rules[0].versions[0].x-deep[0].a[0].b has unsupported YAML tag `!!set` at "
        "line 26, column 27",
    ),
}
EXPLICIT_CASES = [
    pytest.param(content, diagnostic, id=case)
    for case, (content, diagnostic) in EXPLICIT.items()
]


@pytest.mark.parametrize(("content", "diagnostic"), EXPLICIT_CASES)
def test_raw_guard_reports_other_spellings_and_positions(content, diagnostic):
    issues = find_unsupported_yaml_tag_issues(content)

    assert len(issues) == 1
    assert issues[0].startswith(diagnostic + ": ")
    assert _gate(content).issues == issues


def test_loadable_pair_and_binary_tags_change_what_pyyaml_returns():
    """The premise: PyYAML builds tuples and bytes where the text has mappings."""

    pairs = yaml.safe_load(EXPLICIT["loadable-pairs-sources"][0])
    assert pairs["rules"][0]["sources"] == [("citation", CITATION)]
    imports = yaml.safe_load(EXPLICIT["loadable-omap-imports"][0])
    assert imports["imports"] == [("us", "statutes/26/2")]
    binary = yaml.safe_load(EXPLICIT["decodable-binary"][0])
    assert binary["module"]["summary"] == b"Lantern"
    aliased = yaml.safe_load(EXPLICIT["aliased-set-rule"][0])
    assert aliased["rules"][1] == {"name", "kind", "dtype", "versions"}


@pytest.mark.parametrize(
    ("slot", "tag", "error"),
    [
        ("kind", "!!bool", KeyError),
        ("name", "!!timestamp", AttributeError),
        ("excerpt", "!!int", ValueError),
    ],
)
def test_gate_reports_tags_that_make_safe_load_raise(slot, tag, error):
    content = _fill(slot, tag)
    with pytest.raises(error):
        yaml.safe_load(content)

    assert _gate(content).issues == [_issue(slot, tag, tag, SCALAR_REASON)]


def test_null_tag_hides_the_strict_proof_requirement_from_pyyaml():
    content = _fill("required", "!!null")
    assert yaml.safe_load(content)["module"]["proof_validation"]["required"] is None

    assert _gate(content).issues == [
        _issue("required", "!!null", "!!null", SCALAR_REASON)
    ]


def test_interpretation_guard_keeps_the_root_and_module_mapping_tags():
    """Nodes the interpretation scan checks are reported once, by that scan."""

    root = "--- !!set\n" + UNTAGGED
    module = UNTAGGED.replace("module:\n", "module: !!set\n", 1)

    for content, label in ((root, "RuleSpec root"), (module, "module")):
        interpretation_issue = (
            f"{label} has unsupported YAML tag `tag:yaml.org,2002:set`; only an "
            "untagged mapping or `!!map` is supported."
        )
        assert find_interpretation_yaml_issues(content) == [interpretation_issue]
        assert _gate(content).issues == [interpretation_issue]
        # The whole-document scan, run on its own, still covers them.
        assert len(find_unsupported_yaml_tag_issues(content)) == 1


INTERPRETATIONS = """  interpretations:
    - id: weekly-review
      question: !!binary TGFudGVybg==
      texts:
        - corpus_citation_path: us/statute/26/1
          excerpt: The first rate is 0.1.
      reading: Review weekly.
      alternatives:
        - id: daily-review
          reading: Review daily.
      basis: The source supports weekly reviews.
      status: open
      affects:
        - first_rate
"""


def test_interpretation_tags_are_reported_once_by_the_interpretation_scan():
    content = UNTAGGED.replace("module:\n", "module:\n" + INTERPRETATIONS, 1)
    interpretation_issue = (
        "module.interpretations[0].question has unsupported YAML tag "
        "`tag:yaml.org,2002:binary`; only the canonical string tag `!!str` "
        "(`tag:yaml.org,2002:str`) is supported."
    )

    assert find_interpretation_yaml_issues(content) == [interpretation_issue]
    payload, _issues, _paths, tag_issues = _safe_load_with_raw_constraints(content)
    assert tag_issues == []
    assert payload == yaml.safe_load(content)
    assert interpretation_issue in _gate(content).issues


def test_both_scans_report_their_own_nodes_in_one_document():
    content = UNTAGGED.replace("module:\n", "module:\n" + INTERPRETATIONS, 1).replace(
        "  - \n    name: second_rate\n", "  - !!set\n    name: second_rate\n"
    )

    issues = _gate(content).issues

    assert [issue.split(" has ")[0] for issue in issues] == [
        "module.interpretations[0].question",
        "rules[1]",
    ]


def test_a_module_that_is_not_a_mapping_is_checked_by_the_tag_scan():
    content = "format: rulespec/v1\nmodule: !!pairs\n  - summary: x\nrules: []\n"

    assert find_interpretation_yaml_issues(content) == []
    assert _gate(content).issues == [
        f"module has unsupported YAML tag `!!pairs` at line 2, column 9: {PAIRS_REASON}"
    ]


def test_raw_guard_scans_a_recursive_alias_target_once():
    content = "a: &x\n  self: *x\n  tagged: !!set {b: c}\nb: *x\nc: [*x, *x]\n"

    assert find_unsupported_yaml_tag_issues(content) == [
        f"a.tagged has unsupported YAML tag `!!set` at line 3, column 11: {SET_REASON}"
    ]


def test_raw_guard_bounds_its_report():
    content = "m:\n" + "".join(
        f"  k{'x' * 200}{i}: !!set {{a: b}}\n" for i in range(14)
    )

    issues = find_unsupported_yaml_tag_issues(content)

    assert len(issues) == 11
    assert issues[-1] == "4 more unsupported YAML tag(s) not shown."
    assert all(len(issue) < 450 for issue in issues)
    assert issues[0].startswith("m.k" + "x" * 79 + "... has unsupported YAML tag")


def test_raw_guard_keeps_parser_errors():
    with pytest.raises(yaml.YAMLError):
        find_unsupported_yaml_tag_issues("rules: [\n")
    assert find_unsupported_yaml_tag_issues("") == []
    assert find_unsupported_yaml_tag_issues("# only a comment\n") == []


_MAPPING_TAGS = (None, "!", "!!map", "!!set", "!<tag:yaml.org,2002:set>", "!!omap")
_SEQUENCE_TAGS = (None, "!", "!!seq", "!!omap", "!!pairs", "!!set")
_SCALAR_TAGS = (
    None,
    "!",
    "!!str",
    "!<tag:yaml.org,2002:str>",
    "!!binary",
    "!!null",
    "!!bool",
    "!!int",
    "!!float",
    "!!timestamp",
    "!local",
)
_ALLOWED = {
    "mapping": {"!", "tag:yaml.org,2002:map"},
    "sequence": {"!", "tag:yaml.org,2002:seq"},
    "scalar": {"tag:yaml.org,2002:str"},
}
_SCALAR_TEXTS = (
    "first_rate",
    "'0.1'",
    "0.1",
    "2024-01-01",
    "true",
    "~",
    "TGFudGVybg==",
    '"quoted text"',
    "1e5",
)
_KEYS = ("a", "b", "c", "d", "e")
_SCALAR = st.tuples(
    st.just("scalar"), st.sampled_from(_SCALAR_TAGS), st.sampled_from(_SCALAR_TEXTS)
)
_NODES = st.recursive(
    _SCALAR,
    lambda children: (
        st.tuples(
            st.just("sequence"), st.sampled_from(_SEQUENCE_TAGS), st.lists(children)
        )
        | st.tuples(
            st.just("mapping"),
            st.sampled_from(_MAPPING_TAGS),
            st.lists(
                st.tuples(st.sampled_from(_SCALAR_TAGS), children),
                max_size=len(_KEYS),
            ),
        )
    ),
    max_leaves=12,
)


def _tagged(tag, text):
    return f"{tag} {text}" if tag else text


def _render_flow(node):
    kind, tag, value = node
    if kind == "scalar":
        return _tagged(tag, value)
    if kind == "sequence":
        items = ", ".join(_render_flow(item) for item in value)
        return _tagged(tag, f"[{items}]")
    entries = ", ".join(
        f"{_tagged(key_tag, key)}: {_render_flow(child)}"
        for key, (key_tag, child) in zip(_KEYS, value, strict=False)
    )
    return _tagged(tag, "{" + entries + "}")


def _render_block(node, indent):
    """The text after `key:` or `-`, with a block collection's tag on that line."""

    kind, tag, value = node
    if kind == "scalar" or not value:
        return " " + _render_flow(node)
    pad = " " * indent
    text = f" {tag}" if tag else ""
    if kind == "sequence":
        for item in value:
            text += f"\n{pad}-" + _render_block(item, indent + 2)
        return text
    for key, (key_tag, child) in zip(_KEYS, value, strict=False):
        text += f"\n{pad}{_tagged(key_tag, key)}:" + _render_block(child, indent + 2)
    return text


def _rejected_tag_positions(content):
    """Independent oracle: the parser's events, not the composed nodes."""

    kinds = {
        yaml.events.MappingStartEvent: "mapping",
        yaml.events.SequenceStartEvent: "sequence",
        yaml.events.ScalarEvent: "scalar",
    }
    return sorted(
        (event.start_mark.line + 1, event.start_mark.column + 1)
        for event in yaml.parse(content, Loader=yaml.SafeLoader)
        if type(event) in kinds
        and event.tag is not None
        and event.tag not in _ALLOWED[kinds[type(event)]]
    )


def _reported_position(issue):
    line, column = issue.split(" at line ", 1)[1].split(":", 1)[0].split(", column ")
    return int(line), int(column)


def _loaded_types(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return set()
    seen.add(id(value))
    types = {type(value)}
    if isinstance(value, dict):
        for key, item in value.items():
            types |= _loaded_types(key, seen) | _loaded_types(item, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            types |= _loaded_types(item, seen)
    return types


@settings(max_examples=400, deadline=None)
@given(root=_NODES, block=st.booleans())
def test_raw_guard_matches_the_parser_events_and_safe_load(root, block):
    content = (
        "x:" + _render_block(root, 2) + "\n"
        if block
        else "{x: " + _render_flow(root) + "}\n"
    )
    issues = find_unsupported_yaml_tag_issues(content)
    positions = _rejected_tag_positions(content)

    if len(positions) > 10:
        assert (
            issues[-1]
            == f"{len(positions) - 10} more unsupported YAML tag(s) not shown."
        )
        issues = issues[:-1]
        positions = positions[:10]
    assert [_reported_position(issue) for issue in issues] == positions
    payload, _interpretation, _paths, tag_issues = _safe_load_with_raw_constraints(
        content
    )
    assert tag_issues == find_unsupported_yaml_tag_issues(content)
    if not tag_issues:
        loaded = yaml.safe_load(content)
        assert payload == loaded
        # Nothing PyYAML returns can differ in kind from what the engine reads.
        assert not _loaded_types(loaded) & {set, tuple, bytes}


def _engine_compile(binary, root: Path, content: str):
    root = root.resolve() / "rulespec-us"
    module = root / "us/statutes/26/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(content)
    compiled = root.parent / "compiled.json"
    run = subprocess.run(
        [
            str(binary),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(compiled),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    program = (
        json.loads(compiled.read_text())["program"] if run.returncode == 0 else None
    )
    return run, program


def test_engine_compiles_the_untagged_module(real_engine, tmp_path):
    run, _program = _engine_compile(real_engine.binary, tmp_path, UNTAGGED)

    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("slot", MAPPING_SLOTS)
@pytest.mark.parametrize("tag", ["!!set", "!<tag:yaml.org,2002:set>"])
def test_engine_ignores_set_tags_that_pyyaml_reads_as_sets(
    real_engine, tmp_path, slot, tag
):
    """The defect's premise: the engine compiles the mapping the gate skipped."""

    tagged, tagged_program = _engine_compile(
        real_engine.binary, tmp_path / "tagged", _fill(slot, tag)
    )
    untagged, untagged_program = _engine_compile(
        real_engine.binary, tmp_path / "untagged", UNTAGGED
    )

    assert tagged.returncode == 0, tagged.stderr
    assert tagged_program == untagged_program
    assert not _gate(_fill(slot, tag)).passed


ENGINE_COMPILES = (
    "double-quoted-key-omap-value",
    "flow-set-rule",
    "explicit-key-set-rule",
    "aliased-set-rule",
    "single-quoted-key-set-value",
    "loadable-pairs-sources",
    "empty-pairs",
    "decodable-binary",
    "nested-flow-set",
)


@pytest.mark.parametrize("case", ENGINE_COMPILES)
def test_engine_compiles_explicit_spellings_the_encoder_now_rejects(
    real_engine, tmp_path, case
):
    content, _diagnostic = EXPLICIT[case]

    run, _program = _engine_compile(real_engine.binary, tmp_path, content)

    assert run.returncode == 0, run.stderr
    assert not _gate(content).passed


@pytest.mark.parametrize("case", sorted(set(EXPLICIT) - set(ENGINE_COMPILES)))
def test_engine_rejects_the_other_explicit_spellings(real_engine, tmp_path, case):
    content, _diagnostic = EXPLICIT[case]

    run, _program = _engine_compile(real_engine.binary, tmp_path, content)

    assert run.returncode != 0
    assert not _gate(content).passed


def test_engine_reads_binary_tagged_text_that_pyyaml_decodes(real_engine, tmp_path):
    content = EXPLICIT["decodable-binary"][0]

    run, program = _engine_compile(real_engine.binary, tmp_path, content)

    assert run.returncode == 0, run.stderr
    assert program["module"]["summary"] == "TGFudGVybg=="
    assert yaml.safe_load(content)["module"]["summary"] == b"Lantern"


ACCEPTED_SHAPES = [
    *((slot, tag) for slot in MAPPING_SLOTS for tag in ("!!map", "!")),
    *((slot, tag) for slot in SEQUENCE_SLOTS for tag in ("!!seq", "!")),
    *((slot, "!!str") for slot in SCALAR_SLOTS),
]


@pytest.mark.parametrize(("slot", "tag"), ACCEPTED_SHAPES)
def test_engine_reads_every_accepted_tag_as_the_untagged_module(
    real_engine, tmp_path, slot, tag
):
    """No accepted tag changes what the engine compiles."""

    tagged, tagged_program = _engine_compile(
        real_engine.binary, tmp_path / "tagged", _fill(slot, tag)
    )
    _untagged, untagged_program = _engine_compile(
        real_engine.binary, tmp_path / "untagged", UNTAGGED
    )

    assert tagged.returncode == 0, tagged.stderr
    assert tagged_program == untagged_program
    assert _gate(_fill(slot, tag)).issues == _expected_accepted_issues(slot, tag)


def test_engine_reads_non_specific_tagged_text_that_pyyaml_resolves(
    real_engine, tmp_path
):
    """PyYAML resolves `! '0.1'` as a plain scalar; the engine keeps the text."""

    content = _swap(
        (
            "  summary: Fictional lantern review rates.\n",
            "  summary: ! '0.1'\n",
        )
    )

    run, program = _engine_compile(real_engine.binary, tmp_path, content)

    assert run.returncode == 0, run.stderr
    assert program["module"]["summary"] == "0.1"
    assert yaml.safe_load(content)["module"]["summary"] == 0.1
    assert not _gate(content).passed


def test_engine_rejects_non_specific_tags_on_string_like_rule_fields(
    real_engine, tmp_path
):
    content = _fill("formula", "!")

    run, _program = _engine_compile(real_engine.binary, tmp_path, content)

    assert run.returncode != 0
    assert "expected scalar string-like value" in run.stderr
    assert yaml.safe_load(content)["rules"][0]["versions"][0]["formula"] == 0.1
    assert not _gate(content).passed


# A module the apply and eval paths accept: no strict proofs, so the repairs and
# stamping run on it, with one rule `slot` can tag.
APPLY_BASE = """format: rulespec/v1
module:
  summary: Fictional lantern review rates.
rules:
  - {rule}
    name: first_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.1'
"""
APPLY_CLEAN = APPLY_BASE.format(rule="")
APPLY_CASES = [
    pytest.param(
        APPLY_BASE.format(rule="!!set "),
        "rules[0] has unsupported YAML tag `!!set` at line 5, column 5",
        id="set-rule",
    ),
    pytest.param(
        APPLY_CLEAN.replace("    versions:\n", "    versions: !!pairs\n").replace(
            "      - effective_from: '2024-01-01'\n        formula: '0.1'\n",
            "      - formula: '0.1'\n",
        ),
        "rules[0].versions has unsupported YAML tag `!!pairs` at line 9, column 15",
        id="pairs-versions",
    ),
    pytest.param(
        APPLY_CLEAN.replace("    kind: parameter\n", "    kind: !!bool parameter\n"),
        "rules[0].kind has unsupported YAML tag `!!bool` at line 7, column 11",
        id="raising-bool",
    ),
    pytest.param(
        APPLY_CLEAN.replace(
            "  summary: Fictional lantern review rates.\n",
            "  summary: !!binary TGFudGVybg==\n",
        ),
        "module.summary has unsupported YAML tag `!!binary` at line 3, column 12",
        id="binary-summary",
    ),
]


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
        source_attestation=_complete_source_attestation(CITATION),
        context_manifest_file=str(
            _write_test_context_manifest(tmp_path / "context", citation_path=CITATION)
        ),
    )
    return result, generated, output_root


def test_a_load_and_dump_rewrite_drops_every_value_of_a_set_rule():
    """Why stamping must refuse: the dump keeps the keys and drops the values."""

    loaded = yaml.safe_load(APPLY_CASES[0].values[0])
    assert loaded["rules"][0] == {"name", "kind", "dtype", "versions"}

    dumped = safe_dump_yaml(loaded, sort_keys=False)

    assert "!!set" in dumped
    assert "first_rate" not in dumped
    assert "'0.1'" not in dumped


def test_real_stamping_still_rewrites_modules_with_accepted_tags(tmp_path):
    content = APPLY_BASE.format(rule="!!map ").replace(
        "    name: first_rate\n", "    name: !!str first_rate\n"
    )
    result, generated, _ = _candidate(tmp_path, content)

    _stamp_generated_source_attestation_for_apply(result, generated)

    stamped = yaml.safe_load(generated.read_text())
    assert stamped["rules"] == yaml.safe_load(APPLY_CLEAN)["rules"]
    assert stamped["module"]["source_verification"]["corpus_citation_path"] == (
        CITATION
    )


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
def test_real_stamping_rejects_tags_before_rewrite(tmp_path, content, diagnostic):
    result, generated, _ = _candidate(tmp_path, content)
    original = generated.read_bytes()

    with pytest.raises(RuntimeError) as exc_info:
        _stamp_generated_source_attestation_for_apply(result, generated)

    assert str(exc_info.value).startswith(
        "Cannot apply RuleSpec YAML with unsupported tags: " + diagnostic
    )
    assert generated.read_bytes() == original


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
@pytest.mark.parametrize(
    "error",
    [
        "Generated RuleSpec failed compile validation",
        "Generated RuleSpec failed CI validation",
    ],
)
def test_apply_preflight_blocks_tags_before_repairs(
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

    assert issue is not None
    assert issue.startswith(
        "Cannot apply RuleSpec YAML with unsupported tags: " + diagnostic
    )
    assert generated.read_bytes() == original


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
def test_apply_overlay_blocks_tags_with_real_stamping(tmp_path, content, diagnostic):
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


TESTS = """- name: first_rate_applies
  period: 2024-01-01
  input: {}
  output:
    first_rate: 0.1
"""
TAGGED_TESTS = [
    pytest.param(
        TESTS.replace("- name:", "- !!set\n  name:"),
        "document root[0] has unsupported YAML tag `!!set` at line 1, column 3",
        id="set-case",
    ),
    pytest.param(
        TESTS.replace("  output:\n", "  output: !!set\n"),
        "document root[0].output has unsupported YAML tag `!!set` at line 4, column 11",
        id="set-output",
    ),
    pytest.param(
        TESTS.replace("  input: {}\n", "  input: !!pairs []\n"),
        "document root[0].input has unsupported YAML tag `!!pairs` at line 3, "
        "column 10",
        id="pairs-input",
    ),
    pytest.param(
        TESTS.replace("first_rate: 0.1", "first_rate: !!timestamp 0.1"),
        "document root[0].output.first_rate has unsupported YAML tag `!!timestamp` "
        "at line 5, column 17",
        id="raising-timestamp",
    ),
]


@pytest.mark.parametrize(("tests", "diagnostic"), TAGGED_TESTS)
def test_apply_preflight_blocks_companion_test_tags(tmp_path, tests, diagnostic):
    result, generated, output_root = _candidate(tmp_path, APPLY_CLEAN, tests)
    result.success = False
    result.error = "Generated RuleSpec failed CI validation"
    test_file = generated.with_name("1.test.yaml")

    issue = _generated_rulespec_yaml_nonparser_issue(
        result, generated, output_root=output_root
    )

    assert issue is not None
    assert issue.startswith(
        "Cannot apply companion test YAML with unsupported tags: " + diagnostic
    )
    assert generated.read_text() == APPLY_CLEAN
    assert test_file.read_text() == tests


def test_apply_preflight_passes_accepted_tags_in_module_and_tests(tmp_path):
    content = APPLY_BASE.format(rule="! ")
    tests = TESTS.replace("  input: {}\n", "  input: !!map {}\n")
    result, generated, output_root = _candidate(tmp_path, content, tests)

    assert (
        _generated_rulespec_yaml_nonparser_issue(
            result, generated, output_root=output_root
        )
        is None
    )


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
def test_materialization_keeps_raw_tags(tmp_path, content, diagnostic, delivery):
    content = content.replace("module:\n", "module:\n" + MISPLACED_INPUTS, 1)

    materialized, _tests = _materialize(tmp_path, content, delivery)

    assert materialized == content
    assert find_unsupported_yaml_tag_issues(materialized)


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
def test_normalization_keeps_raw_tags_and_still_refuses_non_rulespec_text(
    content, diagnostic
):
    with pytest.raises(ValueError, match="must start with `format: rulespec/v1`"):
        _normalize_main_eval_content(
            "entries: !!set {first: kept}\n",
            target_path=Path("section.yaml"),
            single_amount_table_slice=False,
        )

    normalized = _normalize_main_eval_content(
        "\n" + content + "\n\n",
        target_path=Path("section.yaml"),
        single_amount_table_slice=False,
    )

    assert normalized == content
    assert find_unsupported_yaml_tag_issues(normalized)[0].startswith(diagnostic)


@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_still_repairs_modules_with_accepted_tags(tmp_path, delivery):
    content = APPLY_BASE.format(rule="!!map ").replace(
        "module:\n", "module:\n" + MISPLACED_INPUTS, 1
    )

    materialized, _tests = _materialize(tmp_path, content, delivery)

    payload = yaml.safe_load(materialized)
    assert "inputs" not in payload["module"]
    assert payload["inputs"][0]["name"] == "supplied_review_count"


@pytest.mark.parametrize(("tests", "diagnostic"), TAGGED_TESTS)
@pytest.mark.parametrize("delivery", ["bundle", "workspace"])
def test_materialization_keeps_raw_test_tags(tmp_path, tests, diagnostic, delivery):
    _main, materialized_tests = _materialize(tmp_path, APPLY_CLEAN, delivery, tests)

    assert materialized_tests == tests


@pytest.mark.parametrize("delivery", ["bundle", "workspace"])
def test_materialization_still_normalizes_tests_with_accepted_tags(tmp_path, delivery):
    tests = TESTS.replace("  output:\n", "  output: !!map\n")

    _main, materialized_tests = _materialize(tmp_path, APPLY_CLEAN, delivery, tests)

    assert materialized_tests != tests
    assert yaml.safe_load(materialized_tests)[0]["output"] == {"first_rate": 0.1}


@pytest.mark.parametrize(("content", "diagnostic"), APPLY_CASES)
@pytest.mark.parametrize("tagged_side", ["generated", "preserved"])
def test_retry_overlay_rejects_tags_before_rewriting(
    tmp_path, content, diagnostic, tagged_side
):
    generated = content if tagged_side == "generated" else APPLY_CLEAN
    preserved = content if tagged_side == "preserved" else APPLY_CLEAN
    target = tmp_path / "section.yaml"
    target.write_text(generated, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target,
            artifact_root=tmp_path,
            candidate=ValidationRetryCandidate(rulespec=preserved),
        )

    assert str(exc_info.value).startswith(
        f"{tagged_side} repair overlay has unsupported YAML tags: {diagnostic}"
    )
    assert target.read_text(encoding="utf-8") == generated


@pytest.mark.parametrize(("tests", "diagnostic"), TAGGED_TESTS)
@pytest.mark.parametrize("tagged_side", ["generated", "preserved"])
def test_retry_overlay_rejects_test_tags_before_rewriting(
    tmp_path, tests, diagnostic, tagged_side
):
    target = tmp_path / "section.yaml"
    target.write_text(APPLY_CLEAN, encoding="utf-8")
    test_file = tmp_path / "section.test.yaml"
    generated_tests = tests if tagged_side == "generated" else TESTS
    test_file.write_text(generated_tests, encoding="utf-8")
    candidate = ValidationRetryCandidate(
        rulespec=APPLY_CLEAN,
        tests=tests if tagged_side == "preserved" else TESTS,
    )

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target, artifact_root=tmp_path, candidate=candidate
        )

    assert str(exc_info.value).startswith(
        f"{tagged_side} repair overlay tests have unsupported YAML tags: {diagnostic}"
    )
    assert target.read_text(encoding="utf-8") == APPLY_CLEAN
    assert test_file.read_text(encoding="utf-8") == generated_tests


def test_retry_overlay_still_merges_modules_with_accepted_tags(tmp_path):
    target = tmp_path / "section.yaml"
    target.write_text(APPLY_BASE.format(rule="!!map "), encoding="utf-8")
    preserved = APPLY_CLEAN.replace(
        "module:\n", "imports:\n  - us:statutes/26/2\nmodule:\n"
    )

    repairs = _overlay_validation_retry_candidate(
        target,
        artifact_root=tmp_path,
        candidate=ValidationRetryCandidate(rulespec=preserved),
    )

    assert repairs == ("import:us:statutes/26/2",)
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


def test_eval_repairs_still_run_with_accepted_tags(tmp_path):
    content = _with_unused_import(APPLY_BASE.format(rule="!!map "))

    repairs, repaired = _eval_repairs(tmp_path, content, TESTS)

    assert repairs == [f"unused_import:{UNUSED_IMPORT}"]
    assert UNUSED_IMPORT not in repaired


@pytest.mark.parametrize("tagged_file", ["module", "tests"])
def test_eval_repairs_skip_files_with_unsupported_tags(tmp_path, tagged_file):
    content = _with_unused_import(
        APPLY_BASE.format(rule="!!set " if tagged_file == "module" else "")
    )
    tests = TAGGED_TESTS[0].values[0] if tagged_file == "tests" else TESTS

    repairs, repaired = _eval_repairs(tmp_path, content, tests)

    assert repairs == []
    assert repaired == content


def test_timestamp_tag_on_a_date_loads_like_the_plain_date():
    """`!!timestamp` is rejected even where PyYAML agrees, keeping one rule."""

    content = _fill("effective_from", "!!timestamp")
    loaded = yaml.safe_load(content)["rules"][0]["versions"][0]["effective_from"]

    assert loaded == date(2024, 1, 1)
    assert find_unsupported_yaml_tag_issues(content) == [
        _issue("effective_from", "!!timestamp", "!!timestamp", SCALAR_REASON)
    ]


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
def test_pipeline_compile_check_reports_tags_before_loading(
    tmp_path, content, diagnostic
):
    """Validators read the loaded payload, so the preflight stops first."""

    rules_file = _pipeline_module(tmp_path, content)

    result = _pipeline(tmp_path)._run_compile_check(rules_file)

    assert not result.passed
    assert len(result.issues) == 1
    assert result.issues[0].startswith(
        "1.yaml has unsupported YAML tags: " + diagnostic
    )


@pytest.mark.parametrize(("tests", "diagnostic"), TAGGED_TESTS)
def test_pipeline_ci_reports_companion_test_tags(
    tmp_path, monkeypatch, tests, diagnostic
):
    rules_file = _pipeline_module(tmp_path, APPLY_CLEAN, tests)
    pipeline = _pipeline(tmp_path)

    def fake_compile(rules_file, output_path):
        return (
            subprocess.CompletedProcess(["axiom-rules-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        )

    monkeypatch.setattr(pipeline, "_compile_rulespec_to_artifact", fake_compile)

    result = pipeline._run_rulespec_ci(rules_file)

    assert not result.passed
    assert any(
        issue.startswith("1.test.yaml has unsupported YAML tags: " + diagnostic)
        for issue in result.issues
    ), result.issues


@pytest.mark.parametrize("tag", ["!!bool", "!!timestamp"])
def test_amendment_timing_leaves_raising_tags_to_the_yaml_preflight(tag):
    """The pipeline runs this before its preflight when amendments are bound."""

    content = APPLY_CLEAN.replace(
        "    kind: parameter\n", f"    kind: {tag} parameter\n"
    )
    document = SimpleNamespace(
        citation_path=CITATION, metadata={"commencement_date": "2027-04-01"}
    )
    with pytest.raises((KeyError, AttributeError)):
        yaml.safe_load(content)

    assert amendment_timing_issues(content, [document]) == []


# PyYAML raises a plain ValueError, not a yaml.YAMLError, while composing a
# double-quoted escape beyond U+10FFFF and while constructing an impossible
# implicit date. Callers that caught only YAML errors crashed on such text.
VALUE_ERROR_MODULES = [
    pytest.param(
        APPLY_CLEAN.replace(
            "  summary: Fictional lantern review rates.\n",
            '  summary: "\\U00110000"\n',
        ),
        id="escape-beyond-unicode",
    ),
    pytest.param(
        APPLY_CLEAN.replace(
            "    name: first_rate\n", '    "\\U00110000": first_rate\n'
        ),
        id="escape-in-key",
    ),
    pytest.param(
        APPLY_CLEAN.replace(
            "      - effective_from: '2024-01-01'\n",
            "      - effective_from: 2024-13-01\n",
        ),
        id="impossible-date",
    ),
]
VALUE_ERROR_TESTS = [
    pytest.param(
        TESTS.replace("- name: first_rate_applies\n", '- name: "\\U00110000"\n'),
        id="escape-beyond-unicode",
    ),
    pytest.param(
        TESTS.replace("  period: 2024-01-01\n", "  period: 2024-13-01\n"),
        id="impossible-date",
    ),
]
VALUE_ISSUE = (
    "[generated-yaml:value] invalid YAML: a scalar cannot be loaded, such as a "
    "double-quoted escape beyond U+10FFFF or an impossible date"
)
TEST_VALUE_ISSUE = (
    "[generated-yaml:value] invalid companion test YAML: a scalar cannot be "
    "loaded, such as a double-quoted escape beyond U+10FFFF or an impossible date"
)


@pytest.mark.parametrize(
    "content",
    [
        *(case.values[0] for case in VALUE_ERROR_MODULES),
        *(case.values[0] for case in VALUE_ERROR_TESTS),
    ],
)
def test_safe_load_raises_a_plain_value_error_on_these_texts(content):
    """The premise: these are not yaml.YAMLError, so YAML-only handlers miss them."""

    with pytest.raises(ValueError) as exc_info:
        yaml.safe_load(content)

    assert not isinstance(exc_info.value, yaml.YAMLError)


def test_raw_scans_let_the_compose_value_error_propagate():
    content = VALUE_ERROR_MODULES[0].values[0]

    with pytest.raises(ValueError, match="chr"):
        find_unsupported_yaml_tag_issues(content)
    with pytest.raises(ValueError, match="chr"):
        find_interpretation_yaml_issues(content)


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
def test_proof_gate_and_pipeline_report_value_errors_as_parse_failures(
    tmp_path, content
):
    result = validate_rulespec_proofs(content)
    assert not result.passed
    assert result.issues[0].startswith("RuleSpec proof YAML parse failed: ")

    rules_file = _pipeline_module(tmp_path, content)
    compile_result = _pipeline(tmp_path)._run_compile_check(rules_file)
    assert not compile_result.passed
    assert compile_result.issues[0].startswith("1.yaml YAML parse failed: ")


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
@pytest.mark.parametrize(
    "error",
    [
        "Generated RuleSpec failed compile validation",
        "Generated RuleSpec failed CI validation",
    ],
)
def test_apply_preflight_blocks_value_errors_before_repairs(tmp_path, content, error):
    result, generated, output_root = _candidate(tmp_path, content)
    result.success = False
    result.error = error
    assert _can_attempt_apply(result)
    original = generated.read_bytes()

    issue = _generated_rulespec_yaml_nonparser_issue(
        result, generated, output_root=output_root
    )

    assert issue == VALUE_ISSUE
    assert generated.read_bytes() == original


@pytest.mark.parametrize("tests", VALUE_ERROR_TESTS)
def test_apply_preflight_blocks_companion_test_value_errors(tmp_path, tests):
    result, generated, output_root = _candidate(tmp_path, APPLY_CLEAN, tests)
    result.success = False
    result.error = "Generated RuleSpec failed CI validation"
    test_file = generated.with_name("1.test.yaml")

    issue = _generated_rulespec_yaml_nonparser_issue(
        result, generated, output_root=output_root
    )

    assert issue == TEST_VALUE_ISSUE
    assert test_file.read_text() == tests


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
def test_real_stamping_reports_value_errors_before_rewrite(tmp_path, content):
    result, generated, _ = _candidate(tmp_path, content)
    original = generated.read_bytes()

    with pytest.raises(RuntimeError) as exc_info:
        _stamp_generated_source_attestation_for_apply(result, generated)

    assert str(exc_info.value) == VALUE_ISSUE
    assert generated.read_bytes() == original


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
def test_apply_overlay_reports_value_errors_with_real_stamping(tmp_path, content):
    result, generated, output_root = _candidate(tmp_path, content)
    original = generated.read_bytes()

    ok, issues, _supplemental = _validate_generated_encoding_in_policy_overlay(
        result,
        output_root=output_root,
        policy_repo_path=tmp_path / "rulespec-us/us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=MagicMock(),
        validate_dependents=False,
    )

    assert not ok
    assert any(VALUE_ISSUE in issue for issue in issues), issues
    assert generated.read_bytes() == original


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
def test_retry_overlay_reports_generated_value_errors(tmp_path, content):
    target = tmp_path / "section.yaml"
    target.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target,
            artifact_root=tmp_path,
            candidate=ValidationRetryCandidate(rulespec=APPLY_CLEAN),
        )

    assert str(exc_info.value) == (
        "generated repair overlay RuleSpec must be valid UTF-8 YAML"
    )
    assert target.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("tests", VALUE_ERROR_TESTS)
def test_retry_overlay_reports_generated_test_value_errors(tmp_path, tests):
    target = tmp_path / "section.yaml"
    target.write_text(APPLY_CLEAN, encoding="utf-8")
    test_file = tmp_path / "section.test.yaml"
    test_file.write_text(tests, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target,
            artifact_root=tmp_path,
            candidate=ValidationRetryCandidate(rulespec=APPLY_CLEAN, tests=TESTS),
        )

    assert str(exc_info.value) == "repair overlay tests must be valid UTF-8 YAML"
    assert test_file.read_text(encoding="utf-8") == tests


@pytest.mark.parametrize(
    ("label", "rulespec", "tests"),
    [
        *(
            pytest.param("RuleSpec", case.values[0], None, id=f"module-{case.id}")
            for case in VALUE_ERROR_MODULES
        ),
        *(
            pytest.param("tests", APPLY_CLEAN, case.values[0], id=f"tests-{case.id}")
            for case in VALUE_ERROR_TESTS
        ),
    ],
)
def test_retry_overlay_reports_preserved_value_errors(tmp_path, label, rulespec, tests):
    target = tmp_path / "section.yaml"
    target.write_text(APPLY_CLEAN, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        _overlay_validation_retry_candidate(
            target,
            artifact_root=tmp_path,
            candidate=ValidationRetryCandidate(rulespec=rulespec, tests=tests),
        )

    assert str(exc_info.value).startswith(
        f"preserved repair overlay {label} cannot be loaded: YAML text cannot be loaded"
    )
    assert target.read_text(encoding="utf-8") == APPLY_CLEAN


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
def test_normalization_keeps_text_that_cannot_be_loaded(content):
    with pytest.raises(ValueError, match="must start with `format: rulespec/v1`"):
        _normalize_main_eval_content(
            'entries: "\\U00110000"\n',
            target_path=Path("section.yaml"),
            single_amount_table_slice=False,
        )

    normalized = _normalize_main_eval_content(
        "\n" + content + "\n\n",
        target_path=Path("section.yaml"),
        single_amount_table_slice=False,
    )

    assert normalized == content


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_keeps_modules_that_cannot_be_loaded(
    tmp_path, content, delivery
):
    content = content.replace("module:\n", "module:\n" + MISPLACED_INPUTS, 1)

    materialized, _tests = _materialize(tmp_path, content, delivery)

    assert materialized == content


@pytest.mark.parametrize("tests", VALUE_ERROR_TESTS)
@pytest.mark.parametrize("delivery", ["bundle", "workspace"])
def test_materialization_keeps_tests_that_cannot_be_loaded(tmp_path, tests, delivery):
    _main, materialized_tests = _materialize(tmp_path, APPLY_CLEAN, delivery, tests)

    assert materialized_tests == tests


@pytest.mark.parametrize("tagged_file", ["module", "tests"])
def test_eval_repairs_skip_files_that_cannot_be_loaded(tmp_path, tagged_file):
    content = _with_unused_import(
        VALUE_ERROR_MODULES[0].values[0] if tagged_file == "module" else APPLY_CLEAN
    )
    tests = VALUE_ERROR_TESTS[0].values[0] if tagged_file == "tests" else TESTS

    repairs, repaired = _eval_repairs(tmp_path, content, tests)

    assert repairs == []
    assert repaired == content


@pytest.mark.parametrize("content", VALUE_ERROR_MODULES)
def test_engine_rejects_the_texts_pyyaml_cannot_load(real_engine, tmp_path, content):
    """The escape fails YAML parsing; the date fails `effective_from`."""

    run, _program = _engine_compile(real_engine.binary, tmp_path, content)

    assert run.returncode != 0
    assert "yaml parse error" in run.stderr


# The raw scans compose with one more frame per level than yaml.safe_load, so
# nesting this deep exhausts them while plain loading still succeeds. The engine
# rejects anything past 127 levels.
DEEP = 400
DEEP_MODULE = APPLY_CLEAN.replace(
    "rules:\n", "x-deep: " + "[" * DEEP + "]" * DEEP + "\nrules:\n", 1
)
DEEP_TESTS = TESTS.replace(
    "  input: {}\n", "  input: " + "[" * DEEP + "]" * DEEP + "\n"
)


def test_deep_nesting_exhausts_the_scan_but_not_loading():
    assert yaml.safe_load(DEEP_MODULE)["x-deep"]
    with pytest.raises(RecursionError):
        find_unsupported_yaml_tag_issues(DEEP_MODULE)


def test_pipeline_reports_nesting_too_deep_to_scan(tmp_path, monkeypatch):
    rules_file = _pipeline_module(tmp_path, DEEP_MODULE)
    pipeline = _pipeline(tmp_path)

    compile_result = pipeline._run_compile_check(rules_file)

    assert not compile_result.passed
    assert compile_result.issues == ["1.yaml YAML nesting is too deep to check."]

    rules_file.write_text(APPLY_CLEAN)
    rules_file.with_name("1.test.yaml").write_text(DEEP_TESTS)

    def fake_compile(rules_file, output_path):
        return (
            subprocess.CompletedProcess(["axiom-rules-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        )

    monkeypatch.setattr(pipeline, "_compile_rulespec_to_artifact", fake_compile)
    ci_result = pipeline._run_rulespec_ci(rules_file)

    assert "1.test.yaml YAML nesting is too deep to check." in ci_result.issues


@pytest.mark.parametrize("delivery", ["bundle", "workspace"])
def test_materialization_keeps_tests_too_deep_to_scan(tmp_path, delivery):
    _main, materialized_tests = _materialize(
        tmp_path, APPLY_CLEAN, delivery, DEEP_TESTS
    )

    assert materialized_tests == DEEP_TESTS


def test_rewrite_tag_check_reports_text_too_deep_to_scan():
    assert _rewrite_unsupported_tag_issues(DEEP_MODULE) == [
        "YAML nesting is too deep to check for unsupported tags, so the file is "
        "not rewritten."
    ]


def test_engine_rejects_nesting_too_deep_to_scan(real_engine, tmp_path):
    run, _program = _engine_compile(real_engine.binary, tmp_path, DEEP_MODULE)

    assert run.returncode != 0
    assert "recursion limit exceeded" in run.stderr
