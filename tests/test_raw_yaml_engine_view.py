"""What the raw YAML guards admit loads the same in PyYAML and in the engine.

The guards are the raw scans that run before validation and before any
load-and-dump rewrite: duplicate mapping keys, unsupported tags, merge keys
outside rule metadata, and the interpretation scan. The property: whenever all
of them report nothing, ``yaml.safe_load`` and a model of what the engine reads
give the same value, once rule ``metadata`` is set aside.

The model encodes only engine behaviour this repository records with tests that
run the real engine:

- E1. Two equal keys in one mapping are an error
  (``test_duplicate_key_apply.py::test_engine_rejects_raw_duplicates_that_a_rewrite_would_hide``).
- E2. Merge keys are not applied; ``<<`` is an ordinary key
  (``test_yaml_merge_keys.py::test_engine_compiles_rejected_merges_apart_from_their_rewrite``,
  ``::test_engine_compiles_a_partial_module_where_pyyaml_sees_the_merged_field``).
- E3. A tag on a mapping or sequence is ignored and the collection is read as
  written; an accepted tag reads as the untagged text
  (``test_unsupported_yaml_tags.py::test_engine_ignores_set_tags_that_pyyaml_reads_as_sets``,
  ``::test_engine_reads_every_accepted_tag_as_the_untagged_module``), and a
  ``!!binary`` scalar reads as its text
  (``::test_engine_reads_binary_tagged_text_that_pyyaml_decodes``).
- E4. A rule's ``metadata`` does not change what the engine compiles
  (``test_yaml_merge_keys.py::test_engine_compiles_accepted_merges_like_their_rewrite``).

Plain scalars resolve the same way in both views here because the generated
documents use only scalars whose YAML 1.1 and YAML 1.2 readings agree. The
guards do not cover plain scalars that the two YAML versions read differently
(``yes``, ``0o17``, ``1_000``); that drift is outside this property.
"""

from __future__ import annotations

import pytest
import yaml
from hypothesis import HealthCheck, event, find, given, settings
from hypothesis import strategies as st

from axiom_encode.harness.proof_validator import (
    find_interpretation_yaml_issues,
    find_unsupported_yaml_tag_issues,
    find_yaml_merge_key_issues,
)
from axiom_encode.yaml_serialization import find_duplicate_mapping_key_issues


class _EngineDuplicateKey(ValueError):
    pass


class _EngineViewLoader(yaml.SafeLoader):
    """A model of the engine's reading, built from E1 to E3 above."""

    def flatten_mapping(self, node):
        # E2: merge keys are not applied.
        return None

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                # E1: two equal keys in one mapping are an error.
                raise _EngineDuplicateKey(key)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


# E2: `<<` is an ordinary key.
_EngineViewLoader.add_constructor(
    "tag:yaml.org,2002:merge", lambda loader, node: loader.construct_scalar(node)
)
# E3: collection tags are ignored, and `!!binary` reads as its text.
for _tag in ("tag:yaml.org,2002:set",):
    _EngineViewLoader.add_constructor(_tag, yaml.SafeLoader.construct_yaml_map)
for _tag in ("tag:yaml.org,2002:omap", "tag:yaml.org,2002:pairs"):
    _EngineViewLoader.add_constructor(_tag, yaml.SafeLoader.construct_yaml_seq)
_EngineViewLoader.add_constructor(
    "tag:yaml.org,2002:binary", lambda loader, node: loader.construct_scalar(node)
)


def _without_rule_metadata(value):
    # E4: a rule's metadata does not change what the engine compiles.
    if not isinstance(value, dict) or not isinstance(value.get("rules"), list):
        return value
    return {
        **value,
        "rules": [
            {key: item for key, item in rule.items() if key != "metadata"}
            if isinstance(rule, dict)
            else rule
            for rule in value["rules"]
        ],
    }


def _engine_view(content):
    return _without_rule_metadata(yaml.load(content, Loader=_EngineViewLoader))


def _pyyaml_view(content):
    return _without_rule_metadata(yaml.safe_load(content))


def _guard_issues(content):
    return [
        *find_duplicate_mapping_key_issues(content),
        *find_unsupported_yaml_tag_issues(content),
        *find_yaml_merge_key_issues(content),
        *find_interpretation_yaml_issues(content),
    ]


def _views_differ(content):
    try:
        return _engine_view(content) != _pyyaml_view(content)
    except _EngineDuplicateKey:
        return True
    except yaml.YAMLError:
        # PyYAML cannot construct some rejected shapes at all (for example
        # `!!omap` over a mapping with two keys).
        return True


_ACCEPTED_MAPPING_TAGS = (None, None, "!!map", "!")
_ACCEPTED_SEQUENCE_TAGS = (None, None, "!!seq", "!")
_VIOLATIONS = (
    None,
    None,
    None,
    "duplicate-key",
    "merge-outside-metadata",
    "mapping-tag",
    "sequence-tag",
    "binary-scalar",
    "set-tagged-anchor",
    "metadata-merge-aliased-out",
)
_MERGES = ("<<: *a0", "<<: [*a0, *a1]", "<<: {k9: merged}")


@st.composite
def _modules(draw):
    """A small RuleSpec module whose keys, tags, merges and aliases vary.

    Spellings both loaders read alike vary freely: accepted collection tags,
    `!!str`, quoted keys, and merges inside rule metadata. Each document then
    draws at most one kind of violation and injects it at a random subset of
    the places it can go, so about half the documents are admitted.
    """

    violation = draw(st.sampled_from(_VIOLATIONS))

    def inject(kind):
        return violation == kind and draw(st.booleans())

    lines = ["format: rulespec/v1", "x-shared:"]
    for index in range(2):
        tag = "!!set" if inject("set-tagged-anchor") else None
        lines.append(f"  - &a{index}" + (f" {tag}" if tag else ""))
        lines.append(f"    k{index}: v{index}")
        lines.append(f"    note: a{index}")

    def mapping(indent, fields, *, in_metadata=False):
        pad = " " * indent
        body = []
        if (in_metadata and draw(st.booleans())) or inject("merge-outside-metadata"):
            body.append(pad + draw(st.sampled_from(_MERGES)))
        for field in fields:
            key, value = field.split(": ", 1)
            spelling = draw(st.sampled_from((key, key, f"'{key}'", f"!!str {key}")))
            if inject("binary-scalar"):
                value = "!!binary c3VtbWFyeQ=="
            elif draw(st.booleans()):
                value = f"!!str {value}"
            body.append(f"{pad}{spelling}: {value}")
        if fields and inject("duplicate-key"):
            key, _value = draw(st.sampled_from(fields)).split(": ", 1)
            body.append(f"{pad}{key}: repeated")
        return body

    def tagged(text, accepted, kind, rejected):
        tag = draw(st.sampled_from(accepted))
        if inject(kind):
            tag = draw(st.sampled_from(rejected))
        return text + (f" {tag}" if tag else "")

    def mapping_tag(text):
        return tagged(text, _ACCEPTED_MAPPING_TAGS, "mapping-tag", ("!!set", "!foo"))

    def sequence_tag(text):
        return tagged(
            text, _ACCEPTED_SEQUENCE_TAGS, "sequence-tag", ("!!omap", "!!pairs")
        )

    lines.append(mapping_tag("module:"))
    lines.extend(mapping(2, ["summary: s"]))
    lines.append(sequence_tag("rules:"))
    aliased_out = False
    for rule_index in range(draw(st.integers(1, 2))):
        lines.append(mapping_tag("  -"))
        lines.extend(mapping(4, [f"name: r{rule_index}", "kind: parameter"]))
        if draw(st.booleans()):
            lines.append("    metadata:")
            lines.extend(mapping(6, ["note: m"], in_metadata=True))
            if not aliased_out and inject("metadata-merge-aliased-out"):
                lines.append(f"      x-version: &mv{rule_index}")
                lines.append("        <<: *a0")
                lines.append("        effective_from: '2024-01-01'")
                aliased_out = f"mv{rule_index}"
        lines.append(sequence_tag("    versions:"))
        for _version in range(draw(st.integers(1, 2))):
            lines.append(mapping_tag("      -"))
            lines.extend(mapping(8, ["effective_from: '2024-01-01'", "formula: '1'"]))
        if aliased_out:
            lines.append(f"      - *{aliased_out}")
        elif draw(st.booleans()):
            lines.append("      - *a0")
    return "\n".join(lines) + "\n"


@settings(
    max_examples=600,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(content=_modules())
def test_what_the_guards_admit_loads_the_same_in_pyyaml_and_the_engine(content):
    issues = _guard_issues(content)
    event("admitted" if not issues else "rejected")
    if issues:
        return

    assert _engine_view(content) == _pyyaml_view(content)


@settings(max_examples=600, deadline=None)
@given(content=_modules())
def test_where_the_two_views_differ_a_guard_reports_it(content):
    """The converse, for the shapes generated here.

    It holds because every generated shape that changes a value (a duplicate
    key, a merge outside metadata, a collection or `!!binary` tag) is one the
    guards refuse.
    """

    if _views_differ(content):
        assert _guard_issues(content)


@pytest.mark.parametrize(
    "condition",
    [
        pytest.param(
            lambda c: find_duplicate_mapping_key_issues(c) and _views_differ(c),
            id="duplicate-keys",
        ),
        pytest.param(
            lambda c: (
                find_yaml_merge_key_issues(c)
                and not find_duplicate_mapping_key_issues(c)
                and not find_unsupported_yaml_tag_issues(c)
                and _views_differ(c)
            ),
            id="merge-keys",
        ),
        pytest.param(
            lambda c: (
                find_unsupported_yaml_tag_issues(c)
                and not find_duplicate_mapping_key_issues(c)
                and not find_yaml_merge_key_issues(c)
                and _views_differ(c)
            ),
            id="tags",
        ),
        pytest.param(
            lambda c: (
                not _guard_issues(c)
                and "<<" in c
                and "metadata:" in c
                and "!!str" in c
                and "!!map" in c
            ),
            id="admitted-with-metadata-merge-and-accepted-tags",
        ),
    ],
)
def test_the_generator_reaches_each_kind_of_document(condition):
    """Each axis produces documents the views read differently, which keeps the
    properties above from passing vacuously."""

    find(
        _modules(),
        condition,
        settings=settings(max_examples=5000, deadline=None, database=None),
    )
