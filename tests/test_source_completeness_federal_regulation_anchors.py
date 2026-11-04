"""Federal regulation deferrals anchored at the `<title>-cfr` module root.

The encoder writes `us/regulation/42/457/800` to
`regulations/42-cfr/457/800.yaml`, so models name deferred outputs under
`us:regulations/42-cfr/457/800/...`. The source sub-paragraph coverage gate
only counts that root, while complete-source-unit coverage only counted the
corpus-shaped `us:regulations/42/457/800/...` root and silently dropped the
rest. Nine targeted signed re-encode runs on 2026-09-29 (42 CFR 457.800,
457.622 and 435.927, three each) ended `apply_blocked_validation` this way:
every final candidate anchored all of its deferrals at the `-cfr` root and
received no deferral-specific feedback, only "neither encoded nor precisely
deferred" for the branches those deferrals named.

The fixtures are the byte-exact `final-rejected-candidate` files from the
`targeted-reencode-failure-<run>-1` artifacts of runs 36635442145 and
36610751620 (sha256 values recorded in each `issues.json`).
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import re
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.evals import _source_identifier_to_relative_rulespec_path
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    _deferred_output_covered_subparagraphs,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_source_subparagraph_coverage_issues,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures/source_completeness"

FEDERAL_REGULATION_CITATIONS = tuple(
    f"us/regulation/{title}/{tail}"
    for title, tail in itertools.product(
        ("7", "20", "26", "42", "45"),
        ("273/4", "457/800", "1.36B-2", "1/36B-2", "435", "416/1101", "273/9/d"),
    )
)
OTHER_CITATIONS = (
    "us/statute/26/36B",
    "us/statute/42/1437c-1",
    "us/statute/7/2017/e",
    "us/guidance/irs/rev-proc-2025-32/page-14",
    "us/form/cms/medicaid-chip-bhp-eligibility-levels/block-1",
    "us/regulation/cfr/457",
    "us-ca/regulation/mpp/63-300.1",
    "us-nj/regulation/njac-10-90/10-90-3.19",
    "us-nc/manual/dhhs/fns/appendix-3100/page-1",
    "us-ms/statute/27-7-5",
    "us-la/statute/47:295",
    "us-co/statute/39/39-22-104",
    "de/statute/estg/32a",
)
REPEAL_SOURCE = (
    "(a) Repealed.\n\n(b) General rule.\n\n(1) Repealed.\n\n(2) Rules.\n\n"
    "(i) Repealed.\n\n(ii) Repealed."
)
REPEAL_BRANCHES = recognize_source_structure(REPEAL_SOURCE)
BRANCH_PATHS = tuple(branch.path for branch in REPEAL_BRANCHES)
REPEALED_LEAF_PATHS = tuple(
    branch.path
    for branch in REPEAL_BRANCHES
    if not any(
        len(other.path) > len(branch.path)
        and other.path[: len(branch.path)] == branch.path
        for other in REPEAL_BRANCHES
    )
)


def _analyze(content: str, source: str, citation: str, test_cases=None):
    return analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=test_cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _load_saved_rejection(run_id: str):
    fixture = FIXTURES / f"signed_reencode_{run_id}"
    recorded = json.loads((fixture / "issues.json").read_text())
    content = (fixture / "candidate.yaml").read_text()
    tests = (fixture / "candidate.test.yaml").read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == recorded["rulespec_sha256"]
    assert hashlib.sha256(tests.encode()).hexdigest() == recorded["tests_sha256"]
    source = (fixture / "source.txt").read_text()
    return recorded, content, yaml.safe_load(tests), source


def _module_root(citation: str) -> str:
    relative = _source_identifier_to_relative_rulespec_path(citation)
    return f"{citation.partition('/')[0]}:{relative.with_suffix('').as_posix()}"


def _deferral_payload(citation: str, outputs_and_reasons) -> dict:
    return {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": citation},
            "deferred_outputs": [
                {"output": output, "reason": reason}
                for output, reason in outputs_and_reasons
            ],
        },
        "rules": [],
    }


def test_saved_457_800_rejection_was_only_bare_structure_issues():
    recorded, _content, _tests, _source = _load_saved_rejection("36635442145")

    assert recorded["citation"] == "us/regulation/42/457/800"
    assert [issue.split(" ci: ", 1)[1] for issue in recorded["issues"]] == [
        "[complete-source-unit:structure] Source branch (a) at "
        "us/regulation/42/457/800(a) [Absatz a] is neither encoded nor "
        "precisely deferred.",
        "[complete-source-unit:structure] Source branch (b) at "
        "us/regulation/42/457/800(b) [Absatz b] is neither encoded nor "
        "precisely deferred.",
    ]


def test_saved_457_800_rejection_now_evaluates_generated_cfr_deferrals():
    recorded, content, tests, source = _load_saved_rejection("36635442145")
    payload = yaml.safe_load(content)
    outputs = [record["output"] for record in payload["module"]["deferred_outputs"]]
    assert outputs and all(
        output.startswith("us:regulations/42-cfr/457/800/") for output in outputs
    )

    issues = _analyze(content, source, recorded["citation"], tests).issues

    for index, branch in enumerate(("a", "b")):
        assert any(
            issue.startswith("[complete-source-unit:deferral] ")
            and f"`module.deferred_outputs[{index}]` identifies source branch "
            f"({branch})"
            in issue
            and "literal canonical citation required in `reason` is "
            f"`us/regulation/42/457/800({branch})`"
            in issue
            for issue in issues
        ), issues


def test_saved_457_622_rejection_coverage_is_invariant_to_anchor_root():
    recorded, content, tests, source = _load_saved_rejection("36610751620")
    citation = recorded["citation"]
    source_root_content = re.sub(
        r"us:regulations/42-cfr/457/622(?=[/#])",
        "us:regulations/42/457/622",
        content,
    )
    assert source_root_content != content

    def outcome(text: str) -> list[str]:
        return [
            issue.replace("us:regulations/42-cfr/457/622", "<anchor>").replace(
                "us:regulations/42/457/622", "<anchor>"
            )
            for issue in _analyze(text, source, citation, tests).issues
        ]

    assert outcome(content) == outcome(source_root_content)


def test_saved_457_622_rejection_subparagraph_gate_counts_only_cfr_root():
    recorded, content, _tests, source = _load_saved_rejection("36610751620")
    citation = recorded["citation"]
    rules_file = Path("/rulespec-us/us/regulations/42-cfr/457/622.yaml")
    source_root_content = re.sub(
        r"us:regulations/42-cfr/457/622(?=[/#])",
        "us:regulations/42/457/622",
        content,
    )

    assert not find_source_subparagraph_coverage_issues(
        content, rules_file=rules_file, source_texts={citation: source}
    )
    assert find_source_subparagraph_coverage_issues(
        source_root_content, rules_file=rules_file, source_texts={citation: source}
    )


@pytest.mark.parametrize(
    "root", ("us:regulations/42/457/622", "us:regulations/42-cfr/457/622")
)
def test_precise_federal_regulation_repeal_deferral_covers_branch_at_either_root(
    root,
):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/42/457/622
  deferred_outputs:
    - output: {root}/a#repealed_paragraph
      reason: us/regulation/42/457/622(a) is repealed.
rules: []
"""

    result = _analyze(content, "(a) Repealed.", "us/regulation/42/457/622", [])

    assert not result.issues


def test_federal_regulation_retry_shape_names_literal_canonical_citation():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/42/457/622
  deferred_outputs:
    - output: us:regulations/42-cfr/457/622/a#repealed_paragraph
      reason: 42 CFR 457.622(a) is repealed.
rules: []
"""

    issues = _analyze(content, "(a) Repealed.", "us/regulation/42/457/622", []).issues

    assert any(
        "`module.deferred_outputs[0]` identifies source branch (a)" in issue
        and "literal canonical citation required in `reason` is "
        "`us/regulation/42/457/622(a)`"
        in issue
        for issue in issues
    ), issues


# Invariants, checked exhaustively over the citation and branch grids above.


@pytest.mark.parametrize("citation", FEDERAL_REGULATION_CITATIONS)
def test_deferral_anchors_end_at_the_encoder_module_root(citation):
    anchors = completeness_module._deferral_anchor_bases(citation)

    assert anchors[0] == completeness_module._rulespec_target_base(citation)
    assert anchors[-1] == _module_root(citation)
    assert len(set(anchors)) == len(anchors)


@pytest.mark.parametrize("citation", OTHER_CITATIONS)
def test_deferral_anchors_unchanged_outside_numbered_federal_regulations(citation):
    assert completeness_module._deferral_anchor_bases(citation) == (
        completeness_module._rulespec_target_base(citation),
    )


@pytest.mark.parametrize(
    ("citation", "path", "precise"),
    tuple(itertools.product(FEDERAL_REGULATION_CITATIONS, BRANCH_PATHS, (True, False))),
)
def test_deferred_coverage_is_invariant_to_anchor_root(citation, path, precise):
    fragments = "".join(f"({part})" for part in path)
    reason = (
        f"{citation}{fragments} is repealed."
        if precise
        else "This branch is not available."
    )
    results = []
    for anchor in completeness_module._deferral_anchor_bases(citation):
        payload = _deferral_payload(
            citation, [(f"{anchor}/{'/'.join(path)}#deferred_output", reason)]
        )
        covered, issues = completeness_module._deferred_coverage(
            payload,
            corpus_citation_path=citation,
            source_text=REPEAL_SOURCE,
            branches=REPEAL_BRANCHES,
        )
        results.append(
            (covered, [issue.replace(anchor, "<anchor>") for issue in issues])
        )

    assert len(results) == 2
    assert results[0] == results[1]
    covered, issues = results[0]
    # The repeal predicate does not read dotted CFR section citations
    # (`us/regulation/7/1.36B-2(a)`) at either root; that is unchanged here.
    if precise and path in REPEALED_LEAF_PATHS and "." not in citation:
        assert path in covered and not issues
    elif not precise:
        assert not covered
        assert any("`module.deferred_outputs[0]`" in issue for issue in issues)


@pytest.mark.parametrize(
    ("citation", "path"),
    tuple(itertools.product(FEDERAL_REGULATION_CITATIONS, BRANCH_PATHS)),
)
def test_module_root_deferral_is_never_silently_dropped(citation, path):
    """A branch deferral at the module root is covered or named in an issue."""

    payload = _deferral_payload(
        citation,
        [
            (
                f"{_module_root(citation)}/{'/'.join(path)}#deferred_output",
                "Unavailable.",
            )
        ],
    )

    covered, issues = completeness_module._deferred_coverage(
        payload,
        corpus_citation_path=citation,
        source_text=REPEAL_SOURCE,
        branches=REPEAL_BRANCHES,
    )

    assert path in covered or any(
        "`module.deferred_outputs[0]`" in issue for issue in issues
    )


@pytest.mark.parametrize(
    "citation",
    tuple(
        citation
        for citation in FEDERAL_REGULATION_CITATIONS
        if len(citation.split("/")) == 5
    ),
)
def test_coverage_gates_agree_on_module_root_top_level_deferral(citation):
    source = "(a) Repealed.\n\n(b) Repealed."
    reason = f"{citation}(a) is repealed."
    payload = _deferral_payload(
        citation, [(f"{_module_root(citation)}/a#deferred_output", reason)]
    )
    rules_file = Path("/rulespec-us/us") / _source_identifier_to_relative_rulespec_path(
        citation
    )

    covered, issues = completeness_module._deferred_coverage(
        payload,
        corpus_citation_path=citation,
        source_text=source,
        branches=recognize_source_structure(source),
    )

    assert ("a",) in covered and not issues
    assert ("a",) in _deferred_output_covered_subparagraphs(
        payload, citation, rules_file=rules_file
    )
