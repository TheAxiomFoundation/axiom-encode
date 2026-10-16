"""Coverage retains the baseline's demands and complete physical spans."""

import json
import re
from types import SimpleNamespace

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.cli import (
    _try_repair_generated_mixed_missing_deferred_outputs_for_apply,
)
from axiom_encode.harness import validator_pipeline as vp

MEDICAL_SOURCE = """D. As used in this section:
(4) "medical care expenses" means the amounts paid for:
(f) care in an institution other than a hospital;
(5) "physician" means a medical doctor;
"""


def _coverage_issues(source, citation):
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [],
        }
    )
    return vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}
    )


@pytest.mark.parametrize(
    "source",
    [
        MEDICAL_SOURCE,
        '(5) Definitions:\n(c) "taxpayer" means a resident individual;\n',
    ],
)
def test_nested_letter_retains_the_baseline_coverage_demand(source):
    label = "f" if source == MEDICAL_SOURCE else "c"
    body = source.split(f"({label}) ", 1)[1].strip()
    assert vp._high_signal_top_level_subparagraphs(source) == [(label, body)]


def test_medical_source_retains_baseline_f_demand():
    citation = "us-nm/statute/7-2-18.13"
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [],
        }
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: MEDICAL_SOURCE}
    )
    assert len(issues) == 1, issues
    assert "7-2-18.13(f)" in issues[0]


@pytest.mark.parametrize(
    "source",
    [MEDICAL_SOURCE, '(f) "benefit" means the amount payable.\n'],
)
def test_auto_repair_does_not_write_uncorroborated_deferral(tmp_path, source):
    output_root = tmp_path / "out"
    rules_file = output_root / "codex" / "statutes/7-2-18/13.yaml"
    rules_file.parent.mkdir(parents=True)
    policy_repo = tmp_path / "rulespec-us" / "us-nm"
    policy_repo.mkdir(parents=True)
    citation = "us-nm/statute/7-2-18.13"
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [{"name": "retained", "kind": "derived"}],
        }
    )
    rules_file.write_text(content)
    source_file = tmp_path / "source.txt"
    source_file.write_text(source)
    context = tmp_path / "context-manifest.json"
    context.write_text(json.dumps({"source_text_file": "source.txt"}))
    result = SimpleNamespace(
        runner="codex", output_file=str(rules_file), context_manifest_file=str(context)
    )
    repaired = _try_repair_generated_mixed_missing_deferred_outputs_for_apply(
        result,
        output_root=output_root,
        policy_repo_path=policy_repo,
        issues=[
            "Source sub-paragraph coverage missing: us-nm/statute/7-2-18.13(f) "
            "('care in an institution other than a hospital') has no rule citing "
            "it and no entry in `module.deferred_outputs`."
        ],
    )
    assert repaired == []
    assert rules_file.read_text() == content


@given(
    outer=st.sampled_from(list("ABCDEFGH")),
    number=st.integers(min_value=1, max_value=20),
    letter=st.sampled_from(list("abcdefghjklmnopqr")),
    cue=st.sampled_from(["means", "eligible", "ineligible"]),
)
@settings(deadline=None)
def test_nested_markers_retain_the_baseline_demands(outer, number, letter, cue):
    source = (
        f"{outer}. Definitions:\n({number}) The following terms apply:\n"
        f"({letter}) A person {cue} a beneficiary.\n"
        f'({number + 1}) "physician" means a medical doctor.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        (letter, source.split(f"({letter}) ", 1)[1].strip())
    ]


def test_true_root_letter_remains_demanded_with_bounded_text():
    source = '(a) "benefit" means the amount payable.\n(b) Effective date.\n'
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", '"benefit" means the amount payable.')
    ]


def test_combined_root_markers_keep_the_shallowest_branch_at_the_same_offset():
    first_body = '(1) (A) "benefit" means the amount payable.'
    second_body = "(1) (B) A person is eligible for the benefit."
    source = f"(a) {first_body}\n(b) {second_body}\n"
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", first_body),
        ("b", second_body),
    ]


def test_combined_numeric_child_markers_retain_the_baseline_demand():
    body = '(1) (C) "benefit" means the amount payable.'
    source = f"(1) Definitions:\n(c) {body}\n(2) Effective date.\n"
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", body + "\n(2) Effective date.")
    ]
    for citation in ("us-mi/statute/206.30", "us-mi/statute/206.30/1"):
        assert len(_coverage_issues(source, citation)) == 1


@given(
    number=st.integers(min_value=1, max_value=20),
    cue=st.sampled_from(["means", "is eligible for", "is ineligible for"]),
)
@settings(max_examples=50, deadline=None)
def test_combined_marker_offsets_keep_roots_and_numeric_children(number, cue):
    first_body = f"({number}) (A) A person {cue} the benefit."
    second_body = f"({number}) (B) A person {cue} the benefit."
    root_source = f"(a) {first_body}\n(b) {second_body}\n"
    assert vp._high_signal_top_level_subparagraphs(root_source) == [
        ("a", first_body),
        ("b", second_body),
    ]
    nested_body = f"(1) (C) A person {cue} the benefit."
    nested_source = (
        f"({number}) Definitions:\n(c) {nested_body}\n({number + 1}) Effective date.\n"
    )
    assert vp._high_signal_top_level_subparagraphs(nested_source) == [
        ("c", nested_body + f"\n({number + 1}) Effective date.")
    ]
    citation = "us-mi/statute/206.30"
    for target in (citation, f"{citation}/{number}"):
        assert len(_coverage_issues(nested_source, target)) == 1


@pytest.mark.parametrize(
    "cue", ["Effective date.", '"benefit" means the amount payable.']
)
def test_root_text_retains_baseline_span_across_a_dotted_parent(cue):
    source = f'(a) {cue}\nD. Definitions:\n(1) "physician" means a medical doctor.\n'
    expected = [("a", source.split("(a) ", 1)[1].strip())]
    assert vp._high_signal_top_level_subparagraphs(source) == expected


@pytest.mark.parametrize("cue", ["means", "is eligible for"])
def test_root_keeps_high_signal_content_in_its_indented_numeric_child(cue):
    body = f'Calculation:\n    (1) "benefit" {cue} the amount payable.'
    source = f"(a) {body}\n(b) Effective date.\n"
    assert vp._high_signal_top_level_subparagraphs(source) == [("a", body)]


@pytest.mark.parametrize(
    "citation",
    [
        "us-mi/statute/206.30/5",
        "us-wa/regulation/388/388-450/388-450-0162/5",
    ],
)
@pytest.mark.parametrize("indentation", ["", "    "])
def test_scoped_numeric_unit_keeps_its_letter_child_operative_content(
    citation, indentation
):
    body = f"Calculation:\n{indentation}(1) A person is eligible for the benefit."
    source = f"(5) Definitions:\n(a) {body}\n"
    assert vp._high_signal_top_level_subparagraphs(source) == [("a", body)]
    assert len(_coverage_issues(source, citation)) == 1


@pytest.mark.parametrize(
    "citation",
    [
        "us-mi/statute/206.30/5",
        "us-wa/regulation/388/388-450/388-450-0162/5",
    ],
)
@pytest.mark.parametrize("indentation", ["", "    "])
def test_scoped_numeric_unit_keeps_next_letter_after_its_own_number(
    citation, indentation
):
    body = '"benefit" means the amount payable.'
    source = (
        f"(5) Definitions:\n(a) Calculation:\n{indentation}(1) Details.\n(b) {body}\n"
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [("b", body)]
    assert len(_coverage_issues(source, citation)) == 1


@pytest.mark.parametrize("repealed, successor", [("c", "d"), ("k", "l")])
def test_real_letter_after_bracketed_repeal_remains_demanded(repealed, successor):
    body = '"benefit" means the amount payable.'
    source = (
        "(a) Administrative terms.\n(1) Item.\n(i) Details.\n"
        f"[({repealed}) Repealed.]\n({successor}) {body}\n"
    )
    assert (successor, body) in vp._high_signal_top_level_subparagraphs(source)


@pytest.mark.parametrize(
    "prefix",
    [
        "[(c) Repealed.]\n(5) Definitions:\n",
        "(5) Definitions:\n[(c) Repealed.]\n",
    ],
)
def test_bracketed_repeal_keeps_the_baseline_numeric_child_demand(prefix):
    source = f'{prefix}(d) "benefit" means the amount payable.\n'
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("d", '"benefit" means the amount payable.')
    ]


def test_unrecognized_marker_conservatively_keeps_its_real_demand(monkeypatch):
    monkeypatch.setattr(
        vp, "recognize_source_structure", lambda *args, **kwargs: (), raising=False
    )
    body = '"benefit" means the amount payable.'
    source = f"(a) {body}\n(b) Effective date.\n"
    assert vp._high_signal_top_level_subparagraphs(source) == [("a", body)]


@pytest.mark.parametrize(
    "citation, source",
    [
        (
            "us-mi/statute/206.30/10",
            '(10) Definitions:\n(a) "benefit" means the amount payable.\n',
        ),
        (
            "us/statute/26/21/a/4",
            '(4) Definitions:\n(a) "benefit" means the amount payable.\n',
        ),
        (
            "us-nm/statute/7-2-18.13/d/4",
            'D. Definitions:\n(4) Terms:\n(a) "benefit" means the amount payable.\n',
        ),
        (
            "us-wa/regulation/388/388-450/388-450-0162/2",
            '(2) Definitions:\n(a) "benefit" means the amount payable.\n',
        ),
        (
            "us-ny/regulation/18-nycrr/387/14/a/5",
            '(5) Definitions:\n(a) "benefit" means the amount payable.\n',
        ),
    ],
)
def test_scoped_unit_preserves_real_direct_child_obligation(citation, source):
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [],
        }
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}
    )
    assert len(issues) == 1
    assert "(a)" in issues[0]


def test_section_number_retains_the_baseline_nested_child_demand():
    source = '(10) Definitions:\n(a) "benefit" means the amount payable.\n'
    for citation in [
        "us/statute/26/10",
        "us-ma/statute/62/10",
        "us-ny/regulation/18-nycrr/387/10",
    ]:
        assert vp._high_signal_top_level_subparagraphs(source) == [
            ("a", '"benefit" means the amount payable.')
        ]
        assert len(_coverage_issues(source, citation)) == 1


@given(
    number=st.integers(min_value=1, max_value=20),
    letter=st.sampled_from(list("abcf")),
    citation=st.sampled_from(
        [
            "us-mi/statute/206.30",
            "us-wa/regulation/388/388-450/388-450-0162",
            "us-ny/regulation/18-nycrr/387/14",
            "us/regulation/26/1.36B-2",
            "us-co/regulation/10-ccr-2506-1/4.207.2",
        ]
    ),
)
@settings(deadline=None)
def test_scoped_and_whole_units_retain_baseline_demands(number, letter, citation):
    body = f'({number}) Definitions:\n({letter}) "benefit" means the amount payable.\n'
    expected = [(letter, '"benefit" means the amount payable.')]
    assert vp._high_signal_top_level_subparagraphs(body) == expected
    for target in (citation, f"{citation}/{number}"):
        content = yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {"source_verification": {"corpus_citation_path": target}},
                "rules": [],
            }
        )
        issues = vp.find_source_subparagraph_coverage_issues(
            content, source_texts={target: body}
        )
        assert len(issues) == 1, issues
        assert f"({letter})" in issues[0]


@given(
    number=st.integers(min_value=1, max_value=20),
    own_list_length=st.integers(min_value=1, max_value=3),
    indentation=st.sampled_from(["", "    "]),
    citation=st.sampled_from(
        [
            "us-mi/statute/206.30",
            "us-wa/regulation/388/388-450/388-450-0162",
            "us-ny/regulation/18-nycrr/387/14",
        ]
    ),
)
@settings(max_examples=50, deadline=None)
def test_scoped_numeric_unit_preserves_own_numbered_list_and_letter_sibling(
    number, own_list_length, indentation, citation
):
    own_list = "\n".join(
        f'{indentation}({index}) "benefit" means the amount payable at stage {index}.'
        for index in range(1, own_list_length + 1)
    )
    first_body = f"Calculation:\n{own_list}"
    second_body = "A person is eligible for the benefit."
    source = f"({number}) Definitions:\n(a) {first_body}\n(b) {second_body}\n"
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", first_body),
        ("b", second_body),
    ]
    for target in (citation, f"{citation}/{number}"):
        assert len(_coverage_issues(source, target)) == 2
    prose_source = re.sub(r"stage (\d+)\.", r"stage \1 as applicable.", source)
    assert vp._high_signal_top_level_subparagraphs(prose_source) == [
        ("a", re.sub(r"stage (\d+)\.", r"stage \1 as applicable.", first_body)),
        ("b", second_body),
    ]
    assert len(_coverage_issues(prose_source, citation)) == 2


@given(
    nested=st.booleans(),
    letter=st.sampled_from(list("abcdefghjklmnopqr")),
    cue=st.sampled_from(["means", "eligible", "ineligible"]),
)
@settings(deadline=None)
def test_every_demand_retains_its_baseline_physical_span(nested, letter, cue):
    source = f"({letter}) A person {cue} a beneficiary.\n"
    if nested:
        source = "(5) Definitions:\n" + source
    demands = vp._high_signal_top_level_subparagraphs(source)
    assert demands == [(letter, f"A person {cue} a beneficiary.")]
