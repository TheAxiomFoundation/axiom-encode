"""Authentic notices and finite ancestry negatives through normal coverage."""

import hashlib
from pathlib import Path

import pytest
import yaml

from axiom_encode.editorial_source_ancestry import (
    _certify,
    confirms_bibliographic_editorial_scope,
)
from axiom_encode.harness import coverage_exemptions as ce
from axiom_encode.harness import validator_pipeline as vp

FIXTURES = Path(__file__).parent / "fixtures/editorial_source_ancestry"


def issues(source, summary, scope):
    module = {"source_verification": {"corpus_citation_path": "us/statute/26/21"}}
    if summary is not None:
        module.update(status="deferred", summary=summary)
    return vp.find_source_subparagraph_coverage_issues(
        yaml.safe_dump({"format": "rulespec/v1", "module": module, "rules": []}),
        source_texts={"us/statute/26/21": source},
        requested_source="us/statute/26/21/" + "/".join(scope),
    )


@pytest.mark.parametrize(
    "name,scope,digest",
    [
        (
            "a-9-source.txt",
            ("a", "9"),
            "b559f25fcb8a8e6a2bf4bb37235c24b3065fe118eb7349bc78fbac2cfccbdba3",
        ),
        (
            "b-12-source.txt",
            ("b", "12"),
            "eeb9d8da8084dd18b6bd90f296eff74884335e21781c4c9a23b67e0785b45717",
        ),
    ],
)
def test_authentic_and_adversarial_normal_consumers(name, scope, digest, monkeypatch):
    text = (FIXTURES / name).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == digest
    cert = _certify(text, scope)
    summary = text[cert.notice_span[0] : cert.notice_span[1]]
    required = vp._high_signal_top_level_subparagraphs(text)
    assert issues(text, None, scope)
    # A text ancestry certificate is proposal evidence. Runtime waiver now
    # additionally requires a byte-bound reviewed tombstone or a repealed row.
    assert issues(text, summary, scope) == issues(text, None, scope)
    assert issues(text, "(1) Operative payments.", scope)
    assert vp._high_signal_top_level_subparagraphs(text) == required
    with monkeypatch.context() as m:
        m.setattr(
            vp,
            "_authoritative_deferred_editorial_source_slice",
            lambda *args, **kwargs: False,
        )
        assert issues(text, summary, scope) == issues(text, None, scope)
    first = cert.structural_paths[0][0]
    notice = cert.notice_span[0]
    after = cert.next_sibling_start
    parent = cert.parent_span[0]
    later = next(
        start for start, end, path in cert.structural_paths if path == (scope[0], "2")
    )
    mutations = {
        "curly_quoted_parent": text[:parent]
        + "“\n"
        + text[parent : after + 60]
        + "”"
        + text[after + 60 :],
        "straight_quoted_parent": text[:parent]
        + '"\n'
        + text[parent : after + 60]
        + '"'
        + text[after + 60 :],
        "single_quoted_parent": text[:parent]
        + "'\n"
        + text[parent : after + 60]
        + "'"
        + text[after + 60 :],
        "curly_single_quoted_parent": text[:parent]
        + "‘\n"
        + text[parent : after + 60]
        + "’"
        + text[after + 60 :],
        "reference_parent": text[:parent] + "See\n" + text[parent:],
        "straight_single_later_child": text[:later]
        + "'"
        + text[later : later + 3]
        + "'"
        + text[later + 3 :],
        "curly_single_later_child": text[:later]
        + "‘"
        + text[later : later + 3]
        + "’"
        + text[later + 3 :],
        "duplicate_parent": text[:notice]
        + "\n("
        + scope[0]
        + ") New parent.\n"
        + text[notice:],
        "inline_parent_reset": text[:notice] + "(z) New root. " + text[notice:],
        "unknown_section_heading": text[:notice] + "§ 99 New section. " + text[notice:],
        "unproved_section_heading": text[:notice]
        + "Section 6 Definitions: "
        + text[notice:],
        "unproved_compact_heading": text[:notice]
        + "Sec6 New section. "
        + text[notice:],
        "unknown_roman_heading": text[:notice] + "IV: Definitions. " + text[notice:],
        "unknown_symbol": text[:notice] + "• " + text[notice:],
        "physical_discontinuity": text[:notice] + "\nDefinitions: " + text[notice:],
        "gap_first_child": text[:first] + text[first:].replace("(1)", "(2)", 1),
        "reordered_second_child": text[:first] + text[first:].replace("(2)", "(1)", 1),
        "dropped_child": text[:first] + text[first:].replace("(2)", "", 1),
        "quoted_child": text[:first] + text[first:].replace("(1)", '"(1)"', 1),
        "indirect_quoted_child": text[:first]
        + '"First '
        + text[first:].replace("(1)", '(1)"', 1),
        "duplicate_child": text[:first] + text[first:].replace("(2)", "(2) (2)", 1),
        "unproved_nested_label": text[:notice] + "(Z) Unproved. " + text[notice:],
        "broken_reference": text.replace("section 401(a)", "something 401(a)", 1),
        "unknown_reference_label": text[:notice] + "(unknown) " + text[notice:],
        "notice_extra_prose": text[: cert.notice_span[1]]
        + " Taxes remain payable."
        + text[cert.notice_span[1] :],
        "reference_successor": text[:after] + text[after:].replace(") ", ") see ", 1),
        "wrong_successor": text[:after]
        + text[after:].replace("(" + str(int(scope[1]) + 1) + ")", "(99)", 1),
        "duplicate_notice": text[:notice]
        + text[cert.notice_span[0] : cert.notice_span[1]]
        + " "
        + text[notice:],
        "missing_target": text[:notice]
        + text[notice:].replace("Repealed.", "Operative.", 1),
        "citation_tail_unknown": text[:notice]
        + text[notice:].replace("Stat.", "Stat. subject to eligibility ", 1),
    }
    for control, changed in mutations.items():
        assert changed != text, control
        assert not confirms_bibliographic_editorial_scope(changed, scope), control
        assert not vp._authoritative_deferred_editorial_source_slice(
            changed, summary, scope
        ), control
        baseline = issues(changed, None, scope)
        assert issues(changed, summary, scope) == baseline, control
        assert vp._high_signal_top_level_subparagraphs(
            changed
        ) == vp._high_signal_top_level_subparagraphs(changed)


def test_actual_deferred_policy_consumers_require_reviewed_tombstones(monkeypatch):
    # Exercise the absence-of-review precondition even after these four exact
    # occurrences receive reviewed entries. The full registry has separate tests.
    entries = tuple(
        e
        for e in ce.load_coverage_exemptions()
        if e.owner_evidence["citation_path"]
        not in {
            "us/statute/26/3121/a/9",
            "us/statute/26/3306/b/12",
            "us/statute/26/3121/b/17",
            "us/statute/26/3306/b/8",
        }
    )
    monkeypatch.setattr(ce, "load_coverage_exemptions", lambda: entries)
    for name, citation, relative in [
        ("a-9", "us/statute/26/3121", "us/statutes/26/3121/a/9.yaml"),
        ("b-12", "us/statute/26/3306", "us/statutes/26/3306/b/12.yaml"),
    ]:
        content = (FIXTURES / (name + "-policy.yaml")).read_text()
        source = (FIXTURES / (name + "-source.txt")).read_text()
        deferred_issues = vp.find_source_subparagraph_coverage_issues(
            content, rules_file=Path(relative), source_texts={citation: source}
        )
        assert deferred_issues
        payload = yaml.safe_load(content)
        payload["module"]["status"] = "active"
        assert deferred_issues == vp.find_source_subparagraph_coverage_issues(
            yaml.safe_dump(payload),
            rules_file=Path(relative),
            source_texts={citation: source},
        )
