"""Deferred-output retry overlays preserve omissions and consume exact removals."""

import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import evals, source_completeness
from axiom_encode.harness import validator_pipeline as vp

TARGET = "us-nm:statutes/7-2-18/13/f#deferred_f"
PRESERVED = {"output": TARGET, "reason": "placeholder"}


def _overlay(
    root: Path,
    generated: list[dict],
    preserved: list[dict],
    *,
    source_citation: str | None = None,
):
    def payload(outputs):
        module: dict[str, object] = {"deferred_outputs": outputs}
        if source_citation is not None:
            module["source_verification"] = {"corpus_citation_path": source_citation}
        return {"format": "rulespec/v1", "module": module, "rules": []}

    file = root / "candidate.yaml"
    file.write_text(
        yaml.safe_dump(payload(generated)),
        encoding="utf-8",
    )
    candidate = evals.ValidationRetryCandidate(
        rulespec=yaml.safe_dump(payload(preserved))
    )
    repairs = evals._overlay_validation_retry_candidate(
        file, artifact_root=root, candidate=candidate
    )
    content = file.read_text(encoding="utf-8")
    return yaml.safe_load(content), repairs, content


def test_deferred_overlay_consumes_exact_removal_marker(tmp_path):
    payload, repairs, content = _overlay(
        tmp_path, [{"output": TARGET, "repair_remove": True}], [PRESERVED]
    )

    assert payload["module"].get("deferred_outputs", []) == []
    assert repairs == (f"removed_deferred_output:{TARGET}",)
    assert "repair_remove" not in content
    assert vp.find_deferred_output_issues(content) == []


def test_deferred_overlay_omission_preserves_existing_output(tmp_path):
    payload, repairs, _content = _overlay(tmp_path, [], [PRESERVED])

    assert payload["module"]["deferred_outputs"] == [PRESERVED]
    assert repairs == (f"deferred_output:{TARGET}",)


@pytest.mark.parametrize(
    "item, match",
    [
        ({"output": TARGET, "repair_remove": False}, "repair_remove: true"),
        (
            {"output": TARGET, "repair_remove": True, "reason": "placeholder"},
            "only output and repair_remove",
        ),
        (
            {"output": "us-nm:statutes/7-2-18/13/z#missing", "repair_remove": True},
            "no preserved item",
        ),
        (
            {"output": TARGET, "reason": "placeholder", "repair_replace": True},
            "unsupported keys.*repair_replace",
        ),
        ({"output": TARGET, "repair_remove": 1}, "repair_remove: true"),
    ],
)
def test_deferred_overlay_rejects_invalid_removal_or_unknown_keys(
    tmp_path, item, match
):
    with pytest.raises(ValueError, match=match):
        _overlay(tmp_path, [item], [PRESERVED])


def test_deferred_overlay_accepts_supported_provenance_fields(tmp_path):
    generated = {
        **PRESERVED,
        "blocked_by": ["us-nm:statutes/7-2-18/13/d/4#medical_care_expenses"],
        "source_values": ["us-nm:statutes/7-2-18/13#medical_care_expenses_limit"],
    }
    payload, _repairs, content = _overlay(tmp_path, [generated], [PRESERVED])

    assert payload["module"]["deferred_outputs"] == [generated]
    assert vp.find_deferred_output_issues(content) == []


def test_deferred_overlay_rejects_retained_unsupported_fields(tmp_path):
    invalid = {**PRESERVED, "repair_replace": True}

    with pytest.raises(ValueError, match="unsupported keys.*repair_replace"):
        _overlay(tmp_path, [], [invalid])


@pytest.mark.parametrize(
    "preserved_module",
    [
        pytest.param("omitted", id="no-module"),
        pytest.param({}, id="empty-module"),
        pytest.param(None, id="null-module"),
        pytest.param([], id="non-mapping-module"),
    ],
)
@pytest.mark.parametrize(
    "item, match",
    [
        (
            {**PRESERVED, "repair_replace": True},
            "unsupported keys.*repair_replace",
        ),
        ({"output": TARGET, "repair_remove": True}, "no preserved item"),
    ],
)
def test_deferred_overlay_validates_generated_records_without_preserved_module(
    tmp_path, preserved_module, item, match
):
    preserved = {"format": "rulespec/v1", "rules": []}
    if preserved_module != "omitted":
        preserved["module"] = preserved_module
    file = tmp_path / "candidate.yaml"
    file.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {"deferred_outputs": [item]},
                "rules": [],
            }
        ),
        encoding="utf-8",
    )
    candidate = evals.ValidationRetryCandidate(rulespec=yaml.safe_dump(preserved))

    with pytest.raises(ValueError, match=match):
        evals._overlay_validation_retry_candidate(
            file, artifact_root=tmp_path, candidate=candidate
        )


def test_deferred_overlay_allows_exact_removal_of_invalid_prior_record(tmp_path):
    invalid = {**PRESERVED, "repair_replace": True}
    payload, repairs, content = _overlay(
        tmp_path, [{"output": TARGET, "repair_remove": True}], [invalid]
    )

    assert payload["module"].get("deferred_outputs", []) == []
    assert repairs == (f"removed_deferred_output:{TARGET}",)
    assert vp.find_deferred_output_issues(content) == []


def test_deferred_overlay_allows_clean_replacement_of_invalid_prior_record(tmp_path):
    invalid = {**PRESERVED, "repair_replace": True}
    replacement = {"output": TARGET, "reason": "precise replacement"}
    payload, repairs, content = _overlay(tmp_path, [replacement], [invalid])

    assert payload["module"]["deferred_outputs"] == [replacement]
    assert repairs == ()
    assert vp.find_deferred_output_issues(content) == []


def test_deferred_overlay_still_requires_reason_on_ordinary_replacement(tmp_path):
    _payload, _repairs, content = _overlay(tmp_path, [{"output": TARGET}], [PRESERVED])

    assert vp.find_deferred_output_issues(content) == [
        "module.deferred_outputs[0].reason is required."
    ]


@pytest.mark.parametrize("marker_root", ["source", "destination"])
def test_deferred_overlay_removes_source_normalized_output(tmp_path, marker_root):
    file = tmp_path / "regulations" / "7-cfr" / "273" / "4.yaml"
    file.parent.mkdir(parents=True)
    roots = {
        "source": "us:regulations/7/273/4",
        "destination": "us:regulations/7-cfr/273/4",
    }
    output = f"{roots['source']}/a#obsolete"
    module = {"source_verification": {"corpus_citation_path": "us/regulation/7/273/4"}}
    file.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {
                    **module,
                    "deferred_outputs": [
                        {
                            "output": f"{roots[marker_root]}/a#obsolete",
                            "repair_remove": True,
                        }
                    ],
                },
                "rules": [],
            }
        ),
        encoding="utf-8",
    )
    candidate = evals.ValidationRetryCandidate(
        rulespec=yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {
                    **module,
                    "deferred_outputs": [
                        {
                            "output": f"{roots['destination']}/a#obsolete",
                            "reason": "placeholder",
                        }
                    ],
                },
                "rules": [],
            }
        )
    )

    repairs = evals._overlay_validation_retry_candidate(
        file, artifact_root=tmp_path, candidate=candidate
    )

    assert yaml.safe_load(file.read_text())["module"].get("deferred_outputs", []) == []
    assert f"removed_deferred_output:{output}" in repairs


def test_deferred_overlay_removal_keeps_real_source_obligations(tmp_path):
    citation = "us-nm/statute/7-2-18.13"
    source = '(f) "medical care expenses" means care expenses capped at $100.'
    _payload, _repairs, content = _overlay(
        tmp_path,
        [{"output": TARGET, "repair_remove": True}],
        [PRESERVED],
        source_citation=citation,
    )

    coverage_issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}, requested_source=citation
    )
    assert len(coverage_issues) == 1
    assert "Source sub-paragraph coverage missing" in coverage_issues[0]
    complete_source = source_completeness.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=vp.extract_typed_numeric_inventory_occurrences_from_text,
        extract_numeric_grounding_occurrences=vp.extract_typed_numeric_occurrences_from_text,
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    assert any("numeric value 100" in issue for issue in complete_source.issues)


def test_deferred_overlay_prompt_documents_exact_output_removal():
    section = evals._format_validation_retry_candidate(
        evals.ValidationRetryCandidate(rulespec="format: rulespec/v1\nrules: []\n"),
        target_file_name="section.yaml",
    )

    assert "only `output: <existing output>` and `repair_remove: true`" in section
    assert "Source completeness still applies after removal" in section


@given(
    st.lists(st.sampled_from(["omit", "remove", "replace"]), min_size=1, max_size=10)
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_deferred_overlay_preserves_exactly_the_unremoved_outputs(tmp_path, actions):
    preserved = [
        {"output": f"us:statutes/section/{index}#shared", "reason": "preserved"}
        for index in range(len(actions))
    ]
    added = {"output": "us:statutes/section/added#shared", "reason": "added"}
    generated = [added]
    expected = {added["output"]: added}
    removed = set()
    for action, item in zip(actions, preserved, strict=True):
        if action == "remove":
            generated.append({"output": item["output"], "repair_remove": True})
            removed.add(item["output"])
        elif action == "replace":
            replacement = {**item, "reason": "replacement"}
            generated.append(replacement)
            expected[item["output"]] = replacement
        else:
            expected[item["output"]] = item

    # Each Hypothesis example owns its files within the pytest basetemp fixture.
    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        payload, repairs, content = _overlay(Path(directory), generated, preserved)

    assert {item["output"]: item for item in payload["module"]["deferred_outputs"]} == (
        expected
    )
    assert {
        repair.removeprefix("removed_deferred_output:")
        for repair in repairs
        if repair.startswith("removed_deferred_output:")
    } == removed
    assert "repair_remove" not in content
    assert vp.find_deferred_output_issues(content) == []
