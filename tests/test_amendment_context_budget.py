"""Principal requests reserve exact amendment evidence before optional context."""

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import evals
from tests.test_evals import (
    _workspace_prompt_for_source_unit,
    _write_test_corpus_release,
)

_PRINCIPAL = "de/statute/benefit-act"
_TARGET = f"{_PRINCIPAL}/schedule/4/part/7/clause/4"
_AMENDING_ACT = "de/statute/large-amendment-act"
_REQUIRED = f"{_AMENDING_ACT}/section/4"
_COMMENCEMENT = f"{_AMENDING_ACT}/section/2"
_EXACT_BODY = "In Schedule 4, Part 7, clause 4, replace the amount with 305 euros."
_COMMENCEMENT_BODY = "This Act comes into force on 1 April 2027."


def _large_amendment_rows():
    rows = [
        {
            "citation_path": _PRINCIPAL,
            "source_path": "sources/de/benefit-act.xml",
            "body": "Benefit Act",
            "metadata": {"title": "Benefit Act"},
        },
        {
            "citation_path": _TARGET,
            "source_path": "sources/de/benefit-act.xml",
            "body": "The benefit is 200 euros.",
            "metadata": {"title": "Benefit Act"},
        },
    ]
    for number in range(1, 21):
        metadata = {
            "title": "Large Amendment Act",
            "document_type": "amendment act",
            # Every row names the principal document, just as a real large Act
            # can. This declaration must not make every section required.
            "amended_act": _PRINCIPAL,
        }
        body = f"Optional section {number}. " + "x" * 5_500
        if number == 2:
            body = _COMMENCEMENT_BODY
        else:
            metadata.update(
                amendment_target=(
                    _TARGET if number == 4 else f"{_PRINCIPAL}/section/{number}"
                ),
                commencement_date="2027-04-01",
                commencement_citation_path=_COMMENCEMENT,
            )
        if number == 4:
            body = _EXACT_BODY
        elif number in (5, 8):
            body = f"Oversized optional section {number}. " + "x" * 12_001
        rows.append(
            {
                "citation_path": f"{_AMENDING_ACT}/section/{number}",
                "source_path": "sources/de/large-amendment-act.xml",
                "heading": "Commencement" if number == 2 else f"Section {number}",
                "body": body,
                "metadata": metadata,
            }
        )
    return rows


def _context_bytes(provision_text, amendment_texts):
    return len(provision_text.encode("utf-8")) + len(
        evals._AMENDMENT_CONTEXT_SEPARATOR.join(amendment_texts).encode("utf-8")
    )


def _full_context_bytes(metadata, documents):
    return _context_bytes(
        evals._render_provision_metadata(metadata),
        [
            evals._render_amendment_document(document, body=document.body)
            for document in documents
        ],
    )


def test_principal_large_amending_act_keeps_required_evidence_and_records_optional_drops(
    tmp_path,
):
    release = _write_test_corpus_release(tmp_path, _large_amendment_rows())
    source = evals.resolve_corpus_source_unit(_TARGET, release)
    discovered = {item.citation_path: item for item in source.amendment_documents}

    assert len(discovered) == 20
    assert _full_context_bytes(source.provision_metadata, discovered.values()) > 32_000
    assert len(discovered[f"{_AMENDING_ACT}/section/5"].body) > 12_000
    assert len(discovered[f"{_AMENDING_ACT}/section/8"].body) > 12_000
    assert {
        item.citation_path
        for item in source.amendment_documents
        if item.required_for_target
    } == {_REQUIRED, _COMMENCEMENT}

    workspace, prompt = _workspace_prompt_for_source_unit(tmp_path, source)
    visible = {item.citation_path: item for item in workspace.amendment_documents}
    drops = {
        item["citation_path"]: item for item in workspace.dropped_amendment_documents
    }
    manifest = json.loads(workspace.manifest_file.read_text())
    texts = [
        (workspace.root / item.workspace_path).read_text().removesuffix("\n")
        for item in workspace.context_files
        if item.kind == "corpus_amendment_act"
    ]

    assert visible[_REQUIRED].body == _EXACT_BODY
    assert visible[_COMMENCEMENT].body == _COMMENCEMENT_BODY
    assert _EXACT_BODY in prompt
    assert _COMMENCEMENT_BODY in prompt
    assert drops.keys() == discovered.keys() - visible.keys()
    assert {_REQUIRED, _COMMENCEMENT}.isdisjoint(drops)
    assert drops[f"{_AMENDING_ACT}/section/5"]["reason"] == "body_context_limit"
    assert drops[f"{_AMENDING_ACT}/section/8"]["reason"] == "body_context_limit"
    assert any(item["reason"] == "aggregate_context_limit" for item in drops.values())
    assert manifest["dropped_amendment_documents"] == list(
        workspace.dropped_amendment_documents
    )
    assert all(len(document.body) <= 12_000 for document in visible.values())
    assert _context_bytes(workspace.provision_metadata_text or "", texts) <= 32_000


@pytest.mark.parametrize("required_path", [_REQUIRED, _COMMENCEMENT])
def test_principal_required_body_over_limit_blocks_and_names_evidence(
    tmp_path, required_path
):
    rows = _large_amendment_rows()
    row = next(item for item in rows if item["citation_path"] == required_path)
    row["body"] = str(row["body"]) + "x" * 12_001
    release = _write_test_corpus_release(tmp_path, rows)
    source = evals.resolve_corpus_source_unit(_TARGET, release)

    with pytest.raises(ValueError, match="Unresolved amendment context") as error:
        _workspace_prompt_for_source_unit(tmp_path, source)

    assert required_path in str(error.value)


def test_principal_required_aggregate_over_limit_blocks(tmp_path, monkeypatch):
    release = _write_test_corpus_release(tmp_path, _large_amendment_rows())
    source = evals.resolve_corpus_source_unit(_TARGET, release)
    required = [item for item in source.amendment_documents if item.required_for_target]
    monkeypatch.setattr(
        evals,
        "_INJECTED_CONTEXT_LIMIT",
        _full_context_bytes(source.provision_metadata, required) - 1,
    )

    with pytest.raises(
        ValueError, match="Unresolved amendment context.*context budget"
    ) as error:
        _workspace_prompt_for_source_unit(tmp_path, source)

    assert any(item.citation_path in str(error.value) for item in required)


def test_missing_required_commencement_blocks_and_names_missing_evidence():
    document = evals.CorpusAmendmentDocument(
        citation_path=_REQUIRED,
        title="Required amendment",
        expression_date="2026-01-01",
        metadata={
            "amendment_target": _TARGET,
            "commencement_date": "2027-04-01",
            "commencement_citation_path": _COMMENCEMENT,
        },
        body=_EXACT_BODY,
        match_tier="structured",
        required_for_target=True,
    )

    with pytest.raises(ValueError, match="Unresolved amendment context") as error:
        evals._render_injected_context({}, [document])

    assert _COMMENCEMENT in str(error.value)


@pytest.mark.parametrize("optional_slots", [1, 2])
def test_optional_context_prefers_same_schedule_then_natural_document_order(
    tmp_path, monkeypatch, optional_slots
):
    rows = _large_amendment_rows()
    retained_numbers = {1, 2, 3, 4, 10}
    rows = [
        row
        for row in rows
        if not str(row["citation_path"]).startswith(f"{_AMENDING_ACT}/")
        or int(str(row["citation_path"]).rsplit("/", 1)[1]) in retained_numbers
    ]
    optional_targets = {
        1: f"{_PRINCIPAL}/schedule/8/part/9/clause/1",
        3: f"{_PRINCIPAL}/schedule/4/part/7/clause/5",
        10: f"{_PRINCIPAL}/schedule/4/part/8/clause/1",
    }
    for row in rows:
        number = (
            int(str(row["citation_path"]).rsplit("/", 1)[1])
            if str(row["citation_path"]).startswith(f"{_AMENDING_ACT}/")
            else None
        )
        if number in optional_targets:
            row["metadata"]["amendment_target"] = optional_targets[number]
            row["body"] = f"Optional section {number}. " + "x" * 3_000
    # Release row order must not substitute for the Act's numeric section order.
    release = _write_test_corpus_release(tmp_path, list(reversed(rows)))
    source = evals.resolve_corpus_source_unit(_TARGET, release)
    required = [item for item in source.amendment_documents if item.required_for_target]
    preferred_paths = [f"{_AMENDING_ACT}/section/{number}" for number in (3, 10)]
    preferred = [
        next(item for item in source.amendment_documents if item.citation_path == path)
        for path in preferred_paths[:optional_slots]
    ]
    monkeypatch.setattr(
        evals,
        "_INJECTED_CONTEXT_LIMIT",
        _full_context_bytes(source.provision_metadata, [*required, *preferred]),
    )

    workspace, _ = _workspace_prompt_for_source_unit(tmp_path, source)

    assert [
        item.citation_path
        for item in workspace.amendment_documents
        if not item.required_for_target
    ] == preferred_paths[:optional_slots]


@pytest.mark.parametrize("supplement", ["schedule", "annex", "appendix"])
def test_optional_document_order_places_main_provisions_before_supplements(
    tmp_path, monkeypatch, supplement
):
    rows = _large_amendment_rows()
    paths = {
        _PRINCIPAL,
        _TARGET,
        _REQUIRED,
        _COMMENCEMENT,
        f"{_AMENDING_ACT}/section/1",
        f"{_AMENDING_ACT}/section/3",
    }
    rows = [row for row in rows if row["citation_path"] in paths]
    supplement_path = f"{_AMENDING_ACT}/{supplement}/1"
    for row in rows:
        if row["citation_path"] == f"{_AMENDING_ACT}/section/1":
            row["citation_path"] = supplement_path
            row["body"] = "x" * 3_000
        elif row["citation_path"] == f"{_AMENDING_ACT}/section/3":
            row["body"] = "x" * 3_000
    release = _write_test_corpus_release(tmp_path, rows)
    source = evals.resolve_corpus_source_unit(_TARGET, release)
    required = [item for item in source.amendment_documents if item.required_for_target]
    optional = [
        item for item in source.amendment_documents if not item.required_for_target
    ]
    # Each optional row fits individually, but there is room for only one.
    monkeypatch.setattr(
        evals,
        "_INJECTED_CONTEXT_LIMIT",
        max(
            _full_context_bytes(source.provision_metadata, [*required, item])
            for item in optional
        ),
    )

    workspace, _ = _workspace_prompt_for_source_unit(tmp_path, source)

    assert [
        item.citation_path
        for item in workspace.amendment_documents
        if not item.required_for_target
    ] == [f"{_AMENDING_ACT}/section/3"]
    assert {
        item["citation_path"] for item in workspace.dropped_amendment_documents
    } == {supplement_path}


@pytest.mark.parametrize(
    ("target_key", "target_value"),
    [
        ("amendment_target", _TARGET),
        ("amendment_target", f"{_PRINCIPAL}/schedule/4/part/7"),
        ("amendment_target", f"{_TARGET}/subclause/1"),
        ("amendment_target", _PRINCIPAL),
        ("amendment_targets", [f"{_PRINCIPAL}/section/1", _TARGET]),
        ("amendment_targets", [f"{_TARGET}/subclause/1"]),
    ],
)
@pytest.mark.parametrize("principal_root_present", [False, True])
def test_principal_required_targets_include_exact_ancestors_and_descendants(
    tmp_path, target_key, target_value, principal_root_present
):
    rows = _large_amendment_rows()
    if not principal_root_present:
        rows = [row for row in rows if row["citation_path"] != _PRINCIPAL]
    row = next(item for item in rows if item["citation_path"] == _REQUIRED)
    row["metadata"].pop("amendment_target")
    row["metadata"][target_key] = target_value
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_TARGET, release)

    assert {
        item.citation_path
        for item in source.amendment_documents
        if item.required_for_target
    } == {_REQUIRED, _COMMENCEMENT}


def test_principal_complete_timeline_is_retained_when_all_evidence_fits(tmp_path):
    rows = _large_amendment_rows()
    for row in rows:
        if row["citation_path"] not in {_TARGET, _PRINCIPAL, _COMMENCEMENT, _REQUIRED}:
            row["body"] = "Short optional amendment."
    release = _write_test_corpus_release(tmp_path, rows)
    source = evals.resolve_corpus_source_unit(_TARGET, release)

    workspace, _ = _workspace_prompt_for_source_unit(tmp_path, source)

    assert len(workspace.amendment_documents) == 20
    assert {
        item.citation_path: item.body for item in workspace.amendment_documents
    } == {item.citation_path: item.body for item in source.amendment_documents}
    assert workspace.dropped_amendment_documents == ()


@pytest.mark.parametrize("oversized", [False, True])
def test_optional_structured_row_without_commencement_is_bounded_context(
    tmp_path, oversized
):
    optional_path = f"{_AMENDING_ACT}/section/5"
    kept_paths = {_PRINCIPAL, _TARGET, _REQUIRED, _COMMENCEMENT, optional_path}
    rows = [
        row for row in _large_amendment_rows() if row["citation_path"] in kept_paths
    ]
    optional_row = next(row for row in rows if row["citation_path"] == optional_path)
    optional_row["metadata"].pop("commencement_date")
    optional_row["metadata"].pop("commencement_citation_path")
    optional_row["body"] = "x" * (12_001 if oversized else 100)
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_TARGET, release)
    discovered = {item.citation_path: item for item in source.amendment_documents}
    workspace, _ = _workspace_prompt_for_source_unit(tmp_path, source)
    visible = {item.citation_path: item for item in workspace.amendment_documents}
    drops = {
        item["citation_path"]: item for item in workspace.dropped_amendment_documents
    }
    manifest = json.loads(workspace.manifest_file.read_text())

    assert discovered[optional_path].match_tier == "structured"
    assert discovered[optional_path].required_for_target is False
    if oversized:
        assert optional_path not in visible
        assert drops[optional_path]["reason"] == "body_context_limit"
        assert manifest["dropped_amendment_documents"] == list(
            workspace.dropped_amendment_documents
        )
    else:
        assert visible[optional_path].body == optional_row["body"]
        assert optional_path not in drops


def test_all_optional_canonical_targets_keep_whole_row_drop_accounting(tmp_path):
    rows = _large_amendment_rows()
    oversized_path = f"{_AMENDING_ACT}/section/5"
    for row in rows:
        path = str(row["citation_path"])
        if not path.startswith(f"{_AMENDING_ACT}/"):
            continue
        metadata = row["metadata"]
        metadata.pop("commencement_date", None)
        metadata.pop("commencement_citation_path", None)
        metadata["amendment_target"] = f"{_PRINCIPAL}/section/{path.rsplit('/', 1)[1]}"
        row["body"] = "x" * 12_001 if path == oversized_path else "Short amendment."
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_TARGET, release)
    workspace, _ = _workspace_prompt_for_source_unit(tmp_path, source)
    visible = {item.citation_path: item for item in workspace.amendment_documents}
    manifest = json.loads(workspace.manifest_file.read_text())

    assert len(source.amendment_documents) == 20
    assert all(item.required_for_target is False for item in source.amendment_documents)
    assert len(visible) == 19
    assert oversized_path not in visible
    assert len(workspace.dropped_amendment_documents) == 1
    assert workspace.dropped_amendment_documents[0]["citation_path"] == oversized_path
    assert workspace.dropped_amendment_documents[0]["reason"] == "body_context_limit"
    assert manifest["dropped_amendment_documents"] == list(
        workspace.dropped_amendment_documents
    )
    assert all(item.body == "Short amendment." for item in visible.values())


@pytest.mark.parametrize("missing_mode", ["absent", "different_source"])
def test_discovery_missing_declared_commencement_blocks_and_names_evidence(
    tmp_path, missing_mode
):
    rows = _large_amendment_rows()
    if missing_mode == "absent":
        rows = [row for row in rows if row["citation_path"] != _COMMENCEMENT]
    else:
        commencement_row = next(
            row for row in rows if row["citation_path"] == _COMMENCEMENT
        )
        commencement_row["source_path"] = "sources/de/different-act.xml"
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context") as error:
        evals.resolve_corpus_source_unit(_TARGET, release)

    assert _COMMENCEMENT in str(error.value)


_TARGET_CHOICES = [
    _TARGET,
    f"{_PRINCIPAL}/schedule/4/part/7",
    f"{_TARGET}/subclause/1",
    f"{_PRINCIPAL}/schedule/4/part/7/clause/40",
    f"{_PRINCIPAL}/schedule/4/part/7/clause/5",
    f"{_PRINCIPAL}/section/20",
]


def _targets_requested_provision(target):
    """Independent segment comparison, including protection against 4 versus 40."""
    requested = _TARGET.split("/")
    candidate = target.split("/")
    return (
        requested[: len(candidate)] == candidate
        or candidate[: len(requested)] == requested
    )


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    generated=st.lists(
        st.tuples(
            st.sampled_from(_TARGET_CHOICES),
            st.booleans(),
            st.sampled_from([100, 4_000, 11_500, 12_001]),
            st.sampled_from(["x", "ä"]),
        ),
        min_size=4,
        max_size=4,
    )
)
def test_successful_render_never_drops_required_or_targeting_evidence(
    tmp_path, generated
):
    rows = _large_amendment_rows()
    generated_numbers = (1, 3, 5, 6)
    kept_paths = {
        _PRINCIPAL,
        _TARGET,
        _REQUIRED,
        _COMMENCEMENT,
        *(f"{_AMENDING_ACT}/section/{number}" for number in generated_numbers),
    }
    rows = [row for row in rows if row["citation_path"] in kept_paths]
    expected_required = {_REQUIRED, _COMMENCEMENT}
    for number, (target, plural, body_size, character) in zip(
        generated_numbers, generated, strict=True
    ):
        path = f"{_AMENDING_ACT}/section/{number}"
        row = next(item for item in rows if item["citation_path"] == path)
        metadata = row["metadata"]
        metadata.pop("amendment_target")
        metadata["amendment_targets" if plural else "amendment_target"] = (
            [target] if plural else target
        )
        if _targets_requested_provision(target):
            expected_required.add(path)
            body_size = 100
        row["body"] = character * body_size
    # Each example uses a separate authenticated release and materialized context.
    case_root = tmp_path / f"case-{len(list(tmp_path.iterdir()))}"
    release = _write_test_corpus_release(case_root, rows)
    source = evals.resolve_corpus_source_unit(_TARGET, release)

    workspace, _ = _workspace_prompt_for_source_unit(case_root, source)
    discovered = {item.citation_path: item for item in source.amendment_documents}
    visible = {item.citation_path: item for item in workspace.amendment_documents}
    dropped = {item["citation_path"] for item in workspace.dropped_amendment_documents}

    assert {
        item.citation_path
        for item in source.amendment_documents
        if item.required_for_target
    } == expected_required
    assert expected_required <= visible.keys()
    assert expected_required.isdisjoint(dropped)
    assert dropped == discovered.keys() - visible.keys()
    assert all(
        visible[path].body == discovered[path].body for path in expected_required
    )
    for path in dropped:
        metadata = discovered[path].metadata
        values = metadata.get("amendment_targets", [metadata.get("amendment_target")])
        assert all(not _targets_requested_provision(value) for value in values if value)
