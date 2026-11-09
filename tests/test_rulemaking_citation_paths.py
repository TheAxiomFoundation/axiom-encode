"""Target routing for Federal Register (``<jurisdiction>/rulemaking/...``) paths.

The corpus stores Federal Register instruments under ``us/rulemaking/...``.
Every adapter that maps between corpus citation paths and RuleSpec output
paths must agree on one contract:

* forward: ``<j>/rulemaking/<tail>`` -> ``policies/rulemaking/<tail>.yaml``
* reverse: ``policies/rulemaking/<tail>.yaml`` -> ``<j>/rulemaking/<tail>``
* a corpus *policy* path whose first tail segment is ``rulemaking`` would
  land on the same file, so the forward mapper refuses it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from axiom_encode import corpus_resolver
from axiom_encode.cli import _relative_output_to_child_corpus_citation_path
from axiom_encode.harness import (
    dependency_stubs,
    evals,
    source_completeness,
    validator_pipeline,
)

FR_CITATION = "us/rulemaking/federal-register/2026-02-25/2026-03829"
FR_OUTPUT = Path("policies/rulemaking/federal-register/2026-02-25/2026-03829.yaml")

_SEGMENT = st.from_regex(r"[a-z0-9][a-z0-9-]{0,11}", fullmatch=True)
_TAIL = st.lists(_SEGMENT, min_size=1, max_size=5)
_JURISDICTION = st.sampled_from(["us", "us-ca", "us-tn", "uk", "ca"])


def _rules_repo(tmp_path: Path, jurisdiction: str) -> Path:
    repo = tmp_path / f"rulespec-{jurisdiction}" / jurisdiction
    repo.mkdir(parents=True, exist_ok=True)
    return repo


def test_looks_like_corpus_citation_path_accepts_rulemaking():
    assert evals._looks_like_corpus_citation_path(FR_CITATION)


def test_resolve_eval_output_path_routes_rulemaking_target():
    # On main before this change the target raised either "not a canonical
    # citation path" (predicate miss) or "Unsupported canonical source
    # identifier" (output mapper miss).
    assert evals._resolve_eval_output_path(FR_CITATION) == FR_OUTPUT


def test_resolve_eval_output_path_routes_rulemaking_without_fallback_parser():
    def _fallback(_citation: str) -> Path:
        raise AssertionError("rulemaking targets must not reach the USC fallback")

    assert evals._resolve_eval_output_path(FR_CITATION, fallback=_fallback) == (
        FR_OUTPUT
    )


def test_target_rel_for_eval_identifier_routes_rulemaking_target():
    assert evals._target_rel_for_eval_identifier(FR_CITATION) == FR_OUTPUT


def test_import_prefix_for_rulemaking_target_is_jurisdiction():
    assert evals._import_prefix_for_eval_identifier(FR_CITATION) == "us"


def test_policy_path_under_rulemaking_namespace_is_refused():
    with pytest.raises(ValueError, match="rulemaking"):
        evals._source_identifier_to_relative_rulespec_path(
            "us/policy/rulemaking/federal-register/2026-03829"
        )
    with pytest.raises(ValueError, match="rulemaking"):
        evals._resolve_eval_output_path(
            "us/policy/rulemaking/federal-register/2026-03829"
        )


def test_policy_document_named_rulemaking_keeps_its_file():
    # Only paths *inside* the namespace collide; policies/rulemaking.yaml is a
    # sibling file, not a directory entry.
    assert evals._source_identifier_to_relative_rulespec_path(
        "us/policy/rulemaking"
    ) == Path("policies/rulemaking.yaml")


def test_other_policy_paths_keep_their_mapping():
    assert evals._source_identifier_to_relative_rulespec_path(
        "us/policy/cms/eligibility-levels"
    ) == Path("policies/cms/eligibility-levels.yaml")


def test_validator_normalizes_rulemaking_target_to_output_file():
    expected = (
        "policies",
        "rulemaking",
        "federal-register",
        "2026-02-25",
        "2026-03829",
    )
    assert validator_pipeline._corpus_citation_to_normalized_target(FR_CITATION) == (
        expected
    )
    assert (
        validator_pipeline._rulespec_file_normalized_target(
            Path("/repo/rulespec-us") / FR_OUTPUT
        )
        == expected
    )
    assert validator_pipeline._rulespec_base_parts_for_corpus_path(FR_CITATION) == (
        expected
    )


@pytest.mark.parametrize(
    "citation", ("us/policy/rulemaking.2024", "us-tn/policies/rulemaking.fr.x")
)
def test_validator_refuses_dotted_policy_leaf_in_rulemaking_namespace(citation):
    with pytest.raises(ValueError, match="rulemaking"):
        evals._source_identifier_to_relative_rulespec_path(citation)
    assert validator_pipeline._corpus_citation_to_normalized_target(citation) is None


def test_validator_refuses_policy_path_under_rulemaking_namespace():
    assert (
        validator_pipeline._corpus_citation_to_normalized_target(
            "us/policy/rulemaking/federal-register/2026-03829"
        )
        is None
    )


def test_cli_reverse_maps_rulemaking_output_to_corpus_path(tmp_path):
    repo = _rules_repo(tmp_path, "us")
    assert (
        _relative_output_to_child_corpus_citation_path(FR_OUTPUT, rules_repo_path=repo)
        == FR_CITATION
    )
    assert (
        _relative_output_to_child_corpus_citation_path(
            Path("us") / FR_OUTPUT, rules_repo_path=repo
        )
        == FR_CITATION
    )


def test_cli_reverse_mapping_of_plain_policy_output_is_unchanged(tmp_path):
    repo = _rules_repo(tmp_path, "us")
    assert (
        _relative_output_to_child_corpus_citation_path(
            Path("policies/cms/eligibility-levels.yaml"), rules_repo_path=repo
        )
        == "us/policy/cms/eligibility-levels"
    )


def test_dependency_stub_keeps_rulemaking_citation_verbatim(tmp_path):
    repo = _rules_repo(tmp_path, "us")
    assert dependency_stubs._candidate_corpus_paths_for_import_target(
        FR_CITATION, repo
    ) == (FR_CITATION,)


def test_dependency_stub_maps_rulemaking_import_target_to_corpus_path(tmp_path):
    # Before this change the fallback produced us/policy/rulemaking/...
    repo = _rules_repo(tmp_path, "us")
    assert dependency_stubs._candidate_corpus_paths_for_import_target(
        "us:policies/rulemaking/federal-register/2026-02-25/2026-03829#rate", repo
    ) == (FR_CITATION,)
    assert dependency_stubs._candidate_corpus_paths_for_import_target(
        "policies/rulemaking/federal-register/2026-02-25/2026-03829.yaml", repo
    ) == (FR_CITATION,)


@given(jurisdiction=_JURISDICTION, tail=_TAIL)
def test_rulemaking_forward_and_reverse_mappings_round_trip(jurisdiction, tail):
    citation = "/".join((jurisdiction, "rulemaking", *tail))
    output = evals._resolve_eval_output_path(citation)
    assert output == Path("policies", "rulemaking", *tail[:-1], f"{tail[-1]}.yaml")
    assert validator_pipeline._corpus_citation_to_normalized_target(citation) in (
        None,  # the validator matcher only covers us and us-* jurisdictions
        validator_pipeline._rulespec_file_normalized_target(Path("/r") / output),
    )
    if jurisdiction == "us" or jurisdiction.startswith("us-"):
        assert validator_pipeline._corpus_citation_to_normalized_target(
            citation
        ) == validator_pipeline._rulespec_file_normalized_target(Path("/r") / output)
    assert validator_pipeline._rulespec_base_parts_for_corpus_path(citation) == (
        validator_pipeline._rulespec_relative_path_parts(output)
    )
    assert (
        _relative_output_to_child_corpus_citation_path(
            Path(jurisdiction) / output, rules_repo_path=Path("rulespec-us")
        )
        == citation
    )
    assert dependency_stubs._candidate_corpus_paths_for_import_target(
        f"{jurisdiction}:{output.with_suffix('').as_posix()}", Path("rulespec-us")
    ) == (citation,)


@given(jurisdiction=_JURISDICTION, tail=_TAIL)
def test_policy_paths_outside_rulemaking_namespace_never_reverse_to_rulemaking(
    jurisdiction, tail
):
    if tail[0] == "rulemaking":
        tail = ["x", *tail]
    citation = "/".join((jurisdiction, "policy", *tail))
    output = evals._resolve_eval_output_path(citation)
    reverse = _relative_output_to_child_corpus_citation_path(
        Path(jurisdiction) / output, rules_repo_path=Path("rulespec-us")
    )
    assert reverse == citation


def test_completeness_target_base_uses_rulemaking_output_namespace():
    assert source_completeness._rulespec_target_base(FR_CITATION) == (
        "us:policies/rulemaking/federal-register/2026-02-25/2026-03829"
    )
    assert source_completeness._deferral_anchor_bases(FR_CITATION)[0] == (
        "us:policies/rulemaking/federal-register/2026-02-25/2026-03829"
    )


def test_repair_overlay_keeps_deferral_at_rulemaking_output_root(tmp_path):
    # Before the fix the overlay rewrote this correct deferral to
    # us:rulemakings/..., a root no adapter writes.
    citation = "us/rulemaking/fr/2026-03829"
    output = "us:policies/rulemaking/fr/2026-03829/a#x"
    normalized, repairs = evals._normalize_repair_deferred_source_roots(
        [{"output": output}],
        rulespec_file=tmp_path / "policies/rulemaking/fr/2026-03829.yaml",
        artifact_root=tmp_path,
        corpus_citation_path=citation,
    )
    assert normalized == [{"output": output}]
    assert repairs == []


def test_resolver_maps_rulespec_rulemaking_target_back_to_corpus_path():
    assert (
        corpus_resolver.normalize_corpus_identifier(
            "us:policies/rulemaking/federal-register/2026-02-25/2026-03829"
        )
        == FR_CITATION
    )
    assert corpus_resolver.normalize_corpus_identifier(FR_CITATION) == FR_CITATION
    assert (
        corpus_resolver.normalize_corpus_identifier("us:policies/cms/levels")
        == "us/policy/cms/levels"
    )


def test_validator_splits_dotted_rulemaking_leaf_like_the_encoder():
    citation = "us/rulemaking/fr/eo-14257.1"
    output = evals._resolve_eval_output_path(citation)
    assert output == Path("policies/rulemaking/fr/eo-14257/1.yaml")
    assert validator_pipeline._corpus_citation_to_normalized_target(
        citation
    ) == validator_pipeline._rulespec_file_normalized_target(Path("/r") / output)


@given(
    jurisdiction=st.sampled_from(["us", "us-ca"]),
    tail=_TAIL,
    leaf=st.from_regex(r"[a-z0-9]{1,6}(?:\.[a-z0-9]{1,4}){1,2}", fullmatch=True),
)
def test_validator_matches_encoder_output_for_dotted_leaves(jurisdiction, tail, leaf):
    citation = "/".join((jurisdiction, "rulemaking", *tail, leaf))
    output = evals._resolve_eval_output_path(citation)
    assert validator_pipeline._corpus_citation_to_normalized_target(
        citation
    ) == validator_pipeline._rulespec_file_normalized_target(Path("/r") / output)
    assert source_completeness._rulespec_target_base(citation).startswith(
        f"{jurisdiction}:policies/rulemaking/"
    )


@pytest.mark.parametrize(
    "source_id",
    (
        "us:guidance/rulemaking/fr/x",
        "us:forms/rulemaking/fr/x",
        "us:manuals/rulemaking/fr/x",
        "guidance/rulemaking/fr/x",
        "forms/rulemaking/fr/x",
        "us/form/rulemaking/fr/x",
        "us/guidance/rulemaking/fr/x",
        # A dotted leaf nests like a path segment.
        "forms/rulemaking.2024-1234",
        "us:guidance/rulemaking.2024-1234",
        "us/form/rulemaking.2024-1234",
        "us:manuals/rulemaking.fr.2024",
        "us:policy/rulemaking.x",
        "us/policy/rulemaking.x.y",
        # Mixed case is the same directory on a case-insensitive filesystem.
        "us:forms/Rulemaking/x",
        "us/policy/RULEMAKING/fr/x",
    ),
)
def test_non_rulemaking_ids_never_land_in_rulemaking_namespace(source_id):
    with pytest.raises(ValueError, match="rulemaking"):
        evals._source_identifier_to_relative_rulespec_path(source_id)


@pytest.mark.parametrize(
    "source_id",
    (
        "us:policies/rulemaking/fr/x",
        "us:policy/rulemaking/fr/x",
        "policies/rulemaking/fr/x",
    ),
)
def test_rulespec_form_of_rulemaking_output_maps_to_itself(source_id):
    assert evals._source_identifier_to_relative_rulespec_path(source_id) == Path(
        "policies/rulemaking/fr/x.yaml"
    )


def test_single_segment_rulemaking_tail_in_other_roots_is_a_sibling_file():
    assert evals._source_identifier_to_relative_rulespec_path(
        "us:guidance/rulemaking"
    ) == Path("policies/rulemaking.yaml")


_NON_RULEMAKING_ROOT = st.sampled_from(
    ["form", "forms", "guidance", "manual", "manuals", "policy", "policies"]
)
_LEAF_WITH_RULEMAKING = st.builds(
    lambda head, rest: head + rest,
    st.sampled_from(["rulemaking", "Rulemaking", "RULEMAKING", "rulemaking."]),
    st.from_regex(r"[a-z0-9.\-]{0,8}", fullmatch=True),
)


@given(
    jurisdiction=_JURISDICTION,
    root=_NON_RULEMAKING_ROOT,
    head=_LEAF_WITH_RULEMAKING,
    tail=st.lists(_SEGMENT, max_size=3),
    form=st.sampled_from(["corpus", "colon", "bare"]),
)
def test_only_rulemaking_sources_land_in_rulemaking_namespace(
    jurisdiction, root, head, tail, form
):
    rest = "/".join((head, *tail))
    source_id = {
        "corpus": f"{jurisdiction}/{root}/{rest}",
        "colon": f"{jurisdiction}:{root}/{rest}",
        "bare": f"{root}/{rest}",
    }[form]
    try:
        output = evals._source_identifier_to_relative_rulespec_path(source_id)
    except ValueError:
        return
    in_namespace = (
        len(output.parts) >= 3
        and output.parts[0] == "policies"
        and output.parts[1].casefold() == "rulemaking"
    )
    if in_namespace:
        # Only the output's own RuleSpec name may map there.
        assert form != "corpus"
        assert root in {"policies", "policy"}
        assert rest.split("/", 1)[0] == "rulemaking"


@given(
    jurisdiction=_JURISDICTION,
    tail=_TAIL,
    leaf=st.from_regex(r"[a-z0-9]{1,6}(?:\.[a-z0-9]{1,4}){1,2}", fullmatch=True),
)
def test_dotted_rulemaking_leaf_reverses_to_its_nested_form(jurisdiction, tail, leaf):
    # The round trip is the identity only for dot-free tails. A dotted leaf is
    # written nested (eo-14257.1 -> eo-14257/1.yaml), as main already does for
    # policy paths, and both reverse mappers return that nested path.
    citation = "/".join((jurisdiction, "rulemaking", *tail, leaf))
    nested = "/".join((jurisdiction, "rulemaking", *tail, *leaf.split(".")))
    output = evals._resolve_eval_output_path(citation)

    assert output == Path(
        "policies",
        "rulemaking",
        *tail,
        *leaf.split(".")[:-1],
        f"{leaf.split('.')[-1]}.yaml",
    )
    assert (
        _relative_output_to_child_corpus_citation_path(
            Path(jurisdiction) / output, rules_repo_path=Path("rulespec-us")
        )
        == nested
    )
    assert dependency_stubs._candidate_corpus_paths_for_import_target(
        f"{jurisdiction}:{output.with_suffix('').as_posix()}", Path("rulespec-us")
    ) == (nested,)


def test_dotted_rulemaking_leaf_example():
    output = evals._resolve_eval_output_path("us/rulemaking/fr/eo-14257.1")
    assert output == Path("policies/rulemaking/fr/eo-14257/1.yaml")
    assert (
        _relative_output_to_child_corpus_citation_path(
            Path("us") / output, rules_repo_path=Path("rulespec-us")
        )
        == "us/rulemaking/fr/eo-14257/1"
    )


@pytest.mark.parametrize(
    "import_target",
    ("us:rulemaking/fr/2026-03829", "rulemaking/fr/2026-03829"),
)
def test_dependency_stub_maps_rulemaking_head_to_rulemaking_kind(
    tmp_path, import_target
):
    # Without the rulemaking head this fell back to us/policy/rulemaking/...,
    # the path the encoder refuses.
    repo = _rules_repo(tmp_path, "us")
    assert dependency_stubs._candidate_corpus_paths_for_import_target(
        import_target, repo
    ) == ("us/rulemaking/fr/2026-03829",)


@pytest.mark.parametrize(
    "citation",
    (
        "us/policy/rulemaking/fr/x",
        # The plural class token.
        "us/policies/rulemaking/fr/x",
        # A mixed-case namespace segment.
        "us/policy/RULEMAKING/fr/x",
        "us-tn/policy/Rulemaking/fr/x",
    ),
)
def test_refused_policy_source_has_no_coverage_base_in_rulemaking_namespace(
    citation,
):
    with pytest.raises(ValueError):
        evals._source_identifier_to_relative_rulespec_path(citation)
    assert validator_pipeline._rulespec_base_parts_for_corpus_path(citation) == ()
    assert validator_pipeline._rulespec_base_parts_for_corpus_path(
        "us/policy/cms/levels"
    ) == ("policies", "cms", "levels")


def _single_eval_kwargs(tmp_path, citation, **overrides):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    source_unit = SimpleNamespace(
        requested=citation,
        citation_path=citation,
        body="Federal Register notice text",
        amendment_request=False,
    )
    kwargs = dict(
        citation=citation,
        runner=MagicMock(),
        output_root=tmp_path / "out",
        policy_path=tmp_path / "rulespec-us" / "us",
        runtime_axiom_rules_path=tmp_path / "engine",
        corpus_release=MagicMock(),
        mode="repo-augmented",
        extra_context_paths=[],
        source_unit=source_unit,
    )
    kwargs.update(overrides)
    return kwargs


class _StopAfterOutputPath(Exception):
    pass


def _patched_single_eval_preamble(monkeypatch):
    monkeypatch.setattr(evals, "_validate_corpus_source_unit", lambda *a, **k: None)
    monkeypatch.setattr(evals, "_source_metadata_with_attestation", lambda *a, **k: {})

    def stop(*_args, **_kwargs):
        raise _StopAfterOutputPath

    monkeypatch.setattr(evals, "_eval_review_contract_manifest_payload", stop)


def test_eval_run_refuses_federal_register_target_without_explicit_output(
    tmp_path, monkeypatch
):
    _patched_single_eval_preamble(monkeypatch)
    with pytest.raises(ValueError, match="has no default RuleSpec path"):
        evals._run_single_eval(**_single_eval_kwargs(tmp_path, FR_CITATION))


def test_eval_run_accepts_federal_register_target_with_explicit_output(
    tmp_path, monkeypatch
):
    _patched_single_eval_preamble(monkeypatch)
    with pytest.raises(_StopAfterOutputPath):
        evals._run_single_eval(
            **_single_eval_kwargs(
                tmp_path,
                FR_CITATION,
                target_relative_output=Path(
                    "policies/cbp/de-minimis/postal-flat-rate.yaml"
                ),
            )
        )


@pytest.mark.parametrize(
    "citation",
    ("us/statute/26/1/j/2", "us/regulation/7/273/4", "us/policy/cms/levels"),
)
def test_eval_run_keeps_default_paths_for_other_classes(
    tmp_path, monkeypatch, citation
):
    _patched_single_eval_preamble(monkeypatch)
    with pytest.raises(_StopAfterOutputPath):
        evals._run_single_eval(**_single_eval_kwargs(tmp_path, citation))


@pytest.mark.parametrize(
    ("identifier", "expected"),
    (
        (FR_CITATION, True),
        ("us-ca/rulemaking/notice/1", True),
        ("us/statute/26/1", False),
        ("us/regulation/7/273/4", False),
        ("us/policy/rulemaking/fr/x", False),
        ("26 USC 1", False),
        # RuleSpec-form ids name a module path, not a Federal Register source.
        ("us:statutes/rulemaking/fr/x", False),
        ("statutes/rulemaking/fr/x", False),
        # The predicate reads the corpus form only; the eval backstop catches
        # the document's RuleSpec names separately (tested below).
        ("policies/rulemaking/fr/x", False),
    ),
)
def test_is_rulemaking_source_identifier(identifier, expected):
    assert evals.is_rulemaking_source_identifier(identifier) is expected


def test_statute_and_regulation_default_paths_match_main():
    assert evals._resolve_eval_output_path("us/statute/26/1/j/2") == Path(
        "statutes/26/1/j/2.yaml"
    )
    assert evals._resolve_eval_output_path("us/regulation/7/273/4") == Path(
        "regulations/7-cfr/273/4.yaml"
    )


@pytest.mark.parametrize(
    "citation", ("us:statutes/rulemaking/fr/x", "statutes/rulemaking/fr/x")
)
def test_rulespec_form_ids_keep_mains_mapping_and_pass_the_backstop(
    tmp_path, monkeypatch, citation
):
    # main maps both to statutes/rulemaking/fr/x.yaml; the eval backstop must
    # not refuse them as Federal Register sources.
    assert evals._resolve_eval_output_path(citation) == Path(
        "statutes/rulemaking/fr/x.yaml"
    )
    _patched_single_eval_preamble(monkeypatch)
    with pytest.raises(_StopAfterOutputPath):
        evals._run_single_eval(**_single_eval_kwargs(tmp_path, citation))


@pytest.mark.parametrize(
    "citation",
    (
        "us:policies/rulemaking/federal-register/2026-02-25/2026-03829",
        "policies/rulemaking/federal-register/2026-02-25/2026-03829",
        "us:policy/rulemaking/federal-register/2026-02-25/2026-03829",
    ),
)
def test_eval_run_refuses_rulespec_names_of_a_federal_register_document(
    tmp_path, monkeypatch, citation
):
    _patched_single_eval_preamble(monkeypatch)
    with pytest.raises(ValueError, match="has no default RuleSpec path"):
        evals._run_single_eval(**_single_eval_kwargs(tmp_path, citation))


def test_eval_run_refuses_when_the_resolved_source_is_a_federal_register_document(
    tmp_path, monkeypatch
):
    # Whatever alias the caller passed, the resolved source unit decides.
    _patched_single_eval_preamble(monkeypatch)
    kwargs = _single_eval_kwargs(tmp_path, "some-alias")
    kwargs["source_unit"].requested = FR_CITATION
    with pytest.raises(ValueError, match="has no default RuleSpec path"):
        evals._run_single_eval(**kwargs)


@pytest.mark.parametrize(
    "citation",
    (
        "us:policies/rulemaking/federal-register/2026-02-25/2026-03829",
        "policies/rulemaking/federal-register/2026-02-25/2026-03829",
    ),
)
def test_eval_backstop_hint_names_the_resolved_jurisdiction_for_aliases(
    tmp_path, monkeypatch, citation
):
    # The resolver turns the alias into the corpus document; the hint must name
    # that document's us/policies/ path, not one built from the alias.
    _patched_single_eval_preamble(monkeypatch)
    kwargs = _single_eval_kwargs(tmp_path, citation)
    kwargs["source_unit"].requested = FR_CITATION
    with pytest.raises(ValueError) as refused:
        evals._run_single_eval(**kwargs)

    message = str(refused.value)
    assert message.startswith(f"{FR_CITATION} is a Federal Register")
    assert "in us/policies/." in message
    assert "us/policies/<agency>/<subject>.yaml" in message
    assert "policies/policies/" not in message


def test_eval_backstop_catches_an_alias_only_normalization_resolves(
    tmp_path, monkeypatch
):
    # us:rulemaking/<doc> is neither a corpus path nor a RuleSpec name of the
    # document, and here the source unit still carries the raw alias, so only
    # the resolver-normalized citation identifies the Federal Register target.
    citation = "us:rulemaking/federal-register/2026-02-25/2026-03829"
    assert not evals.is_rulemaking_source_identifier(citation)
    assert not evals._names_rulemaking_output(citation)
    _patched_single_eval_preamble(monkeypatch)
    with pytest.raises(ValueError, match=f"^{FR_CITATION} is a Federal Register"):
        evals._run_single_eval(**_single_eval_kwargs(tmp_path, citation))
