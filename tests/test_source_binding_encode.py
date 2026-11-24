"""A non-apply encode binds the candidate before validation and review."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from axiom_encode.checks import ValidationIssue
from axiom_encode.harness import evals as evals_module
from axiom_encode.harness.evals import (
    EvalArtifactMetrics,
    EvalPromptResponse,
    ValidationRetryCandidate,
    run_model_eval,
)
from axiom_encode.source_verification_binding import SourceBindingError
from tests.test_evals import (
    _canonical_rulespec_content_root,
    _write_test_source_unit,
)

CITATION = "us-nc/statute/105/105-153.7"
TARGET = Path("policies/synthetic/unit.yaml")
MODULE = (
    "format: rulespec/v1\n"
    "module:\n"
    "  summary: |-\n"
    "    Synthetic text: with a colon.\n"
    "  source_verification:\n"
    f"    corpus_citation_path: {CITATION}\n"
    "rules: []\n"
)


def _metrics(issues=()) -> EvalArtifactMetrics:
    return EvalArtifactMetrics(
        compile_pass=not issues,
        compile_issues=list(issues),
        ci_pass=not issues,
        ci_issues=list(issues),
        embedded_source_present=False,
        grounded_numeric_count=0,
        ungrounded_numeric_count=0,
        grounding=[],
    )


def _encode(tmp_path, generated: str, **kwargs):
    """Run one eval whose model writes ``generated``; record what is validated."""
    corpus_release, source_unit = _write_test_source_unit(
        tmp_path, "Authoritative synthetic source.", citation_path=CITATION
    )
    policy_root = _canonical_rulespec_content_root(tmp_path, "us-nc")
    validated: list[bytes] = []

    def generate(**call):
        if kwargs.get("accept_valid_retry_candidate") and not validated:
            raise AssertionError("an accepted retained candidate needs no model")
        output_file = call["output_file"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(generated.encode())
        materialized = {output_file}
        if kwargs.get("repair_candidate_tests_only"):
            companion = output_file.with_name(output_file.stem + ".test.yaml")
            companion.write_text("[]\n")
            materialized.add(companion)
            return (
                EvalPromptResponse(text="generated", duration_ms=1),
                True,
                0,
                frozenset(materialized),
            )
        return (
            EvalPromptResponse(text="generated", duration_ms=1),
            True,
            0,
            frozenset({output_file}),
        )

    def evaluate(**call):
        raw = call["rulespec_file"].read_bytes()
        validated.append(raw)
        try:
            yaml.safe_load(raw)
        except yaml.YAMLError:
            return _metrics(["YAML parse failed"])
        return _metrics()

    with (
        patch(
            "axiom_encode.harness.evals._run_prompt_eval_with_empty_artifact_retry",
            side_effect=generate,
        ),
        patch("axiom_encode.harness.evals.evaluate_artifact", side_effect=evaluate),
    ):
        result = run_model_eval(
            citations=[CITATION],
            runner_specs=["openai:model-a"],
            output_root=tmp_path / "out",
            policy_path=policy_root,
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=corpus_release,
            mode="repo-augmented",
            target_relative_output=TARGET,
            **kwargs,
        )[0]
    return result, validated, source_unit.source_attestation["source_sha256"]


def test_the_validated_bytes_are_the_bound_bytes(tmp_path, capsys):
    result, validated, pin = _encode(tmp_path, MODULE, bind_source_verification=True)
    bound = MODULE.replace(
        f"    corpus_citation_path: {CITATION}\n",
        f"    corpus_citation_path: {CITATION}\n    source_sha256: {pin}\n",
    ).encode()
    assert validated == [bound]
    assert Path(result.output_file).read_bytes() == bound
    assert result.success is True
    assert result.generated_output_sha256 == hashlib.sha256(bound).hexdigest()
    assert result.source_attestation["source_sha256"] == pin
    assert "source_binding_edit=insert_source_sha256" in capsys.readouterr().out


def test_without_the_flag_the_model_bytes_are_untouched(tmp_path):
    result, validated, _pin = _encode(tmp_path, MODULE)
    assert validated == [MODULE.encode()]
    assert Path(result.output_file).read_bytes() == MODULE.encode()
    assert result.success is True


def test_a_module_the_model_bound_itself_is_not_edited(tmp_path, capsys):
    _result, _validated, pin = _encode(tmp_path / "probe", MODULE)
    bound = MODULE.replace("rules: []\n", "").replace(
        f"    corpus_citation_path: {CITATION}\n",
        f"    source_sha256: {pin}\n    corpus_citation_path: {CITATION}\n",
    )
    result, validated, _pin = _encode(tmp_path, bound, bind_source_verification=True)
    assert validated == [bound.encode()]
    assert result.success is True
    assert "source_binding" not in capsys.readouterr().out.split("probe")[-1]


@pytest.mark.parametrize(
    "block, kind",
    [
        (
            f"    corpus_citation_path: {CITATION}\n    source_sha256: {'0f' * 32}\n",
            "conflict",
        ),
        ("    corpus_citation_path: us-nc/statute/105/other\n", "conflict"),
        (f"    <<: {{corpus_citation_path: {CITATION}}}\n", "unsupported_shape"),
    ],
)
def test_a_candidate_that_cannot_be_bound_is_not_clean(tmp_path, capsys, block, kind):
    generated = MODULE.replace(f"    corpus_citation_path: {CITATION}\n", block)
    result, validated, _pin = _encode(
        tmp_path, generated, bind_source_verification=True
    )
    # Nothing is rewritten: the refused bytes are the model's bytes.
    assert validated == [generated.encode()]
    assert Path(result.output_file).read_bytes() == generated.encode()
    assert result.success is False
    assert result.error == "Generated RuleSpec failed CI validation"
    assert result.metrics.ci_pass is False
    [issue] = result.metrics.ci_issues
    assert isinstance(issue, ValidationIssue)
    assert issue.check == "source-binding/unbound"
    assert issue.found == kind
    assert issue.startswith("[source-binding] ")
    assert f"source_binding_refused={kind}" in capsys.readouterr().out


def test_flow_style_is_refused_and_never_redumped(tmp_path):
    generated = (
        "format: rulespec/v1\n"
        f"module: {{source_verification: {{corpus_citation_path: {CITATION}}}}}\n"
    )
    result, validated, _pin = _encode(
        tmp_path, generated, bind_source_verification=True
    )
    assert validated == [generated.encode()]
    assert result.success is False
    assert "flow-style" in result.metrics.ci_issues[0]


def test_a_repaired_file_is_bound_before_it_is_evaluated_again(tmp_path):
    """Round 1 cannot bind unparseable YAML; the repair makes it bindable."""
    broken = MODULE.replace("rules: []\n", "rules: [\n")
    rounds: list[bytes] = []

    def repair(*, rulespec_file, **_kwargs):
        if rulespec_file.read_bytes() == broken.encode():
            rulespec_file.write_bytes(MODULE.encode())
            return ["synthetic_repair"]
        return []

    with patch(
        "axiom_encode.harness.evals._apply_generated_eval_repairs", side_effect=repair
    ):
        result, rounds, pin = _encode(tmp_path, broken, bind_source_verification=True)
    assert rounds[0] == broken.encode()
    assert f"    source_sha256: {pin}\n".encode() in rounds[-1]
    assert rounds[-1] == Path(result.output_file).read_bytes()
    assert result.success is True
    assert result.metrics.ci_issues == []


def test_an_unparseable_candidate_gets_the_binding_issue_too(tmp_path):
    result, _validated, _pin = _encode(
        tmp_path, "module: [unclosed\n", bind_source_verification=True
    )
    assert result.success is False
    assert any(
        getattr(issue, "check", None) == "source-binding/unbound"
        and issue.found == "invalid_input"
        for issue in result.metrics.ci_issues
    )


def _bound(pin: str) -> str:
    return MODULE.replace(
        f"    corpus_citation_path: {CITATION}\n",
        f"    corpus_citation_path: {CITATION}\n    source_sha256: {pin}\n",
    )


def test_a_retained_candidate_is_bound_before_its_preflight(tmp_path):
    """No generation: the retained candidate is bound, validated and accepted."""
    _result, _validated, pin = _encode(tmp_path / "probe", MODULE)
    result, validated, _pin = _encode(
        tmp_path,
        "the model must not be called",
        bind_source_verification=True,
        validation_retry_candidate=ValidationRetryCandidate(
            rulespec=MODULE, tests="[]\n"
        ),
        accept_valid_retry_candidate=True,
    )
    assert validated == [_bound(pin).encode()]
    assert Path(result.output_file).read_bytes() == _bound(pin).encode()
    assert result.success is True


def test_a_retained_candidate_that_names_another_source_is_not_accepted(tmp_path):
    """The preflight rejects it; the fresh generation is bound instead."""
    _result, _validated, pin = _encode(tmp_path / "probe", MODULE)
    retained = _bound("0f" * 32)
    result, validated, _pin = _encode(
        tmp_path,
        MODULE,
        bind_source_verification=True,
        validation_retry_candidate=ValidationRetryCandidate(
            rulespec=retained, tests="[]\n"
        ),
        accept_valid_retry_candidate=True,
    )
    assert validated[0] == retained.encode()
    # The retry overlay re-serializes the fresh output (as before this change);
    # the binder then pins that file, and those are the validated bytes.
    final = Path(result.output_file).read_bytes()
    assert validated[-1] == final
    assert yaml.safe_load(final)["module"]["source_verification"] == {
        "corpus_citation_path": CITATION,
        "source_sha256": pin,
    }
    assert result.success is True


def test_a_tests_only_repair_checks_the_pin_and_keeps_the_rulespec(tmp_path):
    _result, _validated, pin = _encode(tmp_path / "probe", MODULE)
    for name, retained, success in (
        ("unbound", MODULE, False),
        ("bound", _bound(pin), True),
    ):
        result, validated, _pin = _encode(
            tmp_path / name,
            retained,
            bind_source_verification=True,
            validation_retry_candidate=ValidationRetryCandidate(
                rulespec=retained, tests="[]\n"
            ),
            repair_candidate_tests_only=True,
        )
        # The immutable RuleSpec is never edited, bound or not.
        assert validated == [retained.encode()]
        assert Path(result.output_file).read_bytes() == retained.encode()
        assert result.success is success
        if not success:
            [issue] = result.metrics.ci_issues
            assert issue.check == "source-binding/unbound"
            assert "immutable in a tests-only repair" in issue


def _staged(tmp_path, text: str) -> tuple[Path, Path]:
    root = tmp_path / "out"
    target = root / "runner" / "policies" / "unit.yaml"
    target.parent.mkdir(parents=True)
    target.write_bytes(text.encode())
    return root, target


ATTESTATION = {
    "requested_corpus_citation_path": CITATION,
    "resolved_corpus_citation_path": CITATION,
    "source_sha256": "ab" * 32,
}


def test_read_only_checks_and_never_writes(tmp_path):
    root, target = _staged(tmp_path, MODULE)
    refusal = evals_module._bind_generated_source_verification(
        target, artifact_root=root, attestation=ATTESTATION, read_only=True
    )
    assert isinstance(refusal, SourceBindingError)
    assert refusal.kind == "unbound"
    assert "immutable in a tests-only repair" in str(refusal)
    assert "bind-source-verification" in str(refusal)
    assert target.read_bytes() == MODULE.encode()

    assert (
        evals_module._bind_generated_source_verification(
            target, artifact_root=root, attestation=ATTESTATION, read_only=False
        )
        is None
    )
    bound = target.read_bytes()
    assert bound != MODULE.encode()
    assert (
        evals_module._bind_generated_source_verification(
            target, artifact_root=root, attestation=ATTESTATION, read_only=True
        )
        is None
    )
    assert target.read_bytes() == bound


def test_no_attestation_is_a_refusal_not_a_pass(tmp_path):
    root, target = _staged(tmp_path, MODULE)
    refusal = evals_module._bind_generated_source_verification(
        target, artifact_root=root, attestation=None, read_only=False
    )
    assert isinstance(refusal, SourceBindingError)
    assert "no resolver source attestation" in str(refusal)
    assert target.read_bytes() == MODULE.encode()


def test_the_binder_refuses_a_file_outside_the_output_root(tmp_path):
    root, _target = _staged(tmp_path, MODULE)
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text(MODULE)
    with pytest.raises(Exception) as caught:
        evals_module._bind_generated_source_verification(
            outside, artifact_root=root, attestation=ATTESTATION, read_only=False
        )
    assert not isinstance(caught.value, SourceBindingError)
    assert outside.read_text() == MODULE


def test_the_wrapper_binds_before_every_evaluation_and_folds_in_the_refusal(tmp_path):
    rulespec = tmp_path / "unit.yaml"
    rulespec.write_text(MODULE)
    order: list[str] = []
    refusal = SourceBindingError("synthetic refusal")
    refusal.kind = "conflict"

    def bind():
        order.append("bind")
        return refusal

    def evaluate(**_kwargs):
        order.append("evaluate")
        return _metrics()

    with patch("axiom_encode.harness.evals.evaluate_artifact", side_effect=evaluate):
        metrics = evals_module._evaluate_generated_artifact_with_repairs(
            rulespec_file=rulespec,
            policy_repo_root=tmp_path,
            axiom_rules_path=tmp_path,
            source_text="",
            local_corpus_release=None,
            allow_artifact_repairs=False,
            bind_source=bind,
        )
    assert order == ["bind", "evaluate"]
    assert metrics.ci_pass is False
    assert [issue.check for issue in metrics.ci_issues] == ["source-binding/unbound"]
    assert "(conflict): synthetic refusal" in metrics.ci_issues[0]


# --- prompt ---


def _prompt(tmp_path, source_metadata):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    source = root / "source.txt"
    source.write_text("The annual allowance is 100.\n")
    workspace = evals_module.EvalWorkspace(
        root, source, root / "manifest.json", source_metadata=source_metadata
    )
    return evals_module._build_eval_prompt(
        "uk/statute/2020/1/section/1",
        "cold",
        workspace,
        [],
        "statutes/2020/1/1.yaml",
        include_tests=True,
        runner_backend="openai",
    )


OPTIONAL_LINE = "  `corpus_citation_path` and optional `source_sha256` pin.\n"


def test_the_prompt_tells_the_model_to_write_the_pin_from_the_metadata(tmp_path):
    prompt = _prompt(
        tmp_path,
        {
            "source_attestation": {
                "requested_corpus_citation_path": "uk/statute/2020/1/section/1",
                "source_sha256": "ab" * 32,
            }
        },
    )
    assert OPTIONAL_LINE not in prompt
    assert (
        "  `corpus_citation_path` and the `source_sha256` pin. Write both, each\n"
        "  on its own line in plain block form, copied exactly from\n"
        "  `source_attestation` in `./source-metadata.json`:\n"
        "  `corpus_citation_path` is its `requested_corpus_citation_path` and\n"
        "  `source_sha256` is its `source_sha256`. The encoder refuses a module\n"
        "  that names another value.\n"
    ) in prompt


@pytest.mark.parametrize(
    "source_metadata",
    [
        None,
        {},
        {"source_attestation": {"requested_corpus_citation_path": "uk/x/1"}},
        {
            "source_attestation": {
                "requested_corpus_citation_path": "uk/x/1",
                "source_sha256": "not-a-digest",
            }
        },
    ],
)
def test_a_prompt_without_an_attested_pin_keeps_the_earlier_line(
    tmp_path, source_metadata
):
    prompt = _prompt(tmp_path, source_metadata)
    assert OPTIONAL_LINE in prompt
    assert "Write both, each" not in prompt


REQUIRED_LINES = (
    "  `corpus_citation_path` and the `source_sha256` pin. Write both, each\n"
    "  on its own line in plain block form, copied exactly from\n"
    "  `source_attestation` in `./source-metadata.json`:\n"
    "  `corpus_citation_path` is its `requested_corpus_citation_path` and\n"
    "  `source_sha256` is its `source_sha256`. The encoder refuses a module\n"
    "  that names another value.\n"
)


def test_the_two_prompts_differ_only_in_the_pin_requirement(tmp_path):
    metadata = {
        "source_attestation": {
            "requested_corpus_citation_path": "uk/statute/2020/1/section/1",
            "source_sha256": "ab" * 32,
        }
    }
    with_pin = _prompt(tmp_path, metadata)
    metadata["source_attestation"]["source_sha256"] = "absent"
    without_pin = _prompt(tmp_path / "again", metadata)
    assert with_pin.count(REQUIRED_LINES) == 1
    assert (
        with_pin.replace(REQUIRED_LINES, OPTIONAL_LINE).replace("ab" * 32, "absent")
        == without_pin
    )
