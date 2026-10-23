"""Interpretation syntax and string types survive real apply attestation stamping."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from axiom_encode.cli import (
    _can_attempt_apply,
    _generated_rulespec_yaml_nonparser_issue,
    _stamp_generated_source_attestation_for_apply,
    _validate_generated_encoding_in_policy_overlay,
)
from axiom_encode.harness.proof_validator import validate_rulespec_proofs
from tests.test_cli import (
    _complete_source_attestation,
    _write_test_context_manifest,
)
from tests.test_module_interpretations import (
    CITATION,
    EXCERPT,
    RAW_RULES,
    RAW_SHADOWED_INTERPRETATIONS,
    _raw_payload,
    _raw_string_payload,
)

APPLY_CITATION = "us/statute/26/1"


def _candidate(tmp_path, content):
    output_root = tmp_path / "out"
    generated = output_root / "codex-test-model/statutes/26/1.yaml"
    generated.parent.mkdir(parents=True)
    generated.write_text(content.replace(CITATION, APPLY_CITATION))
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


@pytest.mark.parametrize("value", ["1e5", "-0o17", "+.5", "2e10"])
@pytest.mark.parametrize("style", ["quoted", "short-tag", "canonical-tag"])
def test_apply_overlay_preserves_interpretation_strings_with_real_stamping(
    tmp_path, value, style
):
    scalar = {
        "quoted": repr(value),
        "short-tag": "!!str " + value,
        "canonical-tag": "!<tag:yaml.org,2002:str> " + value,
    }[style]
    result, generated, output_root = _candidate(
        tmp_path, _raw_string_payload(("question",), scalar)
    )
    before = validate_rulespec_proofs(
        generated.read_text(), source_texts={APPLY_CITATION: EXCERPT}
    )
    assert before.passed, before.issues
    policy_repo = tmp_path / "rulespec-us/us"
    policy_repo.mkdir(parents=True)
    validated = []

    class MetadataPipeline:
        """Keep the shared proof gate real without requiring an engine build."""

        def __init__(self, **_kwargs):
            pass

        def validate(self, path, *, skip_reviewers):
            assert skip_reviewers
            content = path.read_text()
            proof = validate_rulespec_proofs(
                content, source_texts={APPLY_CITATION: EXCERPT}
            )
            assert proof.passed, proof.issues
            document = yaml.safe_load(content)
            assert document["module"]["interpretations"][0]["question"] == value
            assert document["module"]["source_verification"] == {
                "corpus_citation_path": APPLY_CITATION,
                "source_sha256": result.source_attestation["source_sha256"],
            }
            validated.append(path)
            return SimpleNamespace(all_passed=True, results={})

    # Stamping, context-manifest attestation binding, staging, and the proof
    # gate execute; only the unrelated compiler and validation receipt are seams.
    with (
        patch("axiom_encode.cli.ValidatorPipeline", MetadataPipeline),
        patch("axiom_encode.cli._record_successful_apply_validation") as record,
    ):
        ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
            result,
            output_root=output_root,
            policy_repo_path=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            local_corpus_release=MagicMock(),
            validate_dependents=False,
        )

    assert ok, issues
    assert issues == []
    assert supplemental == {}
    assert validated
    record.assert_called_once()
    after = validate_rulespec_proofs(
        generated.read_text(), source_texts={APPLY_CITATION: EXCERPT}
    )
    assert after.passed, after.issues


@pytest.mark.parametrize("field", ["id", "alternative-id"])
@pytest.mark.parametrize("value", ["1e5", "2e10"])
@pytest.mark.parametrize("style", ["quoted", "short-tag", "canonical-tag"])
def test_real_stamping_preserves_numeric_looking_interpretation_ids(
    tmp_path, field, value, style
):
    scalar = {
        "quoted": repr(value),
        "short-tag": "!!str " + value,
        "canonical-tag": "!<tag:yaml.org,2002:str> " + value,
    }[style]
    content = _raw_payload(
        **{"entry_id" if field == "id" else "alternative_id": scalar}
    )
    result, generated, _ = _candidate(tmp_path, content)

    _stamp_generated_source_attestation_for_apply(result, generated)

    after = validate_rulespec_proofs(
        generated.read_text(), source_texts={APPLY_CITATION: EXCERPT}
    )
    assert after.passed, after.issues


INVALID_RAW = [
    pytest.param(
        _raw_payload().replace(
            "  interpretations:\n", "  interpretations: []\n  interpretations:\n", 1
        ),
        "duplicate key `interpretations`",
        id="duplicate-interpretations",
    ),
    pytest.param(
        _raw_payload().replace(
            "      question:",
            "      question: First discarded question\n      question:",
            1,
        ),
        "duplicate key `question`",
        id="duplicate-question",
    ),
    *[
        pytest.param(
            _raw_string_payload(("question",), scalar),
            "unsupported YAML tag",
            id="tag-" + name,
        )
        for name, scalar in [
            ("exponent", "! 1e5"),
            ("octal", "! -0o17"),
            ("fraction", "! +.5"),
            ("prose", "! Ordinary prose"),
            ("quoted", "! '1e5'"),
        ]
    ],
    pytest.param(
        _raw_payload(entry_id="! weekly-review"),
        "unsupported YAML tag",
        id="tag-entry-id",
    ),
    pytest.param(
        _raw_payload(alternative_id="! daily-review"),
        "unsupported YAML tag",
        id="tag-alternative-id",
    ),
    pytest.param(
        _raw_string_payload(("question",), "1e5"),
        "must be a quoted string",
        id="plain-number",
    ),
    pytest.param(
        _raw_payload(entry_id="1e5"),
        "must be a quoted string",
        id="plain-numeric-id",
    ),
    pytest.param(
        "format: rulespec/v1\nx-defaults: &defaults\n"
        f"{RAW_SHADOWED_INTERPRETATIONS}module:\n  <<: *defaults\n{RAW_RULES}",
        "must not supply `interpretations` through YAML merge keys",
        id="module-merge",
    ),
    pytest.param(
        "format: rulespec/v1\nx-base: &base\n  module:\n"
        + "".join("  " + line for line in RAW_SHADOWED_INTERPRETATIONS.splitlines(True))
        + f"<<: *base\n{RAW_RULES}",
        "must not supply `module` through YAML merge keys",
        id="root-merge",
    ),
    pytest.param(
        _raw_payload().replace("  interpretations:", "  !!merge interpretations:", 1),
        "unsupported YAML tag `tag:yaml.org,2002:merge`",
        id="merge-tagged-interpretations-key",
    ),
    pytest.param(
        _raw_payload() + "module: {}\n",
        "duplicate key `module`",
        id="duplicate-root-module",
    ),
    pytest.param(
        _raw_payload().replace("module:\n", "module: !!set\n", 1),
        "module has unsupported YAML tag `tag:yaml.org,2002:set`",
        id="set-tagged-module",
    ),
    pytest.param(
        "--- !!set\n" + _raw_payload(),
        "RuleSpec root has unsupported YAML tag `tag:yaml.org,2002:set`",
        id="set-tagged-root",
    ),
]


@pytest.mark.parametrize(("content", "diagnostic"), INVALID_RAW)
def test_real_stamping_rejects_raw_interpretation_syntax_before_rewrite(
    tmp_path, content, diagnostic
):
    result, generated, _ = _candidate(tmp_path, content)
    original = generated.read_bytes()

    with pytest.raises(RuntimeError) as exc_info:
        _stamp_generated_source_attestation_for_apply(result, generated)

    assert diagnostic in str(exc_info.value)
    assert generated.read_bytes() == original


@pytest.mark.parametrize(("content", "diagnostic"), INVALID_RAW)
@pytest.mark.parametrize(
    "error",
    [
        "Generated RuleSpec failed compile validation",
        "Generated RuleSpec failed CI validation",
    ],
)
def test_apply_preflight_blocks_raw_syntax_before_repairs(
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
    assert diagnostic in issue
    assert generated.read_bytes() == original


@pytest.mark.parametrize(("content", "diagnostic"), INVALID_RAW)
def test_apply_overlay_blocks_invalid_raw_syntax_with_real_stamping(
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
    assert any(diagnostic in issue for issue in issues)
    assert supplemental == {}
    assert generated.read_bytes() == original
