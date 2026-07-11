from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline as _ValidatorPipeline,
)


class ValidatorPipeline(_ValidatorPipeline):
    """Test convenience wrapper for a corpus-free static import check."""

    def __init__(self, *args, local_corpus_release=None, **kwargs):
        super().__init__(
            *args,
            local_corpus_release=local_corpus_release,
            **kwargs,
        )


def _repo_roots() -> tuple[Path, Path]:
    foundation_root = Path(__file__).resolve().parents[2]
    policy_repo_path = foundation_root / "rulespec-us"
    axiom_rules_path = foundation_root / "axiom-rules-engine"
    if not policy_repo_path.exists():
        pytest.skip("rulespec-us repo not present")
    if not axiom_rules_path.exists():
        pytest.skip("axiom-rules-engine repo not present")
    return policy_repo_path, axiom_rules_path


def test_repo_cross_statute_definitions_are_imported():
    policy_repo_path, axiom_rules_path = _repo_roots()
    statutes_root = policy_repo_path / "statutes"
    if not statutes_root.exists():
        pytest.skip("rulespec-us/statutes repo not present")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
    )

    failures: list[str] = []
    for rulespec_file in sorted(statutes_root.rglob("*.yaml")):
        if rulespec_file.name.endswith(".test.yaml"):
            continue
        issues = pipeline._check_cross_statute_definition_imports(rulespec_file)
        if not issues:
            continue
        relative = rulespec_file.relative_to(policy_repo_path.parent)
        for issue in issues:
            failures.append(f"{relative}: {issue}")

    assert not failures, "Cross-statute definition import failures:\n" + "\n".join(
        failures
    )
