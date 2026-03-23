from pathlib import Path

import pytest

from autorac.harness.validator_pipeline import ValidatorPipeline


def _repo_roots() -> tuple[Path, Path]:
    foundation_root = Path(__file__).resolve().parents[2]
    rac_us_path = foundation_root / "rac-us" / "statute"
    rac_path = foundation_root / "rac"
    if not rac_us_path.exists():
        pytest.skip("rac-us/statute repo not present")
    if not rac_path.exists():
        pytest.skip("rac repo not present")
    return rac_us_path, rac_path


def test_repo_cross_statute_definitions_are_imported():
    rac_us_path, rac_path = _repo_roots()
    pipeline = ValidatorPipeline(
        rac_us_path=rac_us_path,
        rac_path=rac_path,
        enable_oracles=False,
    )

    failures: list[str] = []
    for rac_file in sorted(rac_us_path.rglob("*.rac")):
        issues = pipeline._check_cross_statute_definition_imports(rac_file)
        if not issues:
            continue
        relative = rac_file.relative_to(rac_us_path.parent)
        for issue in issues:
            failures.append(f"{relative}: {issue}")

    assert not failures, "Cross-statute definition import failures:\n" + "\n".join(
        failures
    )
