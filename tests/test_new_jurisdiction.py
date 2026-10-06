from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from axiom_encode.ci_parity import find_caller_workflow, parse_caller_workflow
from axiom_encode.new_jurisdiction import run_new_jurisdiction

ORACLES = Path(__file__).parent / "fixtures/new_jurisdiction"


def _engine(tmp_path: Path, *currencies: str) -> Path:
    engine = tmp_path / "engine"
    registry = engine / "src/formula.rs"
    registry.parent.mkdir(parents=True)
    seeds = "\n".join(
        f'        ("{currency}", UnitKindSpec::Currency {{ minor_units: 2 }}),'
        for currency in currencies
    )
    registry.write_text(
        "fn seed() {\n    for (name, kind) in [\n"
        + seeds
        + '\n        ("count", UnitKindSpec::Count),\n'
        + "    ] {}\n}\n"
    )
    return engine


def _args(
    tmp_path: Path,
    engine: Path,
    *,
    cc: str = "dk",
    output: Path | None = None,
    plan: str | None = None,
    override: str | None = None,
    currency: str = "DKK",
    force: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        cc=cc,
        output=output or tmp_path / f"rulespec-{cc}",
        axiom_oracles_path=ORACLES,
        engine_path=engine,
        jurisdiction_name={"dk": "Denmark", "xx": "Exampleland"}.get(cc, cc),
        currency=currency,
        oracle_plan=plan,
        record_oracle_override=override,
        force=force,
    )


def test_denmark_pilot_regression_runnable_euromod_refuses_without_plan_or_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "dk"
    result = run_new_jurisdiction(
        _args(tmp_path, _engine(tmp_path, "DKK"), output=output)
    )
    captured = capsys.readouterr()
    assert result == 1
    assert "EUROMOD" in captured.err
    assert "ok_training_name" in captured.err
    assert "--oracle-plan" in captured.err
    assert "--record-oracle-override" in captured.err
    assert not output.exists()


def test_runnable_country_plan_scaffolds_and_embeds_readiness_evidence(
    tmp_path: Path,
) -> None:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("Compare the first encoded module against DK_2025.")
    output = tmp_path / "dk"
    assert (
        run_new_jurisdiction(
            _args(
                tmp_path, _engine(tmp_path, "DKK"), output=output, plan=str(plan_file)
            )
        )
        == 0
    )
    text = (output / ".axiom/oracle-plan.md").read_text()
    assert "EUROMOD" in text
    assert "ok_training_name" in text
    assert "axiom_oracles/data/euromod_country_readiness.json" in text
    assert "2026-07-02" in text
    assert "Compare the first encoded module against DK_2025." in text


def test_runnable_country_override_records_verbatim_reason(tmp_path: Path) -> None:
    reason = "Bundle access is awaiting maintainer approval; do not classify as absent."
    output = tmp_path / "dk"
    assert (
        run_new_jurisdiction(
            _args(tmp_path, _engine(tmp_path, "DKK"), output=output, override=reason)
        )
        == 0
    )
    text = (output / ".axiom/oracle-override.md").read_text()
    assert reason in text
    assert "RECORDED: <fill on commit>" in text
    assert "EUROMOD" in text and "ok_training_name" in text


def test_country_absent_from_euromod_and_southmod_scaffolds_status(
    tmp_path: Path,
) -> None:
    output = tmp_path / "xx"
    assert (
        run_new_jurisdiction(
            _args(
                tmp_path,
                _engine(tmp_path, "XXX"),
                cc="xx",
                output=output,
                currency="XXX",
            )
        )
        == 0
    )
    status = (output / ".axiom/oracle-status.md").read_text()
    assert "No runnable oracle was found" in status
    assert "country entry: absent" in status
    assert "no adapter found" in status


def test_scaffold_inventory_waiver_shape_and_caller_placeholder(tmp_path: Path) -> None:
    output = tmp_path / "xx"
    assert (
        run_new_jurisdiction(
            _args(
                tmp_path,
                _engine(tmp_path, "XXX"),
                cc="xx",
                output=output,
                currency="XXX",
            )
        )
        == 0
    )
    inventory = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    )
    assert inventory == [
        ".axiom/oracle-status.md",
        ".axiom/registry.toml",
        ".axiom/repository-structure.yaml",
        ".axiom/toolchain.toml",
        ".github/workflows/repository-checks.yml",
        ".gitignore",
        "CLAUDE.md",
        "README.md",
        "corpus-manifest-skeleton.yaml",
        "data/coverage/tax-benefit-source-map.json",
        "data/oracles/oracle-index.json",
        "known-missing-money-atoms.yaml",
        "known-validation-gaps.yaml",
        "oracle-coverage-pending.yaml",
        "tests/test_repository_layout.py",
        "xx/policies/.gitkeep",
        "xx/regulations/.gitkeep",
        "xx/statutes/.gitkeep",
    ]
    assert yaml.safe_load((output / "known-validation-gaps.yaml").read_text()) == {
        "validate_failures": {}
    }
    with pytest.raises(ValueError, match=r"placeholder pin <pin-me>"):
        parse_caller_workflow(output / ".github/workflows/repository-checks.yml")
    with pytest.raises(ValueError, match=r"placeholder pin <pin-me>"):
        find_caller_workflow(output)


def test_emitted_layout_tests_pass_on_fresh_scaffold(tmp_path: Path) -> None:
    output = tmp_path / "xx"
    assert (
        run_new_jurisdiction(
            _args(
                tmp_path,
                _engine(tmp_path, "XXX"),
                cc="xx",
                output=output,
                currency="XXX",
            )
        )
        == 0
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_repository_layout.py"],
        cwd=output,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_present_currency_writes_no_diff(tmp_path: Path) -> None:
    output = tmp_path / "xx"
    assert (
        run_new_jurisdiction(
            _args(
                tmp_path,
                _engine(tmp_path, "XXX"),
                cc="xx",
                output=output,
                currency="XXX",
            )
        )
        == 0
    )
    assert not (output / "engine-currency-seed.diff").exists()


def test_absent_currency_diff_applies_cleanly_to_engine_copy(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "USD")
    output = tmp_path / "xx"
    assert (
        run_new_jurisdiction(
            _args(tmp_path, engine, cc="xx", output=output, currency="XXX")
        )
        == 0
    )
    diff = output / "engine-currency-seed.diff"
    assert diff.is_file()
    engine_copy = tmp_path / "engine-copy"
    shutil.copytree(engine, engine_copy)
    result = subprocess.run(
        ["git", "apply", "--check", str(diff)],
        cwd=engine_copy,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_nonempty_output_refuses_without_force(tmp_path: Path) -> None:
    output = tmp_path / "xx"
    output.mkdir()
    (output / "keep.txt").write_text("user content")
    args = _args(
        tmp_path, _engine(tmp_path, "XXX"), cc="xx", output=output, currency="XXX"
    )
    assert run_new_jurisdiction(args) == 1
    assert list(output.iterdir()) == [output / "keep.txt"]
    args.force = True
    assert run_new_jurisdiction(args) == 0


@pytest.mark.parametrize("value", ["", "   \n\t"])
@pytest.mark.parametrize("argument", ["plan", "override"])
def test_empty_oracle_evidence_arguments_refuse(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    value: str,
) -> None:
    output = tmp_path / "dk"
    args = _args(tmp_path, _engine(tmp_path, "DKK"), output=output)
    setattr(
        args, "oracle_plan" if argument == "plan" else "record_oracle_override", value
    )

    assert run_new_jurisdiction(args) == 1
    assert (
        f"--{'oracle-plan' if argument == 'plan' else 'record-oracle-override'}"
        in capsys.readouterr().err
    )
    assert not output.exists()


def test_empty_oracle_plan_file_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = tmp_path / "empty-plan.md"
    plan.write_text(" \n")
    output = tmp_path / "dk"

    assert (
        run_new_jurisdiction(
            _args(tmp_path, _engine(tmp_path, "DKK"), output=output, plan=str(plan))
        )
        == 1
    )
    assert "--oracle-plan" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("precreate", [False, True])
def test_unparseable_engine_registry_is_atomic(tmp_path: Path, precreate: bool) -> None:
    engine = _engine(tmp_path, "USD")
    (engine / "src/formula.rs").write_text("fn seed() { /* no insertion marker */ }\n")
    output = tmp_path / "xx"
    if precreate:
        output.mkdir()

    assert (
        run_new_jurisdiction(
            _args(tmp_path, engine, cc="xx", output=output, currency="XXX")
        )
        == 1
    )
    assert (
        (output.is_dir() and not list(output.iterdir()))
        if precreate
        else not output.exists()
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"countries": []},
        {"countries": {"DK": "ok_training_name"}},
    ],
)
def test_malformed_readiness_shapes_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: object,
) -> None:
    oracles = tmp_path / "oracles"
    matrix = oracles / "axiom_oracles/data/euromod_country_readiness.json"
    matrix.parent.mkdir(parents=True)
    matrix.write_text(json.dumps(payload))
    args = _args(tmp_path, _engine(tmp_path, "DKK"))
    args.axiom_oracles_path = oracles

    assert run_new_jurisdiction(args) == 1
    assert "malformed EUROMOD readiness matrix" in capsys.readouterr().err
    assert not args.output.exists()


def test_gh_shaped_southmod_mapping_refuses_without_plan_or_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oracles = tmp_path / "oracles"
    matrix = oracles / "axiom_oracles/data/euromod_country_readiness.json"
    matrix.parent.mkdir(parents=True)
    matrix.write_text(json.dumps({"countries": {}}))
    mapping = oracles / "axiom_oracles/bridges/mappings/gh.yaml"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(
        "prefixes:\n  - legal_id_prefix: 'gh:'\n    country: gh\n"
        "    rationale: The intended household oracle is GHAMOD (UNU-WIDER SOUTHMOD).\n"
    )
    output = tmp_path / "gh"
    args = _args(
        tmp_path, _engine(tmp_path, "GHS"), cc="gh", output=output, currency="GHS"
    )
    args.axiom_oracles_path = oracles

    assert run_new_jurisdiction(args) == 1
    captured = capsys.readouterr()
    assert "SOUTHMOD" in captured.err
    assert not output.exists()
