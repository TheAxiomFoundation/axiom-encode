"""Run the targeted re-encode's axiom-compose ref parser on real inputs."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STEP_NAME = "Parse RuleSpec-pinned axiom-compose reference"
COMPOSE_REF = "0123456789abcdef0123456789abcdef01234567"


def _parser_script() -> str:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == STEP_NAME
    )
    match = re.search(r"<<'PY'\n(.*?)\nPY\n", step["run"], re.DOTALL)
    assert match is not None
    return match.group(1)


def _parse(tmp_path: Path, toolchain: Path) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "parse_compose_ref.py"
    script.write_text(_parser_script(), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-I", str(script), str(toolchain)],
        capture_output=True,
        text=True,
        check=False,
    )


def _toolchain(tmp_path: Path) -> Path:
    axiom = tmp_path / "rulespec-xx" / ".axiom"
    axiom.mkdir(parents=True)
    return axiom / "workflow-toolchain.toml"


def test_missing_workflow_toolchain_pins_no_composer(tmp_path: Path) -> None:
    # rulespec-ca and rulespec-dk main carry only toolchain.toml.
    result = _parse(tmp_path, _toolchain(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout == "\n"


def test_missing_axiom_directory_pins_no_composer(tmp_path: Path) -> None:
    toolchain = tmp_path / "rulespec-xx" / ".axiom" / "workflow-toolchain.toml"

    result = _parse(tmp_path, toolchain)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "\n"


def test_toolchain_without_compose_ref_pins_no_composer(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path)
    toolchain.write_text(
        '[workflow_toolchain]\naxiom_encode_version = "0.2.1"\n', encoding="utf-8"
    )

    result = _parse(tmp_path, toolchain)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "\n"


def test_toolchain_with_compose_ref_pins_that_composer(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path)
    toolchain.write_text(
        f'[workflow_toolchain]\naxiom_compose_ref = "{COMPOSE_REF}"\n',
        encoding="utf-8",
    )

    result = _parse(tmp_path, toolchain)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{COMPOSE_REF}\n"


@pytest.mark.parametrize(
    "body",
    [
        '[workflow_toolchain]\naxiom_compose_ref = "main"\n',
        '[workflow_toolchain]\naxiom_compose_ref = "0123abc"\n',
        "[workflow_toolchain]\naxiom_compose_ref = 1\n",
        "[workflow_toolchain\n",
    ],
)
def test_present_but_invalid_toolchain_still_fails(tmp_path: Path, body: str) -> None:
    toolchain = _toolchain(tmp_path)
    toolchain.write_text(body, encoding="utf-8")

    result = _parse(tmp_path, toolchain)

    assert result.returncode != 0
    assert result.stdout == ""


def test_dangling_toolchain_link_still_fails(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path)
    toolchain.symlink_to(tmp_path / "absent.toml")

    result = _parse(tmp_path, toolchain)

    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


def test_toolchain_directory_still_fails(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path)
    toolchain.mkdir()

    result = _parse(tmp_path, toolchain)

    assert result.returncode != 0
    assert result.stdout == ""
