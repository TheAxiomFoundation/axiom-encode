"""The encoder PR validation-effect declaration (#2094)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_validation_effect",
    Path(__file__).resolve().parents[1] / "scripts/check_validation_effect.py",
)
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)


@pytest.mark.parametrize(
    ("body", "effect"),
    [
        ("Fix a typo.\n\nValidation effect: none\n", "none"),
        ("Validation effect: `none`\n", "none"),
        (
            "validation EFFECT:  Loosen\nChanged checks: proof.source_excerpt\n",
            "loosen",
        ),
        ("Validation effect: tighten\nChanged checks: unknown\n", "tighten"),
        (
            "Validation effect: tighten\nChanged checks: a.b, c-d, e/f\n",
            "tighten",
        ),
    ],
)
def test_valid_declarations(body, effect):
    assert check.validation_effect_issues(body) == (effect, [])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (None, "exactly one line"),
        ("", "exactly one line"),
        ("Validation effect: none\nValidation effect: tighten\n", "found 2"),
        ("Validation effect: maybe\n", "unknown validation effect"),
        ("Validation effect: loosen\n", "Changed checks"),
        ("Validation effect: tighten\nChanged checks: \n", "Changed checks"),
        ("Validation effect: loosen\nChanged checks: a, , b\n", "check ids"),
        (
            "Validation effect: loosen\nChanged checks: a\nChanged checks: b\n",
            "exactly one 'Changed checks:'",
        ),
    ],
)
def test_invalid_declarations(body, message):
    _effect, issues = check.validation_effect_issues(body)
    assert issues and message in issues[0]


def test_main_reads_the_body_from_the_environment(monkeypatch, tmp_path, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("PR_BODY", "Validation effect: none\n")
    assert check.main() == 0
    assert "`none`" in summary.read_text()
    monkeypatch.setenv("PR_BODY", "no declaration")
    assert check.main() == 1
    assert "::error::" in capsys.readouterr().out


def test_template_declares_a_valid_default():
    template = Path(__file__).resolve().parents[1] / ".github/pull_request_template.md"
    assert check.validation_effect_issues(template.read_text()) == ("none", [])


def test_html_comments_are_not_declarations():
    template = (
        Path(__file__).resolve().parents[1] / ".github/pull_request_template.md"
    ).read_text()
    body = template.replace(
        "Validation effect: none", "Validation effect: loosen\nChanged checks: x.y"
    )
    assert check.validation_effect_issues(body) == ("loosen", [])
    assert check.validation_effect_issues("<!-- Validation effect: none -->")[1]
