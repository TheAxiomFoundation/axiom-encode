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


def test_unedited_template_fails_until_the_author_chooses():
    template = Path(__file__).resolve().parents[1] / ".github/pull_request_template.md"
    assert check.validation_effect_issues(template.read_text())[1]


def test_html_comments_are_not_declarations():
    template = (
        Path(__file__).resolve().parents[1] / ".github/pull_request_template.md"
    ).read_text()
    body = template.replace(
        "Validation effect: <none|loosen|tighten>",
        "Validation effect: loosen\nChanged checks: x.y",
    )
    assert check.validation_effect_issues(body) == ("loosen", [])
    assert check.validation_effect_issues("<!-- Validation effect: none -->")[1]


def test_fenced_code_is_not_a_declaration():
    fenced = "```text\nValidation effect: none\n```\n"
    assert check.validation_effect_issues(fenced)[1]
    assert check.validation_effect_issues(
        "~~~\nValidation effect: none\n~~~\nValidation effect: tighten\n"
        "Changed checks: unknown\n"
    ) == ("tighten", [])
    # An unclosed fence hides the rest of the body, as GitHub renders it.
    assert check.validation_effect_issues("````\nValidation effect: none\n")[1]


@pytest.mark.parametrize("checks", ["none", "a.b, none", "NONE"])
def test_loosen_or_tighten_cannot_list_none_as_changed_checks(checks):
    _effect, issues = check.validation_effect_issues(
        f"Validation effect: loosen\nChanged checks: {checks}\n"
    )
    assert issues and "cannot list 'none'" in issues[0]


def test_workflow_runs_the_base_copy_without_pull_request_code():
    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/validation-effect.yml"
        ).read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"pull_request_target"}
    assert workflow["permissions"] == {"contents": "read"}
    steps = workflow["jobs"]["declared"]["steps"]
    checkout = steps[0]
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    run_step = steps[1]
    assert "github.event" not in run_step["run"]
    assert run_step["env"]["PR_BODY"] == "${{ github.event.pull_request.body }}"
