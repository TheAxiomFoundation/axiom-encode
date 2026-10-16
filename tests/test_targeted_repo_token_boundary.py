import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# These are the complete external-command and builtin sets used by the token
# steps today. Keep additions explicit: installed CLI entrypoints also execute
# repository Python and must not run with the publication token.
REPO_TOKEN_EXTERNAL_COMMANDS = {"cp", "gh", "git", "jq", "tr"}
REPO_TOKEN_BUILTINS = {"[", "echo", "exit", "printf", "set", "test"}
REPO_TOKEN_COMMANDS = REPO_TOKEN_EXTERNAL_COMMANDS | REPO_TOKEN_BUILTINS


class _ShellToken(NamedTuple):
    value: str
    plain: bool = True
    assignment: bool = False


def _repo_token_commands(script: str) -> set[str]:
    """Inspect the limited shell grammar used here without executing the script.

    Inspect substitutions recursively, preserve quoted data, and reject shell
    syntax outside this grammar rather than guessing where a command starts.
    In particular, command names cannot come from expansion or quoted data.
    """
    commands: set[str] = set()
    offset = 0

    def check_tokens(tokens: list[_ShellToken]) -> None:
        command_start = True
        redirect_target = False
        for token in tokens:
            if token.plain and token.value in {"\n", ";", "&&", "||", "|"}:
                command_start = True
            elif token.plain and token.value in {"<", ">", ">>", ">&"}:
                redirect_target = True
            elif redirect_target:
                redirect_target = False
            elif command_start:
                if token.plain and token.value in {"if", "then", "else", "fi", "!"}:
                    continue
                if token.assignment:
                    continue
                assert token.plain and token.value in REPO_TOKEN_COMMANDS, (
                    f"command is not allowed with the repository token: {token.value!r}"
                )
                commands.add(token.value)
                command_start = False
        assert not redirect_target, "missing redirection target"

    def word() -> _ShellToken:
        nonlocal offset
        assignment = re.match(r"[A-Za-z_][A-Za-z_0-9]*=", script[offset:]) is not None
        value = ""
        plain = True
        quote = ""
        while offset < len(script):
            char = script[offset]
            if not quote and (char.isspace() or char in ";&|<>()"):
                break
            offset += 1
            if char == quote:
                quote = ""
            elif quote == "'":
                value += char
            elif char in {"'", '"'} and not quote:
                quote = char
                plain = False
            elif char == "\\":
                assert offset < len(script), "unfinished shell escape"
                escaped = script[offset]
                offset += 1
                if escaped != "\n":
                    value += escaped
                    plain = False
            elif char == "`":
                raise AssertionError(
                    "backtick execution is outside the allowed grammar"
                )
            elif char == "$":
                plain = False
                if script.startswith("(", offset):
                    offset += 1
                    check_tokens(tokens(substitution=True))
                else:
                    # Only simple variable references and literal defaults occur
                    # in these scripts. Refuse complex parameter expressions,
                    # including substitutions hidden in a default value.
                    variable = re.match(
                        r"(?:[A-Za-z_][A-Za-z_0-9]*|"
                        r"\{[A-Za-z_][A-Za-z_0-9]*(?::-(?:n/a|\[\]|ad-hoc))?\})",
                        script[offset:],
                    )
                    assert variable is not None, "unsupported shell expansion"
                    offset += len(variable[0])
                value += "$EXPANSION"
            else:
                value += char
        assert not quote, "unfinished shell quote"
        return _ShellToken(value, plain, assignment)

    def tokens(*, substitution: bool = False) -> list[_ShellToken]:
        nonlocal offset
        result: list[_ShellToken] = []
        while offset < len(script):
            char = script[offset]
            if char in " \t\r":
                offset += 1
            elif script.startswith("\\\n", offset):
                offset += 2
            elif char == "#":
                while offset < len(script) and script[offset] != "\n":
                    offset += 1
            elif char == ")" and substitution:
                offset += 1
                return result
            elif char in "()`":
                raise AssertionError(f"unsupported shell execution syntax: {char!r}")
            elif char in "\n;&|<>":
                operator = re.match(r"(?:&&|\|\||>>|>&|[\n;|<>])", script[offset:])
                assert operator is not None, "unsupported shell operator"
                value = operator[0]
                offset += len(value)
                # Here-documents, process substitution, and other redirects need
                # their own parser rules before they can enter this allowlist.
                assert not (value == "<" and script[offset : offset + 1] in {"<", "("})
                result.append(_ShellToken(value))
            else:
                result.append(word())
        assert not substitution, "unfinished command substitution"
        return result

    check_tokens(tokens())
    return commands


def _steps() -> list[dict]:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    return workflow["jobs"]["encode"]["steps"]


def test_repo_token_steps_never_invoke_python_helpers() -> None:
    observed = set()
    for step in _steps():
        if not any(
            "AXIOM_REPO_TOKEN" in str(value).upper()
            for value in step.get("env", {}).values()
        ):
            continue
        try:
            observed.update(_repo_token_commands(step.get("run", "")))
        except AssertionError as error:
            raise AssertionError(f"{step['name']}: {error}") from error
    assert observed == REPO_TOKEN_COMMANDS


@pytest.mark.parametrize(
    "secret_reference",
    [
        "${{ secrets.AXIOM_REPO_TOKEN }}",
        "${{ secrets.axiom_repo_token }}",
        "${{ SeCrEtS.AxIoM_RePo_ToKeN }}",
        "${{ secrets['AXIOM_REPO_TOKEN'] }}",
        "${{ secrets['axiom_repo_token'] }}",
        "${{ SeCrEtS['AxIoM_RePo_ToKeN'] }}",
    ],
)
def test_repo_token_detection_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch, secret_reference: str
) -> None:
    step = {
        "name": "Regression publication",
        "env": {"GH_TOKEN": secret_reference},
        "run": "/opt/axiom-verification/axiom-encode guard-generated --repo .",
    }
    monkeypatch.setattr(f"{__name__}._steps", lambda: [step])
    with pytest.raises(
        AssertionError,
        match="Regression publication: command is not allowed with the repository token",
    ):
        test_repo_token_steps_never_invoke_python_helpers()


@pytest.mark.parametrize(
    "script",
    [
        "/opt/axiom-verification/axiom-encode guard-generated --repo .",
        "python -c pass",
        "uv run axiom-encode guard-generated",
        "ruby -e 'puts ENV'",
        "git status; /opt/axiom-verification/axiom-encode guard-generated --repo .",
        'test -n "$(/opt/axiom-verification/axiom-encode guard-generated)"',
        'result="$(git status; /opt/axiom-verification/axiom-encode guard-generated)"',
        "git status | /opt/axiom-verification/axiom-encode guard-generated",
        'printf "%s" "${VALUE:-$(python -c pass)}"',
        'printf "%s" "${VALUE:-n/a}"; /opt/axiom-verification/axiom-encode guard-generated; printf "%s" "${OTHER:-n/a}"',
        'printf "%s" `python -c pass`',
        "git status > >(python -c pass)",
        '"$HELPER" guard-generated',
        "command /opt/axiom-verification/axiom-encode guard-generated",
        "env /opt/axiom-verification/axiom-encode guard-generated",
        'eval "$HELPER"',
        'source "$HELPER"',
        "git() { python -c pass; }; git",
        '"if" python -c pass',
        '"IGNORED"=value python -c pass',
    ],
)
def test_repo_token_command_allowlist_rejects_other_execution(script: str) -> None:
    with pytest.raises(AssertionError):
        _repo_token_commands(script)


@pytest.mark.parametrize(
    ("script", "commands"),
    [
        ('set -euo pipefail\ntest -n "$GH_TOKEN"', {"set", "test"}),
        ('branch="$PREPARED_BRANCH"\ngit push "$branch"', {"git"}),
        ('test "$(git rev-parse HEAD)" = "$RULESPEC_REF"', {"test", "git"}),
        ("body=\"$(printf '%s\\n' 'python; uv run; `data`')\"", {"printf"}),
        (
            'if [ -z "$REF" ]; then\n echo "disabled" >&2\n exit 0\nfi',
            {"[", "echo", "exit"},
        ),
        (
            "if ! jq -e '(.value | length) > 0' \"$FILE\"; then\n gh api repo || echo failed\nfi",
            {"jq", "gh", "echo"},
        ),
        (
            '# python ignored\nvalue="$(tr -d \'\\n\' < "$FILE")"\ncp "$FILE" "$DEST"',
            {"tr", "cp"},
        ),
        ('git -C "$REPO" \\\n status > "$FILE"', {"git"}),
    ],
)
def test_repo_token_command_allowlist_accepts_used_shell_grammar(
    script: str, commands: set[str]
) -> None:
    assert _repo_token_commands(script) == commands


@pytest.mark.parametrize(
    ("helper", "consumer", "output"),
    [
        (
            "Parse RuleSpec-pinned axiom-compose reference",
            "Checkout RuleSpec-pinned axiom-compose",
            "${{ steps.axiom_compose_request.outputs.ref }}",
        ),
        (
            "Prepare reviewed lane pull request branch",
            "Push lane branch and open draft pull request",
            "${{ steps.reviewed_lane_pull_request_branch.outputs.branch }}",
        ),
        (
            "Prepare retired source metadata pull request branch",
            "Push retired source metadata migration branch and open draft pull request",
            "${{ steps.retired_source_metadata_pull_request_branch.outputs.branch }}",
        ),
    ],
)
def test_repo_token_helpers_run_earlier_without_secrets(
    helper: str, consumer: str, output: str
) -> None:
    steps = _steps()
    by_name = {step.get("name"): step for step in steps}
    preparation = by_name[helper]
    publication = by_name[consumer]
    assert steps.index(preparation) < steps.index(publication)
    assert "secrets." not in str(preparation.get("env", {}))
    assert "python" in preparation["run"]
    assert output in publication["env"].values()


def test_publication_preserves_verification_fetch_and_branch_order() -> None:
    steps = _steps()
    names = [step.get("name") for step in steps]
    for ordered_names in [
        [
            "Commit reviewed lane changes locally",
            "Verify reviewed lane pull request base",
            "Prepare reviewed lane pull request branch",
            "Push lane branch and open draft pull request",
        ],
        [
            "Commit retired source metadata migration locally",
            "Check retired source metadata publication token",
            "Verify retired source metadata migration before publication",
            "Verify retired source metadata pull request base",
            "Prepare retired source metadata pull request branch",
            "Push retired source metadata migration branch and open draft pull request",
        ],
    ]:
        indices = [names.index(name) for name in ordered_names]
        assert indices == sorted(indices)
        conditions = [steps[index]["if"] for index in indices[:-1]]
        assert len(set(conditions)) == 1
        assert conditions[0][4:-3] in steps[indices[-1]]["if"]
    verify = steps[
        names.index("Verify retired source metadata migration before publication")
    ]
    assert "verify-committed" in verify["run"]
    assert "secrets." not in str(verify["env"])


@pytest.mark.parametrize(
    ("publication_id", "preflight_id"),
    [
        ("publish_lane_pull_request", "verify_reviewed_lane_pull_request_base"),
        (
            "publish_retired_source_metadata_pull_request",
            "check_retired_source_metadata_publication_token",
        ),
    ],
)
def test_publication_reports_preparation_failure_without_publishing(
    publication_id: str, preflight_id: str
) -> None:
    by_id = {step.get("id"): step for step in _steps()}
    publication = by_id[publication_id]
    assert "!cancelled()" in publication["if"]
    assert f"steps.{preflight_id}.outcome != 'skipped'" in publication["if"]
    assert publication["env"]["PRIOR_STATUS"] == "${{ job.status }}"
    assert publication["run"].splitlines()[:2] == [
        "set -euo pipefail",
        'test "$PRIOR_STATUS" = success',
    ]
    # An attempted publication reports failure before it reads credentials or
    # reaches network commands when any separated preparation step failed.
    completed = subprocess.run(
        ["bash", "-c", publication["run"]],
        env={"PRIOR_STATUS": "failure"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == ""
