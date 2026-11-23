"""Doc pins for the review of head ``dd8bf4fa2`` of the recommendation.

Each test follows a documented procedure literally, as the review did, or
pins the corrected statement. The documents are the recommendation
(``docs/merge-guard-key-protection.md``), the experiment runbook
(``docs/merge-guard-ruleset-experiment.md``) and the design
(``docs/content-bound-merge-guard.md``). Nothing here touches GitHub.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEY_PROTECTION = "docs/merge-guard-key-protection.md"
EXPERIMENT = "docs/merge-guard-ruleset-experiment.md"
DESIGN = "docs/content-bound-merge-guard.md"
NEW_SHA = "b" * 40
OLD_SHA = "a" * 40


def text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def flat(name: str) -> str:
    return " ".join(text(name).replace("\n> ", "\n").replace("\n>\n", "\n").split())


def creation_body() -> dict:
    (block,) = re.findall(r"```json\n(.*?)```", text(KEY_PROTECTION), re.S)
    return json.loads(block)


def pin_step() -> str:
    document = text(KEY_PROTECTION)
    return document[
        document.index("6. Move the pin by") : document.index(
            "7. Open pull requests keep the result"
        )
    ]


def stored_ruleset(enforcement="active") -> dict:
    """What ``GET /orgs/<org>/rulesets/<id>`` returns after checklist step 11."""
    body = creation_body()
    body["rules"][0]["parameters"]["workflows"][0]["sha"] = OLD_SHA
    return body | {
        "id": 4242,
        "enforcement": enforcement,
        "source_type": "Organization",
        "source": "TheAxiomFoundation",
        "node_id": "RRS_x",
        "created_at": "2026-11-20T00:00:00Z",
        "updated_at": "2026-11-21T00:00:00Z",
        "current_user_can_bypass": "never",
        "_links": {"self": {"href": "https://api.github.com/x"}},
    }


# --- 1. a routine pin move must not turn enforcement off ----------------------


def test_the_review_probe_the_creation_body_with_a_new_sha_says_evaluate():
    # The review parsed the JSON of "Create the ruleset" and changed only
    # the SHA: the body still sets ``evaluate``. That is right for the
    # creation and is the reason a pin move must never reuse the body.
    body = creation_body()
    body["rules"][0]["parameters"]["workflows"][0]["sha"] = NEW_SHA
    assert body["enforcement"] == "evaluate"
    step = " ".join(pin_step().split())
    assert "**updating the stored ruleset**" in step
    assert (
        '**Never send the creation body of "Create the ruleset", step 3, for a '
        "pin move**" in step
    )
    assert "`PUT` the full body of step 3" not in flat(KEY_PROTECTION)
    assert "Verify as in step 4 above." not in step
    assert "**`active` from checklist step 11 on**" in step


def pin_script() -> str:
    """The shell lines of the pin procedure, exactly as the document has them."""
    (block,) = re.findall(r"```sh\n(.*?)```", pin_step(), re.S)
    return "\n".join(
        line[3:] if line.startswith("   ") else line for line in block.splitlines()
    )


GH_STAND_IN = r"""#!/usr/bin/env bash
# A stand-in for `gh api`: GET prints the stored ruleset, PUT records the body
# it is given and answers as GitHub would (the stored ruleset with that body).
set -euo pipefail
if [ "$1" != "api" ]; then exit 64; fi
shift
if [ "$1" = "-X" ]; then
  [ "$2" = "PUT" ] || exit 65
  [ "$3" = "orgs/TheAxiomFoundation/rulesets/4242" ] || exit 66
  [ "$4" = "--input" ] || exit 67
  cp "$5" "$LAB/put.json"
  jq -s '.[0] * .[1] * (if env.ANSWER_ENFORCEMENT != "" then {enforcement: env.ANSWER_ENFORCEMENT} else {} end)' "$LAB/stored.json" "$5"
else
  [ "$1" = "orgs/TheAxiomFoundation/rulesets/4242" ] || exit 68
  cat "$LAB/stored.json"
fi
"""

needs_shell = pytest.mark.skipif(
    shutil.which("jq") is None or shutil.which("bash") is None,
    reason="runs the documented shell lines; needs bash and jq",
)


def run_pin_script(tmp_path, stored: dict, answer_enforcement: str = ""):
    (tmp_path / "bin").mkdir()
    gh = tmp_path / "bin" / "gh"
    gh.write_text(GH_STAND_IN, encoding="utf-8")
    gh.chmod(0o755)
    (tmp_path / "stored.json").write_text(json.dumps(stored), encoding="utf-8")
    script = (
        pin_script().replace("<id>", "4242").replace("<new 40-hex commit SHA>", NEW_SHA)
    )
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=work,
        env={
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "LAB": str(tmp_path),
            "ANSWER_ENFORCEMENT": answer_enforcement,
        },
        capture_output=True,
        text=True,
    )
    return result, work, tmp_path / "put.json"


def test_the_pin_procedure_is_one_script_that_stops_at_the_first_failure():
    script = pin_script()
    lines = script.splitlines()
    assert lines[0] == "set -euo pipefail"
    assert sum("-X PUT" in line for line in lines) == 1 and "-X POST" not in script
    put = next(i for i, line in enumerate(lines) if "-X PUT" in line)
    check = next(i for i, line in enumerate(lines) if line.startswith("jq -e"))
    # The check of the stored ruleset comes before anything is built or sent.
    assert check < put and "move.json" not in "\n".join(lines[: check + 1])
    assert '(.bypass_actors | type) == "array"' in lines[check]
    assert "(.rules | length) == 1" in lines[check]
    assert '.rules[0].type == "workflows"' in lines[check]
    assert "(.rules[0].parameters.workflows | length) == 1" in lines[check]
    assert "ruleset.json" not in script and ";" not in script and "||" not in script
    assert (
        "BODY='{name, target, enforcement, bypass_actors, conditions, rules}'" in script
    )


@needs_shell
@pytest.mark.parametrize("enforcement", ["active", "evaluate", "disabled"])
def test_the_documented_script_moves_the_pin_and_keeps_everything_else(
    tmp_path, enforcement
):
    stored = stored_ruleset(enforcement)
    result, _work, put = run_pin_script(tmp_path, stored)
    assert result.returncode == 0, result.stderr
    sent = json.loads(put.read_text())
    for key in ("name", "target", "enforcement", "bypass_actors", "conditions"):
        assert sent[key] == stored[key], key
    assert sent["enforcement"] == enforcement
    assert set(sent) == set(creation_body())
    (rule,) = sent["rules"]
    assert rule["parameters"]["workflows"] == [
        {
            "repository_id": 1125775893,
            "path": ".github/workflows/content-bound-merge-guard.yml",
            "sha": NEW_SHA,
        }
    ]
    # The last line the owner reads: the enforcement and the new pin.
    assert result.stdout.strip().splitlines()[-1] == f"{enforcement}\t{NEW_SHA}"


MERGE_QUEUE_RULE = {"type": "merge_queue", "parameters": {"merge_method": "SQUASH"}}


def refused_rulesets():
    plain = stored_ruleset()
    workflows_rule = plain["rules"][0]
    two_entries = copy.deepcopy(workflows_rule)
    two_entries["parameters"]["workflows"].append(
        two_entries["parameters"]["workflows"][0] | {"path": "x.yml"}
    )
    return {
        "a merge_queue rule first (the review's case)": plain
        | {"rules": [MERGE_QUEUE_RULE, workflows_rule]},
        "a merge_queue rule second": plain
        | {"rules": [workflows_rule, MERGE_QUEUE_RULE]},
        "only a merge_queue rule": plain | {"rules": [MERGE_QUEUE_RULE]},
        "no rule": plain | {"rules": []},
        "two workflow entries": plain | {"rules": [two_entries]},
        "no workflow entry": plain
        | {"rules": [{"type": "workflows", "parameters": {"workflows": []}}]},
        "bypass_actors is null (a read without the right)": plain
        | {"bypass_actors": None},
        "bypass_actors is absent": {
            key: value for key, value in plain.items() if key != "bypass_actors"
        },
        "enforcement is absent": {
            key: value for key, value in plain.items() if key != "enforcement"
        },
    }


@needs_shell
@pytest.mark.parametrize("label", sorted(refused_rulesets()))
def test_the_documented_script_sends_nothing_for_a_ruleset_it_cannot_move(
    tmp_path, label
):
    result, work, put = run_pin_script(tmp_path, refused_rulesets()[label])
    assert result.returncode != 0
    assert not put.exists(), "no PUT was sent"
    assert not (work / "move.json").exists(), "no body was written"
    assert not (work / "move.json.tmp").exists()


@needs_shell
@pytest.mark.parametrize("answer", ["evaluate", "disabled"])
def test_the_documented_script_fails_when_the_ruleset_comes_back_weaker(
    tmp_path, answer
):
    result, _work, put = run_pin_script(tmp_path, stored_ruleset("active"), answer)
    assert put.exists()
    assert result.returncode != 0
    # ``diff`` printed the difference, and the last line was never reached.
    assert "enforcement" in result.stdout
    assert f"{answer}\t{NEW_SHA}" not in result.stdout


def jq(filter_text: str, value: dict, *arguments: str) -> dict:
    result = subprocess.run(
        ["jq", "-S", *arguments, filter_text],
        input=json.dumps(value),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_the_lab_pin_move_uses_the_same_procedure():
    runbook = text(EXPERIMENT)
    step = " ".join(
        runbook[
            runbook.index("### X19. A pin move with an open green pull request") :
        ].split()
    )
    step = step[: step.index("- **What the result changes.**")]
    assert "`PUT` the full body with the new `sha`" not in step
    assert "by updating the stored ruleset, exactly as the pin procedure does" in step
    assert "Do not send the creation body of §1.3 again" in step
    assert "the enforcement must still be `active`" in step
    assert "the conditions and the bypass list must be unchanged" in step
    assert "**Any other enforcement after the move fails this step**" in step


# --- 2. the rollback shortcut -------------------------------------------------

STALE = {
    DESIGN: [
        "or make sure that no open pull request of `rulespec-us` holds a signed "
        "v6 manifest (close each one)",
        "So the second form of (iii) is enough",
    ],
    KEY_PROTECTION: [
        "or, as the weaker form, make sure immediately before the next act that "
        "no open pull request",
        "except inside an environment with required reviewers",
        "moves it into an environment with required reviewers",
        "**The exception is required reviewers only**",
    ],
    EXPERIMENT: ["`PUT` the full body with the new `sha`"],
}
REQUIRED = {
    DESIGN: [
        "**This is required whenever a signed v6 manifest is outstanding or its "
        "disposition is unknown.**",
        "Closing pull requests is not a substitute",
        "a closed pull request leaves the signed bytes on its branch",
        "One run that cannot be accounted for makes the disposition unknown, and "
        "then the pin is reverted.",
        # 3. credentials
        "- **An environment is no general exception** (review of head `dd8bf4fa2`).",
        "shows `can_admins_bypass: false`, required reviewers with "
        "`prevent_self_review: true`",
        # 4. health checks
        "The record holds each name **with its `updated_at`**",
        "A read is whole only when one page was printed and `listed` equals "
        "`total_count`",
        "the **effective list of target repositories**",
        "An incomplete or refused read is not a record: the owner verifies it by hand.",
        # 6. acceptance
        "**The ruleset, read back, is not the evidence of an owner's decision**",
        "(1) **the decision**, recorded independently of the ruleset",
        "(2) **actor evidence for each privileged change**",
        # queue settings (coordination note)
        "**Four settings of the `merge_queue` rule are required, not optional, "
        "and are read back:**",
    ],
    KEY_PROTECTION: [
        "This step is required whenever a signed v6 manifest is outstanding or "
        "its disposition is unknown**",
        "An environment is no general exception**",
        "must show `can_admins_bypass: false`, a `required_reviewers` rule with "
        "`prevent_self_review: true`",
        "**The existing token is revoked in every case, also when a workflow "
        "still needs one**",
        "moving the same credential elsewhere leaves every copy that was already "
        "read valid",
    ],
    EXPERIMENT: [
        "**and the record shows all three of these on the same group commit SHA**",
        "is **not a pass and not a fail: the part did not test the question; run "
        "it again**",
        "A pass with a `repository_id` condition says nothing about the "
        "name-based configuration and does not count.",
        "**a refused rename is valid evidence**, together with proof that the "
        "guard still applies",
        "A refused rename without that second observation is not a pass.",
        '`grouping_strategy: "ALLGREEN"`, `max_entries_to_merge: 1`, '
        "`min_entries_to_merge: 1` and `max_entries_to_build: 1`",
    ],
}


# Pins added after the independent review of ab311944e, which found nine doc
# mutants that no pin caught, and for the corrections of that round.
REQUIRED[DESIGN] += [
    '`grouping_strategy: "ALLGREEN"`, `max_entries_to_merge: 1`, '
    "`min_entries_to_merge: 1`, `max_entries_to_build: 1`.",
    "{grouping_strategy, max_entries_to_merge, min_entries_to_merge, "
    "max_entries_to_build}",
    "any other value, or a missing field, stops the activation",
    "Max states it himself, outside any agent session, to the second maintainer",
    "the deployment branch policy lists `main` only",
    "Leaving the revert out, which (iii) allows only with a complete inventory, "
    "leaves that gap open",
    "[.secrets[] | {name, visibility, updated_at}]",
    "**A residual that only the owner's experiment can settle** [inferred, untested]",
]
REQUIRED[KEY_PROTECTION] += [
    "on an open pull request, on a closed one, or on any branch",
    'an inventory with one unknown entry is "unknown"',
    'must print `["main"]`',
    "**nothing is written and nothing is sent unless the check before it passed**",
    '`null` means "could not read", not "empty"',
    "which stops new v6 signing; allowlisted and eligible earlier-lane (v5) "
    "changes still pass, and only a guard that cannot pass blocks every merge "
    "(§4.4, R13).",
    "that stops new v6 signing, while allowlisted and eligible v5 changes still "
    "pass [code: `SUCCESS_VERDICTS`]; if the guard cannot pass, it blocks every "
    "merge, which is the fail-closed state.",
    "(required whenever a signed v6 manifest is outstanding or unaccounted for, "
    "on an open or closed pull request or on any branch)",
    "[.secrets[] | {name, visibility, updated_at}]",
]
REQUIRED[EXPERIMENT] += [
    "the imitation's `merge_group` job completed with `success`",
    "the ordinary check `lab-slow` completed with `success`",
    "failed for the dismissed approval",
    "**Any other enforcement after the move fails this step**",
    "- **A re-run after the guard job ended (record only; it has no pass rule "
    "and no class of its own).**",
]
STALE[DESIGN] += ['The form "no open pull request holds a signed v6 manifest"']
STALE[KEY_PROTECTION] += [
    "(or, as the weaker form, no open pull request",
    "which blocks every merge (§4.4, R13)",
    "it then blocks every merge",
]


def problems(name: str, document: str) -> list[str]:
    found = [
        f"{name}: lost or doubled {phrase!r}"
        for phrase in REQUIRED[name]
        if document.count(phrase) != 1
    ]
    found += [
        f"{name}: the corrected text is back: {phrase!r}"
        for phrase in STALE[name]
        if phrase in document
    ]
    return found


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_each_document_states_the_corrections_once_and_not_the_defects(name):
    assert problems(name, flat(name)) == []


@pytest.mark.parametrize(
    ("name", "phrase"),
    [(name, phrase) for name in sorted(REQUIRED) for phrase in REQUIRED[name]],
)
def test_the_pin_reports_each_lost_correction(name, phrase):
    document = flat(name)
    for changed in (document.replace(phrase, ""), document + " " + phrase):
        found = problems(name, changed)
        assert len(found) == 1 and repr(phrase) in found[0]


@pytest.mark.parametrize(
    ("name", "phrase"),
    [(name, phrase) for name in sorted(STALE) for phrase in STALE[name]],
)
def test_the_pin_reports_each_defect_that_comes_back(name, phrase):
    found = problems(name, flat(name) + " " + phrase)
    assert len(found) == 1 and repr(phrase) in found[0]


# --- 4. the health-check projections see what they promise --------------------


def commands(name: str, start: str) -> list[str]:
    return [
        command.replace("\\|", "|")
        for command in re.findall(r"`(gh api [^`]*)`", text(name))
        if command.startswith(start)
    ]


def test_every_read_of_secret_names_also_reads_the_dates():
    reads = [
        command
        for command in commands(DESIGN, "gh api ")
        if "/secrets" in command and "--jq" in command
    ]
    assert len(reads) >= 4
    for command in reads:
        if "orgs/TheAxiomFoundation/actions/secrets" in command:
            continue  # the owner's read of visibility, not a record of dates
        assert "{name, updated_at}" in command, command
        assert "--paginate" in command and "total_count" in command, command
    assert "--jq '[.secrets[].name]'" not in text(DESIGN)


@pytest.mark.skipif(shutil.which("jq") is None, reason="needs jq")
def test_the_secret_projection_differs_when_only_the_date_changes():
    # The review's fixture: one name, two dates. The old projection printed
    # ``["EXAMPLE"]`` for both.
    (filter_text,) = {
        re.search(r"--jq '(.*)'$", command).group(1)
        for command in commands(DESIGN, "gh api --paginate repos/")
        if "/secrets" in command
    }
    first = {"total_count": 1, "secrets": [{"name": "EXAMPLE", "updated_at": "1"}]}
    second = {"total_count": 1, "secrets": [{"name": "EXAMPLE", "updated_at": "2"}]}
    assert jq("[.secrets[].name]", first) == jq("[.secrets[].name]", second)
    assert jq(filter_text, first) != jq(filter_text, second)
    assert jq(filter_text, first)["listed"] == 1
    cut = {"total_count": 3, "secrets": first["secrets"]}
    printed = jq(filter_text, cut)
    assert printed["listed"] != printed["total_count"]


def test_the_app_inventory_records_the_target_repositories():
    reads = [
        command
        for command in commands(DESIGN, "gh api ")
        if "orgs/TheAxiomFoundation/installations" in command
    ]
    assert len(reads) == 2
    for command in reads:
        assert "--paginate" in command and "total_count" in command
        assert "{id, app_slug, repository_selection, updated_at, permissions}" in (
            command
        )
    assert "user/installations/<installation id>/repositories" in text(DESIGN).replace(
        "\n        ", ""
    )


# --- 5. the lab record names every part of X2 ---------------------------------


def x2_rows(runbook: str) -> list[str]:
    record = runbook[runbook.index("## 4. Record") : runbook.index("## 5. What")]
    return re.findall(r"^\| X2 \(([a-z])\) \|", record, re.M)


def test_the_record_has_each_part_of_x2_exactly_once():
    assert x2_rows(text(EXPERIMENT)) == ["a", "b", "c", "d", "e"]


def test_the_review_probe_a_second_x2_a_in_place_of_x2_c_is_reported():
    # The review replaced the X2 (c) label by a second X2 (a); the count of
    # five rows still held. The set of parts does not.
    runbook = text(EXPERIMENT).replace("| X2 (c) |", "| X2 (a) |")
    assert len(x2_rows(runbook)) == 5
    assert x2_rows(runbook) != ["a", "b", "c", "d", "e"]
