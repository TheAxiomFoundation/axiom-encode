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
BODY_FIELDS = "{name, target, enforcement, bypass_actors, conditions, rules}"
MOVE_FILTER = BODY_FIELDS + " | .rules[0].parameters.workflows[0].sha = $sha"
COMPARE_FILTER = BODY_FIELDS + ' | .rules[0].parameters.workflows[0].sha = "PIN"'


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


def test_the_pin_procedure_holds_the_filters_this_file_exercises():
    step = pin_step()
    assert step.count("'" + MOVE_FILTER + "' before.json > move.json") == 1
    assert step.count("'" + COMPARE_FILTER + "' before.json > before.cmp") == 1
    assert step.count("'" + COMPARE_FILTER + "' after.json > after.cmp") == 1
    assert "gh api orgs/TheAxiomFoundation/rulesets/<id> > before.json" in step
    assert (
        "gh api -X PUT orgs/TheAxiomFoundation/rulesets/<id> --input move.json" in step
    )
    assert "diff before.cmp after.cmp" in step
    # One PUT, of the body built from the stored ruleset, and no POST.
    assert step.count("-X PUT") == 1 and "-X POST" not in step
    assert "ruleset.json" not in step


def model_move(stored: dict, sha: str) -> dict:
    """The move filter, written out: six fields of the stored ruleset, one SHA."""
    body = {
        key: copy.deepcopy(stored[key])
        for key in (
            "name",
            "target",
            "enforcement",
            "bypass_actors",
            "conditions",
            "rules",
        )
    }
    body["rules"][0]["parameters"]["workflows"][0]["sha"] = sha
    return body


def jq(filter_text: str, value: dict, *arguments: str) -> dict:
    result = subprocess.run(
        ["jq", "-S", *arguments, filter_text],
        input=json.dumps(value),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("enforcement", ["active", "evaluate", "disabled"])
@pytest.mark.parametrize(
    "bypass", [[], [{"actor_id": 1, "actor_type": "Team", "bypass_mode": "always"}]]
)
def test_the_documented_move_keeps_enforcement_targeting_and_bypass(
    enforcement, bypass
):
    stored = stored_ruleset(enforcement) | {"bypass_actors": bypass}
    expected = model_move(stored, NEW_SHA)
    assert expected["enforcement"] == enforcement
    assert expected["bypass_actors"] == bypass
    assert expected["conditions"] == stored["conditions"]
    assert expected["target"] == stored["target"]
    assert expected["rules"][0]["parameters"]["workflows"][0] == {
        "repository_id": 1125775893,
        "path": ".github/workflows/content-bound-merge-guard.yml",
        "sha": NEW_SHA,
    }
    assert set(expected) == set(creation_body())
    if shutil.which("jq") is None:  # pragma: no cover - the model stands in
        return
    # Where jq exists, the documented filter itself is run on the fixture.
    moved = jq(MOVE_FILTER, stored, "--arg", "sha", NEW_SHA)
    assert moved == expected
    # The comparison of the procedure: nothing but the SHA differs.
    after = stored | moved | {"updated_at": "2026-11-22T00:00:00Z"}
    assert jq(COMPARE_FILTER, stored) == jq(COMPARE_FILTER, after)
    # And it does notice a ruleset that came back weaker.
    for weaker in (
        {"enforcement": "evaluate" if enforcement != "evaluate" else "disabled"},
        {"bypass_actors": [*bypass, {"actor_id": 9, "actor_type": "Integration"}]},
        {"conditions": stored["conditions"] | {"ref_name": {"include": []}}},
    ):
        assert jq(COMPARE_FILTER, stored) != jq(COMPARE_FILTER, after | weaker)


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
