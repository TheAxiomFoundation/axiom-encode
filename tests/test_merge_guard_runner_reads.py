"""The merge guard runner: opaque tokens, incomplete reads, per-PR checks.

Regression and hostile cases for the review of head ``36bab7533``:

1. the runner refused a current GitHub App installation token (longer than
   512 characters);
2. an evaluation that read a truncated tree, or a listing past its page
   bound, was reported as a completed run;
3. two pull requests with one head commit overwrote each other's shadow
   check.

API payloads are fixtures: the ``World`` fake of ``test_content_merge_guard``
and the opener fakes of ``test_merge_guard_runner``. No test makes a network
call.
"""

from __future__ import annotations

import copy
import io
import json
import urllib.error

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode import merge_guard_runner as runner
from tests.test_content_merge_guard import (
    COMPANION,
    NAME,
    PAGE,
    PR,
    PRIMARY,
    ROOT,
    H,
    M,
    P,
    T,
    build,
    plain,
    v5_manifest,
)
from tests.test_merge_guard_runner import (
    CLIENT,
    OUR_APP,
    PULLS,
    SHADOW_LISTING,
    Exploding,
    Response,
    Writer,
    arguments,
    pulls,
    shadow_world,
    sweep,
)

# --- 1. the token is opaque ----------------------------------------------------

LEGACY_TOKEN = "ghs_" + "x" * 36  # 40 characters, the earlier format
# About 520 characters with two dots: the stateless installation token.
CURRENT_TOKEN = "ghs_" + "a" * 200 + "." + "b" * 200 + "." + "c" * 114
FOUR_KB_TOKEN = "ghs_" + "t" * (4096 - 4)
ACCEPTED_TOKENS = {
    "legacy-40": LEGACY_TOKEN,
    "current-520": CURRENT_TOKEN,
    "512": "g" * 512,
    "513": "g" * 513,
    "4-kb": FOUR_KB_TOKEN,
    "upper-bound": "u" * runner.MAX_TOKEN_CHARS,
    "lower-bound": "l" * runner.MIN_TOKEN_CHARS,
    "every-allowed-character": "AZaz09-._~+/=" * 4,
    "jwt-shaped": "eyJhbGciOiJ" + "A" * 300 + "." + "B" * 400 + "." + "C_-" * 100,
}
REFUSED_TOKENS = {
    "empty": "",
    "one-character": "x",
    "below-lower-bound": "l" * (runner.MIN_TOKEN_CHARS - 1),
    "above-upper-bound": "u" * (runner.MAX_TOKEN_CHARS + 1),
    "one-megabyte": "m" * (1024 * 1024),
    "inner-space": "ghs_" + "x" * 20 + " " + "x" * 20,
    "leading-space": " " + LEGACY_TOKEN,
    "trailing-space": LEGACY_TOKEN + " ",
    "trailing-newline": LEGACY_TOKEN + "\n",
    "trailing-carriage-return": LEGACY_TOKEN + "\r",
    "header-injection": LEGACY_TOKEN + "\r\nX-Injected: 1",
    "folded-header": LEGACY_TOKEN + "\n " + "x" * 20,
    "tab": LEGACY_TOKEN + "\t" + "x" * 20,
    "nul": LEGACY_TOKEN + "\x00",
    "escape": "\x1b[31m" + LEGACY_TOKEN,
    "delete": LEGACY_TOKEN + "\x7f",
    "vertical-tab": LEGACY_TOKEN + "\x0b",
    "non-breaking-space": LEGACY_TOKEN + "\u00a0" + "x" * 20,
    "latin-1-letter": "ghs_" + "é" * 40,
    "fullwidth-digits": "ghs_" + "１２３４" * 10,
    "arabic-indic-digits": "ghs_" + "٣" * 40,
    "cyrillic-lookalike": "ghs_" + "а" * 40,  # U+0430, not ASCII "a"
    "emoji": "ghs_" + "x" * 30 + "\U0001f511",
    "zero-width-joiner": "ghs_" + "x" * 20 + "\u200d" + "x" * 20,
    "line-separator": LEGACY_TOKEN + "\u2028",
    "bearer-prefix": "Bearer " + LEGACY_TOKEN,
    "comma": LEGACY_TOKEN + "," + LEGACY_TOKEN,
    "double-quote": '"' + LEGACY_TOKEN + '"',
    "colon": "user:" + LEGACY_TOKEN,
    "semicolon": LEGACY_TOKEN + ";x",
    "percent-encoded-newline": LEGACY_TOKEN + "%0d%0a",
    "backslash": LEGACY_TOKEN + "\\",
    "surrogate": LEGACY_TOKEN + "\udc80",
    "bytes": LEGACY_TOKEN.encode(),
    "none": None,
    "integer": 12345678901234567890123,
    "list": [LEGACY_TOKEN],
}


class Recording:
    """An opener fake that records each request and answers ``raw``."""

    def __init__(self, status=200, raw=b"{}"):
        self.status = status
        self.raw = raw
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return Response(self.status, self.raw)


class WorldOpener:
    """Answers reads from a ``World`` and requires one exact bearer token."""

    ORIGIN = "https://api.github.com"

    def __init__(self, world, token):
        self.world = world
        self.token = token
        self.paths = []

    def open(self, request, timeout):
        assert request.get_header("Authorization") == "Bearer " + self.token
        assert request.get_method() == "GET"
        path = request.full_url[len(self.ORIGIN) :]
        self.paths.append(path)
        try:
            value = self.world.fetch(path)
        except KeyError:
            raise urllib.error.HTTPError(
                "https://api.github.com/x", 404, "nf", {}, io.BytesIO(b"")
            ) from None
        return Response(200, json.dumps(value).encode())


def command_report(capsys, token, opener, **changes):
    environ = {} if token is None else {runner.TOKEN_ENV: token}
    code = runner.command(arguments(**changes), environ=environ, opener=opener)
    out = capsys.readouterr()
    return code, json.loads(out.out), out


def test_the_review_probe_a_520_character_token_is_refused_no_longer(capsys):
    # The reviewer's probe: a synthetic 520-character, two-dot ``ghs_`` token
    # gave "MERGE_GUARD_TOKEN is absent or malformed", exit 1, zero requests.
    assert len(CURRENT_TOKEN) == 520 and CURRENT_TOKEN.count(".") == 2
    opener = WorldOpener(plain({"README.md": b"x\n"}), CURRENT_TOKEN)
    code, report, _out = command_report(capsys, CURRENT_TOKEN, opener)
    assert code == 0 and report["complete"] is True and "error" not in report
    assert opener.paths, "the evaluation made its reads with the token"


@pytest.mark.parametrize("label", sorted(ACCEPTED_TOKENS))
def test_command_evaluates_with_a_token_of_any_accepted_length(capsys, label):
    token = ACCEPTED_TOKENS[label]
    opener = WorldOpener(plain({"README.md": b"x\n"}), token)
    code, report, out = command_report(capsys, token, opener)
    assert code == 0 and report["complete"] is True
    (entry,) = report["results"]
    assert entry["verdict"]["reason"] == guard.R_NO_GUARDED_CONTENT
    # Every read carried the whole token (``WorldOpener`` asserts it) and the
    # report never does.
    assert len(opener.paths) >= 4
    assert token not in out.out and token not in out.err


@pytest.mark.parametrize("label", sorted(ACCEPTED_TOKENS))
def test_reader_and_writer_send_an_accepted_token_verbatim(label):
    token = ACCEPTED_TOKENS[label]
    opener = Recording()
    runner.token_reader(token, opener=opener)(ROOT + "/pulls/1")
    body = runner.shadow_check_body(
        guard.Verdict(guard.FAIL, guard.R_NO_LANE, "", {"pull_request": 42}), H, 42
    )
    created = Recording(status=201, raw=b'{"id": 1}')
    runner.check_writer(token, opener=created)(ROOT + "/check-runs", body)
    for request in (*opener.requests, *created.requests):
        assert request.get_header("Authorization") == "Bearer " + token
        assert request.full_url.startswith("https://api.github.com/")
    assert len(opener.requests) == 1 and len(created.requests) == 1


@pytest.mark.parametrize("label", sorted(REFUSED_TOKENS))
def test_reader_and_writer_refuse_a_hostile_token_before_any_request(label):
    token = REFUSED_TOKENS[label]
    opener = Recording()
    with pytest.raises(runner.RunnerError, match=runner.TOKEN_ENV) as caught:
        runner.token_reader(token, opener=opener)(ROOT + "/pulls/1")
    body = runner.shadow_check_body(
        guard.Verdict(guard.FAIL, guard.R_NO_LANE, "", {"pull_request": 42}), H, 42
    )
    with pytest.raises(runner.RunnerError, match=runner.TOKEN_ENV):
        runner.check_writer(token, opener=opener)(ROOT + "/check-runs", body)
    assert opener.requests == []
    if isinstance(token, str) and token:
        assert token not in str(caught.value)


@pytest.mark.parametrize(
    "label",
    sorted(label for label, token in REFUSED_TOKENS.items() if isinstance(token, str)),
)
@pytest.mark.parametrize("mode", ["dry-run", "shadow"])
def test_command_refuses_a_hostile_token_before_any_request(capsys, label, mode):
    token = REFUSED_TOKENS[label]
    code, report, out = command_report(capsys, token, Exploding(), mode=mode)
    assert code == runner.EXIT_RUN_ERROR
    assert report["complete"] is False and report["results"] == []
    assert runner.TOKEN_ENV in report["error"]
    if len(token) >= 8:
        assert token not in out.out and token not in out.err


def test_the_token_rule_has_no_format_knowledge_and_stays_header_safe():
    pattern = runner._TOKEN.pattern
    # No prefix, no dot count, no fixed length: only a character class and
    # transport bounds.
    assert pattern == "[A-Za-z0-9._~+/=-]{20,16384}"
    assert runner.MAX_TOKEN_CHARS >= 8 * 520
    for code in [*range(0, 0x21), 0x7F, *range(0x80, 0x100), 0x2028, 0xFF11]:
        hostile = "x" * 30 + chr(code) + "x" * 30
        assert runner._TOKEN.fullmatch(hostile) is None, hex(code)
    for forbidden in " \"\\,;:%()<>@[]{}?&#!$^*|`'":
        assert runner._TOKEN.fullmatch("x" * 30 + forbidden + "x" * 30) is None


# --- 2. an incomplete read is never a completed evaluation -------------------

DECIDED = {
    (verdict, reason)
    for verdict, reasons in runner._DECIDED_REASONS.items()
    for reason in reasons
}
ALL_REASONS = sorted(
    value
    for name, value in vars(guard).items()
    if name.startswith("R_") and isinstance(value, str)
)
VERDICT_KINDS = (guard.PASS, guard.FAIL, guard.NOT_APPLICABLE)


def test_every_reason_of_the_evaluator_is_decided_or_incomplete_never_both():
    decided = {reason for _verdict, reason in DECIDED}
    incomplete = set(guard.INCOMPLETE_READ_REASONS)
    assert decided | incomplete == set(ALL_REASONS)
    assert decided & incomplete == set()
    assert len(DECIDED) == len(decided), "one verdict kind for each reason"
    assert runner._UNREAD_REASONS == guard.INCOMPLETE_READ_REASONS
    # The reasons the review named.
    for reason in (guard.R_DELTA_INCOMPLETE, guard.R_REVIEWS, guard.R_COMMITS):
        assert reason in incomplete
    # A success never comes from the incomplete side.
    assert runner._DECIDED_REASONS[guard.PASS] == {
        guard.R_PASS,
        guard.R_CONTROL_PLANE_PASS,
    }


MARKS = {
    "no-mark": {},
    "mark-true": {guard.EVIDENCE_INCOMPLETE_READ: True},
    "mark-false": {guard.EVIDENCE_INCOMPLETE_READ: False},
    "mark-none": {guard.EVIDENCE_INCOMPLETE_READ: None},
    "mark-text": {guard.EVIDENCE_INCOMPLETE_READ: "no"},
}


@pytest.mark.parametrize("mark", sorted(MARKS))
@pytest.mark.parametrize("kind", VERDICT_KINDS)
@pytest.mark.parametrize("reason", ALL_REASONS + ["a_reason_added_later", ""])
def test_every_reason_crossed_with_every_read_outcome(monkeypatch, reason, kind, mark):
    # The evaluator's result is replaced, so that every reason is crossed
    # with every verdict kind and every form of the incomplete-read mark. The
    # run is complete for exactly one cell of each decided reason.
    evidence = {"pull_request": 42, "head": H} | MARKS[mark]
    monkeypatch.setattr(
        guard,
        "evaluate",
        lambda *_a, **_k: guard.Verdict(kind, reason, "", dict(evidence)),
    )
    for mode, post in ((runner.DRY_RUN, None), (runner.SHADOW, Writer())):
        world = shadow_world(plain({"README.md": b"x\n"}))
        report, code = sweep(NAME, [42], mode, fetch=world.fetch, post=post)
        complete = (kind, reason) in DECIDED and mark == "no-mark"
        assert report["complete"] is complete
        assert code == (runner.EXIT_OK if complete else runner.EXIT_RUN_ERROR)
        # The verdict itself is carried unchanged either way.
        (entry,) = report["results"]
        assert entry["verdict"]["verdict"] == kind
        assert entry["verdict"]["reason"] == reason
        if not complete:
            assert report["error"] == "an evaluation could not read GitHub"


@pytest.mark.parametrize(
    "verdict",
    [
        None,
        [],
        "pass",
        {},
        {"verdict": "pass"},
        {"verdict": "pass", "reason": guard.R_PASS},
        {"verdict": "pass", "reason": guard.R_PASS, "evidence": None},
        {"verdict": "pass", "reason": guard.R_PASS, "evidence": []},
        {"verdict": "PASS", "reason": guard.R_PASS, "evidence": {}},
        {"verdict": True, "reason": guard.R_PASS, "evidence": {}},
        {"verdict": "pass", "reason": None, "evidence": {}},
        {"verdict": "pass", "reason": [guard.R_PASS], "evidence": {}},
        {"verdict": "pass", "reason": guard.R_NO_LANE, "evidence": {}},
        {"verdict": "fail", "reason": guard.R_PASS, "evidence": {}},
        {"verdict": "not_applicable", "reason": guard.R_PASS, "evidence": {}},
        {"verdict": "neutral", "reason": guard.R_PASS, "evidence": {}},
        {"verdict": "neutral", "reason": guard.R_NO_LANE, "evidence": {}},
        {"verdict": "skipped", "reason": guard.R_CI, "evidence": {}},
        {"verdict": "", "reason": guard.R_NO_APPROVAL, "evidence": {}},
        {"verdict": ["pass"], "reason": guard.R_PASS, "evidence": {}},
    ],
)
def test_a_verdict_of_an_unexpected_shape_is_not_a_completed_evaluation(verdict):
    assert runner._evaluation_incomplete(verdict) is True


def test_a_decided_verdict_is_a_completed_evaluation():
    for kind, reason in sorted(DECIDED):
        assert not runner._evaluation_incomplete(
            {"verdict": kind, "reason": reason, "evidence": {"pull_request": 1}}
        )


def assert_incomplete(report, code):
    """A run whose one evaluation could not read: closed twice."""
    assert code == runner.EXIT_RUN_ERROR
    assert report["complete"] is False
    assert report["error"] == "an evaluation could not read GitHub"
    (entry,) = report["results"]
    assert entry["verdict"]["verdict"] == "fail"
    assert entry["verdict"]["evidence"][guard.EVIDENCE_INCOMPLETE_READ] is True
    return entry


def test_the_review_probe_a_truncated_tree_is_a_run_error(capsys):
    # The reviewer's probe: ``command()`` through the fake opener with the
    # ``truncated=True`` fixture returned exit 0, ``complete: true`` and
    # ``fail / delta_incomplete``.
    world = plain({"README.md": b"x\n"})
    world.tree(H, {"README.md": b"x\n"}, truncated=True)
    code, report, _out = command_report(
        capsys, LEGACY_TOKEN, WorldOpener(world, LEGACY_TOKEN)
    )
    entry = assert_incomplete(report, code)
    assert entry["verdict"]["reason"] == guard.R_DELTA_INCOMPLETE


@pytest.mark.parametrize("which", ["base", "head"])
@pytest.mark.parametrize("mode", ["dry-run", "shadow"])
def test_a_truncated_tree_keeps_its_fail_verdict_and_fails_the_run(which, mode):
    world = shadow_world(build())
    path = f"{ROOT}/git/trees/{H if which == 'head' else world.api[PR]['base']['sha']}"
    world.api[path + "?recursive=1"]["truncated"] = True
    writer = Writer()
    report, code = sweep(
        NAME, [42], mode, fetch=world.fetch, post=writer if mode == "shadow" else None
    )
    entry = assert_incomplete(report, code)
    assert entry["verdict"]["reason"] == guard.R_DELTA_INCOMPLETE
    if mode == "shadow":
        # The fail verdict is still shown on the head; it is never a pass.
        ((_path, body),) = writer.calls
        assert "fail" in body["output"]["title"]
        assert body["conclusion"] == "neutral"


def full_pages(world, listing, pages):
    """Make ``listing`` answer ``pages`` full pages of 100 distinct entries."""
    (first,) = world.api[PR + listing + PAGE][:1] or [{}]
    for page in range(1, pages + 1):
        world.api[f"{PR}{listing}?per_page=100&page={page}"] = [
            dict(first, id=1000 * page + index, sha=f"{page:020x}{index:020x}")
            for index in range(100)
        ]


@pytest.mark.parametrize(
    ("listing", "reason"),
    [
        ("/reviews", guard.R_REVIEWS),
        ("/commits", guard.R_COMMITS),
        ("/files", guard.R_DELTA_DISAGREES),
    ],
)
def test_evaluator_pagination_exhaustion_is_a_run_error(listing, reason):
    # Ten full pages: the evaluator's page bound. It did not see the list
    # whole, so the fail verdict is not a completed evaluation.
    world = build()
    assert world.evaluate().verdict == "pass"
    full_pages(world, listing, 10)
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    entry = assert_incomplete(report, code)
    assert entry["verdict"]["reason"] == reason


def test_a_review_list_of_one_full_page_and_a_short_page_is_read_whole():
    # The other side: pagination that ends inside the bound is complete.
    world = build()
    first = world.api[PR + "/reviews" + PAGE]
    world.api[PR + "/reviews" + PAGE] = [
        dict(first[0], id=100 + index, state="COMMENTED") for index in range(100)
    ]
    world.api[PR + "/reviews?per_page=100&page=2"] = first
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert code == 0 and report["complete"] is True
    assert report["results"][0]["verdict"]["verdict"] == "pass"


# Every read of an evaluation, replaced by every incomplete outcome.


def http_error(*_args):
    raise urllib.error.HTTPError("https://api.github.com/x", 502, "x", {}, None)


def rate_limited(*_args):
    raise urllib.error.HTTPError("https://api.github.com/x", 403, "x", {}, None)


def network_error(*_args):
    raise urllib.error.URLError("reset")


def timeout_error(*_args):
    raise TimeoutError("timed out")


def truncated_tree(value):
    if not (isinstance(value, dict) and "tree" in value):
        return None
    return dict(value, truncated=True)


def tree_without_flag(value):
    if not (isinstance(value, dict) and "tree" in value):
        return None
    return {key: item for key, item in value.items() if key != "truncated"}


def overfull_page(value):
    if not isinstance(value, list):
        return None
    return [dict(value[0]) if value else {} for _ in range(101)]


def miscounted_listing(value):
    if not (isinstance(value, dict) and "check_runs" in value):
        return None
    return dict(value, total_count=value["total_count"] + 1)


def uncounted_listing(value):
    if not (isinstance(value, dict) and "check_runs" in value):
        return None
    return {"check_runs": value["check_runs"]}


def cut_listing(value):
    if not (isinstance(value, dict) and "check_runs" in value):
        return None
    return dict(value, check_runs=None)


# name -> (applies to a response and returns the replacement, or raises)
INCOMPLETE_OUTCOMES = {
    "http-502": http_error,
    "rate-limit-403": rate_limited,
    "network-error": network_error,
    "timeout": timeout_error,
    "null": lambda value: "null",
    "text": lambda value: "<html>",
    "wrong-container": lambda value: (
        [value] if isinstance(value, dict) else {"items": value}
    ),
    "truncated-tree": truncated_tree,
    "tree-without-truncated-flag": tree_without_flag,
    "page-above-100": overfull_page,
    "check-count-mismatch": miscounted_listing,
    "check-count-absent": uncounted_listing,
    "check-runs-absent": cut_listing,
}
EVALUATED_WORLDS = {
    "v6-pass": build,
    "earlier-lane-v5": lambda: plain({P: PRIMARY, T: COMPANION, M: v5_manifest(P, T)}),
    "allowlist-only": lambda: plain({"README.md": b"x\n"}),
    "no-lane-fail": lambda: plain({".github/workflows/x.yml": b"x\n"}),
}


def injected_cases():
    cases = []
    for label, make in EVALUATED_WORLDS.items():
        reference = make()
        answers = []
        plain_fetch = reference.fetch

        def fetch(path, plain_fetch=plain_fetch, answers=answers):
            answers.append(plain_fetch(path))
            return answers[-1]

        guard.evaluate(NAME, 42, fetch=fetch)
        for index, answer in enumerate(answers):
            for outcome, change in INCOMPLETE_OUTCOMES.items():
                try:
                    applies = change(copy.deepcopy(answer)) is not None
                except OSError:
                    applies = True  # the outcome is a failed request
                if applies:
                    cases.append((label, index, outcome))
    return cases


INJECTED = injected_cases()


def test_the_injection_matrix_covers_every_read_of_every_evaluation():
    reads = {}
    for label, index, _outcome in INJECTED:
        reads[label] = max(reads.get(label, 0), index + 1)
    # The passing v6 evaluation is the longest: 25 reads at this change.
    for label, make in EVALUATED_WORLDS.items():
        world = make()
        world.evaluate()
        assert reads[label] == len(world.calls), label
    assert reads["v6-pass"] == 25
    # Every read is replaced by every failed-request outcome and by the
    # two wrong-shape outcomes; the others apply to the responses they name.
    always = 7
    assert len(INJECTED) >= sum(reads.values()) * always
    for outcome in INCOMPLETE_OUTCOMES:
        assert any(case[2] == outcome for case in INJECTED), outcome


@pytest.mark.parametrize(("label", "index", "outcome"), INJECTED)
def test_no_incomplete_read_yields_a_completed_evaluation(label, index, outcome):
    """Read number ``index`` of the evaluation has an incomplete outcome.

    Whatever the evaluation was going to decide, the run is then an error
    with ``complete: false`` and a ``fail`` verdict that carries the mark:
    never ``pass``, never ``not_applicable``, never a clean ``fail``.
    """
    world = EVALUATED_WORLDS[label]()
    change = INCOMPLETE_OUTCOMES[outcome]
    count = {"reads": 0}
    plain_fetch = world.fetch

    def fetch(path):
        position = count["reads"]
        count["reads"] += 1
        if position != index:
            return plain_fetch(path)
        value = plain_fetch(path)
        replaced = change(copy.deepcopy(value))
        assert replaced is not None, "the case list holds applicable cases only"
        return None if replaced == "null" else replaced

    report, code = runner.run(NAME, [42], runner.DRY_RUN, fetch=fetch)
    assert count["reads"] > index, "the injected read was reached"
    assert_incomplete(report, code)


def test_an_absent_required_check_is_a_decision_and_a_cut_listing_is_not():
    # ``ci_not_green`` is mixed. No run with the required name, in a listing
    # that adds up: CI has not reported, a decision. A listing that does not
    # add up: the evaluator did not see every run.
    listing = f"{ROOT}/commits/{H}/check-runs?check_name=validate%20%2F%20validate"
    listing += "&filter=latest&per_page=100"
    world = build()
    world.api[listing] = {"total_count": 0, "check_runs": []}
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert code == 0 and report["complete"] is True
    verdict = report["results"][0]["verdict"]
    assert (verdict["verdict"], verdict["reason"]) == ("fail", guard.R_CI)
    assert verdict["detail"] == "required check is absent"
    assert guard.EVIDENCE_INCOMPLETE_READ not in verdict["evidence"]

    world = build()
    world.api[listing]["total_count"] = 101
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    entry = assert_incomplete(report, code)
    assert entry["verdict"]["reason"] == guard.R_CI
    assert entry["verdict"]["detail"] == "required check listing is incomplete"


def test_a_red_required_check_stays_a_decision():
    world = build()
    world.ci(H, conclusion="failure")
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert code == 0 and report["complete"] is True
    verdict = report["results"][0]["verdict"]
    assert (verdict["verdict"], verdict["reason"]) == ("fail", guard.R_CI)
    assert guard.EVIDENCE_INCOMPLETE_READ not in verdict["evidence"]


def test_a_deleted_fork_is_a_decision_and_a_cut_pull_request_object_is_not():
    world = plain({"README.md": b"x\n"})
    world.api[PR]["head"]["repo"] = None
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert code == 0 and report["complete"] is True
    assert report["results"][0]["verdict"]["reason"] == guard.R_PR_IDENTITY
    for field in ("head", "base", "user"):
        world = plain({"README.md": b"x\n"})
        del world.api[PR][field]
        report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
        entry = assert_incomplete(report, code)
        assert entry["verdict"]["reason"] == guard.R_PR_IDENTITY


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda api: api[PR]["head"].pop("sha"), guard.R_PR_IDENTITY),
        (lambda api: api[PR]["head"].update(sha="main"), guard.R_PR_IDENTITY),
        (lambda api: api[PR]["head"].update(ref=""), guard.R_PR_IDENTITY),
        (lambda api: api[PR]["user"].update(id="110687043"), guard.R_PR_IDENTITY),
        (lambda api: api[PR]["user"].update(id=0), guard.R_PR_IDENTITY),
        (
            lambda api: api[ROOT + "/git/ref/heads/main"]["object"].pop("sha"),
            guard.R_BASE_NOT_CURRENT,
        ),
        (
            lambda api: api[ROOT + "/git/ref/heads/main"]["object"].update(sha="x"),
            guard.R_BASE_NOT_CURRENT,
        ),
    ],
)
def test_a_response_without_a_field_github_always_sends_is_incomplete(change, reason):
    # Mixed reasons: the reason alone says "refused". The response lacks an
    # exact commit or a numeric identity, so the evaluator did not read the
    # state it judges.
    world = plain({"README.md": b"x\n"})
    change(world.api)
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    entry = assert_incomplete(report, code)
    assert entry["verdict"]["reason"] == reason


def test_a_pull_request_behind_main_stays_a_decision():
    world = plain({"README.md": b"x\n"})
    base = world.api[PR]["base"]["sha"]
    world.api[f"{ROOT}/compare/{base}...{H}"]["status"] = "diverged"
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert code == 0 and report["complete"] is True
    verdict = report["results"][0]["verdict"]
    assert (verdict["verdict"], verdict["reason"]) == (
        "fail",
        guard.R_BASE_NOT_CURRENT,
    )


def test_an_unforeseen_error_of_the_evaluator_carries_the_mark(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(guard, "_evaluate", broken)
    verdict = guard.evaluate(NAME, 42, fetch=lambda _path: {})
    assert (verdict.verdict, verdict.reason) == ("fail", guard.R_EVALUATION_ERROR)
    assert verdict.evidence[guard.EVIDENCE_INCOMPLETE_READ] is True


def test_a_success_verdict_never_carries_the_incomplete_mark():
    for make in EVALUATED_WORLDS.values():
        verdict = make().evaluate()
        if verdict.verdict != guard.FAIL:
            assert guard.EVIDENCE_INCOMPLETE_READ not in verdict.evidence
    refused = guard.evaluate("not-a-repository", 42, fetch=lambda _path: {})
    assert refused.reason == guard.R_INVALID_REQUEST
    assert refused.evidence[guard.EVIDENCE_INCOMPLETE_READ] is True


# --- 3. two pull requests with one head commit -------------------------------


def shared_head_world():
    """Pull requests 42 and 43 have the head ``H``; only 42 has its CI run."""
    world = shadow_world(plain({P: PRIMARY, T: COMPANION, M: v5_manifest(P, T)}))
    other = copy.deepcopy(world.api[PR])
    other["number"] = 43
    other["head"]["ref"] = "codex/other-branch"
    world.api[ROOT + "/pulls/43"] = other
    world.api[PULLS + "1"] = pulls([42, 43])
    return world


class ListingWriter(Writer):
    """A check-run POST whose checks then appear in the shadow listing."""

    def __init__(self, world):
        super().__init__()
        self.world = world
        self.listed = []

    def __call__(self, path, body):
        created = super().__call__(path, body)
        created["id"] = 9000 + len(self.calls)
        self.listed.append(created | {"app": OUR_APP, "status": "completed"})
        self.world.api[SHADOW_LISTING] = {
            "total_count": len(self.listed),
            "check_runs": list(self.listed),
        }
        return created


def test_the_review_probe_two_pull_requests_with_one_head_keep_their_checks():
    # The reviewer's probe: two sweeps over a shared head wrote four checks
    # with alternating verdicts and no pull request number in their titles.
    world = shared_head_world()
    writer = ListingWriter(world)
    report, code = sweep(NAME, None, runner.SHADOW, fetch=world.fetch, post=writer)
    assert code == 0
    assert [
        (entry["pull_request"], entry["verdict"]["reason"], entry["publication"])
        for entry in report["results"]
    ] == [(42, guard.R_EARLIER_LANE, "created"), (43, guard.R_CI, "created")]
    titles = [body["output"]["title"] for _path, body in writer.calls]
    assert titles == [
        "shadow verdict for #42: not_applicable (earlier_lane_v5)",
        "shadow verdict for #43: fail (ci_not_green)",
    ]
    identities = [body["external_id"] for _path, body in writer.calls]
    assert identities[0].startswith("pr-42:") and identities[1].startswith("pr-43:")
    assert all(body["head_sha"] == H for _path, body in writer.calls)
    for (_path, body), number in zip(writer.calls, (42, 43), strict=True):
        assert f"pull request #{number} at this head" in body["output"]["summary"]
    # The second sweep finds each pull request's own newest check: no write.
    report, code = sweep(NAME, None, runner.SHADOW, fetch=world.fetch, post=writer)
    assert code == 0 and len(writer.calls) == 2
    assert [entry["publication"] for entry in report["results"]] == [
        "unchanged",
        "unchanged",
    ]


def identity_of(world, number=42):
    verdict = guard.evaluate(NAME, number, fetch=world.fetch)
    return verdict, runner.shadow_check_body(verdict, H, number)["external_id"]


def ours(external_id, run_id=6, **changes):
    return {
        "id": run_id,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": external_id,
        "status": "completed",
        "app": OUR_APP,
    } | changes


def published(world, existing, number=42):
    writer = Writer()
    report, _code = sweep(
        NAME,
        [number],
        runner.SHADOW,
        fetch=shadow_world(world, existing).fetch,
        post=writer,
    )
    return report["results"][0]["publication"], writer.calls


def test_the_same_verdict_for_two_pull_requests_has_two_identities():
    verdict = guard.Verdict(guard.FAIL, guard.R_NO_LANE, "d", {"pull_request": 42})
    other = guard.Verdict(guard.FAIL, guard.R_NO_LANE, "d", {"pull_request": 43})
    first = runner.shadow_check_body(verdict, H, 42)
    second = runner.shadow_check_body(other, H, 43)
    assert first["external_id"] != second["external_id"]
    # The digest part differs too, not only the readable prefix.
    assert first["external_id"].split(":")[1] != second["external_id"].split(":")[1]
    assert runner._EXTERNAL_ID.fullmatch(first["external_id"])
    assert "#42" in first["output"]["title"] and "#43" in second["output"]["title"]


@pytest.mark.parametrize("number", [43, 0, -42, "42", 42.0, True, None, [42], 4, 420])
def test_a_check_body_never_carries_the_verdict_of_another_pull_request(number):
    verdict = guard.Verdict(guard.FAIL, guard.R_NO_LANE, "", {"pull_request": 42})
    with pytest.raises(runner.RunnerError):
        runner.shadow_check_body(verdict, H, number)


def test_a_verdict_without_its_pull_request_has_no_check_body():
    for evidence in ({}, {"pull_request": None}, {"pull_request": "42"}):
        verdict = guard.Verdict(guard.FAIL, guard.R_NO_LANE, "", evidence)
        with pytest.raises(runner.RunnerError):
            runner.shadow_check_body(verdict, H, 42)


def test_our_check_for_another_pull_request_does_not_suppress_this_verdict():
    world = plain({"README.md": b"x\n"})
    verdict, identity = identity_of(world)
    # The same verdict text, published for pull request 43 on this head.
    twin = guard.Verdict(
        verdict.verdict,
        verdict.reason,
        verdict.detail,
        verdict.evidence | {"pull_request": 43},
    )
    foreign = runner.shadow_check_body(twin, H, 43)["external_id"]
    assert foreign != identity
    publication, calls = published(world, [ours(foreign)])
    assert publication == "created" and len(calls) == 1


def test_a_newer_check_for_another_pull_request_does_not_hide_ours():
    # Pull request 43's check is newer (higher id). Pull request 42's own
    # newest check still carries this verdict: nothing is written again.
    world = plain({"README.md": b"x\n"})
    _verdict, identity = identity_of(world)
    other = "pr-43:" + "0" * 64
    publication, calls = published(world, [ours(identity, 5), ours(other, 9)])
    assert publication == "unchanged" and calls == []


@pytest.mark.parametrize(
    "newer",
    [
        "pr-42:" + "0" * 64,  # this pull request, another verdict
        "pr-42:",  # no digest
        "pr-42",  # no separator
        "pr-420:" + "0" * 63,  # not the exact form: kept in the comparison
        "PR-43:" + "0" * 64,
        "pr-43:" + "0" * 63,
        "pr-043:" + "0" * 64,
        "pr-43:" + "g" * 64,
        " pr-43:" + "0" * 64,
        "pr-43:" + "0" * 64 + "\n",
        "",
        None,
        42,
        ["pr-43:" + "0" * 64],
    ],
)
def test_a_newer_check_of_ours_in_any_other_form_writes_the_verdict_again(newer):
    # Only an exact ``pr-<other number>:<digest>`` is set aside. Everything
    # else of ours that is newer means "the newest check differs".
    world = plain({"README.md": b"x\n"})
    _verdict, identity = identity_of(world)
    publication, calls = published(world, [ours(identity, 5), ours(newer, 9)])
    assert publication == "created" and len(calls) == 1


def test_a_pull_request_number_that_is_a_prefix_of_another_is_not_confused():
    # ``pr-4:`` and ``pr-42:`` and ``pr-420:``.
    world = plain({"README.md": b"x\n"})
    _verdict, identity = identity_of(world)
    assert identity.startswith("pr-42:")
    digest = identity.split(":")[1]
    for other in ("pr-4:" + digest, "pr-420:" + digest, "pr-142:" + digest):
        publication, calls = published(world, [ours(other)])
        assert publication == "created" and len(calls) == 1, other


def test_another_app_cannot_suppress_a_verdict_with_the_new_identity():
    world = plain({"README.md": b"x\n"})
    _verdict, identity = identity_of(world)
    forged = ours(identity, app={"id": 15368, "client_id": "Iv1.other"})
    publication, calls = published(world, [forged])
    assert publication == "created" and len(calls) == 1
    assert CLIENT != "Iv1.other"


@pytest.mark.parametrize(
    "changes",
    [
        {"external_id": "pr-43:" + "0" * 64},
        {"external_id": None},
        {"external_id": ""},
    ],
)
def test_a_created_check_with_another_identity_is_a_publication_error(changes):
    world = shadow_world(plain({"README.md": b"x\n"}))
    writer = Writer(**changes)
    report, code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
    assert code == runner.EXIT_RUN_ERROR and report["complete"] is False
    assert report["results"][0]["publication"] == "error"


# --- the work list is not a census (non-blocking finding) --------------------


def test_a_pull_request_that_moves_across_a_page_boundary_can_be_missed():
    # This test records a limit; it is not a guard. Pull request 1 closes
    # after page 1 was read, so pull request 101 moves to page 1 and page 2
    # is empty: 100 numbers, 101 still open and not listed. ``complete``
    # means "the work list this run obtained", as the module says.
    def fetch(path):
        if path == PULLS + "1":
            return pulls(range(1, 101))
        if path == PULLS + "2":
            return []
        raise KeyError(path)

    numbers = runner.open_pull_requests(NAME, fetch)
    assert len(numbers) == 100 and 101 not in numbers
    assert "It does not mean that the work list held every open pull request" in (
        " ".join(runner.__doc__.split())
    )
