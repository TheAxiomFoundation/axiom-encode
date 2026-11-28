"""One admission read is retried only for a short rate limit or a 5xx answer.

Every error here is a real ``urllib.error.HTTPError`` whose headers are an
``email.message.Message``, as ``urllib`` raises it.
"""

from __future__ import annotations

import email.message
import email.utils
import http.client
import io
import json
import urllib.error
import urllib.request

import pytest

from axiom_encode import reviewed_pr_admission as admission

PATH = "/repos/TheAxiomFoundation/rulespec-us/pulls/1"
URL = "https://api.github.com" + PATH
NOW = 1_800_000_000.0
LIMIT_BODY = b'{"message":"API rate limit exceeded for this address."}'
SECONDARY_BODY = b'{"message":"You have exceeded a secondary rate limit"}'
POLICY_BODY = b'{"message":"Resource not accessible"}'
HTTP_DATE = "Wed, 21 Oct 2037 07:28:00 GMT"


def http_error(code, reason="Forbidden", headers=None, body=b"{}"):
    message = email.message.Message()
    for name, value in (headers or {}).items():
        message[name] = value
    return urllib.error.HTTPError(URL, code, reason, message, io.BytesIO(body))


class Response:
    def __init__(self, payload, status=200):
        self.status = status
        self._raw = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, limit):
        return self._raw[:limit]


class World:
    """A scripted API: each read takes the next answer; sleeps move the clock."""

    def __init__(self, monkeypatch, answers, *, jitter=0.0):
        self.answers = list(answers)
        self.requests = []
        self.sleeps = []
        self.clock = NOW
        monkeypatch.setattr(
            admission.urllib.request, "build_opener", lambda *handlers: self
        )
        monkeypatch.setattr(admission.time, "sleep", self.sleep)
        monkeypatch.setattr(admission.time, "time", lambda: self.clock)
        monkeypatch.setattr(admission, "_jitter_seconds", lambda: jitter)
        monkeypatch.setattr(admission, "_process_waited", 0.0)

    def open(self, request, timeout):
        self.requests.append(request.full_url)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds


def primary_limit(reset_in=20):
    return http_error(
        403,
        "rate limit exceeded",
        {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Used": "60",
            "X-RateLimit-Reset": str(int(NOW) + reset_in),
            "X-GitHub-Request-Id": "AB12:CD34",
        },
        LIMIT_BODY,
    )


def test_rate_limit_403_then_success_returns_the_later_answer(monkeypatch):
    world = World(monkeypatch, [primary_limit(), Response({"n": 1})])
    assert admission.github_json(PATH) == {"n": 1}
    assert world.requests == [URL, URL]
    # The distance to the stated reset plus the margin.
    assert admission._RETRY_RESET_MARGIN_SECONDS == 3.0
    assert world.sleeps == [23.0]


RETRYABLE = {
    "403 Retry-After": (403, "Forbidden", {"Retry-After": "7"}, b"{}"),
    "403 remaining 0 and reason": (
        403,
        "rate limit exceeded",
        {"X-RateLimit-Remaining": "0"},
        b"{}",
    ),
    "403 remaining 0 and body": (
        403,
        "Forbidden",
        {"X-RateLimit-Remaining": "0"},
        SECONDARY_BODY,
    ),
    "429": (429, "Too Many Requests", {}, b"{}"),
    "429 Retry-After": (429, "Too Many Requests", {"Retry-After": "7"}, b"{}"),
    "500": (500, "Internal Server Error", {}, b"{}"),
    "502": (502, "Bad Gateway", {}, b"{}"),
    "503 Retry-After": (503, "Service Unavailable", {"Retry-After": "7"}, b"{}"),
    "504": (504, "Gateway Timeout", {}, b"{}"),
}


@pytest.mark.parametrize("case", RETRYABLE, ids=list(RETRYABLE))
def test_each_retryable_answer_is_read_again(monkeypatch, case):
    error = http_error(*RETRYABLE[case])
    world = World(monkeypatch, [error, Response([1])])
    assert admission.github_json(PATH) == [1]
    expected = 7.0 if error.headers.get("Retry-After") else 2.0
    assert world.sleeps == [expected]


RAISED_AT_ONCE = {
    "403 policy": (403, "Forbidden", {}, POLICY_BODY),
    # A policy 403 that happens to be the last request of the budget.
    "403 remaining 0 only": (403, "Forbidden", {"X-RateLimit-Remaining": "0"}, b"{}"),
    "403 remaining 0 and policy body": (
        403,
        "Forbidden",
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(NOW) + 5)},
        POLICY_BODY,
    ),
    # The words alone, with budget left and no Retry-After.
    "403 words in body, remaining 41": (
        403,
        "Forbidden",
        {"X-RateLimit-Remaining": "41"},
        b'{"message":"This endpoint has a rate limit; see the documentation"}',
    ),
    "403 words in reason, no header": (403, "rate limit exceeded", {}, LIMIT_BODY),
    "403 malformed Retry-After": (403, "Forbidden", {"Retry-After": "soon"}, b"{}"),
    "403 negative Retry-After": (403, "Forbidden", {"Retry-After": "-5"}, b"{}"),
    "403 HTTP-date Retry-After": (403, "Forbidden", {"Retry-After": HTTP_DATE}, b"{}"),
    "401": (401, "Unauthorized", {}, b"{}"),
    "404": (404, "Not Found", {}, b"{}"),
    "422 remaining 0": (422, "Unprocessable", {"X-RateLimit-Remaining": "0"}, b"{}"),
    "501": (501, "Not Implemented", {}, b"{}"),
}


@pytest.mark.parametrize("case", RAISED_AT_ONCE, ids=list(RAISED_AT_ONCE))
def test_other_answers_are_raised_at_once(monkeypatch, capsys, case):
    code, reason, headers, body = RAISED_AT_ONCE[case]
    world = World(
        monkeypatch,
        [http_error(code, reason, headers, body), Response({"never": "read"})],
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert len(world.requests) == 1 and world.sleeps == []
    # The raised error is what the server said: status, reason, headers, body.
    assert (caught.value.code, caught.value.reason) == (code, reason)
    assert caught.value.filename == URL
    assert dict(caught.value.headers) == headers
    assert caught.value.read() == body
    err = capsys.readouterr().err
    if code == 403:
        # Every 403 leaves one line, so a failed run shows what the server said.
        (line,) = err.splitlines()
        assert f"HTTP 403 for {PATH}; attempt 1 of 4; " in line
        assert "; not a rate limit, not retried; x-ratelimit-limit=" in line
    else:
        assert err == ""


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
    ],
    ids=["URLError", "timeout", "reset"],
)
def test_a_failure_that_is_not_an_http_answer_is_not_retried(monkeypatch, failure):
    world = World(monkeypatch, [failure, Response({"never": "read"})])
    with pytest.raises(type(failure)) as caught:
        admission.github_json(PATH)
    assert caught.value is failure
    assert len(world.requests) == 1 and world.sleeps == []


def test_attempts_used_up_raises_the_last_rate_limit_error(monkeypatch, capsys):
    errors = [primary_limit(reset_in=0) for _ in range(admission._RETRY_MAX_ATTEMPTS)]
    world = World(monkeypatch, [*errors, Response({"never": "read"})])
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert caught.value.code == 403 and caught.value.reason == "rate limit exceeded"
    # The body that was read to classify the answer is still on the error.
    assert caught.value.read() == LIMIT_BODY
    assert dict(caught.value.headers) == dict(errors[-1].headers)
    assert len(world.requests) == admission._RETRY_MAX_ATTEMPTS == 4
    assert len(world.sleeps) == 3
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 4
    assert all("; retry in " in line for line in lines[:3])
    assert "attempt 4 of 4; attempts used up, not retried; " in lines[3]


def test_attempts_used_up_on_5xx_backs_off_exponentially(monkeypatch):
    errors = [http_error(503, "Service Unavailable") for _ in range(4)]
    world = World(monkeypatch, [*errors, Response({"never": "read"})])
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert caught.value is errors[-1]
    assert world.sleeps == [2.0, 4.0, 8.0]


def test_the_hourly_reset_is_not_waited_for(monkeypatch, capsys):
    """The primary anonymous limit is not fixed here: it fails at once."""
    world = World(
        monkeypatch, [primary_limit(reset_in=55 * 60), Response({"never": "read"})]
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH + "/reviews?per_page=100&page=1")
    assert caught.value.code == 403 and caught.value.read() == LIMIT_BODY
    assert len(world.requests) == 1 and world.sleeps == []
    (line,) = capsys.readouterr().err.splitlines()
    assert line == (
        f"GitHub admission read: HTTP 403 for {PATH}/reviews; attempt 1 of 4; "
        "a wait of 3303s is beyond the bound of one read, not retried; "
        "x-ratelimit-limit=60 x-ratelimit-remaining=0 "
        f"x-ratelimit-reset={int(NOW) + 3300} x-ratelimit-used=60 "
        "retry-after=- date=- x-github-request-id=AB12:CD34"
    )


def test_the_log_line_holds_bounded_printable_header_text_only(monkeypatch, capsys):
    error = http_error(
        503,
        "Service Unavailable",
        {"Date": HTTP_DATE, "X-GitHub-Request-Id": "id with\ttab " + "x" * 200},
    )
    World(monkeypatch, [error, Response({})])
    assert admission.github_json(PATH + "?per_page=100") == {}
    (line,) = capsys.readouterr().err.splitlines()
    assert f"HTTP 503 for {PATH}; attempt 1 of 4; retry in 2s; " in line
    assert "date=Wed,_21_Oct_2037_07:28:00_GMT " in line
    request_id = line.rsplit("x-github-request-id=", 1)[1]
    assert request_id.startswith("id_with_tab_x") and len(request_id) == 64
    assert "?" not in line and line.isascii() and line.isprintable()


def test_the_wait_of_one_read_may_reach_the_bound_exactly(monkeypatch):
    assert admission._RETRY_MAX_TOTAL_WAIT_SECONDS == 300.0
    errors = [
        http_error(429, "Too Many Requests", {"Retry-After": "150"}) for _ in range(2)
    ]
    world = World(monkeypatch, [*errors, Response({"n": 2})])
    assert admission.github_json(PATH) == {"n": 2}
    assert world.sleeps == [150.0, 150.0]


def test_the_total_wait_of_one_read_is_bounded(monkeypatch, capsys):
    errors = [
        http_error(429, "Too Many Requests", {"Retry-After": "140"}) for _ in range(3)
    ]
    world = World(monkeypatch, [*errors, Response({"never": "read"})])
    with pytest.raises(urllib.error.HTTPError):
        admission.github_json(PATH)
    # Two waits of 140 s fit the bound; the third does not start.
    assert world.sleeps == [140.0, 140.0] and len(world.requests) == 3
    assert "beyond the bound of one read" in capsys.readouterr().err.splitlines()[-1]


def test_the_total_wait_of_one_process_is_bounded(monkeypatch, capsys):
    assert admission._RETRY_MAX_PROCESS_WAIT_SECONDS == 900.0

    def limited():
        return http_error(429, "Too Many Requests", {"Retry-After": "290"})

    world = World(
        monkeypatch,
        [limited(), Response(1), limited(), Response(2), limited(), Response(3)]
        + [limited(), Response({"never": "read"})],
    )
    assert [admission.github_json(PATH) for _ in range(3)] == [1, 2, 3]
    assert admission._process_waited == 870.0
    with pytest.raises(urllib.error.HTTPError):
        admission.github_json(PATH)
    assert world.sleeps == [290.0, 290.0, 290.0]
    assert "beyond the bound of this process" in capsys.readouterr().err


def test_the_process_wait_may_reach_its_bound_exactly(monkeypatch):
    world = World(
        monkeypatch,
        [http_error(429, "Too Many Requests", {"Retry-After": "30"}), Response(1)],
    )
    monkeypatch.setattr(admission, "_process_waited", 870.0)
    assert admission.github_json(PATH) == 1
    assert world.sleeps == [30.0] and admission._process_waited == 900.0


@pytest.mark.parametrize("value", ["-5", "1e9", "soon", "9" * 13, "", HTTP_DATE])
def test_a_malformed_wait_header_is_not_trusted(monkeypatch, value):
    error = http_error(
        429, "Too Many Requests", {"Retry-After": value, "X-RateLimit-Reset": value}
    )
    world = World(monkeypatch, [error, Response({})])
    assert admission.github_json(PATH) == {}
    assert world.sleeps == [2.0]


def test_the_reset_distance_is_measured_from_the_answer_date(monkeypatch):
    # The local clock runs 100 s ahead of the server: by the local clock the
    # reset is in the past, by the server's own Date it is 20 s away.
    server_now = NOW - 100
    error = http_error(
        403,
        "rate limit exceeded",
        {
            "Date": email.utils.formatdate(server_now, usegmt=True),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(server_now) + 20),
        },
    )
    world = World(monkeypatch, [error, Response({})])
    assert admission.github_json(PATH) == {}
    assert world.sleeps == [23.0]


@pytest.mark.parametrize("date", [None, "not a date", "Wed, 21 Oct 2037 07:28:00"])
def test_without_a_usable_date_the_local_clock_is_used(monkeypatch, date):
    headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(NOW) + 20)}
    if date is not None:
        headers["Date"] = date
    world = World(
        monkeypatch, [http_error(403, "rate limit exceeded", headers), Response({})]
    )
    assert admission.github_json(PATH) == {}
    assert world.sleeps == [23.0]


def test_a_reset_in_the_past_waits_the_margin(monkeypatch):
    world = World(monkeypatch, [primary_limit(reset_in=-30), Response({})])
    assert admission.github_json(PATH) == {}
    assert world.sleeps == [3.0]


def test_the_jitter_is_added_to_every_wait_and_is_small(monkeypatch):
    world = World(
        monkeypatch,
        [
            http_error(429, "Too Many Requests", {"Retry-After": "7"}),
            http_error(503, "Service Unavailable"),
            primary_limit(),
            Response({}),
        ],
        jitter=0.5,
    )
    assert admission.github_json(PATH) == {}
    # Retry-After; second backoff step; reset distance less the 7.5 + 4.5 s
    # that the clock moved, plus the margin.
    assert world.sleeps == [7.5, 4.5, 20 - 12 + 3 + 0.5]
    monkeypatch.undo()
    assert admission._RETRY_MAX_JITTER_SECONDS == 1.0
    assert all(0.0 <= admission._jitter_seconds() <= 1.0 for _ in range(200))


def test_a_retry_does_not_relax_the_checks_of_the_answer(monkeypatch):
    # A non-200 success status after a retry is still refused.
    World(monkeypatch, [http_error(503, "Service Unavailable"), Response({}, 206)])
    with pytest.raises(admission.AdmissionError, match="did not succeed"):
        admission.github_json(PATH)
    # A redirect refusal is not an HTTP error and is never retried.
    refusal = admission.AdmissionError("GitHub admission requests must not redirect")
    world = World(monkeypatch, [refusal, Response({"never": "read"})])
    with pytest.raises(admission.AdmissionError, match="must not redirect"):
        admission.github_json(PATH)
    assert len(world.requests) == 1 and world.sleeps == []
    # A path outside the fixed organization makes no request at all.
    world = World(monkeypatch, [Response({"never": "read"})])
    with pytest.raises(admission.AdmissionError, match="invalid admission API path"):
        admission.github_json("/repos/other/x")
    assert world.requests == []


def test_header_text_cannot_form_a_workflow_command(monkeypatch, capsys):
    hostile = {
        "X-GitHub-Request-Id": "##[add-mask]secret",
        "X-RateLimit-Limit": "::error::stop",
        "X-RateLimit-Used": "%0A::set-output name=a::b",
        "Date": "[group]<x>;$(id)`id`|&'\"\\/",
    }
    World(monkeypatch, [http_error(403, "Forbidden", hostile, POLICY_BODY)])
    with pytest.raises(urllib.error.HTTPError):
        admission.github_json(PATH)
    (line,) = capsys.readouterr().err.splitlines()
    fields = line.split("; ")[-1]
    assert "x-github-request-id=___add-mask_secret" in fields
    assert "x-ratelimit-limit=::error::stop " in fields
    assert "x-ratelimit-used=_0A::set-output_name_a::b " in fields
    assert "date=_group__x____id__id_______ " in fields
    assert set(fields) <= set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,:._-= "
    )
    assert "#" not in line and "[" not in line and "]" not in line


# --- the same path over real ``http.client`` responses, without a network ---


class _Wire:
    """A socket stand-in that hands ``http.client`` the bytes of one answer."""

    def __init__(self, raw):
        self.raw = raw

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self.raw)


class _ScriptedHTTPS(urllib.request.HTTPSHandler):
    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.responses = []

    def https_open(self, request):
        response = http.client.HTTPResponse(_Wire(self.answers.pop(0)), method="GET")
        response.begin()
        response.url = request.full_url
        response.msg = response.reason
        self.responses.append(response)
        return response


def real_world(monkeypatch, answers):
    """The real opener of ``github_json`` with only the transport replaced."""
    transport = _ScriptedHTTPS(answers)
    build = urllib.request.build_opener
    monkeypatch.setattr(
        admission.urllib.request,
        "build_opener",
        lambda *handlers: build(*handlers, transport),
    )
    sleeps = []
    monkeypatch.setattr(admission.time, "sleep", sleeps.append)
    monkeypatch.setattr(admission, "_jitter_seconds", lambda: 0.0)
    monkeypatch.setattr(admission, "_process_waited", 0.0)
    return transport, sleeps


def wire(status, headers, body):
    lines = [f"HTTP/1.1 {status}", *[f"{k}: {v}" for k, v in headers.items()]]
    return "\r\n".join(lines).encode() + b"\r\n\r\n" + body


def with_length(status, headers, body):
    return wire(status, {**headers, "Content-Length": len(body)}, body)


def test_a_real_403_response_is_read_closed_and_raised_with_its_body(monkeypatch):
    transport, sleeps = real_world(
        monkeypatch,
        [with_length("403 Forbidden", {"X-RateLimit-Remaining": "41"}, POLICY_BODY)],
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert caught.value.code == 403 and caught.value.reason == "Forbidden"
    assert caught.value.headers["X-RateLimit-Remaining"] == "41"
    assert caught.value.read() == POLICY_BODY
    # The response of the wire was read and closed; nothing was read again.
    (response,) = transport.responses
    assert response.closed and sleeps == []


def test_a_real_rate_limit_response_then_success(monkeypatch, capsys):
    limited = with_length(
        "403 rate limit exceeded",
        {"X-RateLimit-Remaining": "0", "Retry-After": "9"},
        LIMIT_BODY,
    )
    transport, sleeps = real_world(
        monkeypatch, [limited, with_length("200 OK", {}, b'{"n": 3}')]
    )
    assert admission.github_json(PATH) == {"n": 3}
    assert sleeps == [9.0] and all(r.closed for r in transport.responses)
    assert "; retry in 9s; " in capsys.readouterr().err


def test_a_truncated_403_body_still_raises_the_403(monkeypatch, capsys):
    # Chunked body cut inside its first chunk: the read raises IncompleteRead.
    cut = wire("403 Forbidden", {"Transfer-Encoding": "chunked"}, b'40\r\n{"mess')
    transport, sleeps = real_world(monkeypatch, [cut])
    with pytest.raises(http.client.IncompleteRead):
        _read_all(cut)
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert caught.value.code == 403 and caught.value.read() == b""
    assert transport.responses[0].closed and sleeps == []
    assert "not a rate limit, not retried" in capsys.readouterr().err


def _read_all(raw):
    response = http.client.HTTPResponse(_Wire(raw), method="GET")
    response.begin()
    return response.read(admission._RETRY_ERROR_BODY_BYTES)
