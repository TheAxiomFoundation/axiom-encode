"""One admission read is retried only for a rate-limit or transient answer."""

from __future__ import annotations

import email.message
import io
import json
import urllib.error

import pytest

from axiom_encode import reviewed_pr_admission as admission

PATH = "/repos/TheAxiomFoundation/rulespec-us/pulls/1"
URL = "https://api.github.com" + PATH
NOW = 1_800_000_000.0


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

    def __init__(self, monkeypatch, answers):
        self.answers = list(answers)
        self.requests = []
        self.sleeps = []
        self.clock = NOW
        monkeypatch.setattr(
            admission.urllib.request, "build_opener", lambda *handlers: self
        )
        monkeypatch.setattr(admission.time, "sleep", self.sleep)
        monkeypatch.setattr(admission.time, "time", lambda: self.clock)

    def open(self, request, timeout):
        self.requests.append(request.full_url)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds


PRIMARY_LIMIT = {
    "reason": "rate limit exceeded",
    "headers": {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(NOW) + 20)},
    "body": b'{"message":"API rate limit exceeded for this address."}',
}


def test_rate_limit_403_then_success_returns_the_later_answer(monkeypatch):
    world = World(monkeypatch, [http_error(403, **PRIMARY_LIMIT), Response({"n": 1})])
    assert admission.github_json(PATH) == {"n": 1}
    assert world.requests == [URL, URL]
    # The wait is the time to the stated reset plus one second.
    assert world.sleeps == [21.0]


@pytest.mark.parametrize(
    "error",
    [
        http_error(403, "Forbidden", {"Retry-After": "7"}),
        http_error(403, "Forbidden", {"X-RateLimit-Remaining": "0"}),
        http_error(403, "rate limit exceeded"),
        http_error(403, body=b'{"message":"You have exceeded a secondary rate limit"}'),
        http_error(429, "Too Many Requests"),
        http_error(429, "Too Many Requests", {"Retry-After": "7"}),
        http_error(500, "Internal Server Error"),
        http_error(502, "Bad Gateway"),
        http_error(503, "Service Unavailable", {"Retry-After": "7"}),
        http_error(504, "Gateway Timeout"),
    ],
    ids=lambda error: f"{error.code}-{error.reason}-{dict(error.headers)}",
)
def test_each_retryable_answer_is_read_again(monkeypatch, error):
    world = World(monkeypatch, [error, Response([1])])
    assert admission.github_json(PATH) == [1]
    expected = 7.0 if error.headers.get("Retry-After") else 2.0
    assert world.sleeps == [expected]


@pytest.mark.parametrize(
    "error",
    [
        http_error(403, "Forbidden", body=b'{"message":"Resource not accessible"}'),
        http_error(403, "Forbidden", {"X-RateLimit-Remaining": "41"}),
        http_error(401, "Unauthorized"),
        http_error(404, "Not Found"),
        http_error(422, "Unprocessable Entity", {"X-RateLimit-Remaining": "0"}),
        http_error(501, "Not Implemented"),
    ],
    ids=lambda error: f"{error.code}-{dict(error.headers)}",
)
def test_other_answers_are_raised_at_once(monkeypatch, error):
    world = World(monkeypatch, [error, Response({"never": "read"})])
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert caught.value is error
    assert len(world.requests) == 1 and world.sleeps == []


def test_attempts_used_up_raises_the_last_rate_limit_error(monkeypatch):
    errors = [
        http_error(403, "rate limit exceeded")
        for _ in range(admission._RETRY_MAX_ATTEMPTS)
    ]
    world = World(monkeypatch, [*errors, Response({"never": "read"})])
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert caught.value is errors[-1] and caught.value.code == 403
    assert len(world.requests) == admission._RETRY_MAX_ATTEMPTS == 4
    assert world.sleeps == [2.0, 4.0, 8.0]


def test_a_reset_beyond_the_wait_bound_is_not_waited_for(monkeypatch):
    far = int(NOW + admission._RETRY_MAX_TOTAL_WAIT_SECONDS + 60)
    error = http_error(
        403,
        "rate limit exceeded",
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(far)},
    )
    world = World(monkeypatch, [error, Response({"never": "read"})])
    with pytest.raises(urllib.error.HTTPError) as caught:
        admission.github_json(PATH)
    assert caught.value is error
    assert len(world.requests) == 1 and world.sleeps == []


def test_the_total_wait_of_one_read_is_bounded(monkeypatch):
    assert admission._RETRY_MAX_TOTAL_WAIT_SECONDS == 300.0
    errors = [http_error(429, "Too Many Requests", {"Retry-After": "140"})] * 3
    world = World(monkeypatch, [*errors, Response({"never": "read"})])
    with pytest.raises(urllib.error.HTTPError):
        admission.github_json(PATH)
    # Two waits of 140 s fit the bound; the third does not start.
    assert world.sleeps == [140.0, 140.0] and len(world.requests) == 3


@pytest.mark.parametrize("value", ["-5", "1e9", "soon", "9" * 13, ""])
def test_a_malformed_wait_header_is_not_trusted(monkeypatch, value):
    error = http_error(
        429, "Too Many Requests", {"Retry-After": value, "X-RateLimit-Reset": value}
    )
    world = World(monkeypatch, [error, Response({})])
    assert admission.github_json(PATH) == {}
    assert world.sleeps == [2.0]


def test_a_reset_in_the_past_waits_one_second(monkeypatch):
    error = http_error(
        403,
        "rate limit exceeded",
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(NOW) - 30)},
    )
    world = World(monkeypatch, [error, Response({})])
    assert admission.github_json(PATH) == {}
    assert world.sleeps == [1.0]


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
