"""Subprocess progress renews a hang ceiling without losing captured streams."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests import process_limits


@pytest.fixture
def fake_child(monkeypatch):
    class Pipe:
        encoding = "utf-8"

        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.closed = False

        def fileno(self):
            return self.descriptor

        def close(self):
            self.closed = True

    class Child:
        stdout = Pipe(1)
        stderr = Pipe(2)
        returncode = None
        killed = False
        communicated = False

        def wait(self, timeout):
            assert timeout == 1
            self.returncode = 0

        def kill(self):
            self.killed = True

        def communicate(self):
            self.communicated = True
            return "", ""

    child = Child()
    monkeypatch.setattr(process_limits.subprocess, "Popen", lambda *_, **__: child)
    return child


def install_selector(monkeypatch, events, clock):
    class Selector:
        def __init__(self):
            self.entries = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def register(self, pipe, _event, name):
            self.entries[name] = SimpleNamespace(fileobj=pipe, data=name)

        def unregister(self, pipe):
            name = next(
                name for name, key in self.entries.items() if key.fileobj is pipe
            )
            del self.entries[name]

        def get_map(self):
            return self.entries

        def select(self, remaining):
            name, elapsed, expected_budget = next(events)
            # Advance when I/O becomes ready, independently of how many times
            # the implementation reads its clock between readiness events.
            assert remaining == expected_budget, (remaining, expected_budget)
            clock[0] += elapsed
            if name is None:
                return []
            return [(self.entries[name], 1)]

    monkeypatch.setattr(process_limits.selectors, "DefaultSelector", Selector)


def test_progress_on_either_stream_restarts_the_hang_ceiling(monkeypatch, fake_child):
    clock = [0.0]
    # Data on either stream renews five seconds. EOF is not progress: after
    # the first EOF arrives four seconds later, only one second remains.
    events = iter(
        (("stdout", 4, 5), ("stderr", 4, 5), ("stdout", 4, 5), ("stderr", 0, 1))
    )
    install_selector(monkeypatch, events, clock)
    monkeypatch.setattr(process_limits.time, "monotonic", lambda: clock[0])
    chunks = iter((b"hello\r\n", b"error\r", b"", b""))
    monkeypatch.setattr(process_limits.os, "read", lambda *_: next(chunks))

    result = process_limits.run_with_progress_timeout(["child"], timeout=5)

    assert (result.returncode, result.stdout, result.stderr) == (
        0,
        "hello\n",
        "error\n",
    )
    assert not fake_child.killed
    assert fake_child.stdout.closed and fake_child.stderr.closed


def test_inactivity_kills_and_reaps_the_child(monkeypatch, fake_child):
    clock = [0.0]
    install_selector(monkeypatch, iter(((None, 5, 5),)), clock)
    monkeypatch.setattr(process_limits.time, "monotonic", lambda: clock[0])

    with pytest.raises(subprocess.TimeoutExpired) as error:
        process_limits.run_with_progress_timeout(["child"], timeout=5)

    assert error.value.timeout == 5
    assert fake_child.killed and fake_child.communicated
    assert fake_child.stdout.closed and fake_child.stderr.closed


def test_real_child_drains_both_streams_and_preserves_text_capture(tmp_path):
    code = (
        "import os, sys; assert os.getcwd() == sys.argv[1]; "
        "sys.stdout.write('a' * 100_000 + 'café\\r\\n'); "
        "sys.stderr.write('b' * 100_000 + 'error\\r'); "
        "sys.exit(7)"
    )
    result = process_limits.run_with_progress_timeout(
        [sys.executable, "-c", code, str(tmp_path)], cwd=tmp_path
    )
    assert result.returncode == 7
    assert result.stdout == "a" * 100_000 + "café\n"
    assert result.stderr == "b" * 100_000 + "error\n"
