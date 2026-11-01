"""Subprocess progress renews a hang ceiling without losing captured streams."""

from __future__ import annotations

import os
import select
import subprocess
import sys
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests import process_limits


@pytest.fixture
def fake_child(monkeypatch):
    return make_child(monkeypatch)


def make_child(monkeypatch):
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
        waited = False

        def wait(self, timeout=None):
            assert timeout == (None if self.killed else 1)
            self.waited = True
            self.returncode = -9 if self.killed else 0

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
    assert fake_child.killed and fake_child.waited
    assert not fake_child.communicated
    assert fake_child.stdout.closed and fake_child.stderr.closed


@settings(deadline=None)
@given(
    stdout=st.binary(min_size=1), stderr=st.binary(min_size=1), after_eof=st.booleans()
)
def test_timeout_keeps_captured_bytes_without_draining(stdout, stderr, after_eof):
    with pytest.MonkeyPatch.context() as monkeypatch:
        child = make_child(monkeypatch)
        clock = [0.0]
        events = [("stdout", 0, 5), ("stderr", 0, 5)]
        if after_eof:
            events.extend((("stdout", 4, 5), ("stderr", 0, 1)))
            original_wait = child.wait

            def wait(timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired(["child"], timeout)
                return original_wait()

            monkeypatch.setattr(child, "wait", wait)
        else:
            events.append((None, 5, 5))
        install_selector(monkeypatch, iter(events), clock)
        monkeypatch.setattr(process_limits.time, "monotonic", lambda: clock[0])
        chunks = iter((stdout, stderr, b"", b""))
        monkeypatch.setattr(process_limits.os, "read", lambda *_: next(chunks))

        with pytest.raises(subprocess.TimeoutExpired) as error:
            process_limits.run_with_progress_timeout(["child"], timeout=5)

        # Like subprocess.run, timeout output stays bytes even in text mode,
        # including incomplete multibyte characters at the timeout boundary.
        assert error.value.output == error.value.stdout == stdout
        assert error.value.stderr == stderr
        assert child.killed and child.waited and not child.communicated
        assert child.stdout.closed and child.stderr.closed


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX inherited pipes")
def test_timeout_does_not_drain_pipes_inherited_by_a_descendant(monkeypatch):
    # The extra pipe is a readiness/exit handshake, so no elapsed-time
    # assertion depends on how promptly this test is scheduled. EOF waits for
    # the descendant's fixed lifetime even when the cleanup assertion fails.
    read_fd, write_fd = os.pipe()
    child = None
    drains = 0
    original_popen = subprocess.Popen

    class ObservedChild(original_popen):
        def __init__(self, *args, **kwargs):
            nonlocal child
            super().__init__(*args, **kwargs)
            child = self
            assert select.select(
                [read_fd], [], [], process_limits.HANG_TIMEOUT_SECONDS
            )[0], "descendant not ready"
            assert os.read(read_fd, 1) == b"R"

        def communicate(self, *args, **kwargs):
            nonlocal drains
            drains += 1
            return super().communicate(*args, **kwargs)

    monkeypatch.setattr(process_limits.subprocess, "Popen", ObservedChild)
    code = """
import os, sys, time
fd = int(sys.argv[1])
os.write(1, b'partial stdout\\n')
os.write(2, b'partial stderr\\n')
if os.fork() == 0:
    os.write(fd, b'R')
    time.sleep(0.25)
    os._exit(0)
os.close(fd)
time.sleep(300)
"""
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            process_limits.run_with_progress_timeout(
                [sys.executable, "-c", code, str(write_fd)],
                pass_fds=(write_fd,),
                timeout=0.01,
            )
        assert drains == 0, ("unbounded pipe drains", drains)
        assert error.value.stdout == b"partial stdout\n"
        assert error.value.stderr == b"partial stderr\n"
        assert child is not None and child.returncode is not None
        assert child.stdout.closed and child.stderr.closed
    finally:
        os.close(write_fd)
        if child is not None:
            child.kill()
            child.wait()
            child.stdout.close()
            child.stderr.close()
        try:
            assert select.select(
                [read_fd], [], [], process_limits.HANG_TIMEOUT_SECONDS
            )[0], "descendant did not exit"
            assert os.read(read_fd, 1) == b""
        finally:
            os.close(read_fd)


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
