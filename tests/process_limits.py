"""Hang ceilings and progress-aware capture for real test subprocesses."""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from collections.abc import Mapping, Sequence

# Catch hangs, not slowness: scheduling delays in shared runners and lanes can
# dominate elapsed time, so ordinary completion has a generous five minutes.
HANG_TIMEOUT_SECONDS = 300


def run_with_progress_timeout(
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    pass_fds: tuple[int, ...] = (),
    timeout: float = HANG_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Capture a POSIX child, restarting its hang ceiling on stdout/stderr bytes.

    Preserve separate streams and subprocess.run's text newline conversion. A
    child can finish slow work while making progress; a silent hang still dies.
    """
    child = subprocess.Popen(
        args,
        env=env,
        cwd=cwd,
        pass_fds=pass_fds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    assert child.stdout is not None and child.stderr is not None
    try:
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ, "stdout")
            ready.register(child.stderr, selectors.EVENT_READ, "stderr")
            while ready.get_map():
                events = ready.select(max(0, deadline - time.monotonic()))
                if not events:
                    raise subprocess.TimeoutExpired(args, timeout)
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        output[key.data].append(chunk)
                        deadline = time.monotonic() + timeout
                    else:
                        ready.unregister(key.fileobj)
        child.wait(timeout=max(0, deadline - time.monotonic()))
    except BaseException as error:
        child.kill()
        # Match subprocess.run on POSIX: a descendant can hold these pipes
        # open after the direct child dies, so never drain them on timeout.
        child.wait()
        if isinstance(error, subprocess.TimeoutExpired):
            error.output = b"".join(output["stdout"]) if output["stdout"] else None
            error.stderr = b"".join(output["stderr"]) if output["stderr"] else None
        raise
    finally:
        child.stdout.close()
        child.stderr.close()

    def text(stream: str) -> str:
        pipe = getattr(child, stream)
        decoded = b"".join(output[stream]).decode(pipe.encoding)
        return decoded.replace("\r\n", "\n").replace("\r", "\n")

    return subprocess.CompletedProcess(
        args, child.returncode, text("stdout"), text("stderr")
    )
