"""E2E overlay stub: this encoder pin predates corpus_materialize (#1742).

NO_FETCH_ENV and fetch_disabled match the PR head's; the replays run online,
so remote_fetch_disabled is never entered and remote_fetch_is_disabled is
always False.
"""

import contextlib
import os

NO_FETCH_ENV = "AXIOM_CORPUS_NO_FETCH"


def fetch_disabled(environ=os.environ):
    return environ.get(NO_FETCH_ENV, "").strip().lower() in {"1", "true", "yes"}


def remote_fetch_is_disabled():
    return False


@contextlib.contextmanager
def remote_fetch_disabled():
    raise RuntimeError("overlay stub: --offline is not supported on this pin")
    yield
