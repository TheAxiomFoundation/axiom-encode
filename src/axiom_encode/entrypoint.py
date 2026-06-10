"""Console-script dispatch for the ``axiom-encode`` command.

``pyproject.toml`` points the ``axiom-encode`` script here. Subcommands
implemented outside :mod:`axiom_encode.cli` are dispatched first; every
other invocation delegates to :func:`axiom_encode.cli.main` with
``sys.argv`` untouched, so the main command surface behaves exactly as if
it were registered directly.
"""

from __future__ import annotations

import sys


def main() -> int | None:
    argv = sys.argv[1:]
    if argv and argv[0] == "check-source-staleness":
        from axiom_encode.source_hash import run_check_source_staleness

        return run_check_source_staleness(argv[1:])
    from axiom_encode.cli import main as cli_main

    return cli_main()
