"""Compare environments in tests without printing their values.

When an assert fails, pytest prints the value of each name, attribute, call
and operand in it, and a failed dict comparison lists the differing entries.
An assert that mentions a whole environment (``os.environ``, a copy of it,
or one a fake recorded from the code under test) therefore prints its
variables, credentials included: at ``-vv`` every one, and below it as many as
pytest's length limits let through. Reduce the environment to names, or to one
value, in a statement before the assert; an argument to a call inside the
assert is printed too.

pytest's traceback also shows, within similar limits, the arguments of the
failing test and of the function the failure was raised in. A fixture or a
helper argument holding an environment is printed whatever the assert says,
and so is the ``env`` a fake standing in for a launcher is called with.
Such a fake checks its call with ``expect`` and raises with ``throw``, each of
which fails in a frame of its own. A fake that keeps what it is passed, to be
compared whole, wraps each environment in ``Names``.
``tests/test_environment_assertions.py`` keeps the suite to this.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import NoReturn

_MISSING = object()


def changed_names(
    before: Mapping[str, object], after: Mapping[str, object]
) -> list[str]:
    """The names whose values differ between two environments, sorted.

    A name set in only one of them counts as changed, whatever its value.
    """
    return sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name, _MISSING) != after.get(name, _MISSING)
    )


class Names(Mapping[str, str]):
    """An environment that compares by its values but prints only its names.

    A recorded invocation holding one compares equal to the expected
    invocation exactly when every value matches, while a failing comparison
    shows names alone.
    """

    def __init__(self, environment: Mapping[str, str]) -> None:
        self._environment = dict(environment)

    def __getitem__(self, name: str) -> str:
        return self._environment[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._environment)

    def __len__(self) -> int:
        return len(self._environment)

    def __repr__(self) -> str:
        return f"Names({sorted(self._environment)!r})"


def expect(condition: object, detail: object = None) -> None:
    """Fail unless ``condition`` holds, showing ``detail``.

    For a fake whose arguments hold an environment: the failure is raised in
    this frame, whose arguments are ``condition`` and ``detail`` alone.
    """
    if not condition:
        raise AssertionError() if detail is None else AssertionError(detail)


def throw(error: BaseException) -> NoReturn:
    """Raise ``error`` in this frame, whose one argument is ``error``.

    For a fake whose arguments hold an environment: should the code under test
    stop handling ``error``, the traceback ends here rather than in the fake.
    """
    raise error
