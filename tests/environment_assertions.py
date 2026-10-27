"""Compare environments in tests without printing their values.

When an assert fails, pytest prints the value of each name, attribute, call
and operand in it, and a failed dict comparison lists the differing entries.
An assert that mentions a whole environment (``os.environ``, a copy of it,
or one a fake recorded from the code under test) therefore prints every
variable in it, credentials included. Reduce the environment to names, or to
one value, in a statement before the assert; an argument to a call inside the
assert is printed too.

pytest's traceback also shows the arguments of the failing test and of the
function the assert failed in. A fixture or a helper argument holding an
environment is printed whatever the assert says, and so is the ``env`` a fake
standing in for a launcher is called with. Such a fake checks its call with
``expect``, which fails in a frame of its own.
``tests/test_environment_assertions.py`` keeps the suite to this.
"""

from __future__ import annotations

from collections.abc import Mapping

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


def expect(condition: object, detail: object = None) -> None:
    """Fail unless ``condition`` holds, showing ``detail``.

    For a fake whose arguments hold an environment: the failure is raised in
    this frame, whose arguments are ``condition`` and ``detail`` alone.
    """
    if not condition:
        raise AssertionError() if detail is None else AssertionError(detail)
