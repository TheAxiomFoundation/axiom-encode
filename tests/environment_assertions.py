"""Compare environments in tests without printing their values.

When an assert fails, pytest prints the value of each name, attribute, call
and operand in it, and a failed dict comparison lists the differing entries.
An assert that mentions a whole environment (``os.environ``, a copy of it,
or one a fake recorded from the code under test) therefore prints every
variable in it, credentials included. Reduce the environment to names, or to
one value, in a statement before the assert; an argument to a call inside the
assert is printed too. ``tests/test_environment_assertions.py`` keeps the
suite to this.
"""

from __future__ import annotations

from collections.abc import Mapping


def changed_names(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """The names whose values differ between two environments, sorted.

    A name set in only one of them counts as changed.
    """
    return sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )
