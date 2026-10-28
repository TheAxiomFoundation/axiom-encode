"""Compare environments in tests without printing their values.

When an assert fails, pytest prints the value of each name, attribute, call
and operand in it, and a failed dict comparison lists the differing entries.
An assert that mentions a whole environment (``os.environ``, a copy of it,
or one a fake recorded from the code under test) therefore prints its
variables, credentials included: at ``-vvv`` every one, and below it as many as
pytest's length limits let through. Reduce the environment to names, or to one
value, in a statement before the assert; an argument to a call inside the
assert is printed too.

pytest's traceback also prints, within similar limits, the arguments of the
failing test and of every frame an exception passes through that it does not
shorten to a line: the frame it was raised in, and, for each exception it was
chained from (``raise ... from``, or one raised while handling another), the
first frame of that exception in the test's own file. A fake standing in for
a launcher is such a frame whenever it raises, by design or by accident, and
whenever the code under test wraps what it raised. So a function in a test
module that is passed an environment (a fake's ``env``, a helper's argument, a
record holding one) takes it through ``hides_environment``, which hands it a
``Names`` in each environment's place; a mock standing in for a launcher
raises through ``raising``. A fake that keeps what it is passed, to be compared
whole, then keeps ``Names`` too. ``tests/test_environment_assertions.py`` keeps
the suite to this.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable, Iterator, Mapping
from typing import NoReturn, TypeVar

_MISSING = object()
# Arguments, keyword arguments, mapping keys and record fields that hold a
# process environment.
_ENVIRONMENTS = frozenset({"env", "environment"})

_Function = TypeVar("_Function", bound=Callable[..., object])


def changed_names(
    before: Mapping[str, object], after: Mapping[str, object]
) -> list[str]:
    """The names whose values differ between two environments, sorted.

    A name set in only one of them counts as changed, whatever its value.
    """
    # Should a caller pass something else, the error is its caller's to show:
    # this frame's arguments are the environments.
    __tracebackhide__ = True
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
        __tracebackhide__ = True
        self._environment = dict(environment)

    def __getitem__(self, name: str) -> str:
        return self._environment[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._environment)

    def __len__(self) -> int:
        return len(self._environment)

    def __repr__(self) -> str:
        return f"Names({sorted(self._environment)!r})"


def _masked(value: object, name: object = None) -> object:
    """``value`` with each environment it holds replaced by ``Names``.

    An environment is a mapping passed, or held, under one of
    ``_ENVIRONMENTS``: as an argument, a mapping's key or a record's field.
    Mappings, lists, tuples and dataclass records are read through, and
    copied only when something in them changes (a mapping into a ``dict``);
    any other value is passed as it is.
    """
    if isinstance(value, Names):
        return value
    if isinstance(value, Mapping):
        if name in _ENVIRONMENTS:
            return Names(value)
        items = {key: _masked(item, key) for key, item in value.items()}
        changed = any(items[key] is not item for key, item in value.items())
        return items if changed else value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: _masked(getattr(value, field.name), field.name)
            for field in dataclasses.fields(value)
            if field.init and field.name in _ENVIRONMENTS
        }
        changes = {
            field: masked
            for field, masked in changes.items()
            if masked is not getattr(value, field)
        }
        return dataclasses.replace(value, **changes) if changes else value
    if type(value) in (list, tuple):
        elements = [_masked(element) for element in value]
        changed = any(new is not old for new, old in zip(elements, value))
        return type(value)(elements) if changed else value
    return value


def _refuse(name: str) -> NoReturn:
    # Raised in a frame whose one argument is ``name``: the wrapper's frames,
    # which hold the environments, are hidden, and the error is not chained.
    raise TypeError(f"could not hide the environments {name} is passed") from None


def hides_environment(function: _Function) -> _Function:
    """Pass ``function`` a ``Names`` in place of each environment it is passed.

    For a fake standing in for a launcher, and for any other function a test
    module passes an environment: should an exception pass through its frame,
    pytest prints its arguments, and they now hold names alone. What it calls
    with them (a recorder, the real launcher) is passed the same. The wrapper's
    own frames, which hold the environments themselves, are hidden from
    pytest's traceback, as ``pytest.fail``'s is. A method's ``self`` (or
    ``cls``) is passed as it is. It wraps functions, coroutine functions and
    generator functions (a fixture that yields, say).
    """
    if inspect.isasyncgenfunction(function):
        raise TypeError(f"hides_environment does not wrap {function.__qualname__}")
    parameters = [
        parameter.name
        for parameter in inspect.signature(function).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    kept = 1 if parameters[:1] in (["self"], ["cls"]) else 0

    def masked(
        args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[list[object], dict[str, object]]:
        __tracebackhide__ = True
        try:
            positional = [
                argument
                if position < kept
                else _masked(argument, (parameters[position:] or [None])[0])
                for position, argument in enumerate(args)
            ]
            named = {name: _masked(value, name) for name, value in kwargs.items()}
        except Exception:
            _refuse(function.__qualname__)
        return positional, named

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def hidden_coroutine(*args: object, **kwargs: object) -> object:
            __tracebackhide__ = True
            positional, named = masked(args, kwargs)
            return await function(*positional, **named)

        return hidden_coroutine  # type: ignore[return-value]

    if inspect.isgeneratorfunction(function):

        @functools.wraps(function)
        def hidden_generator(*args: object, **kwargs: object) -> object:
            __tracebackhide__ = True
            positional, named = masked(args, kwargs)
            return (yield from function(*positional, **named))

        return hidden_generator  # type: ignore[return-value]

    @functools.wraps(function)
    def hidden(*args: object, **kwargs: object) -> object:
        __tracebackhide__ = True
        positional, named = masked(args, kwargs)
        return function(*positional, **named)

    return hidden  # type: ignore[return-value]


def _raise(error: BaseException | type[BaseException]) -> NoReturn:
    raise error


def raising(error: BaseException | type[BaseException]) -> Callable[..., NoReturn]:
    """A side effect that raises ``error``, for a mock standing in for a launcher.

    A mock raises the exception it is given as its side effect in a frame of
    ``unittest.mock``'s whose printed arguments are its call's, environment
    included. This side effect raises it from a frame whose one argument is
    the exception; its own frame, which holds the call's arguments, is hidden.
    """

    def side_effect(*args: object, **kwargs: object) -> NoReturn:
        __tracebackhide__ = True
        _raise(error)

    return side_effect
