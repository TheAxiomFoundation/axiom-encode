"""Find the commands a GitHub Actions ``run:`` script executes, and parse uv calls.

This is a small bash reader for auditing how this repository's workflows
install Python. The threat model is an accidental regression in a reviewed
workflow, written in ordinary shell. It is not a sandbox against deliberately
obfuscated bash: it cannot see a command assembled in a variable
(``"$cmd" sync``), a script run by path, or one fetched at run time.

Within that model it follows:

- quoting (a quoted ``;`` or ``>`` is data), ``$'...'`` strings, comments and
  line continuations;
- ``$(...)``, backtick and ``<(...)`` substitutions, also inside ``$((...))``
  and ``((...))``. Each runs just before the command that contains it;
- heredocs: bodies are data, though ``$(...)`` in an unquoted body runs;
- ``case`` patterns, array literals and ``[[ ... ]]`` tests (data), and
  ``function`` and ``coproc`` bodies (commands);
- ``bash``/``sh -c``, ``su -c``, ``eval`` and ``trap`` scripts, and ``find -exec``
  commands. Nested scripts inherit the caller's ``NAME=value`` assignments;
- the wrappers ``sudo``, ``env`` (including ``-S``), ``timeout``, ``nice``,
  ``nohup``, ``time``, ``exec``, ``stdbuf``, ``xargs`` and ``command``, with
  their full option syntax;
- through :func:`expand`, the command ``uv run``, ``uv tool run`` or ``uvx``
  launches.

``uv`` calls are parsed against the option table of the pinned uv release
(``tests/fixtures/uv_cli_options.json``), so a value is never mistaken for a
subcommand or for ``uv run``'s command.

The reader fails closed by raising :class:`UnanalyzableScript` on:

- an unterminated quote, substitution or heredoc, or a ``case`` left open
  inside a substitution;
- an unknown uv option or subcommand, or an unknown option to a wrapper;
- an ``env -S`` value with its own escapes, ``${VAR}`` expansions or a
  ``#`` comment, and a ``mapfile``/``readarray -C`` callback;
- an ``env`` or ``sudo`` assignment whose name is computed (``"${p}_X=1"``);
- a shell that reads its script from stdin or a file, ``source``/``.``, or
  ``su`` without ``-c``;
- a program produced by a substitution or a ``${{ }}`` expression;
- a program in ``UNSUPPORTED_RUNNERS``, which run a command this reader does
  not follow.

Regenerate the option table after moving the uv pin::

    .venv/bin/python -m tests.workflow_shell > tests/fixtures/uv_cli_options.json
"""

from __future__ import annotations

import itertools
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

UV_OPTIONS_PATH = Path(__file__).parent / "fixtures" / "uv_cli_options.json"
# `uv <command> --help` pages the option table covers ("" is `uv --help`).
UV_COMMANDS = (
    "",
    "add",
    "export",
    "lock",
    "pip compile",
    "pip freeze",
    "pip install",
    "pip show",
    "pip sync",
    "pip uninstall",
    "python install",
    "remove",
    "run",
    "sync",
    "tool install",
    "tool run",
    "venv",
    "version",
)
_UV_GROUPS = frozenset({"pip", "python", "tool"})
# Hidden clap aliases that `--help` does not list: command -> alias -> option.
# `uv_option_table` probes each one against the uv it builds the table from.
UV_HIDDEN_ALIASES = {
    "pip compile": {"--constraint": "--constraints", "--override": "--overrides"},
    "pip install": {
        "--constraint": "--constraints",
        "--override": "--overrides",
        "--requirement": "--requirements",
    },
}
# Hidden options that are not aliases: command -> {"flags": [...], "valued": [...]}.
# `uv_option_table` probes each one as well.
UV_HIDDEN_OPTIONS: dict[str, dict[str, list[str]]] = {}
_REGENERATE = (
    "regenerate it with `.venv/bin/python -m tests.workflow_shell > "
    "tests/fixtures/uv_cli_options.json`"
)
_HELP_OPTION = re.compile(
    r"^ {2,6}(?:(-[A-Za-z0-9]), )?(--[a-z0-9][a-z0-9-]*)(?:\.\.\.)?"
    r"(?: (<[^>]+>)(?:\.\.\.)?)?(?:\s|$)"
)
# Programs that run another command in a way this reader does not follow.
UNSUPPORTED_RUNNERS = frozenset(
    {
        "arch",
        "busybox",
        "caffeinate",
        "chroot",
        "chrt",
        "doas",
        "expect",
        "fakeroot",
        "faketime",
        "firejail",
        "flock",
        "gdb",
        "ionice",
        "ltrace",
        "newgrp",
        "nsenter",
        "parallel",
        "pkexec",
        "runuser",
        "script",
        "setsid",
        "sg",
        "ssh",
        "strace",
        "systemd-run",
        "taskset",
        "unbuffer",
        "unshare",
        "valgrind",
        "watch",
        "xvfb-run",
    }
)

_GHA_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?\+?=.*", re.DOTALL)
# Substitution placeholders use private-use characters, so script text cannot
# collide with them. The scope keeps a nested script's own substitutions apart
# from its caller's, which have already run.
_PLACEHOLDER = re.compile("(\\d+)\\.(\\d+)")
_SCOPES = itertools.count()
# Longest first, so `;;` is not read as two `;`.
_OPERATORS = (
    ";;&",
    "<<<",
    "<<-",
    "&>>",
    ";;",
    ";&",
    "&&",
    "||",
    "|&",
    "<<",
    ">>",
    "<&",
    ">&",
    "&>",
    "<>",
    ">|",
    ";",
    "&",
    "|",
    "(",
    ")",
    "<",
    ">",
)
_OPERATOR_CHARS = frozenset("();<>|&")
_CASE_ARM_ENDS = frozenset({";;", ";&", ";;&"})
_KEYWORDS = frozenset(
    {"!", "{", "}", "if", "then", "else", "elif", "fi", "do", "done", "while"}
    | {"until"}
)
_HEADERS = frozenset({"for", "select"})
_SHELLS = frozenset({"bash", "sh", "dash", "zsh"})
_SHELL_VALUED = frozenset({"--init-file", "--rcfile"})
_FIND_ACTIONS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


@dataclass(frozen=True)
class _Wrapper:
    flags: frozenset[str]
    valued: frozenset[str]
    positionals: int = 0  # words between the options and the command
    numeric_flags: bool = False  # `nice -10`
    # Options whose argument is optional and only ever attached:
    # `--replace[=R]`, `-i[R]`.
    optional: frozenset[str] = frozenset()


def _options(*names: str) -> frozenset[str]:
    return frozenset(names)


_WRAPPERS = {
    "sudo": _Wrapper(
        _options(
            *("-A", "-b", "-E", "-e", "-H", "-i", "-K", "-k", "-l", "-n", "-P", "-S"),
            *("-s", "-V", "-v", "--askpass", "--background", "--edit", "--help"),
            *("--list", "--login", "--non-interactive", "--preserve-env"),
            *("--preserve-groups", "--remove-timestamp", "--reset-timestamp"),
            *("--set-home", "--shell", "--stdin", "--validate", "--version"),
        ),
        _options(
            *("-C", "-D", "-g", "-h", "-p", "-R", "-r", "-T", "-t", "-U", "-u"),
            *("--chdir", "--chroot", "--close-from", "--command-timeout", "--group"),
            *("--host", "--other-user", "--prompt", "--role", "--type", "--user"),
        ),
    ),
    "exec": _Wrapper(_options("-c", "-l"), _options("-a")),
    "nice": _Wrapper(_options(), _options("-n", "--adjustment"), numeric_flags=True),
    "nohup": _Wrapper(_options(), _options()),
    "stdbuf": _Wrapper(
        _options(), _options("-e", "-i", "-o", "--error", "--input", "--output")
    ),
    "time": _Wrapper(
        _options("-a", "-p", "-q", "-v", "--append", "--portability", "--quiet"),
        _options("-f", "-o", "--format", "--output"),
    ),
    "timeout": _Wrapper(
        _options("-v", "--foreground", "--preserve-status", "--verbose"),
        _options("-k", "-s", "--kill-after", "--signal"),
        positionals=1,
    ),
    "xargs": _Wrapper(
        _options(
            *("-0", "-o", "-p", "-r", "-t", "-x", "--exit", "--interactive"),
            *("--no-run-if-empty", "--null", "--open-tty", "--verbose"),
        ),
        _options(
            *("-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s", "--arg-file"),
            *("--delimiter", "--max-args", "--max-chars", "--max-procs"),
            "--process-slot-var",
        ),
        optional=_options("-e", "-i", "-l", "--eof", "--max-lines", "--replace"),
    ),
}


class UnanalyzableScript(ValueError):
    """A run script uses syntax this reader cannot follow."""


@dataclass(frozen=True)
class Command:
    """One simple command, with wrappers such as ``sudo`` removed."""

    words: tuple[str, ...]
    line: str
    # NAME=value words that set the command's environment, as written,
    # including those inherited from a caller such as `A=1 bash -c '...'`.
    assignments: tuple[str, ...] = ()

    @property
    def program(self) -> str:
        """The program's file name, or "" for a line that only assigns."""
        return self.words[0].rsplit("/", 1)[-1] if self.words else ""


@dataclass(frozen=True)
class UvInvocation:
    subcommand: str  # "sync", "pip install", "tool run", "uvx", or "" for `uv -V`
    # Long option names, e.g. "--with" for "-wrequests" or "--with=requests".
    options: tuple[str, ...]
    command: Command
    # The command `uv run`, `uv tool run` or `uvx` runs, if any.
    runs: tuple[str, ...] = ()


@dataclass
class _Substitution:
    lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Token:
    text: str
    operator: bool
    spaced: bool  # whitespace precedes it


@dataclass(frozen=True)
class _Script:
    text: str
    assignments: tuple[str, ...]


class _Reader:
    """Split a script into logical lines.

    Each substitution becomes a placeholder, and its own lines are stored at
    that index of the shared ``registry``.
    """

    def __init__(self, text: str, registry: list[_Substitution], scope: int) -> None:
        self.text = text
        self.i = 0
        self.registry = registry
        self.scope = scope

    def _error(self, message: str) -> UnanalyzableScript:
        start = self.text.rfind("\n", 0, self.i) + 1
        end = self.text.find("\n", self.i)
        line = self.text[start : end if end != -1 else len(self.text)]
        return UnanalyzableScript(f"{message}: {line.strip()!r}")

    def _register(self, substitution: _Substitution) -> str:
        self.registry.append(substitution)
        return f"{self.scope}.{len(self.registry) - 1}"

    def _single_quoted(self) -> str:
        end = self.text.find("'", self.i + 1)
        if end == -1:
            raise self._error("unterminated single quote")
        quoted = self.text[self.i : end + 1]
        self.i = end + 1
        return quoted

    def _ansi_c_quoted(self) -> str:
        text = self.text
        end = self.i + 2
        while end < len(text) and text[end] != "'":
            end += 2 if text[end] == "\\" else 1
        if end >= len(text):
            raise self._error("unterminated $'...' string")
        quoted = text[self.i : end + 1]
        self.i = end + 1
        return quoted

    def lines(self, closer: str | None = None) -> list[str]:
        """Read logical lines until the end, or an unmatched ``closer``."""
        lines: list[str] = []
        out: list[str] = []
        heredocs: list[tuple[int, str, bool, bool]] = []
        depth = 0
        word_start = True
        text = self.text
        while self.i < len(text):
            c = text[self.i]
            if closer is not None and c == closer and depth == 0:
                if _open_cases("\n".join([*lines, "".join(out)])):
                    # The `)` ends a case pattern, not the substitution.
                    out.append(c)
                    self.i += 1
                    continue
                self.i += 1
                if heredocs:
                    raise self._error("heredoc left open inside a substitution")
                lines.append("".join(out))
                return lines
            if c == "\\":
                if text.startswith("\\\n", self.i):
                    self.i += 2
                    continue
                out.append(text[self.i : self.i + 2])
                self.i += 2
                word_start = False
                continue
            if text.startswith("$'", self.i):
                out.append(self._ansi_c_quoted())
                word_start = False
                continue
            if c == "'":
                out.append(self._single_quoted())
                word_start = False
                continue
            if c == '"':
                out.append(self._double_quoted())
                word_start = False
                continue
            if word_start and text.startswith("((", self.i):
                # An arithmetic command runs nothing but its substitutions.
                out.append(": " + self._substitution())
                word_start = False
                continue
            if (
                c == "`"
                or text.startswith("$(", self.i)
                or (c in "<>" and text.startswith("(", self.i + 1))
            ):
                out.append(self._substitution())
                word_start = False
                continue
            if text.startswith("<<<", self.i):
                out.append(" <<< ")  # a here-string: its word is data
                self.i += 3
                word_start = True
                continue
            if text.startswith("<<", self.i):
                delimiter, expands, strip_tabs = self._heredoc_header()
                placeholder = self._register(_Substitution())
                index = len(self.registry) - 1
                heredocs.append((index, delimiter, expands, strip_tabs))
                out.append(f" << {placeholder} ")
                word_start = True
                continue
            if c == "#" and word_start:
                end = text.find("\n", self.i)
                self.i = len(text) if end == -1 else end
                continue
            if c == "\n":
                self.i += 1
                lines.append("".join(out))
                out = []
                for index, delimiter, expands, strip_tabs in heredocs:
                    self.registry[index] = self._heredoc_body(
                        delimiter, expands, strip_tabs
                    )
                heredocs = []
                word_start = True
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            out.append(c)
            self.i += 1
            word_start = c in " \t" or c in _OPERATOR_CHARS
        if closer is not None:
            raise self._error(f"unterminated substitution (missing {closer!r})")
        if heredocs:
            raise self._error("heredoc without a body")
        lines.append("".join(out))
        return lines

    def _double_quoted(self) -> str:
        text = self.text
        out = ['"']
        self.i += 1
        while self.i < len(text):
            c = text[self.i]
            if c == "\\":
                out.append(text[self.i : self.i + 2])
                self.i += 2
            elif c == '"':
                self.i += 1
                out.append('"')
                return "".join(out)
            elif c == "`" or text.startswith("$(", self.i):
                out.append(self._substitution())
            else:
                out.append(c)
                self.i += 1
        raise self._error("unterminated double quote")

    def substitutions(self) -> list[str]:
        """Placeholders for the substitutions in plain text (a heredoc body)."""
        found: list[str] = []
        text = self.text
        while self.i < len(text):
            if text[self.i] == "\\":
                self.i += 2
            elif text[self.i] == "`" or text.startswith("$(", self.i):
                found.append(self._substitution())
            else:
                self.i += 1
        return found

    def _substitution(self) -> str:
        text = self.text
        if text.startswith("$((", self.i) or text.startswith("((", self.i):
            # Arithmetic runs no command, but a substitution inside it does.
            self.i += 3 if text[self.i] == "$" else 2
            depth = 2
            inner: list[str] = []
            while self.i < len(text):
                if text[self.i] == "`" or (
                    text.startswith("$(", self.i) and not text.startswith("$((", self.i)
                ):
                    inner.append(self._substitution())
                    continue
                if text[self.i] == "(":
                    depth += 1
                elif text[self.i] == ")":
                    depth -= 1
                    if depth == 0:
                        self.i += 1
                        return "0" + "".join(inner)
                self.i += 1
            raise self._error("unterminated arithmetic expansion")
        if text[self.i] == "`":
            end = self.i + 1
            while end < len(text) and text[end] != "`":
                end += 2 if text[end] == "\\" else 1
            if end >= len(text):
                raise self._error("unterminated backtick substitution")
            # Inside backticks, a backslash quotes only `$`, a backtick or itself.
            source = re.sub(r"\\([$`\\])", r"\1", text[self.i + 1 : end])
            self.i = end + 1
            inner = _Reader(source, self.registry, self.scope)
            return self._register(_Substitution(inner.lines()))
        self.i += 2  # "$(", "<(" or ">("
        return self._register(_Substitution(self.lines(closer=")")))

    def _heredoc_header(self) -> tuple[str, bool, bool]:
        text = self.text
        self.i += 2
        strip_tabs = text.startswith("-", self.i)
        if strip_tabs:
            self.i += 1
        while self.i < len(text) and text[self.i] in " \t":
            self.i += 1
        match = re.match(
            r"""'([^']+)'|"([^"]+)"|\\?([^\s;&|<>()'"`]+)""", text[self.i :]
        )
        if not match:
            raise self._error("unreadable heredoc delimiter")
        self.i += match.end()
        delimiter = next(group for group in match.groups() if group)
        expands = match.group(3) is not None and not match.group(0).startswith("\\")
        return delimiter, expands, strip_tabs

    def _heredoc_body(
        self, delimiter: str, expands: bool, strip_tabs: bool
    ) -> _Substitution:
        text = self.text
        body: list[str] = []
        while self.i < len(text):
            end = text.find("\n", self.i)
            end = len(text) if end == -1 else end
            line = text[self.i : end]
            self.i = min(end + 1, len(text))
            # Like bash: the whole line, less leading tabs for `<<-`.
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                if not expands:
                    return _Substitution()
                # An unquoted body is data, but its substitutions still run.
                joined = "\n".join(body)
                found = _Reader(joined, self.registry, self.scope).substitutions()
                return _Substitution(
                    lines=[
                        line
                        for placeholder in found
                        for _, index in _PLACEHOLDER.findall(placeholder)
                        for line in self.registry[int(index)].lines
                    ]
                )
            body.append(line)
        raise UnanalyzableScript(f"heredoc {delimiter!r} is never closed")


_CASE_WORD = re.compile(
    r"(?:^|[;&|(\n]|\b(?:then|do|else|elif|if|while|until|!))\s*(case|esac)(?=\s|;|$)"
)


_SIMPLE_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}
_NUMERIC_ESCAPES = {"x": 2, "u": 4, "U": 8}  # hex digits each may take


def _ansi_c(raw: str) -> str:
    """Decode the body of a bash ``$'...'`` string as bash does."""
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] != "\\" or i + 1 >= len(raw):
            out.append(raw[i])
            i += 1
            continue
        escape = raw[i + 1]
        if escape in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[escape])
            i += 2
        elif escape in "01234567":
            digits = re.match(r"[0-7]{1,3}", raw[i + 1 :]).group()
            out.append(chr(int(digits, 8) & 0xFF))
            i += 1 + len(digits)
        elif escape in _NUMERIC_ESCAPES:
            width = _NUMERIC_ESCAPES[escape]
            digits = re.match(f"[0-9A-Fa-f]{{1,{width}}}", raw[i + 2 :])
            if digits and int(digits.group(), 16) <= 0x10FFFF:
                out.append(chr(int(digits.group(), 16)))
                i += 2 + len(digits.group())
            else:
                out.append(raw[i : i + 2])  # bash keeps a malformed escape
                i += 2
        elif escape == "c" and i + 2 < len(raw):
            out.append(chr(ord(raw[i + 2]) & 0x1F))
            i += 3
        else:
            out.append(raw[i : i + 2])
            i += 2
    return "".join(out)


def _split(value: str) -> list[str]:
    """Split an `env -S` value; its own escapes and ${VAR} are not followed."""
    if any(ch in value for ch in "\\$#"):
        raise UnanalyzableScript(f"env -S escapes, variables or comments in {value!r}")
    try:
        return shlex.split(value)
    except ValueError as error:
        raise UnanalyzableScript(f"{error}: {value!r}") from None


def _open_cases(text: str) -> bool:
    """Whether ``text`` opens more ``case`` statements than it closes.

    Only unquoted words in command position count, so `printf 'use case'`
    and `echo "if case x in"` do not.
    """
    unquoted = re.sub(r"'[^']*'|\"(?:\\\\.|[^\"\\\\])*\"", "''", text)
    words = _CASE_WORD.findall(unquoted)
    return words.count("case") > words.count("esac")


def _normalize_expressions(script: str) -> str:
    # GitHub substitutes ${{ }} before bash runs; keep each one a single word.
    return _GHA_EXPRESSION.sub(
        lambda m: "${{" + re.sub(r"\s+", "", m.group(1)) + "}}", script
    )


def _tokens(line: str) -> list[_Token]:
    """Split a logical line into words and operators, keeping quoting apart."""
    tokens: list[_Token] = []
    word: list[str] = []
    quoted = False
    started = False
    spaced = True

    def flush() -> None:
        nonlocal word, quoted, started, spaced
        if started:
            tokens.append(_Token("".join(word), False, spaced))
            spaced = False
        word, quoted, started = [], False, False

    i = 0
    while i < len(line):
        c = line[i]
        if c in " \t\n":
            flush()
            spaced = True
            i += 1
        elif c == "\\":
            word.append(line[i + 1 : i + 2])
            quoted = started = True
            i += 2
        elif line.startswith("$'", i):
            end = i + 2
            while end < len(line) and line[end] != "'":
                end += 2 if line[end] == "\\" else 1
            if end >= len(line):
                raise UnanalyzableScript(f"unterminated $'...': {line.strip()!r}")
            word.append(_ansi_c(line[i + 2 : end]))
            quoted = started = True
            i = end + 1
        elif c == "'":
            end = line.find("'", i + 1)
            if end == -1:
                raise UnanalyzableScript(f"unterminated single quote: {line.strip()!r}")
            word.append(line[i + 1 : end])
            quoted = started = True
            i = end + 1
        elif c == '"' or line.startswith('$"', i):
            i += 2 if c == "$" else 1
            while i < len(line) and line[i] != '"':
                if line.startswith("\\\n", i):
                    i += 2  # a line continuation, removed as bash removes it
                    continue
                if line[i] == "\\" and line[i + 1 : i + 2] in {"$", "`", '"', "\\"}:
                    i += 1
                word.append(line[i])
                i += 1
            if i >= len(line):
                raise UnanalyzableScript(f"unterminated double quote: {line.strip()!r}")
            quoted = started = True
            i += 1
        elif c in _OPERATOR_CHARS:
            operator = next(op for op in _OPERATORS if line.startswith(op, i))
            if (
                operator[0] in "<>"
                and started
                and not quoted
                and "".join(word).isdigit()
            ):
                word, started = [], False  # a file-descriptor prefix: 2>&1
            flush()
            tokens.append(_Token(operator, True, spaced))
            spaced = False
            i += len(operator)
        else:
            word.append(c)
            started = True
            i += 1
    flush()
    return tokens


def _read_wrapper_options(name: str, spec: _Wrapper, words: list[str]) -> int:
    """Index of the first word after ``name``'s options; unknown ones raise."""
    index = 1
    while index < len(words) and words[index].startswith("-") and words[index] != "-":
        arg = words[index]
        if arg == "--":
            return index + 1
        if arg.startswith("--"):
            option, has_value, _ = arg.partition("=")
            if option in spec.valued:
                index += 1 if has_value else 2
            elif option in spec.flags or option in spec.optional:
                index += 1
            else:
                raise UnanalyzableScript(f"unknown {name} option {arg!r}")
            continue
        if spec.numeric_flags and arg[1:].isdigit():
            index += 1
            continue
        for offset, letter in enumerate(arg[1:], start=1):
            option = "-" + letter
            if option in spec.valued:
                index += 1 if offset < len(arg) - 1 else 2
                break
            if option in spec.optional:
                index += 1  # any rest of the cluster is its argument
                break
            if option not in spec.flags:
                raise UnanalyzableScript(f"unknown {name} option {option!r} in {arg!r}")
        else:
            index += 1
    return index


_ENV_FLAGS = frozenset({"-", "--debug", "--ignore-environment", "--null"})
# NAME=value where NAME is built from an expansion: `"${p}_X=1"`.
_COMPUTED_ASSIGNMENT = re.compile(r"[^=\s]*[$`\ue000][^=\s]*=")
_ENV_VALUED = frozenset({"--chdir", "--unset"})


def _env_command(words: list[str]) -> tuple[list[str], list[str]]:
    """Split ``env ...`` into its assignments and the command it runs.

    Options are read in order as GNU env does. `-S` splits its value into
    words that are read as if they had been written in its place.
    """
    args = list(words[1:])
    assignments: list[str] = []
    while args:
        arg = args[0]
        if arg == "--":
            args = args[1:]
            break
        if arg in _ENV_FLAGS:
            args = args[1:]
        elif arg in _ENV_VALUED:
            args = args[2:]
        elif arg == "--split-string":
            if len(args) < 2:
                raise UnanalyzableScript("env --split-string without a value")
            args = [*_split(args[1]), *args[2:]]
        elif arg.startswith("--split-string="):
            args = [*_split(arg.partition("=")[2]), *args[1:]]
        elif arg.startswith(("--chdir=", "--unset=")):
            args = args[1:]
        elif arg.startswith("--"):
            raise UnanalyzableScript(f"unknown env option {arg!r}")
        elif arg.startswith("-") and len(arg) > 1:
            rest = args[1:]
            for offset, letter in enumerate(arg[1:], start=1):
                if letter in "0iv":
                    continue
                if letter not in "CPSu":
                    raise UnanalyzableScript(f"unknown env option -{letter} in {arg!r}")
                value = arg[offset + 1 :]
                if not value:
                    if not rest:
                        raise UnanalyzableScript(f"env -{letter} without a value")
                    value, rest = rest[0], rest[1:]
                if letter == "S":
                    rest = [*_split(value), *rest]
                break
            args = rest
        elif _ASSIGNMENT.fullmatch(arg):
            assignments.append(arg)
            args = args[1:]
        elif _COMPUTED_ASSIGNMENT.match(arg):
            raise UnanalyzableScript(f"env assigns a computed name: {arg!r}")
        else:
            break
    return assignments, args


def _shell_script(words: list[str]) -> str:
    """The script a shell runs with ``-c``; it cannot see stdin or files."""
    name = words[0].rsplit("/", 1)[-1]
    runs_string = False
    index = 1
    while index < len(words):
        arg = words[index]
        if arg in {"-", "--"}:
            index += 1
            break
        if arg.startswith("--"):
            index += 2 if arg in _SHELL_VALUED else 1
        elif re.fullmatch(r"[-+][A-Za-z]+", arg):
            runs_string = runs_string or (arg[0] == "-" and "c" in arg)
            # -o/-O (and +o/+O) take the next word as their value.
            index += 1 + sum(letter in "oO" for letter in arg[1:])
        else:
            break
    if runs_string and index < len(words):
        return words[index]  # bash runs the first non-option argument
    raise UnanalyzableScript(
        f"{name} reads its script from stdin or a file: {' '.join(words)!r}"
    )


def _unwrap(words: list[str]) -> list[tuple[list[str], list[str]] | _Script]:
    """Strip keywords and wrappers from one simple command.

    Returns the commands it runs, each as (words, assignments), and the
    scripts it runs (``bash -c``, ``eval``, ``trap``) with the assignments
    they inherit.
    """
    assignments: list[str] = []

    def script(text: str) -> list[tuple[list[str], list[str]] | _Script]:
        return [_Script(text, tuple(assignments))]

    while words:
        word = words[0]
        name = word.rsplit("/", 1)[-1]
        if _ASSIGNMENT.fullmatch(word):
            assignments.append(word)
            words = words[1:]
        elif word in _KEYWORDS:
            words = words[1:]
        elif word == "function":
            words = words[2:]  # `function NAME`; its body follows `{`
        elif word == "coproc":
            # `coproc NAME { ...; }` or `coproc command`.
            words = words[2:] if words[2:3] == ["{"] else words[1:]
        elif word in _HEADERS:
            return []
        elif name in UNSUPPORTED_RUNNERS:
            raise UnanalyzableScript(f"{name} runs a command this reader cannot follow")
        elif name in _SHELLS:
            return script(_shell_script(words))
        elif name == "su":
            for index, arg in enumerate(words[1:], start=1):
                if arg.startswith("--command="):
                    return script(arg.partition("=")[2])
                if arg in {"-c", "--command"} and index + 1 < len(words):
                    return script(words[index + 1])
            raise UnanalyzableScript(f"su without -c: {' '.join(words)!r}")
        elif name == "eval":
            args = words[2:] if words[1:2] == ["--"] else words[1:]
            return script(" ".join(args))
        elif name == "trap":
            args = words[2:] if words[1:2] == ["--"] else words[1:]
            # `trap - SIG` resets, `trap '' SIG` ignores, `trap -p` prints.
            if args and args[0] and not args[0].startswith("-"):
                return script(args[0])
            return []
        elif name == "find":
            found: list[tuple[list[str], list[str]] | _Script] = []
            plain: list[str] = []
            index = 0
            while index < len(words):
                if words[index] in _FIND_ACTIONS:
                    end = index + 1
                    while end < len(words) and words[end] not in {";", "+"}:
                        end += 1
                    found.extend(_unwrap(words[index + 1 : end]))
                    index = end + 1
                else:
                    plain.append(words[index])
                    index += 1
            return [(plain, assignments), *found]
        elif name == "command":
            args = words[1:]
            while args and args[0] in {"-p", "--"}:
                args = args[1:]
            if args and args[0] in {"-v", "-V"}:
                return []
            words = args
        elif name == "env":
            env_assignments, words = _env_command(words)
            assignments.extend(env_assignments)
        elif name in {"source", "."}:
            raise UnanalyzableScript(f"{name} runs a script file: {' '.join(words)!r}")
        elif name in {"mapfile", "readarray"} and any(
            re.match(r"-[A-Za-z]*C", arg) for arg in words[1:]
        ):
            raise UnanalyzableScript(f"{name} -C runs a callback: {' '.join(words)!r}")
        elif name == "builtin":
            words = words[2:] if words[1:2] == ["--"] else words[1:]
        elif word.startswith("${{"):
            raise UnanalyzableScript(
                f"the program comes from a workflow expression: {' '.join(words)!r}"
            )
        elif name in _WRAPPERS:
            spec = _WRAPPERS[name]
            index = _read_wrapper_options(name, spec, words)
            while index < len(words) and name == "sudo":
                if _ASSIGNMENT.fullmatch(words[index]):
                    assignments.append(words[index])
                    index += 1
                elif _COMPUTED_ASSIGNMENT.match(words[index]):
                    raise UnanalyzableScript(
                        f"sudo assigns a computed name: {words[index]!r}"
                    )
                else:
                    break
            words = words[index + spec.positionals :]
        else:
            if _PLACEHOLDER.search(word):
                raise UnanalyzableScript(
                    f"the program comes from a substitution: {' '.join(words)!r}"
                )
            return [(words, assignments)]
    return [([], assignments)] if assignments else []


def _commands_in(
    lines: list[str],
    registry: list[_Substitution],
    scope: int,
    inherited: tuple[str, ...],
) -> list[Command]:
    found: list[Command] = []
    words: list[str] = []
    line = ""
    in_header = False  # between `case` and `in`
    in_pattern = False  # a case pattern, up to its `)`
    closer: str | None = None  # `)` of an array literal or `]]` of a test
    depth = 0  # nested `case` statements

    def run(text: str) -> None:
        for placeholder_scope, index in _PLACEHOLDER.findall(text):
            if int(placeholder_scope) != scope:
                continue  # the caller's substitution, which already ran
            if int(index) >= len(registry):
                raise UnanalyzableScript(f"unknown substitution in {line.strip()!r}")
            substitution = registry[int(index)]
            found.extend(_commands_in(substitution.lines, registry, scope, inherited))

    def at_start() -> bool:
        return all(word in _KEYWORDS for word in words)

    def finish() -> None:
        nonlocal words
        for word in words:
            run(word)
        for item in _unwrap(words):
            if isinstance(item, _Script):
                for command in commands(item.text):
                    found.append(
                        Command(
                            command.words,
                            command.line,
                            (*inherited, *item.assignments, *command.assignments),
                        )
                    )
            else:
                program, assignments = item
                found.append(
                    Command(tuple(program), line.strip(), (*inherited, *assignments))
                )
        words = []

    for line in lines:
        tokens = _tokens(line)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if closer is not None:
                run(token.text)
                if token.text == closer and (closer == ")") == token.operator:
                    closer = None
                continue
            if in_header:
                run(token.text)
                in_header = not (not token.operator and token.text == "in")
                in_pattern = not in_header
            elif in_pattern:
                run(token.text)  # bash expands pattern words
                if not token.operator and token.text == "esac":
                    depth -= 1
                    in_pattern = False
                elif token.operator and token.text == ")":
                    in_pattern = False
            elif token.operator and any(ch in token.text for ch in "<>"):
                # A redirection: its target is data, but may hold a substitution.
                if index < len(tokens) and not tokens[index].operator:
                    run(tokens[index].text)
                    index += 1
            elif (
                token.operator
                and token.text == "("
                and not token.spaced
                and words
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\+?=", words[-1])
            ):
                closer = ")"  # an array literal: data
            elif token.operator:
                finish()
                in_pattern = depth > 0 and token.text in _CASE_ARM_ENDS
            elif at_start() and token.text == "case":
                depth += 1
                in_header = True
            elif at_start() and token.text == "esac" and depth:
                depth -= 1
            elif at_start() and token.text == "[[":
                words.append(token.text)
                closer = "]]"  # a test: its operators are data
            else:
                words.append(token.text)
        finish()
    if closer is not None:
        raise UnanalyzableScript(f"unterminated {closer!r} in {line.strip()!r}")
    return found


def commands(script: str) -> list[Command]:
    """Every simple command ``script`` runs, in the order bash runs them.

    A substitution runs just before the command that contains it, and a
    ``bash -c`` script is expanded where it runs.
    """
    scope = next(_SCOPES)
    registry: list[_Substitution] = []
    lines = _Reader(_normalize_expressions(script), registry, scope).lines()
    return _commands_in(lines, registry, scope, ())


def expand(found: list[Command], table: dict) -> list[Command]:
    """``found`` plus, after each ``uv run``/``uvx`` call, what it launches."""
    expanded: list[Command] = []
    for command in found:
        expanded.append(command)
        invocation = uv_invocation(command, table)
        if invocation is not None and invocation.runs:
            runs = invocation.runs
            if "--module" in invocation.options:
                runs = ("python", "-m", *runs)
            inner = [
                Command(c.words, command.line, (*command.assignments, *c.assignments))
                for c in commands(shlex.join(runs))
            ]
            expanded.extend(expand(inner, table))
    return expanded


def python_module(words: tuple[str, ...]) -> str | None:
    """The module ``python ... -m MODULE`` runs, or None for a script or -c."""
    index = 1
    while index < len(words):
        arg = words[index]
        if arg == "--" or not arg.startswith("-") or arg == "-":
            return None
        if arg.startswith("--"):
            index += 2 if arg == "--check-hash-based-pycs" else 1
            continue
        for offset, letter in enumerate(arg[1:], start=1):
            rest = arg[offset + 1 :]
            if letter == "c":
                return None
            if letter == "m":
                if rest:
                    return rest
                return words[index + 1] if index + 1 < len(words) else None
            if letter in "WX":
                index += 0 if rest else 1  # its value is the rest or the next word
                break
        index += 1
    return None


def load_uv_options() -> dict:
    return json.loads(UV_OPTIONS_PATH.read_text(encoding="utf-8"))


def uv_option_table(uv: str = "uv") -> dict:
    """Build the option table from ``uv <command> --help`` for each command."""

    def table(command: str) -> dict:
        page = subprocess.run(
            [uv, *command.split(), "--help"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        flags: set[str] = set()
        valued: set[str] = set()
        shorts: dict[str, str] = {}
        for line in page.splitlines():
            match = _HELP_OPTION.match(line)
            if match:
                short, long, value = match.groups()
                (valued if value else flags).update(n for n in (short, long) if n)
                if short:
                    shorts[short] = long
        hidden = UV_HIDDEN_OPTIONS.get(command, {})
        for kind, names in (("flags", flags), ("valued", valued)):
            for name in hidden.get(kind, ()):
                probe = [uv, *command.split(), name]
                probe += ["x"] if kind == "valued" else []
                if subprocess.run([*probe, "--help"], capture_output=True).returncode:
                    raise RuntimeError(f"{' '.join(probe)} is not accepted")
                names.add(name)
        aliases = UV_HIDDEN_ALIASES.get(command, {})
        for alias, option in aliases.items():
            probe = [uv, *command.split(), alias]
            probe += ["x"] if option in valued else []
            accepted = subprocess.run(
                [*probe, "--help"], capture_output=True, text=True
            ).returncode
            if option not in flags | valued or accepted != 0:
                raise RuntimeError(f"{' '.join(probe)} is not an alias of {option}")
        return {
            "aliases": dict(sorted(aliases.items())),
            "flags": sorted(flags),
            "short": dict(sorted(shorts.items())),
            "valued": sorted(valued),
        }

    version = subprocess.run(
        [uv, "--version"], capture_output=True, text=True, check=True
    ).stdout.split()[1]
    return {
        "uv_version": version,
        "commands": {command: table(command) for command in UV_COMMANDS},
    }


def _read_option(
    args: tuple[str, ...], index: int, table: dict, options: list[str], where: str
) -> int:
    arg = args[index]
    if arg.startswith("--"):
        name, has_value, _ = arg.partition("=")
        name = table.get("aliases", {}).get(name, name)
        if name in table["valued"]:
            if not has_value and index + 1 >= len(args):
                raise UnanalyzableScript(f"uv option {arg!r} needs a value: {where!r}")
            options.append(name)
            return index + (1 if has_value else 2)
        if name in table["flags"] and not has_value:
            options.append(name)
            return index + 1
        raise UnanalyzableScript(
            f"unknown uv option {arg!r} in {where!r}; if the pinned uv accepts "
            "it, add it to UV_HIDDEN_ALIASES or UV_HIDDEN_OPTIONS and "
            f"{_REGENERATE}"
        )
    for offset, letter in enumerate(arg[1:], start=1):
        name = "-" + letter
        canonical = table.get("short", {}).get(name, name)
        if name in table["valued"]:
            attached = offset < len(arg) - 1
            if not attached and index + 1 >= len(args):
                raise UnanalyzableScript(f"uv option {name!r} needs a value: {where!r}")
            options.append(canonical)
            return index + (1 if attached else 2)
        if name not in table["flags"]:
            raise UnanalyzableScript(
                f"unknown uv option {name!r} in {where!r}; if the pinned uv "
                f"accepts it, add it to UV_HIDDEN_OPTIONS and {_REGENERATE}"
            )
        options.append(canonical)
    return index + 1


def uv_invocation(command: Command, table: dict) -> UvInvocation | None:
    """Parse ``command`` as a uv call, or return None if it does not run uv."""
    tables = table["commands"]
    args = command.words[1:]
    options: list[str] = []
    index = 0
    if command.program == "uvx":
        # `uvx` is `uv tool run`.
        while index < len(args) and args[index].startswith("-"):
            if args[index] == "--":
                index += 1
                break
            index = _read_option(args, index, tables["tool run"], options, command.line)
        return UvInvocation("uvx", tuple(options), command, tuple(args[index:]))
    if command.program != "uv":
        return None
    while index < len(args) and args[index].startswith("-"):
        index = _read_option(args, index, tables[""], options, command.line)
    if index == len(args):
        return UvInvocation("", tuple(options), command)
    subcommand = args[index]
    index += 1
    if subcommand in _UV_GROUPS and index < len(args):
        subcommand = f"{subcommand} {args[index]}"
        index += 1
    if subcommand not in tables:
        raise UnanalyzableScript(
            f"`uv {subcommand}` is not in {UV_OPTIONS_PATH.name}: {command.line!r}; "
            f"add it to UV_COMMANDS and {_REGENERATE}"
        )
    runs: tuple[str, ...] = ()
    while index < len(args):
        arg = args[index]
        if arg == "--":
            if subcommand in {"run", "tool run"}:
                runs = tuple(args[index + 1 :])
            break
        if arg.startswith("-") and arg != "-":
            index = _read_option(args, index, tables[subcommand], options, command.line)
        elif subcommand in {"run", "tool run"}:
            runs = tuple(args[index:])  # the command uv runs, with its arguments
            break
        else:
            index += 1
    return UvInvocation(subcommand, tuple(options), command, runs)


if __name__ == "__main__":
    print(json.dumps(uv_option_table(), indent=1, sort_keys=True))
