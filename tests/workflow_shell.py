"""Find the commands a GitHub Actions ``run:`` script executes, and parse uv calls.

This is a small bash reader, just enough to audit how workflows install
Python. It follows:

- quoting (a quoted ``;`` or ``>`` is data), comments and line continuations;
- heredocs, whose bodies are data, though ``$(...)`` in an unquoted body runs;
- ``$(...)``, backtick and ``<(...)`` substitutions, also inside ``$((...))``.
  Each runs just before the command that contains it;
- ``case`` patterns (data) and ``function`` bodies (commands);
- ``bash``/``sh -c``, ``su -c``, ``eval`` and ``trap`` scripts, and
  ``find -exec`` commands;
- the wrappers ``sudo``, ``env`` (including ``-S``), ``timeout``, ``nice``,
  ``nohup``, ``time``, ``exec``, ``stdbuf``, ``xargs`` and ``command``;
- through :func:`expand`, the command ``uv run``, ``uv tool run`` or ``uvx``
  launches.

``uv`` calls are parsed against the option table of the pinned uv release
(``tests/fixtures/uv_cli_options.json``), so a value is never mistaken for a
subcommand or for ``uv run``'s command.

Anything this reader cannot follow raises :class:`UnanalyzableScript`, so a
guard built on it fails closed instead of silently skipping a command. That
covers unterminated quotes, substitutions and heredocs, an unknown uv option
or subcommand, a shell that reads its script from stdin or a file, and a
program produced by a substitution. A program named by a variable
(``"$PYBIN" ...``) is kept as written, because its identity is out of reach.

Regenerate the option table after moving the uv pin::

    .venv/bin/python -m tests.workflow_shell > tests/fixtures/uv_cli_options.json
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
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

_GHA_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?\+?=.*", re.DOTALL)
_PLACEHOLDER = re.compile(r"__SUBST(\d+)__")
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
# wrapper -> (options that take a value, positional words before the command)
_WRAPPERS = {
    "env": (frozenset({"-u", "-C", "--unset", "--chdir"}), 0),
    "exec": (frozenset({"-a"}), 0),
    "nice": (frozenset({"-n", "--adjustment"}), 0),
    "nohup": (frozenset(), 0),
    "stdbuf": (frozenset({"-i", "-o", "-e"}), 0),
    "sudo": (
        frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-R", "-t", "-T", "-U"}),
        0,
    ),
    "time": (frozenset({"-f", "-o"}), 0),
    "timeout": (frozenset({"-k", "-s", "--kill-after", "--signal"}), 1),
    "xargs": (
        frozenset({"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s", "--arg-file"}),
        0,
    ),
}


class UnanalyzableScript(ValueError):
    """A run script uses syntax this reader cannot follow."""


@dataclass(frozen=True)
class Command:
    """One simple command, with wrappers such as ``sudo`` removed."""

    words: tuple[str, ...]
    line: str
    # NAME=value words that set the command's environment, as written.
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


@dataclass(frozen=True)
class _Token:
    text: str
    operator: bool


class _Reader:
    """Split a script into logical lines.

    Each substitution is replaced by a ``__SUBST<n>__`` placeholder, and its
    own lines are stored at index n of the shared ``registry``.
    """

    def __init__(self, text: str, registry: list[list[str]]) -> None:
        self.text = text
        self.i = 0
        self.registry = registry

    def _error(self, message: str) -> UnanalyzableScript:
        start = self.text.rfind("\n", 0, self.i) + 1
        end = self.text.find("\n", self.i)
        line = self.text[start : end if end != -1 else len(self.text)]
        return UnanalyzableScript(f"{message}: {line.strip()!r}")

    def _register(self, lines: list[str]) -> str:
        self.registry.append(lines)
        return f"__SUBST{len(self.registry) - 1}__"

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
            if c == "'":
                end = text.find("'", self.i + 1)
                if end == -1:
                    raise self._error("unterminated single quote")
                out.append(text[self.i : end + 1])
                self.i = end + 1
                word_start = False
                continue
            if c == '"':
                out.append(self._double_quoted())
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
                out.append("<<<")  # a here-string: its word is data
                self.i += 3
                word_start = True
                continue
            if text.startswith("<<", self.i):
                delimiter, expands, strip_tabs = self._heredoc_header()
                index = len(self.registry)
                self.registry.append([])  # filled once the body is read
                heredocs.append((index, delimiter, expands, strip_tabs))
                out.append(f" << __SUBST{index}__ ")
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
        if text.startswith("$((", self.i):
            # Arithmetic runs no command, but a substitution inside it does.
            self.i += 3
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
            source = text[self.i + 1 : end].replace("\\`", "`")
            self.i = end + 1
            return self._register(_Reader(source, self.registry).lines())
        self.i += 2  # "$(", "<(" or ">("
        return self._register(self.lines(closer=")"))

    def _heredoc_header(self) -> tuple[str, bool, bool]:
        text = self.text
        self.i += 2
        strip_tabs = text.startswith("-", self.i)
        if strip_tabs:
            self.i += 1
        while self.i < len(text) and text[self.i] in " \t":
            self.i += 1
        match = re.match(
            r"""'([^']+)'|"([^"]+)"|\\?([A-Za-z_][A-Za-z0-9_]*)""", text[self.i :]
        )
        if not match:
            raise self._error("unreadable heredoc delimiter")
        self.i += match.end()
        delimiter = next(group for group in match.groups() if group)
        expands = match.group(3) is not None and not match.group(0).startswith("\\")
        return delimiter, expands, strip_tabs

    def _heredoc_body(
        self, delimiter: str, expands: bool, strip_tabs: bool
    ) -> list[str]:
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
                    return []
                # An unquoted body is data, but its substitutions still run.
                found = _Reader("\n".join(body), self.registry).substitutions()
                return [
                    line
                    for placeholder in found
                    for index in _PLACEHOLDER.findall(placeholder)
                    for line in self.registry[int(index)]
                ]
            body.append(line)
        raise UnanalyzableScript(f"heredoc {delimiter!r} is never closed")


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

    def flush() -> None:
        nonlocal word, quoted, started
        if started:
            tokens.append(_Token("".join(word), False))
        word, quoted, started = [], False, False

    i = 0
    while i < len(line):
        c = line[i]
        if c in " \t\n":
            flush()
            i += 1
        elif c == "\\":
            word.append(line[i + 1 : i + 2])
            quoted = started = True
            i += 2
        elif c == "'":
            end = line.find("'", i + 1)
            if end == -1:
                raise UnanalyzableScript(f"unterminated single quote: {line.strip()!r}")
            word.append(line[i + 1 : end])
            quoted = started = True
            i = end + 1
        elif c == '"':
            i += 1
            while i < len(line) and line[i] != '"':
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
                any(ch in operator for ch in "<>")
                and started
                and not quoted
                and "".join(word).isdigit()
            ):
                word, started = [], False  # a file-descriptor prefix: 2>&1
            flush()
            tokens.append(_Token(operator, True))
            i += len(operator)
        else:
            word.append(c)
            started = True
            i += 1
    flush()
    return tokens


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


def _unwrap(words: list[str]) -> list[tuple[list[str], list[str]] | str]:
    """Strip keywords and wrappers from one simple command.

    Returns the commands it runs, each as (words, assignments), and the
    scripts it runs (``bash -c``, ``eval``, ``trap``) as strings.
    """
    assignments: list[str] = []
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
        elif word in _HEADERS:
            return []
        elif name in _SHELLS:
            return [_shell_script(words)]
        elif name == "su":
            for index, arg in enumerate(words[1:], start=1):
                if arg.startswith("--command="):
                    return [arg.partition("=")[2]]
                if arg in {"-c", "--command"} and index + 1 < len(words):
                    return [words[index + 1]]
            raise UnanalyzableScript(f"su without -c: {' '.join(words)!r}")
        elif name == "eval":
            return [" ".join(words[1:])]
        elif name == "trap":
            args = words[2:] if words[1:2] == ["--"] else words[1:]
            # `trap - SIG` resets, `trap '' SIG` ignores, `trap -p` prints.
            if args and args[0] and not args[0].startswith("-"):
                return [args[0]]
            return []
        elif name == "find":
            found: list[tuple[list[str], list[str]] | str] = []
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
        elif name in _WRAPPERS:
            valued, positionals = _WRAPPERS[name]
            index = 1
            while index < len(words) and words[index].startswith("-"):
                arg = words[index]
                if arg == "--":
                    index += 1
                    break
                if name == "env" and arg in {"-S", "--split-string"}:
                    return [" ".join(words[index + 1 :])]
                if name == "env" and arg.startswith("--split-string="):
                    return [" ".join([arg.partition("=")[2], *words[index + 1 :]])]
                if name == "env" and arg.startswith("-S"):
                    return [" ".join([arg[2:], *words[index + 1 :]])]
                index += 2 if arg in valued else 1
            while (
                index < len(words)
                and name in {"env", "sudo"}
                and _ASSIGNMENT.fullmatch(words[index])
            ):
                assignments.append(words[index])
                index += 1
            words = words[index + positionals :]
        else:
            if _PLACEHOLDER.search(word):
                raise UnanalyzableScript(
                    f"the program comes from a substitution: {' '.join(words)!r}"
                )
            return [(words, assignments)]
    return [([], assignments)] if assignments else []


def _commands_in(lines: list[str], registry: list[list[str]]) -> list[Command]:
    found: list[Command] = []
    words: list[str] = []
    line = ""
    in_header = False  # between `case` and `in`
    in_pattern = False  # a case pattern, up to its `)`
    depth = 0  # nested `case` statements

    def run(indexes: list[str]) -> None:
        for index in indexes:
            found.extend(_commands_in(registry[int(index)], registry))

    def finish() -> None:
        nonlocal words
        if words:
            run([i for w in words for i in _PLACEHOLDER.findall(w)])
            for item in _unwrap(words):
                if isinstance(item, str):
                    found.extend(commands(item))
                else:
                    program, assignments = item
                    found.append(
                        Command(tuple(program), line.strip(), tuple(assignments))
                    )
        words = []

    for line in lines:
        tokens = _tokens(line)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if in_header:
                run(_PLACEHOLDER.findall(token.text))
                in_header = not (not token.operator and token.text == "in")
                in_pattern = not in_header
            elif in_pattern:
                if not token.operator and token.text == "esac":
                    depth -= 1
                    in_pattern = False
                elif token.operator and token.text == ")":
                    in_pattern = False
            elif token.operator and any(ch in token.text for ch in "<>"):
                # A redirection: its target is data, but may hold a substitution.
                if index < len(tokens) and not tokens[index].operator:
                    run(_PLACEHOLDER.findall(tokens[index].text))
                    index += 1
            elif token.operator:
                finish()
                in_pattern = depth > 0 and token.text in _CASE_ARM_ENDS
            elif not words and token.text == "case":
                depth += 1
                in_header = True
            elif not words and token.text == "esac" and depth:
                depth -= 1
            else:
                words.append(token.text)
        finish()
    return found


def commands(script: str) -> list[Command]:
    """Every simple command ``script`` runs, in the order bash runs them.

    A substitution runs just before the command that contains it, and a
    ``bash -c`` script is expanded where it runs.
    """
    registry: list[list[str]] = []
    lines = _Reader(_normalize_expressions(script), registry).lines()
    return _commands_in(lines, registry)


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
                Command(c.words, command.line, c.assignments)
                for c in commands(shlex.join(runs))
            ]
            expanded.extend(expand(inner, table))
    return expanded


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
            options.append(canonical)
            return index + (1 if offset < len(arg) - 1 else 2)
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
