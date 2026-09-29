"""Find the commands a GitHub Actions ``run:`` script executes, and parse uv calls.

This is a small bash reader, just enough to audit how workflows install
Python. It handles:

- quote-aware comments and line continuations;
- heredocs (bodies are data, but ``$(...)`` in an unquoted body still runs);
- ``$(...)``, backtick and ``<(...)`` substitutions;
- ``bash -c``/``sh -c``/``eval`` scripts;
- the wrappers ``sudo``, ``env``, ``timeout``, ``nice``, ``nohup``, ``time``,
  ``exec``, ``stdbuf``, ``xargs`` and ``command``.

``uv`` calls are parsed against the option table of the pinned uv release
(``tests/fixtures/uv_cli_options.json``), so a value is never mistaken for a
subcommand or for ``uv run``'s command. Anything this reader cannot follow
raises :class:`UnanalyzableScript`, so a guard built on it fails closed
instead of silently skipping a command.
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
_HELP_OPTION = re.compile(
    r"^ {2,6}(?:(-[A-Za-z0-9]), )?(--[a-z0-9][a-z0-9-]*)(?:\.\.\.)?"
    r"(?: (<[^>]+>)(?:\.\.\.)?)?(?:\s|$)"
)

_GHA_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?\+?=.*", re.DOTALL)
_PUNCTUATION = "();<>|&"
_KEYWORDS = frozenset(
    {"!", "{", "}", "if", "then", "else", "elif", "fi", "do", "done", "while"}
    | {"until", "esac"}
)
_HEADERS = frozenset({"for", "case", "select", "function", "in"})
_SHELLS = frozenset({"bash", "sh", "dash", "zsh"})
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
_SUBSTITUTION = "__SUBSTITUTION__"


class UnanalyzableScript(ValueError):
    """A run script uses syntax this reader cannot follow."""


@dataclass(frozen=True)
class Command:
    """One simple command, with wrappers such as ``sudo`` removed."""

    words: tuple[str, ...]
    line: str

    @property
    def program(self) -> str:
        return self.words[0].rsplit("/", 1)[-1]


@dataclass(frozen=True)
class UvInvocation:
    subcommand: str  # "sync", "pip install", "tool run", "uvx", or "" for `uv -V`
    options: tuple[str, ...]  # canonical option names, e.g. "-w" for "-wrequests"
    command: Command


class _Reader:
    """Split a script into logical lines, lifting substitutions out of them."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0

    def _error(self, message: str) -> UnanalyzableScript:
        start = self.text.rfind("\n", 0, self.i) + 1
        end = self.text.find("\n", self.i)
        line = self.text[start : end if end != -1 else len(self.text)]
        return UnanalyzableScript(f"{message}: {line.strip()!r}")

    def lines(self, closer: str | None = None) -> list[str]:
        """Read logical lines until the end, or an unmatched ``closer``."""
        lines: list[str] = []
        out: list[str] = []
        lifted: list[str] = []
        heredocs: list[tuple[str, bool]] = []
        depth = 0
        word_start = True
        text = self.text
        while self.i < len(text):
            c = text[self.i]
            if closer is not None and c == closer and depth == 0:
                self.i += 1
                lines.extend(lifted)
                lines.append("".join(out))
                if heredocs:
                    raise self._error("heredoc left open inside a substitution")
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
                out.append(self._double_quoted(lifted))
                word_start = False
                continue
            if (
                c == "`"
                or (c == "$" and text.startswith("$(", self.i))
                or (c in "<>" and text.startswith("(", self.i + 1))
            ):
                out.append(self._substitution(lifted))
                word_start = False
                continue
            if c == "<" and text.startswith("<<", self.i):
                if text.startswith("<<<", self.i):
                    out.append("<<<")
                    self.i += 3
                    continue
                heredocs.append(self._heredoc_header())
                out.append(" <<HEREDOC ")
                word_start = True
                continue
            if c == "#" and word_start:
                end = text.find("\n", self.i)
                self.i = len(text) if end == -1 else end
                continue
            if c == "\n":
                self.i += 1
                lines.extend(lifted)
                lines.append("".join(out))
                lifted, out = [], []
                for delimiter, expands in heredocs:
                    lines.extend(self._heredoc_body(delimiter, expands))
                heredocs = []
                word_start = True
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            out.append(c)
            self.i += 1
            word_start = c in " \t" + _PUNCTUATION
        if closer is not None:
            raise self._error(f"unterminated substitution (missing {closer!r})")
        if heredocs:
            raise self._error("heredoc without a body")
        lines.extend(lifted)
        lines.append("".join(out))
        return lines

    def substitutions(self) -> list[str]:
        """The command lines inside the substitutions of plain text."""
        lifted: list[str] = []
        text = self.text
        while self.i < len(text):
            if text[self.i] == "\\":
                self.i += 2
            elif text[self.i] == "`" or text.startswith("$(", self.i):
                self._substitution(lifted)
            else:
                self.i += 1
        return lifted

    def _double_quoted(self, lifted: list[str]) -> str:
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
                out.append(self._substitution(lifted))
            else:
                out.append(c)
                self.i += 1
        raise self._error("unterminated double quote")

    def _substitution(self, lifted: list[str]) -> str:
        text = self.text
        if text.startswith("$((", self.i):
            # Arithmetic expansion runs no command.
            depth = 0
            while self.i < len(text):
                if text[self.i] == "(":
                    depth += 1
                elif text[self.i] == ")":
                    depth -= 1
                    if depth == 0:
                        self.i += 1
                        return "0"
                self.i += 1
            raise self._error("unterminated arithmetic expansion")
        if text[self.i] == "`":
            end = self.i + 1
            while end < len(text) and text[end] != "`":
                end += 2 if text[end] == "\\" else 1
            if end >= len(text):
                raise self._error("unterminated backtick substitution")
            inner = _Reader(text[self.i + 1 : end].replace("\\`", "`"))
            lifted.extend(inner.lines())
            self.i = end + 1
            return _SUBSTITUTION
        self.i += 2  # "$(", "<(" or ">("
        lifted.extend(self.lines(closer=")"))
        return _SUBSTITUTION

    def _heredoc_header(self) -> tuple[str, bool]:
        text = self.text
        self.i += 2
        if text.startswith("-", self.i):
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
        return delimiter, match.group(3) is not None and not match.group(0).startswith(
            "\\"
        )

    def _heredoc_body(self, delimiter: str, expands: bool) -> list[str]:
        text = self.text
        body: list[str] = []
        while self.i < len(text):
            end = text.find("\n", self.i)
            end = len(text) if end == -1 else end
            line = text[self.i : end]
            self.i = min(end + 1, len(text))
            if line.strip() == delimiter:
                if not expands:
                    return []
                # An unquoted body is data, but its substitutions still run.
                return _Reader("\n".join(body)).substitutions()
            body.append(line)
        raise UnanalyzableScript(f"heredoc {delimiter!r} is never closed")


def _normalize_expressions(script: str) -> str:
    # GitHub substitutes ${{ }} before bash runs; keep each one a single word.
    return _GHA_EXPRESSION.sub(
        lambda m: "${{" + re.sub(r"\s+", "", m.group(1)) + "}}", script
    )


def _words(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as error:
        raise UnanalyzableScript(f"{error}: {line.strip()!r}") from None


def _simple_commands(words: list[str]) -> list[list[str]]:
    commands: list[list[str]] = [[]]
    skip_next = False
    for word in words:
        if skip_next:
            skip_next = False
            continue
        if word and all(ch in _PUNCTUATION for ch in word):
            if "<" in word or ">" in word:
                # A redirection: drop its target, and a file-descriptor number.
                if commands[-1] and commands[-1][-1].isdigit():
                    commands[-1].pop()
                skip_next = word not in {"<(", ">("}
                continue
            commands.append([])
            continue
        commands[-1].append(word)
    return [command for command in commands if command]


def _unwrap(words: list[str]) -> list[str] | str | None:
    """Strip keywords and wrappers; return the words, a nested script, or None."""
    while words:
        word = words[0]
        name = word.rsplit("/", 1)[-1]
        if word in _KEYWORDS or _ASSIGNMENT.fullmatch(word):
            words = words[1:]
        elif word in _HEADERS:
            return None
        elif name in _SHELLS:
            for index, arg in enumerate(words[1:], start=1):
                if arg == "-c" or (re.fullmatch(r"-[a-z]*c[a-z]*", arg) is not None):
                    if index + 1 >= len(words):
                        raise UnanalyzableScript(f"{name} -c without a script")
                    return words[index + 1]
                if not arg.startswith("-"):
                    break
            return words
        elif name == "eval":
            return " ".join(words[1:])
        elif name == "command":
            if len(words) > 1 and words[1] in {"-v", "-V"}:
                return None
            words = [w for w in words[1:2] if w != "-p"] + words[2:]
        elif name in _WRAPPERS:
            valued, positionals = _WRAPPERS[name]
            index = 1
            while index < len(words) and words[index].startswith("-"):
                if words[index] == "--":
                    index += 1
                    break
                if name == "env" and words[index] in {"-S", "--split-string"}:
                    return " ".join(words[index + 1 :])
                index += 2 if words[index] in valued else 1
            while (
                index < len(words)
                and name in {"env", "sudo"}
                and _ASSIGNMENT.fullmatch(words[index])
            ):
                index += 1
            words = words[index + positionals :]
        else:
            return words
    return None


def commands(script: str) -> list[Command]:
    """Every simple command ``script`` runs, in the order it appears.

    A command inside ``$(...)`` is listed before the command it feeds, and a
    ``bash -c`` script is expanded where it runs.
    """
    found: list[Command] = []
    for line in _Reader(_normalize_expressions(script)).lines():
        for words in _simple_commands(_words(line)):
            unwrapped = _unwrap(words)
            if isinstance(unwrapped, str):
                found.extend(commands(unwrapped))
            elif unwrapped:
                found.append(Command(tuple(unwrapped), line.strip()))
    return found


def load_uv_options() -> dict:
    return json.loads(UV_OPTIONS_PATH.read_text(encoding="utf-8"))


def uv_option_table(uv: str = "uv") -> dict:
    """Build the option table from ``uv <command> --help`` for each command."""

    def table(command: str) -> dict[str, list[str]]:
        page = subprocess.run(
            [uv, *command.split(), "--help"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        flags: set[str] = set()
        valued: set[str] = set()
        for line in page.splitlines():
            match = _HELP_OPTION.match(line)
            if match:
                short, long, value = match.groups()
                (valued if value else flags).update(n for n in (short, long) if n)
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
            "it as a hidden alias, add it to UV_HIDDEN_ALIASES and regenerate "
            f"{UV_OPTIONS_PATH.name}"
        )
    for offset, letter in enumerate(arg[1:], start=1):
        name = "-" + letter
        if name in table["valued"]:
            options.append(name)
            return index + (1 if offset < len(arg) - 1 else 2)
        if name not in table["flags"]:
            raise UnanalyzableScript(f"unknown uv option {name!r} in {where!r}")
        options.append(name)
    return index + 1


def uv_invocation(command: Command, table: dict) -> UvInvocation | None:
    """Parse ``command`` as a uv call, or return None if it does not run uv."""
    if command.program == "uvx":
        return UvInvocation("uvx", (), command)
    if command.program != "uv":
        return None
    tables = table["commands"]
    args = command.words[1:]
    options: list[str] = []
    index = 0
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
            f"`uv {subcommand}` is not in {UV_OPTIONS_PATH.name}: {command.line!r}"
        )
    while index < len(args):
        arg = args[index]
        if arg == "--":
            break
        if arg.startswith("-") and arg != "-":
            index = _read_option(args, index, tables[subcommand], options, command.line)
        elif subcommand in {"run", "tool run"}:
            break  # the command uv runs; its arguments are its own
        else:
            index += 1
    return UvInvocation(subcommand, tuple(options), command)
