"""Mirror the engine's RuleSpec expression lexer and parser.

This port follows ``src/formula.rs`` at engine commit
``e0bda6475f30ac487b7be544f5b9742c716e4bba``. Keep the byte-based lexer,
precedence and expression entry point in step with that source. Rule formulas
use ``parse_expr`` followed by EOF, through ``non_exhaustive_match_count``;
leading statements and let bindings therefore refuse, rather than introducing
another statement language into formula strings.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


class FormulaSyntaxError(ValueError):
    """Locate a formula that the engine's expression grammar cannot parse."""

    def __init__(self, line: int, col: int, message: str):
        self.line, self.col, self.message = line, col, message
        super().__init__(f"formula parse error at line {line}, col {col}: {message}")


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    col: int


@dataclass(frozen=True)
class Expr:
    """Retain the engine's expression shapes and their source positions."""

    kind: str
    value: str = ""
    children: tuple[Expr, ...] = ()
    line: int = 1
    col: int = 1
    cases: tuple[tuple[Expr, Expr], ...] = ()
    default: Expr | None = None


KEYWORDS = frozenset(
    {
        "entity",
        "amend",
        "from",
        "to",
        "match",
        "if",
        "elif",
        "else",
        "and",
        "or",
        "not",
        "true",
        "True",
        "false",
        "False",
    }
)
_PAIRS = frozenset({"=>", "<=", ">=", "==", "!=", "->"})
_SINGLES = frozenset(":,.=+-*/<>()[]")


def _digit(c: int) -> bool:
    return 48 <= c <= 57


def _initial(c: int) -> bool:
    return 65 <= c <= 90 or 97 <= c <= 122 or c == 95


def tokenise(source: str) -> tuple[Token, ...]:
    """Port ``Lexer::tokenise``, including UTF-8 byte columns and slash paths."""

    try:
        raw = source.encode("utf-8")
    except UnicodeEncodeError as error:
        prefix = source[: error.start]
        raise FormulaSyntaxError(
            prefix.count("\n") + 1,
            len(prefix.rsplit("\n", 1)[-1].encode("utf-8")) + 1,
            "source contains a character that UTF-8 cannot encode",
        ) from None
    pos, line, col = 0, 1, 1
    tokens: list[Token] = []

    def advance(count: int) -> None:
        nonlocal pos, line, col
        for c in raw[pos : pos + count]:
            if c == 10:
                line, col = line + 1, 1
            else:
                col += 1
        pos += count

    while pos < len(raw):
        if raw[pos : pos + 3] == b'"""':
            closing = raw.find(b'"""', pos + 3)
            advance(len(raw) - pos if closing < 0 else closing + 3 - pos)
            continue
        c = raw[pos]
        if c == 35:
            closing = raw.find(b"\n", pos)
            advance(len(raw) - pos if closing < 0 else closing - pos)
            continue
        # Rust u8::is_ascii_whitespace excludes vertical tab.
        if c in b"\t\n\x0c\r ":
            advance(1)
            continue
        start, token_line, token_col = pos, line, col
        date = raw[pos : pos + 10]
        if (
            _digit(c)
            and len(date) == 10
            and date[4] == date[7] == 45
            and all(i in {4, 7} or _digit(b) for i, b in enumerate(date))
        ):
            advance(10)
            tokens.append(Token("date", date.decode(), token_line, token_col))
            continue
        if _digit(c):
            end = pos
            while end < len(raw) and (_digit(raw[end]) or raw[end] == 95):
                end += 1
            kind = "int"
            if end + 1 < len(raw) and raw[end] == 46 and _digit(raw[end + 1]):
                kind = "float"
                end += 1
                while end < len(raw) and (_digit(raw[end]) or raw[end] == 95):
                    end += 1
            advance(end - pos)
            tokens.append(Token(kind, raw[start:end].decode(), token_line, token_col))
            continue
        if c in {34, 39}:
            quote = c
            advance(1)
            value: list[str] = []
            segment = pos
            while pos < len(raw):
                if raw[pos] == quote:
                    value.append(raw[segment:pos].decode())
                    advance(1)
                    tokens.append(
                        Token("string", "".join(value), token_line, token_col)
                    )
                    break
                if raw[pos] == 92:
                    value.append(raw[segment:pos].decode())
                    if pos + 1 == len(raw):
                        raise FormulaSyntaxError(
                            token_line, token_col, "unterminated string"
                        )
                    escaped = raw[pos + 1 :].decode()[0]
                    value.append(
                        {"n": "\n", "r": "\r", "t": "\t"}.get(escaped, escaped)
                    )
                    advance(1 + len(escaped.encode()))
                    segment = pos
                else:
                    advance(1)
            else:
                raise FormulaSyntaxError(token_line, token_col, "unterminated string")
            continue
        if _initial(c):
            end = pos + 1
            while end < len(raw) and (_initial(raw[end]) or _digit(raw[end])):
                end += 1
            path_end, is_path = end, False
            while (
                path_end + 1 < len(raw)
                and raw[path_end] == 47
                and _initial(raw[path_end + 1])
            ):
                is_path = True
                path_end += 2
                while path_end < len(raw) and (
                    _initial(raw[path_end]) or _digit(raw[path_end])
                ):
                    path_end += 1
            end = path_end if is_path else end
            value = raw[pos:end].decode()
            kind = "path" if is_path else value if value in KEYWORDS else "ident"
            if kind in {"True", "False"}:
                kind = kind.lower()
            advance(end - pos)
            tokens.append(Token(kind, value, token_line, token_col))
            continue
        pair = raw[pos : pos + 2].decode("ascii", errors="replace")
        if pair in _PAIRS:
            advance(2)
            tokens.append(Token(pair, pair, token_line, token_col))
            continue
        if chr(c) in _SINGLES:
            advance(1)
            tokens.append(Token(chr(c), chr(c), token_line, token_col))
            continue
        unexpected = raw[pos:].decode()[0]
        raise FormulaSyntaxError(line, col, f"unexpected char {unexpected!r}")
    tokens.append(Token("eof", "", line, col))
    return tuple(tokens)


ParseStep = Generator["ParseStep", Expr, Expr]


class Parser:
    """Port ``Parser::parse_expr`` through ``parse_primary``."""

    def __init__(self, tokens: tuple[Token, ...]):
        self.tokens, self.pos = tokens, 0

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def consume(self, kind: str) -> Token:
        token = self.peek()
        if token.kind != kind:
            raise FormulaSyntaxError(
                token.line,
                token.col,
                f"expected {kind}, got {token.kind} ({token.value})",
            )
        self.pos += 1
        return token

    def match_one(self, *kinds: str) -> Token | None:
        if self.peek().kind in kinds:
            token = self.peek()
            self.pos += 1
            return token
        return None

    def parse_expr(self) -> ParseStep:
        if self.peek().kind == "ident" and self.peek(1).kind == "=":
            token = self.peek()
            raise FormulaSyntaxError(
                token.line,
                token.col,
                "let-bindings are not supported in RuleSpec formulas yet",
            )
        if self.peek().kind == "match":
            return (yield self.parse_match())
        if self.peek().kind == "if":
            return (yield self.parse_cond())
        return (yield self.parse_or())

    def parse_match(self) -> ParseStep:
        token = self.consume("match")
        subject = yield self.parse_or()
        self.consume(":")
        cases: list[tuple[Expr, Expr]] = []
        default = None
        while (
            self.peek().kind in {"int", "float", "string", "true", "false", "ident"}
            and self.peek(1).kind == "=>"
        ):
            if default is not None:
                at = self.peek()
                raise FormulaSyntaxError(
                    at.line, at.col, "a match wildcard arm must be final"
                )
            pattern = yield self.parse_primary()
            self.consume("=>")
            result = yield self.parse_expr()
            if pattern.kind == "var" and pattern.value == "_":
                default = result
            else:
                cases.append((pattern, result))
        return Expr(
            "match",
            children=(subject,),
            line=token.line,
            col=token.col,
            cases=tuple(cases),
            default=default,
        )

    def parse_cond(self) -> ParseStep:
        token = self.consume("if")
        condition = yield self.parse_or()
        self.consume(":")
        then_expr = yield self.parse_expr()
        return (yield self.parse_elif_chain(condition, then_expr, token))

    def parse_elif_chain(
        self, condition: Expr, then_expr: Expr, token: Token
    ) -> ParseStep:
        if self.match_one("elif"):
            next_condition = yield self.parse_or()
            self.consume(":")
            next_then = yield self.parse_expr()
            otherwise = yield self.parse_elif_chain(next_condition, next_then, token)
        else:
            self.consume("else")
            self.consume(":")
            otherwise = yield self.parse_expr()
        return Expr(
            "cond",
            children=(condition, then_expr, otherwise),
            line=token.line,
            col=token.col,
        )

    def _binary(
        self, parse_child, operators: frozenset[str], *, once: bool = False
    ) -> ParseStep:
        left = yield parse_child()
        while self.peek().kind in operators:
            token = self.peek()
            self.pos += 1
            right = yield parse_child()
            left = Expr("binary", token.value, (left, right), token.line, token.col)
            if once:
                break
        return left

    def parse_or(self) -> ParseStep:
        return (yield self._binary(self.parse_and, frozenset({"or"})))

    def parse_and(self) -> ParseStep:
        return (yield self._binary(self.parse_cmp, frozenset({"and"})))

    def parse_cmp(self) -> ParseStep:
        return (
            yield self._binary(
                self.parse_add, frozenset({"<", ">", "<=", ">=", "==", "!="}), once=True
            )
        )

    def parse_add(self) -> ParseStep:
        return (yield self._binary(self.parse_mul, frozenset({"+", "-"})))

    def parse_mul(self) -> ParseStep:
        return (yield self._binary(self.parse_unary, frozenset({"*", "/"})))

    def parse_unary(self) -> ParseStep:
        token = self.match_one("-", "not")
        if token:
            return Expr(
                "unary",
                token.value,
                ((yield self.parse_unary()),),
                token.line,
                token.col,
            )
        return (yield self.parse_postfix())

    def parse_postfix(self) -> ParseStep:
        expr = yield self.parse_primary()
        while True:
            if self.peek().kind == "(":
                if expr.kind != "var":
                    token = self.peek()
                    raise FormulaSyntaxError(
                        token.line, token.col, "can only call named functions"
                    )
                self.consume("(")
                args: list[Expr] = []
                if self.peek().kind != ")":
                    args.append((yield self.parse_expr()))
                    while self.match_one(","):
                        args.append((yield self.parse_expr()))
                self.consume(")")
                expr = Expr("call", expr.value, tuple(args), expr.line, expr.col)
            elif self.match_one("."):
                field = self.consume("ident")
                expr = Expr("field", field.value, (expr,), expr.line, expr.col)
            elif self.match_one("["):
                index = yield self.parse_expr()
                self.consume("]")
                expr = Expr(
                    "index", children=(expr, index), line=expr.line, col=expr.col
                )
            else:
                return expr

    def parse_primary(self) -> ParseStep:
        token = self.peek()
        if token.kind in {"int", "float", "string", "true", "false", "ident", "path"}:
            self.pos += 1
            kind = "var" if token.kind in {"ident", "path"} else token.kind
            if kind == "int":
                digits = token.value.replace("_", "").lstrip("0") or "0"
                if len(digits) > 19 or (
                    len(digits) == 19 and digits > "9223372036854775807"
                ):
                    raise FormulaSyntaxError(
                        token.line, token.col, "number too large to fit in target type"
                    )
            if kind == "float":
                try:
                    value = Decimal(token.value.replace("_", ""))
                    if value.to_integral_value(rounding=ROUND_HALF_UP) > Decimal(
                        "79228162514264337593543950335"
                    ):
                        raise InvalidOperation
                except InvalidOperation:
                    raise FormulaSyntaxError(
                        token.line, token.col, "invalid decimal"
                    ) from None
            return Expr(kind, token.value, line=token.line, col=token.col)
        if self.match_one("("):
            expr = yield self.parse_expr()
            self.consume(")")
            return expr
        raise FormulaSyntaxError(
            token.line,
            token.col,
            f"unexpected token in expression: {token.kind} ({token.value})",
        )


def parse_formula(source: str) -> Expr:
    """Use the same expression-and-EOF gate as RuleSpec rule formulas."""

    parser = Parser(tokenise(source))
    stack = [parser.parse_expr()]
    result = None
    while stack:
        try:
            stack.append(stack[-1].send(result))
            result = None
        except StopIteration as complete:
            stack.pop()
            result = complete.value
    expr = result
    if parser.peek().kind != "eof":
        token = parser.peek()
        raise FormulaSyntaxError(
            token.line, token.col, "unexpected trailing token in formula"
        )
    return expr
