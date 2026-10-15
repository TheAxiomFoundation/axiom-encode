"""Authenticate classification-preserving clause-boundary measurement reorders.

The measurement runner uses this adapter only in memory. Production files are
unchanged, and only the exact complete helper ASTs authenticated below qualify.
"""

from __future__ import annotations

import ast
import hashlib


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _node_bytes(node: ast.AST) -> bytes:
    return ast.dump(node, include_attributes=False).encode()


_EXPECTED_HELPERS = {
    "_source_clause_spans": (
        "b2f4e05e419e4b776a55b1425310da2345a8b7107a45eecbe214e8ba366ef185",
        r"""match.group() == "."
and re.search(r"(?<!\w)\d+\.$", source_text[: match.end()])
and re.match(r"\s+Lebensjahr(?:es|e|en)?\b", source_text[match.end() :])""",
        "not",
    ),
    "_source_clause_boundary_splits_state_code_citation": (
        "ea97e740782d8e6d3487320f6383fcd8af7479ae9c606feaae2b2ed07cac2121",
        r"""boundary.group(0) == "."
and re.search(r"\bR\.S\.$", source_text[: boundary.end()], re.IGNORECASE)
and re.match(r"\s*\d+[A-Za-z]?:\d+[A-Za-z]?", source_text[boundary.end() :])""",
        "bool",
    ),
}


def accelerated_clause_boundaries_source(source: bytes) -> tuple[bytes, dict]:
    """Swap two existing pure regex operands in two Boolean-observed guards.

    Exact complete helper AST hashes, complete conjunction ASTs, and Boolean
    observation contexts must match. Valid string/match inputs retain the
    same Boolean decisions; match-object identity, call counts, exceptions,
    and resource timeouts are outside this proof.
    """

    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert len(functions) == sum(
        isinstance(node, ast.FunctionDef) for node in tree.body
    ), "Duplicate function binding"
    original_module_ast_sha256 = _digest(_node_bytes(tree))
    replacements = []
    helper_receipts = {}

    for name, (
        expected_helper_digest,
        expression,
        observation,
    ) in _EXPECTED_HELPERS.items():
        assert name in functions, ("Missing helper", name)
        function = functions[name]
        original_helper = _node_bytes(function)
        assert _digest(original_helper) == expected_helper_digest, (
            "Unknown complete helper AST",
            name,
        )
        assert function.col_offset == 0 and not function.decorator_list
        expected_boolean = ast.parse("(" + expression + ")", mode="eval").body
        assert isinstance(expected_boolean, ast.BoolOp)
        assert isinstance(expected_boolean.op, ast.And)
        assert len(expected_boolean.values) == 3
        targets = [
            child
            for child in ast.walk(function)
            if isinstance(child, ast.BoolOp)
            and _node_bytes(child) == _node_bytes(expected_boolean)
        ]
        assert len(targets) == 1, ("Unknown or duplicate complete conjunction", name)
        conjunction = targets[0]
        parents = {
            id(child): parent
            for parent in ast.walk(function)
            for child in ast.iter_child_nodes(parent)
        }
        parent = parents[id(conjunction)]
        if observation == "not":
            assert isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not)
            assert parent.operand is conjunction
            enclosing = parents[id(parent)]
            assert isinstance(enclosing, ast.BoolOp)
            assert isinstance(enclosing.op, ast.And)
            generator = parents[id(enclosing)]
            assert isinstance(generator, ast.comprehension)
            assert len(generator.ifs) == 1 and generator.ifs[0] is enclosing
        else:
            assert observation == "bool"
            assert isinstance(parent, ast.Call)
            assert isinstance(parent.func, ast.Name) and parent.func.id == "bool"
            assert len(parent.args) == 1 and parent.args[0] is conjunction
            assert not parent.keywords
            assert isinstance(parents[id(parent)], ast.Return)

        original_boolean = _node_bytes(conjunction)
        original_operands = tuple(conjunction.values)
        assert len(original_operands) == 3
        # Both calls are authenticated in full, including every regex, slice,
        # bound, flag, and argument. Reuse their exact AST objects unchanged.
        for operand, method in zip(
            original_operands[1:], ("search", "match"), strict=True
        ):
            assert isinstance(operand, ast.Call)
            assert isinstance(operand.func, ast.Attribute)
            assert isinstance(operand.func.value, ast.Name)
            assert operand.func.value.id == "re" and operand.func.attr == method
        conjunction.values[1], conjunction.values[2] = (
            conjunction.values[2],
            conjunction.values[1],
        )
        assert tuple(map(id, conjunction.values)) == tuple(
            id(original_operands[index]) for index in (0, 2, 1)
        )
        assert all(
            _node_bytes(conjunction.values[index])
            == _node_bytes(original_operands[original])
            for index, original in enumerate((0, 2, 1))
        )
        helper_receipts[name] = {
            "original_helper_ast_sha256": _digest(original_helper),
            "execution_helper_ast_sha256": _digest(_node_bytes(function)),
            "original_complete_boolop_ast_sha256": _digest(original_boolean),
            "execution_complete_boolop_ast_sha256": _digest(_node_bytes(conjunction)),
            "original_operand_ast_sha256": [
                _digest(_node_bytes(node)) for node in original_operands
            ],
            "execution_operand_order": [0, 2, 1],
            "boolean_observation": observation,
        }
        replacements.append((function.lineno - 1, function.end_lineno, function))

    lines = source.splitlines(keepends=True)
    output = []
    untouched_receipts = []
    offset = 0
    for start, end, function in sorted(replacements):
        assert start >= offset
        untouched = b"".join(lines[offset:start])
        output.append(untouched)
        untouched_receipts.append(_digest(untouched))
        replacement = (ast.unparse(function) + "\n").encode()
        rendered_lines = len(replacement.splitlines())
        assert rendered_lines <= end - start
        replacement += b"\n" * (end - start - rendered_lines)
        replacement_tree = ast.parse(replacement)
        assert len(replacement_tree.body) == 1
        assert _node_bytes(replacement_tree.body[0]) == _node_bytes(function)
        output.append(replacement)
        offset = end
    untouched = b"".join(lines[offset:])
    output.append(untouched)
    untouched_receipts.append(_digest(untouched))
    execution = b"".join(output)
    assert len(execution.splitlines()) == len(lines)
    assert _node_bytes(ast.parse(execution)) == _node_bytes(tree), (
        "Unexpected module AST change outside the two authenticated swaps"
    )
    return execution, {
        "source_sha256": _digest(source),
        "execution_source_sha256": _digest(execution),
        "original_module_ast_sha256": original_module_ast_sha256,
        "execution_module_ast_sha256": _digest(_node_bytes(tree)),
        "changed_bindings": sorted(_EXPECTED_HELPERS),
        "helpers": helper_receipts,
        "unchanged_region_sha256": untouched_receipts,
        "line_count_preserved": True,
        "proof": (
            "Exact complete helper and conjunction AST authentication; only original operands "
            "1 and 2 exchange positions. Both conjunctions are observed only as Boolean values, "
            "so bool(a and b and c) equals bool(a and c and b). All operands remain unchanged."
        ),
        "scope": (
            "Successful Boolean classification on valid string/match inputs; match-object "
            "identity, call counts, exceptions and resource-timeout equivalence are not asserted."
        ),
    }
