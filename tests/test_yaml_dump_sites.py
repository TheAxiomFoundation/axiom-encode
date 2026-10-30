"""Every YAML the encoder writes goes through the YAML 1.2-safe dumper.

A plain `yaml.safe_dump` writes strings such as '1e5', '0o17' and '+.5'
unquoted; YAML 1.2 (and the engine) then read them as numbers. So production
code must serialize with `safe_dump_yaml`, or `yaml.dump` with a dumper that
subclasses `Yaml12SafeDumper`. This scan fails if any site regresses.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from axiom_encode.yaml_serialization import Yaml12SafeDumper

SRC = Path(__file__).resolve().parents[1] / "src" / "axiom_encode"
SERIALIZER = SRC / "yaml_serialization.py"


def _yaml_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"yaml", "_yaml"}
            and node.func.attr in {"safe_dump", "dump", "safe_dump_all", "dump_all"}
        ):
            yield node


def test_no_production_code_calls_plain_safe_dump():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path == SERIALIZER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in _yaml_calls(tree):
            if call.func.attr in {"safe_dump", "safe_dump_all"}:
                offenders.append(
                    f"{path.relative_to(SRC)}:{call.lineno} yaml.{call.func.attr}"
                )
            else:
                dumper = next(
                    (kw.value for kw in call.keywords if kw.arg == "Dumper"), None
                )
                if not isinstance(dumper, ast.Name):
                    offenders.append(
                        f"{path.relative_to(SRC)}:{call.lineno} yaml.{call.func.attr} without a named Dumper"
                    )
    assert offenders == []


def test_named_dumpers_inherit_the_yaml12_safe_dumper():
    """Every Dumper passed to yaml.dump in production code subclasses Yaml12SafeDumper."""
    import importlib

    checked = 0
    for path in sorted(SRC.rglob("*.py")):
        if path == SERIALIZER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            kw.value.id
            for call in _yaml_calls(tree)
            if call.func.attr in {"dump", "dump_all"}
            for kw in call.keywords
            if kw.arg == "Dumper" and isinstance(kw.value, ast.Name)
        }
        if not names:
            continue
        module = importlib.import_module(
            "axiom_encode." + ".".join(path.relative_to(SRC).with_suffix("").parts)
        )
        for name in names:
            dumper = getattr(module, name)
            assert issubclass(dumper, Yaml12SafeDumper), f"{path.name}: {name}"
            checked += 1
    assert checked >= 1


def test_the_safe_dumper_quotes_yaml12_numbers():
    for value in [
        "1e5",
        "-0o17",
        "+.5",
        "2e10",
        "0x1F",
        "1_000",
        ".inf",
        "null",
        "true",
    ]:
        dumped = yaml.dump({"v": value}, Dumper=Yaml12SafeDumper)
        assert yaml.safe_load(dumped) == {"v": value}
        assert f"v: {value}\n" not in dumped
