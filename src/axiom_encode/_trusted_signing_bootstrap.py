"""Pre-attachment bootstrap for the compiled signing supervisor.

This file intentionally imports only frozen/builtin ``os`` and ``sys`` until
the interpreter's complete initial import surface has been admitted.
"""

from __future__ import annotations

import os
import sys

_FORBIDDEN_PUBLIC_ROOT_ENVIRONMENT = (
    "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY",
    "AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY",
    "AXIOM_CORPUS_RELEASE_PUBLIC_KEY",
)

for _name in _FORBIDDEN_PUBLIC_ROOT_ENVIRONMENT:
    if _name in os.environ:
        raise RuntimeError(
            "public signing roots must come from the authenticated broker, "
            f"not environment variable {_name}"
        )

if os.__spec__ is None or os.__spec__.origin != "frozen":
    raise RuntimeError("trusted bootstrap requires CPython's frozen os module")
if sys.__spec__ is None or sys.__spec__.origin != "built-in":
    raise RuntimeError("trusted bootstrap requires the built-in sys module")


def _canonical(path: str, *, label: str) -> str:
    if not path or not os.path.isabs(path):
        raise RuntimeError(f"{label} must be absolute")
    canonical = os.path.realpath(path)
    if canonical != path or os.path.normpath(path) != path:
        raise RuntimeError(f"{label} must be canonical and contain no symlink")
    return canonical


def _contains(root: str, path: str) -> bool:
    try:
        return os.path.commonpath((root, path)) == root
    except ValueError:
        return False


def _parse(arguments: list[str]) -> tuple[str, str, list[str], list[str]]:
    runtime_root = ""
    package_root = ""
    import_roots: list[str] = []
    cursor = 0
    while cursor < len(arguments):
        value = arguments[cursor]
        if value == "--":
            return runtime_root, package_root, import_roots, arguments[cursor + 1 :]
        if cursor + 1 >= len(arguments):
            raise RuntimeError("trusted Python bootstrap arguments are incomplete")
        selected = arguments[cursor + 1]
        if value == "--runtime-root" and not runtime_root:
            runtime_root = selected
        elif value == "--package-root" and not package_root:
            package_root = selected
        elif value == "--import-root":
            import_roots.append(selected)
        else:
            raise RuntimeError(
                f"unsupported trusted Python bootstrap argument: {value}"
            )
        cursor += 2
    raise RuntimeError("trusted Python bootstrap is missing --")


def _preflight() -> list[str]:
    runtime, package, imports, cli_arguments = _parse(sys.argv[1:])
    runtime = _canonical(runtime, label="runtime root")
    package = _canonical(package, label="package root")
    imports = [_canonical(path, label="import root") for path in imports]
    if not imports or len(imports) != len(set(imports)):
        raise RuntimeError("trusted Python import roots must be explicit and unique")
    if os.path.dirname(package) not in imports:
        raise RuntimeError("axiom_encode parent must be an explicit import root")
    for path in (package, *imports):
        if not _contains(runtime, path):
            raise RuntimeError("trusted Python paths escape the runtime root")
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError("trusted Python must start with exact -I -S isolation")
    prefixes = (
        sys.prefix,
        sys.exec_prefix,
        sys.base_prefix,
        sys.base_exec_prefix,
    )
    canonical_prefixes = tuple(
        _canonical(path, label="Python compiled prefix") for path in prefixes
    )
    if len(set(canonical_prefixes)) != 1 or not _contains(
        runtime, canonical_prefixes[0]
    ):
        raise RuntimeError("Python virtual or compiled prefix escapes runtime")
    executable = _canonical(sys.executable, label="Python executable")
    if not _contains(runtime, executable):
        raise RuntimeError("Python executable escapes runtime")
    initial_paths: list[str] = []
    for raw_path in sys.path:
        path = _canonical(raw_path, label="initial sys.path root")
        if not _contains(runtime, path):
            raise RuntimeError(f"initial sys.path root escapes runtime: {path}")
        initial_paths.append(path)

    # This stdlib import occurs only after every initial search root is admitted.
    import sysconfig

    for name, raw_path in sysconfig.get_paths().items():
        path = _canonical(raw_path, label=f"sysconfig {name}")
        if not _contains(runtime, path):
            raise RuntimeError(f"sysconfig {name} escapes runtime")
    sys.path[:] = list(dict.fromkeys([*initial_paths, *imports]))

    import axiom_encode

    expected_initializer = os.path.join(package, "__init__.py")
    actual_initializer = _canonical(
        str(axiom_encode.__file__), label="axiom_encode origin"
    )
    if actual_initializer != expected_initializer or tuple(
        _canonical(path, label="axiom_encode package path")
        for path in axiom_encode.__path__
    ) != (package,):
        raise RuntimeError("axiom_encode resolved outside the exact trusted package")
    return cli_arguments


def main() -> int | None:
    cli_arguments = _preflight()
    # Import and attach only after the entire preflight above succeeds.
    from axiom_encode import signing_broker

    signing_broker.reject_direct_private_signing_environment()
    signing_broker.attach_signing_broker_from_environment()
    sys.argv[:] = ["axiom-encode", *cli_arguments]
    from axiom_encode.entrypoint import main as entrypoint

    return entrypoint()


if __name__ == "__main__":
    raise SystemExit(main())
