"""Launcher of the stdlib guard bundle: run the enforcing merge guard.

``.github/workflows/content-bound-merge-guard.yml`` copies this file and the
modules of ``BUNDLE_MODULES`` into an empty directory::

    <bundle>/launch.py                      (this file)
    <bundle>/axiom_encode/__init__.py       (empty)
    <bundle>/axiom_encode/<module>.py       (one per BUNDLE_MODULES entry)

and runs ``python3 -I -S <bundle>/launch.py``. Nothing else of the encoder is
on the runner: no package install, no dependency, no console script, no
``axiom_encode/__init__.py`` of the encoder, no ``cli.py``.

This file is the single source of truth for what the bundle holds
(``BUNDLE_MODULES``, ``SOURCE_FILES``, ``BUNDLE_FILES``). The workflow's
sparse checkout, its copy step and its digest literal are pinned to these
lists by ``tests/test_merge_guard_enforce_workflow.py``.

It lives in ``scripts/`` and not in the ``axiom_encode`` package because it is
a program, not a module of the encoder: nothing imports it, and the console
script cannot reach it.

The launcher takes no argument and reads no environment variable of its own.
It fails closed (exit 1) when:

- the interpreter was not started with ``-I`` and ``-S`` (an environment
  variable, the working directory or a site directory could add code);
- the bundle directory holds anything except the files of ``BUNDLE_FILES``,
  or one of them is not a regular file;
- an import asks for an ``axiom_encode`` module outside ``BUNDLE_MODULES``;
- an ``axiom_encode`` module was loaded from outside the bundle directory.

The bundle directory is appended to ``sys.path``, after the standard library,
so that no file of the bundle can replace a standard-library module.
"""

import os
import sys

BUNDLE_MODULES = (
    "applied_manifest_v6",
    "content_merge_guard",
    "merge_guard_runner",
    "reviewed_pr_admission",
    "rulespec_citation_paths",
    "statute",
)
PACKAGE = "axiom_encode"
LAUNCHER_SOURCE = "scripts/merge_guard_launch.py"
LAUNCHER_NAME = "launch.py"
# Repository paths the workflow checks out, in bundle order.
SOURCE_FILES = (LAUNCHER_SOURCE,) + tuple(
    f"src/{PACKAGE}/{module}.py" for module in BUNDLE_MODULES
)
# Paths inside the bundle directory, in the order of the bundle digest.
BUNDLE_FILES = (LAUNCHER_NAME, f"{PACKAGE}/__init__.py") + tuple(
    f"{PACKAGE}/{module}.py" for module in BUNDLE_MODULES
)
ALLOWED_MODULES = frozenset(
    {PACKAGE} | {f"{PACKAGE}.{module}" for module in BUNDLE_MODULES}
)


class LaunchError(Exception):
    """The bundle is not what the launcher was reviewed to run."""


class _RefuseUnlistedModules:
    """A ``sys.meta_path`` finder: no ``axiom_encode`` module off the list."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if (
            fullname == PACKAGE or fullname.startswith(PACKAGE + ".")
        ) and fullname not in ALLOWED_MODULES:
            raise ImportError(f"{fullname} is not a module of the merge guard bundle")
        return None  # the standard finders load listed and stdlib modules


def _bundle_files(directory):
    found = []
    for parent, directories, files in os.walk(directory):
        for name in directories + files:
            path = os.path.join(parent, name)
            if os.path.islink(path):
                raise LaunchError("the bundle directory holds a link")
        for name in files:
            relative = os.path.relpath(os.path.join(parent, name), directory)
            found.append(relative.replace(os.sep, "/"))
    return sorted(found)


def prepare():
    """Check the interpreter and the bundle, then make the bundle importable.

    Returns the bundle directory. Raises ``LaunchError``.
    """
    if not (sys.flags.isolated and sys.flags.no_site):
        raise LaunchError("the launcher runs under python -I -S only")
    directory = os.path.dirname(os.path.abspath(__file__))
    if _bundle_files(directory) != sorted(BUNDLE_FILES):
        raise LaunchError("the bundle directory does not hold exactly the bundle")
    if os.path.getsize(os.path.join(directory, PACKAGE, "__init__.py")) != 0:
        raise LaunchError("the bundle package __init__.py is not empty")
    # No __pycache__ in the bundle directory: its listing stays exact.
    sys.dont_write_bytecode = True
    if not any(isinstance(finder, _RefuseUnlistedModules) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RefuseUnlistedModules())
    if directory not in sys.path:
        sys.path.append(directory)
    return directory


def loaded_outside(directory):
    """Loaded ``axiom_encode`` modules that are off the list or the bundle."""
    package = os.path.join(directory, PACKAGE)
    problems = []
    for name, module in sorted(sys.modules.items()):
        if name != PACKAGE and not name.startswith(PACKAGE + "."):
            continue
        origin = getattr(module, "__file__", None)
        if (
            name not in ALLOWED_MODULES
            or not isinstance(origin, str)
            or os.path.dirname(os.path.abspath(origin)) != package
        ):
            problems.append(name)
    return problems


def main():
    try:
        directory = prepare()
        from axiom_encode import merge_guard_runner

        if loaded_outside(directory):
            raise LaunchError("a guard module was loaded from outside the bundle")
        return merge_guard_runner.enforce_command()
    except LaunchError as exc:
        print("merge guard launcher refused to run: " + str(exc), file=sys.stderr)
        return 1
    except BaseException as exc:  # fail closed: any error is a failed job
        # Only the type: a message can carry what the failing code handled.
        print("merge guard launcher failed: " + type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
