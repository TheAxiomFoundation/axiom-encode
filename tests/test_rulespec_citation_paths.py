"""Pin the standard-library path helper that the merge guard bundle will run.

The guard key-protection proposal (axiom-encode#2257; section 4, steps 1, 4
and 5 of ``docs/merge-guard-key-protection.md``) recommends running the
content-bound merge guard from a small set of files under ``python -I -S``,
without installing the encoder. Step 1 moves the citation-to-path rule out of
``harness/evals.py`` and ``prepare_signed_backfill.py`` into
``rulespec_citation_paths.py``. These tests pin the parts of steps 4 and 5
that apply before the shadow runner (#2237) merges:

* the new module and ``statute.py`` import only the standard library, at
  module and function level, and the guard files on ``main`` import nothing
  outside the bundle;
* under ``python -I -S``, with only those files beside an empty
  ``axiom_encode/__init__.py``, the path rule and the guard load, and no other
  encoder module loads;
* the move changed no behaviour: ``harness/evals.py`` re-exports the same
  objects, and ``prepare_signed_backfill.py``, which this change does not
  edit, holds the same rule as the new module;
* ``CODEOWNERS`` owns the new module, ``statute.py`` and this file.

The launcher, the runner file and the key-bearing job wait for #2237. This
file activates nothing.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from axiom_encode import content_merge_guard, prepare_signed_backfill
from axiom_encode import rulespec_citation_paths as paths
from axiom_encode.harness import evals

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "axiom_encode"
CODE_OWNERS = ("@PavelMakarchuk", "@MaxGhenis")

PATH_MODULE = "rulespec_citation_paths"
STATUTE_MODULE = "statute"
# The bundle files of proposal section 4, step 2, that exist on main. The
# runner (merge_guard_runner.py) and the launcher join with #2237.
GUARD_MODULES = ("content_merge_guard", "reviewed_pr_admission", "applied_manifest_v6")
BUNDLE_MODULES = (*GUARD_MODULES, PATH_MODULE, STATUTE_MODULE)
# The only names the path module may import from this package.
STATUTE_IMPORTS = frozenset(
    {"citation_to_citation_path", "normalize_rulespec_path_segment"}
)
# Modules that load other code by name or by path at run time. A static
# import check cannot see what they load, so the bundle files may not use them.
DYNAMIC_LOADERS = frozenset({"importlib", "imp", "pkgutil", "runpy", "zipimport"})
# Builtins that run or load code a static import check cannot see.
DYNAMIC_BUILTINS = frozenset({"__import__", "compile", "eval", "exec"})
# The exact standard-library modules the two new bundle files may import.
STATUTE_STDLIB = frozenset({"__future__", "dataclasses", "pathlib", "re"})
PATH_MODULE_STDLIB = frozenset({"__future__", "pathlib", "re", "typing"})

# What harness/evals.py defined before the move: the resolver, its 12 helper
# functions and its 8 constants.
MOVED_FROM_EVALS = (
    "_resolve_eval_output_path",
    "_citation_to_corpus_citation_path",
    "_cfr_citation_to_corpus_citation_path",
    "_looks_like_corpus_citation_path",
    "_source_identifier_to_relative_rulespec_path",
    "_names_rulemaking_output",
    "_mapped_relative_rulespec_path",
    "_expand_louisiana_title_section_tail",
    "_dotted_leaf_to_nested_yaml_path",
    "_preserve_dotted_leaf_filename",
    "_preserve_state_statute_dotted_leaf",
    "_canonical_us_regulation_tail",
    "_canonical_uk_legislation_tail",
    "_RULESPEC_SOURCE_ROOT_TOKENS",
    "_RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN",
    "_RULEMAKING_SOURCE_TOKEN",
    "_RULEMAKING_OUTPUT_ROOT",
    "_UK_LEGISLATION_DOMAIN_TOKEN",
    "_UK_LEGISLATION_SECTION_TOKENS",
    "_CFR_CITATION_RE",
    "_USC_CITATION_RE",
)

# One citation for each branch of the path rule, with the path on main before
# the move.
CITATION_PATHS = {
    "us/statute/26/1": "us/statutes/26/1.yaml",
    "us/statute/7/2014/e/2/B": "us/statutes/7/2014/e/2/B.yaml",
    "us/regulation/7/273/9": "us/regulations/7-cfr/273/9.yaml",
    "us/rulemaking/fr/2025-12345": "us/policies/rulemaking/fr/2025-12345.yaml",
    "us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-13": (
        "us-az/policies/des/faa5/na-utility-expenses-and-allowances/block-13.yaml"
    ),
    "us-ca/regulation/mpp/63-503.132": "us-ca/regulations/mpp/63-503/132.yaml",
    "us-co/regulation/10-ccr-2506-1/4.207.2": (
        "us-co/regulations/10-ccr-2506-1/4.207.2.yaml"
    ),
    "us-co/statute/26/26-2-703.5": "us-co/statutes/26/26-2-703.5.yaml",
    "us-la/statute/47:294": "us-la/statutes/47/294.yaml",
    "uk/statute/legislation.gov.uk/ukpga/2012/5/section/4": (
        "uk/statutes/ukpga/2012/5/4.yaml"
    ),
    "uk/regulation/uksi/2013/376/regulation/22": (
        "uk/regulations/uksi/2013/376/22.yaml"
    ),
    "us/form/irs/1040": "us/policies/irs/1040.yaml",
}


def _imports(module: str) -> list[tuple[int, int, str, tuple[str, ...]]]:
    """Return every import in the file: (line, relative level, module, names).

    ``ast.walk`` visits the whole tree, so imports inside functions, classes,
    ``try`` blocks and ``if TYPE_CHECKING`` blocks are included.
    """

    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, 0, alias.name, ()))
        elif isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            found.append((node.lineno, node.level, node.module or "", names))
    return found


def import_problems(
    module: str,
    *,
    allowed_package_imports: dict,
    stdlib: frozenset[str] | None = None,
) -> list[str]:
    """Return every import of ``module`` that leaves the standard library.

    ``allowed_package_imports`` maps an ``axiom_encode`` module name to the
    names that may be imported from it (``None`` allows any name).
    ``from axiom_encode import <module>`` gives every name of that module, so
    it needs a ``None`` entry. With ``stdlib``, a standard-library import must
    also be one of those modules. The dynamic builtins and loaders are
    refused for every file.
    """

    problems = []
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DYNAMIC_BUILTINS:
            problems.append(f"{module}.py:{node.lineno}: uses {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr == "__import__":
            problems.append(f"{module}.py:{node.lineno}: uses __import__")
    for line, level, name, names in _imports(module):
        where = f"{module}.py:{line}"
        top = name.split(".")[0]
        if level:
            problems.append(f"{where}: relative import of {name or '.'!r}")
        elif top in DYNAMIC_LOADERS:
            problems.append(f"{where}: imports the dynamic loader {name!r}")
        elif name == "axiom_encode" and names:
            # ``from axiom_encode import a, b``: each name is a module.
            for target in names:
                if target not in allowed_package_imports:
                    problems.append(
                        f"{where}: imports 'axiom_encode.{target}', outside the bundle"
                    )
                elif allowed_package_imports[target] is not None:
                    problems.append(f"{where}: imports all of 'axiom_encode.{target}'")
        elif top == "axiom_encode":
            if not names:
                problems.append(f"{where}: plain import of {name!r}")
                continue
            target = name.removeprefix("axiom_encode").removeprefix(".")
            if target not in allowed_package_imports:
                problems.append(f"{where}: imports {name!r}, outside the bundle")
                continue
            allowed = allowed_package_imports[target]
            extra = sorted(set(names) - allowed) if allowed is not None else []
            if extra:
                problems.append(f"{where}: imports {extra} from {name!r}")
        elif top not in sys.stdlib_module_names:
            problems.append(f"{where}: imports {name!r}, not standard library")
        elif stdlib is not None and name not in stdlib:
            problems.append(f"{where}: imports {name!r}, not on the allowlist")
    return problems


def test_the_statute_helpers_import_only_the_standard_library():
    assert (
        import_problems(
            STATUTE_MODULE, allowed_package_imports={}, stdlib=STATUTE_STDLIB
        )
        == []
    )
    used = {name for _line, _level, name, _names in _imports(STATUTE_MODULE)}
    assert used == STATUTE_STDLIB


def test_the_path_module_imports_the_standard_library_and_two_statute_functions():
    assert (
        import_problems(
            PATH_MODULE,
            allowed_package_imports={STATUTE_MODULE: STATUTE_IMPORTS},
            stdlib=PATH_MODULE_STDLIB,
        )
        == []
    )
    used = {name for _line, _level, name, _names in _imports(PATH_MODULE)}
    assert used == PATH_MODULE_STDLIB | {"axiom_encode.statute"}
    package_imports = [
        (name, set(names))
        for _line, _level, name, names in _imports(PATH_MODULE)
        if name.split(".")[0] == "axiom_encode"
    ]
    assert package_imports == [("axiom_encode.statute", set(STATUTE_IMPORTS))]


@pytest.mark.parametrize("module", GUARD_MODULES)
def test_the_guard_files_on_main_import_nothing_outside_the_bundle(module):
    allowed = {name: None for name in BUNDLE_MODULES}
    assert import_problems(module, allowed_package_imports=allowed) == []


@pytest.mark.parametrize(
    ("source", "problems"),
    [
        pytest.param(
            "from axiom_encode import reviewed_pr_admission as _admission\n",
            0,
            id="a-bundle-module-from-the-package",
        ),
        pytest.param(
            "from axiom_encode import content_merge_guard as guard\n",
            0,
            id="the-guard-from-the-package",
        ),
        pytest.param(
            "from axiom_encode import statute, applied_manifest_v6\n",
            0,
            id="two-bundle-modules-from-the-package",
        ),
        pytest.param(
            "from axiom_encode import prepare_signed_backfill\n",
            1,
            id="the-backfill-helpers-from-the-package",
        ),
        pytest.param(
            "from axiom_encode import statute, cli, harness\n",
            2,
            id="one-inside-two-outside",
        ),
        pytest.param(
            "def f():\n    from axiom_encode import harness\n",
            1,
            id="outside-the-bundle-in-a-function",
        ),
    ],
)
def test_the_import_check_reads_modules_imported_from_the_package(
    monkeypatch, tmp_path, source, problems
):
    """``from axiom_encode import <module>`` names a module, not the package."""

    (tmp_path / "probe.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", tmp_path)
    allowed = {name: None for name in BUNDLE_MODULES}
    found = import_problems("probe", allowed_package_imports=allowed)
    assert len(found) == problems, found
    assert all("outside the bundle" in problem for problem in found)


def test_the_guard_takes_the_path_rule_from_the_path_module():
    imported = {
        name: set(names)
        for _line, _level, name, names in _imports("content_merge_guard")
        if name.split(".")[0] == "axiom_encode"
    }
    assert imported["axiom_encode.rulespec_citation_paths"] == {
        "RULESPEC_ATOMIC_ROOTS",
        "citation_rulespec_path",
    }
    assert "axiom_encode.prepare_signed_backfill" not in imported
    assert content_merge_guard.citation_rulespec_path is paths.citation_rulespec_path
    assert content_merge_guard.RULESPEC_ATOMIC_ROOTS is paths.RULESPEC_ATOMIC_ROOTS


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        pytest.param("import yaml\n", "not standard library", id="third-party"),
        pytest.param(
            "def f():\n    import requests\n",
            "not standard library",
            id="third-party-in-a-function",
        ),
        pytest.param(
            "def f():\n    from axiom_encode.harness.evals import x\n",
            "outside the bundle",
            id="the-edge-this-change-cut",
        ),
        pytest.param(
            "from axiom_encode.prepare_signed_backfill import x\n",
            "outside the bundle",
            id="the-signed-backfill-helpers",
        ),
        pytest.param(
            "from axiom_encode.statute import parse_usc_citation\n",
            "['parse_usc_citation']",
            id="another-statute-name",
        ),
        pytest.param("import axiom_encode.statute\n", "plain import", id="plain"),
        pytest.param("from . import statute\n", "relative import", id="relative"),
        pytest.param(
            "from .statute import citation_to_citation_path\n",
            "relative import",
            id="relative-module",
        ),
        pytest.param("import importlib\n", "dynamic loader", id="importlib"),
        pytest.param(
            "def f():\n    from importlib import util\n",
            "dynamic loader",
            id="importlib-in-a-function",
        ),
        pytest.param("import runpy\n", "dynamic loader", id="runpy"),
        pytest.param(
            "def f():\n    return __import__('yaml')\n",
            "uses __import__",
            id="dunder-import",
        ),
        pytest.param(
            "try:\n    import yaml\nexcept ImportError:\n    yaml = None\n",
            "not standard library",
            id="guarded-third-party",
        ),
        pytest.param(
            "import re\n\nload = re.__builtins__.__import__\n",
            "uses __import__",
            id="dunder-import-attribute",
        ),
        pytest.param("x = eval('1')\n", "uses eval", id="eval"),
        pytest.param("def f():\n    exec('import yaml')\n", "uses exec", id="exec"),
        pytest.param(
            "code = compile('1', 'x', 'eval')\n", "uses compile", id="compile"
        ),
        pytest.param("run = exec\n", "uses exec", id="exec-by-reference"),
        pytest.param(
            "from axiom_encode import statute\n",
            "imports all of 'axiom_encode.statute'",
            id="the-whole-statute-module",
        ),
        pytest.param("import subprocess\n", "not on the allowlist", id="subprocess"),
        pytest.param(
            "def f():\n    import os\n", "not on the allowlist", id="os-in-a-function"
        ),
        pytest.param(
            "from urllib import request\n", "not on the allowlist", id="urllib"
        ),
        pytest.param("import ctypes\n", "not on the allowlist", id="ctypes"),
    ],
)
def test_the_import_check_reports_each_way_out_of_the_bundle(
    monkeypatch, tmp_path, source, problem
):
    (tmp_path / "probe.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", tmp_path)
    problems = import_problems(
        "probe",
        allowed_package_imports={STATUTE_MODULE: STATUTE_IMPORTS},
        stdlib=PATH_MODULE_STDLIB,
    )
    assert len(problems) == 1 and problem in problems[0], problems


def test_the_import_check_accepts_the_allowed_forms(monkeypatch, tmp_path):
    (tmp_path / "probe.py").write_text(
        "from __future__ import annotations\n\nimport re\n"
        "from pathlib import Path\nfrom typing import Callable\n\n"
        "from axiom_encode.statute import citation_to_citation_path\n\n"
        "PATTERN = re.compile('x')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", tmp_path)
    assert (
        import_problems(
            "probe",
            allowed_package_imports={STATUTE_MODULE: STATUTE_IMPORTS},
            stdlib=PATH_MODULE_STDLIB,
        )
        == []
    )


def _isolated_bundle(tmp_path: Path, modules: tuple[str, ...]) -> Path:
    """Copy ``modules`` beside an empty ``axiom_encode/__init__.py``."""

    package = tmp_path / "bundle" / "axiom_encode"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    for module in modules:
        shutil.copyfile(PACKAGE / f"{module}.py", package / f"{module}.py")
    return package.parent


_ISOLATED_CHILD = """
import json, sys
bundle, entry, citations = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
path_before = list(sys.path)
sys.path.insert(0, bundle)
module = __import__("axiom_encode." + entry, fromlist=["citation_rulespec_path"])
resolved = {c: str(module.citation_rulespec_path(c)) for c in citations}
refused = {}
for citation in ("no-separator", "us/free text", "us/statute/26/../1"):
    try:
        module.citation_rulespec_path(citation)
    except ValueError as error:
        refused[citation] = str(error)
print(json.dumps({
    "flags": [sys.flags.isolated, sys.flags.no_site],
    "path_before": path_before,
    "encoder_modules": sorted(
        name for name in sys.modules if name.split(".")[0] == "axiom_encode"
    ),
    "other_modules": sorted(
        name for name in sys.modules
        if name.split(".")[0] not in sys.stdlib_module_names
        and name.split(".")[0] not in ("axiom_encode", "__main__")
    ),
    "resolved": resolved,
    "refused": refused,
}))
"""


def _run_isolated(tmp_path: Path, modules: tuple[str, ...], entry: str) -> dict:
    bundle = _isolated_bundle(tmp_path, modules)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _ISOLATED_CHILD,
            str(bundle),
            entry,
            json.dumps(sorted(CITATION_PATHS)),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["flags"] == [1, 1]
    # -S: no site-packages directory is on the path, so a third-party import
    # in a bundle file would fail instead of loading.
    assert not [entry for entry in report["path_before"] if "site-packages" in entry]
    assert report["other_modules"] == []
    return report


def _expected_refusals() -> dict[str, str]:
    refused = {}
    for citation in ("no-separator", "us/free text", "us/statute/26/../1"):
        with pytest.raises(ValueError) as error:
            paths.citation_rulespec_path(citation)
        refused[citation] = str(error.value)
    assert len(refused) == 3
    return refused


def test_the_path_rule_runs_isolated_with_two_files(tmp_path):
    report = _run_isolated(tmp_path, (PATH_MODULE, STATUTE_MODULE), PATH_MODULE)
    assert report["encoder_modules"] == [
        "axiom_encode",
        "axiom_encode.rulespec_citation_paths",
        "axiom_encode.statute",
    ]
    assert report["resolved"] == CITATION_PATHS
    assert report["refused"] == _expected_refusals()


def test_the_guard_loads_isolated_without_the_harness_or_the_backfill_helpers(
    tmp_path,
):
    """The edge this change cuts: the guard's path rule reached ``evals``."""

    report = _run_isolated(tmp_path, BUNDLE_MODULES, "content_merge_guard")
    assert report["encoder_modules"] == sorted(
        ["axiom_encode", *(f"axiom_encode.{module}" for module in BUNDLE_MODULES)]
    )
    assert report["resolved"] == CITATION_PATHS
    assert report["refused"] == _expected_refusals()


def test_the_isolated_bundle_refuses_a_file_outside_it(tmp_path):
    """Without ``statute.py`` the import fails: nothing else supplies it."""

    bundle = _isolated_bundle(tmp_path, (PATH_MODULE,))
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _ISOLATED_CHILD,
            str(bundle),
            PATH_MODULE,
            "[]",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "No module named 'axiom_encode.statute'" in completed.stderr


@pytest.mark.parametrize("citation", sorted(CITATION_PATHS))
def test_the_path_rule_gives_the_paths_it_gave_before_the_move(citation):
    expected = PurePosixPath(CITATION_PATHS[citation])
    assert paths.citation_rulespec_path(citation) == expected
    assert prepare_signed_backfill.citation_rulespec_path(citation) == expected
    jurisdiction, relative = paths._citation_rulespec_path(citation)
    assert PurePosixPath(jurisdiction) / relative == expected
    assert prepare_signed_backfill._citation_rulespec_path(citation) == (
        jurisdiction,
        relative,
    )
    assert evals._resolve_eval_output_path(citation) == Path(relative)


@pytest.mark.parametrize(
    ("citation", "message"),
    [
        ("no-separator", "citation must be a canonical corpus citation path"),
        ("", "citation must be a canonical corpus citation path"),
        ("us/free text", "Eval source identity is not a canonical citation path"),
        ("us/statute/26/../1", "Unsafe RuleSpec path component"),
        ("us-la/statute/47::294", "Invalid Louisiana"),
        ("us/form/rulemaking/x", "reserved rulemaking output namespace"),
    ],
)
def test_the_path_rule_refuses_what_it_refused_before_the_move(citation, message):
    for resolve in (
        paths.citation_rulespec_path,
        prepare_signed_backfill.citation_rulespec_path,
    ):
        with pytest.raises(ValueError, match=message):
            resolve(citation)


def _outcome(resolve, citation: str) -> tuple[str, str]:
    try:
        return ("path", str(resolve(citation)))
    except Exception as error:  # noqa: BLE001 - the error is the outcome
        return (type(error).__name__, str(error))


def test_the_two_copies_agree_on_a_generated_corpus():
    """Every head, source token and tail shape, accepted or refused."""

    heads = (
        "us", "us-co", "us-la", "us-az", "uk", "", "US", "us:statutes",
        "us:policy", "statutes", "policies", "rulemaking", "..",
    )  # fmt: skip
    tokens = (
        "statute", "statutes", "regulation", "regulations", "policy",
        "policies", "manual", "manuals", "form", "forms", "guidance",
        "rulemaking", "legislation", "programs", "Statute", "", ".", "..",
    )  # fmt: skip
    tails = (
        "", "26/1", "26/1/a/2/B", "7/273/9", "47:294", "47::294", "47:294:1",
        "26/26-2-703.5", "10-ccr-2506-1/4.207.2", "mpp/63-503.132",
        "rulemaking/x", "Rulemaking/x.y", "legislation.gov.uk/ukpga/2012/5/section/4",
        "uksi/2013/376/regulation/22", "a/../b", "a/./b", "a//b", ".hidden",
        "trailing.", "x.yaml", "a b", "26/1\u20132", "/abs", "a/b/",
    )  # fmt: skip
    corpus = [
        "/".join(part for part in (head, token, tail) if part or part is head)
        for head in heads
        for token in tokens
        for tail in tails
    ]
    corpus += ["26 USC 32(a)", "42 CFR 435.119", "42 C.F.R. § 435.119(b)(5)", "x"]
    assert len(set(corpus)) > 4000
    outcomes = set()
    for citation in corpus:
        ours = _outcome(paths.citation_rulespec_path, citation)
        theirs = _outcome(prepare_signed_backfill.citation_rulespec_path, citation)
        assert ours == theirs, citation
        assert ours[0] in {"path", "ValueError"}, (citation, ours)
        outcomes.add(ours[0])
    assert outcomes == {"path", "ValueError"}


@pytest.mark.parametrize("name", MOVED_FROM_EVALS)
def test_evals_re_exports_the_same_object(name):
    assert getattr(evals, name) is getattr(paths, name)


def test_the_moved_names_are_the_resolver_twelve_helpers_and_eight_constants():
    functions = [name for name in MOVED_FROM_EVALS if callable(getattr(paths, name))]
    constants = [name for name in MOVED_FROM_EVALS if name not in functions]
    assert len(functions) == 13 and functions[0] == "_resolve_eval_output_path"
    assert len(constants) == 8
    assert len(set(MOVED_FROM_EVALS)) == 21
    # evals.py no longer defines them: each is an import there.
    tree = ast.parse((PACKAGE / "harness" / "evals.py").read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    } | {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert defined.isdisjoint(MOVED_FROM_EVALS)
    # Every definition of the path module is one of the moved names.
    tree = ast.parse((PACKAGE / f"{PATH_MODULE}.py").read_text(encoding="utf-8"))
    path_module_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    } | {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert path_module_names == set(MOVED_FROM_EVALS) | {
        "RULESPEC_ATOMIC_ROOTS",
        "_citation_rulespec_path",
        "citation_rulespec_path",
    }


def definition_problems(source: str, name: str) -> list[str]:
    """Return every reason ``name`` is not bound exactly once, at module level.

    A comparison of the module-level definition proves nothing if the name is
    bound again later: in a nested scope with ``global``, by an import, by a
    second assignment, or through ``globals()[...]`` or ``setattr`` with the
    name as a string. So every other binding, and every string equal to the
    name, is a problem.
    """

    tree = ast.parse(source)
    top = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == name)
        or (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        )
    ]
    problems = []
    if len(top) != 1:
        problems.append(f"{name}: {len(top)} plain module-level definitions")
    bindings = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings += node.name == name
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bindings += node.id == name
        elif isinstance(node, ast.alias):
            bindings += (node.asname or node.name.split(".")[0]) == name
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bindings += name in node.names
        elif isinstance(node, ast.arg):
            bindings += node.arg == name
        elif isinstance(node, ast.ExceptHandler):
            bindings += node.name == name
        elif isinstance(node, ast.Constant) and node.value == name:
            problems.append(f"{name}: named by a string on line {node.lineno}")
    if bindings != 1:
        problems.append(f"{name}: bound {bindings} times")
    return problems


def _definition(module: str, name: str) -> ast.AST:
    """Return the one definition of ``name`` in ``module``."""

    source = (PACKAGE / f"{module}.py").read_text(encoding="utf-8")
    assert definition_problems(source, name) == [], module
    return next(
        node
        for node in ast.parse(source).body
        if (isinstance(node, ast.FunctionDef) and node.name == name)
        or (isinstance(node, ast.Assign) and node.targets[0].id == name)
    )


COPIED_NAMES = (
    "RULESPEC_ATOMIC_ROOTS",
    "_citation_rulespec_path",
    "citation_rulespec_path",
)


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param(
            "\n\ndef citation_rulespec_path(citation):\n    return 1\n", id="second-def"
        ),
        pytest.param("\n\ncitation_rulespec_path = len\n", id="reassignment"),
        pytest.param(
            "\n\ndef swap():\n    global citation_rulespec_path\n"
            "    citation_rulespec_path = len\n",
            id="nested-global-rebinding",
        ),
        pytest.param(
            "\n\nglobals()['citation_rulespec_path'] = len\n", id="globals-rebinding"
        ),
        pytest.param(
            "\n\nimport sys\nsetattr(sys.modules[__name__], "
            "'citation_rulespec_path', len)\n",
            id="setattr-rebinding",
        ),
        pytest.param(
            "\n\nfrom os.path import basename as citation_rulespec_path\n",
            id="import-rebinding",
        ),
        pytest.param(
            "\n\nclass citation_rulespec_path:\n    pass\n", id="class-rebinding"
        ),
        pytest.param(
            "\n\nif True:\n    def citation_rulespec_path(citation):\n"
            "        return 1\n",
            id="conditional-def",
        ),
        pytest.param(
            "\n\nfor citation_rulespec_path in ():\n    pass\n", id="loop-target"
        ),
        pytest.param(
            "\n\ncitation_rulespec_path: object = len\n", id="annotated-assignment"
        ),
    ],
)
def test_the_copy_comparison_refuses_a_name_that_is_bound_again(tail):
    source = "def citation_rulespec_path(citation):\n    return citation\n"
    assert definition_problems(source, "citation_rulespec_path") == []
    assert definition_problems(source + tail, "citation_rulespec_path") != []


def test_the_copy_comparison_refuses_a_missing_or_decorated_name():
    assert definition_problems("x = 1\n", "citation_rulespec_path") != []
    assert (
        definition_problems(
            "a = RULESPEC_ATOMIC_ROOTS = frozenset()\n", "RULESPEC_ATOMIC_ROOTS"
        )
        != []
    )


@pytest.mark.parametrize("module", ["prepare_signed_backfill", PATH_MODULE])
@pytest.mark.parametrize("name", COPIED_NAMES)
def test_each_copied_name_is_bound_once_in_each_file(module, name):
    source = (PACKAGE / f"{module}.py").read_text(encoding="utf-8")
    assert definition_problems(source, name) == []


def test_the_backfill_helpers_and_the_path_module_hold_the_same_rule():
    """``prepare_signed_backfill.py`` keeps its own copy; the copies must agree.

    The guard checks a manifest's primary path with the copy in the path
    module. Signing computed that path with the copy in
    ``prepare_signed_backfill.py``. The two copies differ in one statement:
    the backfill copy imports the resolver from ``harness/evals.py`` inside
    the function, and ``evals.py`` re-exports the path module's resolver.
    """

    for name in ("RULESPEC_ATOMIC_ROOTS", "citation_rulespec_path"):
        assert ast.dump(_definition("prepare_signed_backfill", name)) == ast.dump(
            _definition(PATH_MODULE, name)
        ), name
    theirs = _definition("prepare_signed_backfill", "_citation_rulespec_path")
    ours = _definition(PATH_MODULE, "_citation_rulespec_path")
    assert isinstance(theirs, ast.FunctionDef) and isinstance(ours, ast.FunctionDef)
    resolver_import, *rest = theirs.body
    assert ast.unparse(resolver_import) == (
        "from axiom_encode.harness.evals import _resolve_eval_output_path"
    )
    assert [ast.dump(node) for node in rest] == [ast.dump(node) for node in ours.body]
    assert ast.dump(theirs.args) == ast.dump(ours.args)
    assert ast.unparse(theirs.returns) == ast.unparse(ours.returns)
    for name in ("_citation_rulespec_path", "citation_rulespec_path"):
        for module in ("prepare_signed_backfill", PATH_MODULE):
            definition = _definition(module, name)
            assert isinstance(definition, ast.FunctionDef)
            assert definition.decorator_list == [], (module, name)
            assert definition.type_params == [], (module, name)
    assert evals._resolve_eval_output_path is paths._resolve_eval_output_path
    assert (
        prepare_signed_backfill.RULESPEC_ATOMIC_ROOTS
        == paths.RULESPEC_ATOMIC_ROOTS
        == frozenset({"legislation", "policies", "regulations", "statutes"})
    )


def test_the_backfill_helpers_still_load_without_the_package():
    """Why ``prepare_signed_backfill.py`` keeps its own copy of the rule.

    The signing workflow runs ``python scripts/prepare_signed_backfill.py``
    on the runner's interpreter before it installs the encoder, and that
    script loads ``src/axiom_encode/prepare_signed_backfill.py`` by path. A
    module-level import of the package there would stop the first step of
    every signing run. Later steps run the same script on the provisioned
    verification interpreter, whose ``axiom_encode`` package is a separate
    install, so the file also cannot rely on a module that an installed
    package may not have yet.
    """

    tree = ast.parse(
        (PACKAGE / "prepare_signed_backfill.py").read_text(encoding="utf-8")
    )
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, node.lineno
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] in sys.stdlib_module_names, (node.lineno, name)
    # Not inside a function either: the installed package may predate the
    # path module.
    assert not [
        (line, name)
        for line, _level, name, names in _imports("prepare_signed_backfill")
        if name.split(".")[-1] == PATH_MODULE or PATH_MODULE in names
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "scripts" / "prepare_signed_backfill.py"),
            "validate-country",
            "us",
        ],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (completed.returncode, completed.stdout) == (0, "us\n"), completed.stderr


def code_owners_of(text: str, path: str) -> tuple[str, ...] | None:
    """Return the owners of ``path``: the last matching pattern wins.

    Reads root-anchored literal patterns only (a file, or a directory with a
    trailing ``/``), which is every pattern in this ``CODEOWNERS``. Returns
    ``None`` when the file has a pattern of another form, because such a
    pattern could match ``path`` and this check cannot evaluate it.
    """

    owners: tuple[str, ...] = ()
    for line in text.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pattern, *rule_owners = line.split()
        if not pattern.startswith("/") or any(mark in pattern for mark in "*?[]!\\"):
            return None
        body = pattern[1:]
        target = path.lstrip("/")
        if (
            target.startswith(body)
            if body.endswith("/")
            else target == body or target.startswith(body + "/")
        ):
            owners = tuple(rule_owners)
    return owners


OWNED_PATHS = (
    "/src/axiom_encode/rulespec_citation_paths.py",
    "/src/axiom_encode/statute.py",
    "/tests/test_rulespec_citation_paths.py",
)


@pytest.mark.parametrize("path", OWNED_PATHS)
def test_code_owners_own_the_path_module_the_statute_helpers_and_this_file(path):
    assert (ROOT / path.lstrip("/")).is_file()
    text = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert f"{path} {' '.join(CODE_OWNERS)}" in text.split("\n")
    assert code_owners_of(text, path) == CODE_OWNERS


BOTH_OWNERS = " ".join(CODE_OWNERS)


@pytest.mark.parametrize(
    "later",
    [
        pytest.param("/src/axiom_encode/statute.py @someone-else", id="same-path"),
        pytest.param("/src/axiom_encode/statute.py", id="same-path-no-owner"),
        pytest.param("/src/ @someone-else", id="a-parent-directory"),
        pytest.param("/src/axiom_encode/ @PavelMakarchuk", id="one-owner"),
        pytest.param("/src/axiom_encode @someone-else", id="directory-without-slash"),
        pytest.param("*.py @someone-else", id="a-wildcard-this-check-cannot-read"),
        pytest.param("src/ @someone-else", id="an-unanchored-pattern"),
    ],
)
def test_a_later_code_owners_entry_takes_the_file(later):
    path = "/src/axiom_encode/statute.py"
    entry = f"{path} {BOTH_OWNERS}"
    assert code_owners_of(f"# c\n\n{entry}\n", path) == CODE_OWNERS
    assert code_owners_of(f"/src/ @someone-else\n{entry}\n", path) == CODE_OWNERS
    assert code_owners_of(f"{entry}\n/tests/ @someone-else\n", path) == CODE_OWNERS
    assert code_owners_of(f"{entry}\n{later}\n", path) != CODE_OWNERS
