"""Read-only, independently compiled engine-pin differential evidence."""

from __future__ import annotations

import calendar
import copy
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import yaml

from .engine_binding import (
    EnginePin,
    require_engine_ref_sha,
    resolve_pinned_engine_binary,
)
from .repo_routing import _rulespec_routing_cache_scope

CLASSES = (
    "PASS",
    "EXPECTED_PRESTART_REFUSAL",
    "DIFF",
    "PARITY_BREAK",
    "OLD_PARITY_BREAK",
)
REFUSAL = re.compile(r"derived `([^`]+)` has no formula version at (\d{4}-\d{2}-\d{2})")
# An old engine that already refused the same date: a dated parameter without a
# value yet, or another dated rule. The new engine refusing earlier is no change.
OLD_DATE_REFUSAL = re.compile(
    r"(?:derived `[^`]+` has no formula version"
    r"|parameter `[^`]+` has no value(?: for key `[^`]*`)?)"
    r" at (\d{4}-\d{2}-\d{2})"
)


class ConfinedOutput:
    """Publish through held directory descriptors, never through linked paths."""

    def __init__(self, path: Path):
        self.path = path.absolute()
        if ".." in self.path.parts:
            raise ValueError(f"Unsafe --out-dir parent component: {self.path}")
        self.fd = None

    @staticmethod
    @contextmanager
    def directory(path: Path, *, parent_fd: int | None = None):
        fd = (
            os.open("/", os.O_RDONLY | os.O_DIRECTORY)
            if parent_fd is None
            else os.dup(parent_fd)
        )
        try:
            for part in path.parts:
                if part == "/":
                    continue
                if part in (".", ".."):
                    raise ValueError(f"Unsafe output component: {part}")
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = child
            yield fd
        finally:
            os.close(fd)

    @classmethod
    def reject_links(cls, fd: int):
        with os.scandir(fd) as entries:
            for entry in entries:
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Symlink in --out-dir: {entry.name}")
                if stat.S_ISDIR(mode):
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=fd,
                    )
                    try:
                        cls.reject_links(child)
                    finally:
                        os.close(child)

    def __enter__(self):
        try:
            with self.directory(self.path) as fd:
                self.reject_links(fd)
                self.fd = os.dup(fd)
        except OSError as error:
            raise ValueError(f"Unsafe --out-dir: {self.path}: {error}") from error
        return self

    def __exit__(self, *exc):
        os.close(self.fd)

    def write(self, relative: Path, data: bytes):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe output path: {relative}")
        with self.directory(relative.parent, parent_fd=self.fd) as fd:
            try:
                mode = os.stat(relative.name, dir_fd=fd, follow_symlinks=False).st_mode
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(mode):
                    raise ValueError(f"Non-regular output file: {relative}")
            temporary = f".engine-pin-diff-{secrets.token_hex(16)}"
            file_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=fd,
            )
            try:
                with os.fdopen(file_fd, "wb") as stream:
                    stream.write(data)
                # replace does not dereference the destination (including a
                # link planted after the check); held dirfds fix the parent.
                os.replace(temporary, relative.name, src_dir_fd=fd, dst_dir_fd=fd)
            finally:
                try:
                    os.unlink(temporary, dir_fd=fd)
                except FileNotFoundError:
                    pass


def inside_checkout(path: Path, checkouts: tuple[Path, ...]) -> bool:
    if any(path.resolve().is_relative_to(root.resolve()) for root in checkouts):
        return True
    # resolve() need not canonicalize case/Unicode aliases on macOS. Existing
    # ancestor identities also protect descendants that do not exist yet.
    for ancestor in (path, *path.parents):
        for root in checkouts:
            try:
                if ancestor.samefile(root):
                    return True
            except FileNotFoundError:
                pass
    return False


def add_parser(subparsers):
    parser = subparsers.add_parser("engine-pin-diff", help=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    for side in ("old", "new"):
        parser.add_argument(f"--{side}-engine", type=Path, required=True)
        parser.add_argument(f"--{side}-engine-ref", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--rulespec-dependency-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--axiom-compose-path", type=Path)
    parser.add_argument("paths", nargs="*", type=Path)


def effective_versions(rule: dict) -> list[dict]:
    """Mirror RuleDefinition.effective_versions, including shorthand aliases."""
    versions = rule.get("versions") or []
    if versions:
        return versions
    if (
        rule.get("formula") is not None
        or rule.get("effective_from", rule.get("from")) is not None
    ):
        return [rule]
    return []


def inventory(payload: dict, module: str) -> list[dict]:
    found = []
    for rule in payload.get("rules", []):
        versions = effective_versions(rule)
        if rule.get("kind") != "derived" or len(versions) != 1:
            continue
        version = versions[0]
        start = str(version.get("effective_from", version.get("from")) or "0001-01-01")
        end = version.get("effective_to", version.get("to"))
        if start == "0001-01-01" and end is None:
            continue
        date.fromisoformat(start)
        if end is not None:
            date.fromisoformat(str(end))
        found.append(
            dict(
                module=module,
                rule=rule["name"],
                effective_from=start,
                effective_to=str(end) if end is not None else None,
            )
        )
    return found


def shift_request(request: dict, delta: int) -> dict:
    """Translate query periods and all input/relation intervals, not date facts."""
    shifted = copy.deepcopy(request)
    intervals = [query["period"] for query in shifted["queries"]]
    for collection in ("inputs", "relations"):
        intervals.extend(item["interval"] for item in shifted["dataset"][collection])
    unique_intervals = {id(interval): interval for interval in intervals}
    for interval in unique_intervals.values():
        for endpoint in ("start", "end"):
            interval[endpoint] = (
                date.fromisoformat(interval[endpoint]) + timedelta(days=delta)
            ).isoformat()
    return shifted


def current_start(period: dict, as_of: date) -> date:
    kind = period["period_kind"]
    if kind == "month":
        return as_of.replace(day=1)
    start, end = (date.fromisoformat(period[key]) for key in ("start", "end"))
    if kind == "tax_year":
        anniversary = _anniversary(start, as_of.year)
        return (
            anniversary if anniversary <= as_of else _anniversary(start, as_of.year - 1)
        )
    if kind == "benefit_week":
        return as_of - timedelta(days=(as_of - start).days % 7)
    length = (end - start).days + 1
    return start + timedelta(days=((as_of - start).days // length) * length)


def _anniversary(start: date, year: int) -> date:
    return date(
        year, start.month, min(start.day, calendar.monthrange(year, start.month)[1])
    )


def current_request(request: dict, as_of: date) -> dict:
    """Preserve fixture margins around the final current calendar period."""
    period = request["queries"][0]["period"]
    start = current_start(period, as_of)
    start_delta = start - date.fromisoformat(period["start"])
    if period["period_kind"] == "month":
        end = start.replace(day=calendar.monthrange(start.year, start.month)[1])
    elif period["period_kind"] == "tax_year":
        end = _anniversary(
            date.fromisoformat(period["start"]), start.year + 1
        ) - timedelta(days=1)
    else:
        return shift_request(request, start_delta.days)
    end_delta = end - date.fromisoformat(period["end"])
    shifted = copy.deepcopy(request)
    intervals = [query["period"] for query in shifted["queries"]]
    for collection in ("inputs", "relations"):
        intervals.extend(item["interval"] for item in shifted["dataset"][collection])
    # The request translator can share one interval between queries and facts.
    # Translate each object once, retaining both leading and trailing margins.
    for interval in {id(item): item for item in intervals}.values():
        for endpoint, delta in (("start", start_delta), ("end", end_delta)):
            interval[endpoint] = (
                date.fromisoformat(interval[endpoint]) + delta
            ).isoformat()
    return shifted


def classify(old: dict, new: dict, dated_rules: list[dict], period_start: str) -> str:
    if old == new:
        return "PASS"
    if old.get("companion_parameters", {}) != new.get("companion_parameters", {}):
        return "DIFF"
    match = REFUSAL.fullmatch(new.get("error", ""))
    old_refusal = OLD_DATE_REFUSAL.fullmatch(old.get("error", ""))
    old_ok = "value" in old or (old_refusal and old_refusal[1] == period_start)
    if old_ok and match and match[2] == period_start:
        # Entries with the same legal id and window are one rule. Matches that
        # differ in legal id or window stay ambiguous and leave the difference
        # visible.
        matches = {}
        for rule in dated_rules:
            qualified = f"{rule['module'].removesuffix('.yaml').replace('/', ':', 1)}#{rule['rule']}"
            if match[1] in (rule.get("runtime_name", rule["rule"]), qualified):
                key = (qualified, rule["effective_from"], rule["effective_to"])
                matches[key] = rule
        if len(matches) == 1:
            (rule,) = matches.values()
            if period_start < rule["effective_from"] or (
                rule["effective_to"] is not None and period_start > rule["effective_to"]
            ):
                return "EXPECTED_PRESTART_REFUSAL"
    return "DIFF"


def qualified_rule(rule: dict) -> str:
    return f"{str(Path(rule['module']).with_suffix('')).replace('/', ':', 1)}#{rule['rule']}"


def inventory_context(artifact: dict, dated: list[dict]) -> list[dict]:
    by_id = {rule.get("id"): rule for rule in artifact["program"].get("derived", [])}
    return [
        dict(rule, runtime_name=by_id[qualified_rule(rule)]["name"])
        for rule in dated
        if qualified_rule(rule) in by_id
    ]


def select_seed(cases: list[dict], output: str, artifact: dict) -> dict:
    rules = artifact.get("program", {}).get("derived", [])
    by_reference = {
        reference: rule
        for rule in rules
        for reference in (rule.get("id"), rule.get("name"))
        if reference
    }
    relations = {
        relation["name"]: relation
        for relation in artifact.get("program", {}).get("relations", [])
    }

    def reaches(reference, visited):
        if reference == output:
            return True
        rule = by_reference.get(reference)
        if not rule or rule["name"] in visited:
            return False
        if rule.get("id") == output:
            return True
        visited = visited | {rule["name"]}
        visited_relations = set()

        def walk(node):
            if isinstance(node, list):
                return any(walk(child) for child in node)
            if not isinstance(node, dict):
                return False
            if node.get("kind") == "derived" and reaches(node.get("name"), visited):
                return True
            for field in ("relation", "source_relation"):
                reference = node.get(field)
                relation = relations.get(reference)
                if (
                    relation is None
                    and isinstance(reference, str)
                    and "#relation." in reference
                ):
                    relation = relations.get(reference.rsplit("#relation.", 1)[1])
                if relation is not None and relation["name"] not in visited_relations:
                    visited_relations.add(relation["name"])
                    if walk(relation.get("derivation")):
                        return True
            return any(walk(child) for child in node.values())

        return walk(rule.get("expr")) or walk(rule.get("versions", []))

    for case in cases:
        if output in (case.get("output") or {}):
            return case
    for case in cases:
        if any(reaches(reference, set()) for reference in case.get("output", {})):
            return case
    return cases[0] if cases else dict(name="no-case", period="2026-01", input={})


def comparisons(
    outcomes: dict, dated_rules: list[dict], period_start: str
) -> list[dict]:
    engine_outcomes = {
        side: {
            mode: {
                key: value
                for key, value in outcome.items()
                if key != "companion_parameters"
            }
            for mode, outcome in modes.items()
        }
        for side, modes in outcomes.items()
    }
    rows = [
        dict(
            comparison="old/new explain",
            classification=classify(
                engine_outcomes["old"]["explain"],
                engine_outcomes["new"]["explain"],
                dated_rules,
                period_start,
            ),
        )
    ]
    for side, label in (("old", "OLD_PARITY_BREAK"), ("new", "PARITY_BREAK")):
        if "fast" in outcomes[side]:
            rows.append(
                dict(
                    comparison=f"{side} explain/fast",
                    classification=(
                        "PASS"
                        if engine_outcomes[side]["explain"]
                        == engine_outcomes[side]["fast"]
                        else label
                    ),
                )
            )
    old_parameters = outcomes["old"]["explain"].get("companion_parameters", {})
    new_parameters = outcomes["new"]["explain"].get("companion_parameters", {})
    if old_parameters or new_parameters:
        rows.append(
            dict(
                comparison="old/new companion parameters",
                classification="PASS" if old_parameters == new_parameters else "DIFF",
            )
        )
    pairs = {
        "old/new explain": (
            engine_outcomes["old"]["explain"],
            engine_outcomes["new"]["explain"],
        ),
        **{
            f"{side} explain/fast": (modes["explain"], modes["fast"])
            for side, modes in engine_outcomes.items()
            if "fast" in modes
        },
    }
    return [
        row
        for row in rows
        if not (
            row["classification"] == "PASS"
            and row["comparison"] in pairs
            and all(missing_input(outcome) for outcome in pairs[row["comparison"]])
        )
    ]


def missing_input(outcome: dict) -> bool:
    return bool(re.search(r"\bmissing input\b", outcome.get("error", "")))


def companion_needs_repair(kind: str, rows: list[dict], old_outcome: dict) -> bool:
    labels = {row["classification"] for row in rows}
    return (
        kind == "companion"
        and "EXPECTED_PRESTART_REFUSAL" in labels
        and not labels.intersection({"DIFF", "PARITY_BREAK"})
        and not any(
            "error" in result
            for result in old_outcome.get("companion_parameters", {}).values()
        )
    )


def execute(binary: Path, artifact: Path, request: dict, env: dict) -> dict:
    result = subprocess.run(
        [str(binary), "run-compiled", "--artifact", str(artifact)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if result.returncode:
        return {"error": result.stderr.strip() or result.stdout.strip()}
    payload = json.loads(result.stdout)
    return {
        "value": [
            {
                key: {
                    field: value
                    for field, value in output.items()
                    if field in ("value", "outcome", "kind", "dtype", "unit")
                }
                for key, output in row["outputs"].items()
            }
            for row in payload["results"]
        ]
    }


def run_modes(
    binary: Path,
    artifact: Path,
    request: dict,
    env: dict,
    modes: tuple[str, ...],
    *,
    executor=execute,
) -> dict:
    outcomes = {}
    for mode in modes:
        mode_request = copy.deepcopy(request)
        mode_request["mode"] = mode
        try:
            outcomes[mode] = executor(binary, artifact, mode_request, env)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.TimeoutExpired,
        ) as error:
            outcomes[mode] = {"error": str(error)}
    return outcomes


def write_report(report: dict, output: ConfinedOutput) -> Counter:
    """Publish report.json and report.md through the run's held output.

    The artifacts use the same descriptor, so all evidence of one run lands in
    one directory even if its path is renamed while the run is writing.
    """
    counts = Counter(
        row["classification"] for case in report["cases"] for row in case["comparisons"]
    )
    report["counts"] = {label: counts[label] for label in CLASSES}
    output.write(
        Path("report.json"),
        (json.dumps(report, indent=2, default=str) + "\n").encode(),
    )
    lines = [
        "# Engine pin differential",
        "",
        f"As of: {report['as_of']}",
        "",
        f"Old: `{report['engines']['old']['ref']}`",
        f"New: `{report['engines']['new']['ref']}`",
        "",
        "## Counts",
        "",
        *[f"- {label}: {counts[label]}" for label in CLASSES],
        f"- Incomplete coverage: {len(report.get('incomplete', []))}",
        "",
        "## Non-PASS comparisons",
        "",
        "| Module / case / output | Comparison | Class | Outcomes |",
        "| --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        for row in case["comparisons"]:
            if row["classification"] == "PASS":
                continue
            detail = (
                json.dumps(case["outcomes"], ensure_ascii=False)
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            label = f"{case['module']} / {case['name']} / {case['output']}"
            lines.append(
                f"| {label} | {row['comparison']} | {row['classification']} | {detail} |"
            )
    if report.get("incomplete"):
        lines.extend(["", "## Incomplete coverage", ""])
        lines.extend(f"- {json.dumps(error)}" for error in report["incomplete"])
    output.write(Path("report.md"), ("\n".join(lines) + "\n").encode())
    return counts


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def bind(root: Path, ref: str) -> Path:
    require_engine_ref_sha(ref, description="engine ref")
    if _git(root, "rev-parse", "HEAD") != ref:
        raise ValueError(f"Engine checkout HEAD does not match {ref}: {root}")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Engine checkout has tracked changes: {root}")
    return resolve_pinned_engine_binary(
        root, EnginePin(ref, Path("<engine-pin-diff>")), allow_build=False
    )


def build_request(
    test_file: Path, case: dict, artifact: dict, root: Path, roots: tuple[Path, ...]
) -> dict:
    from . import cli

    def index(items):
        return {
            key: item
            for item in items
            for key in cli._rulespec_public_item_keys(item, policy_repo_path=root)
        }

    program = artifact["program"]
    derived = index(program.get("derived", []))
    # Reuse the caller's guarded module scope, or bound standalone resolution
    # to this preparation. Never carry resolver state into another module.
    with cli._rulespec_resolution_cache_scope():
        result = cli._execute_rulespec_test_case(
            test_file,
            case,
            case_name=str(case.get("name", "no-case")),
            compiled_path=Path("<request-only>"),
            binary=Path("<request-only>"),
            axiom_rules_path=Path("<request-only>"),
            env={},
            parameter_by_id=index(program.get("parameters", [])),
            derived_ids=set(derived),
            derived_by_id=derived,
            declared_relation_names={
                str(item["name"]) for item in program.get("relations", [])
            },
            declared_relation_slots=cli._rulespec_declared_relation_slots(artifact),
            compiled_relations=program.get("relations", []),
            policy_repo_path=root,
            rulespec_roots=roots,
            request_only=True,
        )
    if not isinstance(result, dict):
        raise ValueError(json.dumps(result))
    return result


@_rulespec_routing_cache_scope()
def run(args, *, executor=execute) -> int:
    from . import cli

    root = args.root.resolve(strict=True)
    if cli.canonical_rulespec_root_identity(root) is None:
        raise ValueError("--root must be an exact RuleSpec jurisdiction content root")
    checkout = Path(_git(root, "rev-parse", "--show-toplevel"))
    confined_output = ConfinedOutput(args.out_dir)
    out_dir = confined_output.path
    # Sibling subtrees of one checkout name the same checkout. Authorize each
    # checkout once: every module resolution re-validates each authorized root
    # (its symlink audit and jurisdiction listing), so a repeated root
    # multiplies that work without changing what resolves. Keep the active
    # checkout here when a sibling names it: collect_modules inventories the
    # sibling subtrees from these dependency roots, not from the active root.
    roots = (
        checkout,
        *dict.fromkeys(
            Path(_git(path.resolve(strict=True), "rev-parse", "--show-toplevel"))
            for path in args.rulespec_dependency_root
        ),
    )
    if inside_checkout(out_dir, roots):
        raise ValueError("--out-dir must be outside all authorized RuleSpec checkouts")
    # gettempdir() probes candidates by writing a file. Check containment
    # before any such probe, including when TMPDIR itself is a checkout.
    temporary_root = Path(
        os.environ.get("TMPDIR")
        or os.environ.get("TEMP")
        or os.environ.get("TMP")
        or "/tmp"
    ).resolve(strict=True)
    if inside_checkout(temporary_root, roots):
        raise ValueError(
            "Compiler temporary directory must be outside RuleSpec checkouts"
        )
    # Compiler/composer writers do not expose no-follow flags. Give them only
    # fresh private paths, never reused evidence paths, and publish their bytes
    # through ConfinedOutput. TemporaryDirectory creates its directory with
    # exclusive mkdir and mode 0700, outside all admitted checkouts.
    with (
        confined_output as output,
        tempfile.TemporaryDirectory(
            prefix="hub-epm-compile-", dir=temporary_root
        ) as workspace,
    ):
        return _run(args, root, checkout, roots, output, Path(workspace), executor)


def collect_modules(checkout, root, roots, selected):
    """Return the selected modules under root and the dated rules reachable from it.

    Dependency roots resolve to their checkouts, so a sibling subtree of the
    root's own checkout lists files already read. Each file is inventoried
    once: a repeated entry would make its rules look ambiguous to classify().
    """
    modules = []
    classification_inventory = []
    inventoried = set()

    def rulespec(path):
        if path.suffix not in (".yaml", ".yml") or path.name.endswith(
            (".test.yaml", ".test.yml")
        ):
            return None
        payload = yaml.safe_load(path.read_text())
        if isinstance(payload, dict) and payload.get("format") == "rulespec/v1":
            return payload
        return None

    def admit(path, name, payload):
        if path.resolve() not in inventoried:
            inventoried.add(path.resolve())
            classification_inventory.extend(inventory(payload, name))

    for name in _git(checkout, "ls-files", "-z").split("\0"):
        path = checkout / name
        if not path.is_relative_to(root) or (payload := rulespec(path)) is None:
            continue
        admit(path, name, payload)
        if selected and not any(
            path == candidate or path.is_relative_to(candidate)
            for candidate in selected
        ):
            continue
        modules.append((name, path, payload))
    # Sibling roots of one checkout resolve to the same top; list it once.
    for dependency in dict.fromkeys(path.resolve() for path in roots[1:]):
        for name in _git(dependency, "ls-files", "-z").split("\0"):
            path = dependency / name
            if path.resolve() in inventoried:
                continue
            if (payload := rulespec(path)) is not None:
                admit(path, name, payload)
    return modules, classification_inventory


def _run(args, root, checkout, roots, confined_output, workspace, executor):
    from . import cli

    out_dir = confined_output.path
    selected = []
    for candidate in args.paths:
        candidate = (
            candidate if candidate.is_absolute() else root / candidate
        ).resolve(strict=True)
        if candidate.name.endswith(".test.yaml"):
            candidate = cli._rulespec_program_for_test_file(candidate)
        if not candidate.is_relative_to(root):
            raise ValueError(f"Selected path is outside --root: {candidate}")
        selected.append(candidate)
    engines = {}
    for side in ("old", "new"):
        engine_root = getattr(args, f"{side}_engine").resolve(strict=True)
        ref = getattr(args, f"{side}_engine_ref")
        binary = bind(engine_root, ref)
        engines[side] = dict(
            ref=ref,
            checkout=str(engine_root),
            binary=str(binary),
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        )
    modules, classification_inventory = collect_modules(checkout, root, roots, selected)
    if not modules:
        raise ValueError("No RuleSpec modules selected")
    dated = [rule for name, _, payload in modules for rule in inventory(payload, name)]
    report = dict(
        as_of=args.as_of.isoformat(),
        engines=engines,
        inventory=dated,
        classification_inventory=classification_inventory,
        modules=[],
        cases=[],
        incomplete=[],
        root=str(root),
        repository_commit=_git(checkout, "rev-parse", "HEAD"),
    )
    pipelines = {
        side: cli.ValidatorPipeline(
            policy_repo_path=root,
            axiom_rules_path=Path(engine["checkout"]),
            axiom_rules_engine_ref=engine["ref"],
            local_corpus_release=None,
            axiom_compose_path=getattr(args, "axiom_compose_path", None),
            enable_oracles=False,
            rulespec_dependency_roots=roots[1:],
        )
        for side, engine in engines.items()
    }
    env = pipelines["new"]._rulespec_engine_env()

    @cli._rulespec_resolution_cache_scope()
    def compare_module(module_index, module):
        name, path, payload = module
        module_record = dict(
            path=name, blob_sha=_git(checkout, "hash-object", str(path))
        )
        report["modules"].append(module_record)
        artifacts, compilation = {}, {}
        for side in engines:
            directory = workspace / "artifacts" / side
            directory.mkdir(parents=True, exist_ok=True)
            artifact = directory / f"{module_index}.json"
            compiled, compiled_payload = pipelines[side]._compile_rulespec_to_artifact(
                path, artifact
            )
            for generated in directory.iterdir():
                if generated.is_symlink() or not generated.is_file():
                    raise ValueError(f"Unsafe compiler output: {generated}")
                confined_output.write(
                    generated.relative_to(workspace), generated.read_bytes()
                )
            compilation[side] = (
                None
                if compiled.returncode == 0
                else compiled.stderr.strip() or compiled.stdout.strip()
            )
            if compilation[side] is None:
                artifacts[side] = compiled_payload
            else:
                report["incomplete"].append(
                    dict(module=name, engine=side, error=compilation[side])
                )
        module_record["compile_errors"] = compilation
        test_file = path.with_suffix(".test.yaml")
        cases = cli._load_rulespec_test_cases(test_file) if test_file.exists() else []
        seeds = [("companion", case, None, None) for case in cases]
        for rule in inventory(payload, name):
            output = qualified_rule(rule)
            seed = select_seed(
                cases, output, artifacts.get("new", artifacts.get("old", {}))
            )
            boundary_dates = []
            if rule["effective_from"] != "0001-01-01":
                start = date.fromisoformat(rule["effective_from"])
                boundary_dates.extend(
                    start + timedelta(days=delta) for delta in (-1, 0, 1)
                )
            if rule["effective_to"]:
                end = date.fromisoformat(rule["effective_to"])
                boundary_dates.extend(
                    end + timedelta(days=delta) for delta in (-1, 0, 1)
                )
            for boundary in boundary_dates:
                case = copy.deepcopy(seed)
                case["output"] = {output: None}
                case["name"] = f"{rule['rule']}@{boundary} ({seed.get('name', 'case')})"
                seeds.append(("boundary", case, boundary, not cases))
        for kind, case, boundary, no_case in seeds:
            period = cli._rulespec_period_spec(case.get("period", "2026-01"))
            variants = [(kind, boundary)]
            if kind == "companion":
                variants.append(("current", None))
            for variant, anchor in variants:
                start = (
                    anchor.isoformat()
                    if anchor
                    else current_start(period, args.as_of).isoformat()
                    if variant == "current"
                    else period["start"]
                )
                outcomes, saved_requests = {}, {}
                for side in engines:
                    outcomes[side] = {}
                    modes = (
                        ("explain",) if variant == "current" else ("explain", "fast")
                    )
                    preparation_error = None
                    try:
                        if compilation[side]:
                            raise ValueError(f"compile: {compilation[side]}")
                        if not case.get("output"):
                            raise ValueError("companion case has no output assertions")
                        translated_case = copy.deepcopy(case)
                        delta = (
                            date.fromisoformat(start)
                            - date.fromisoformat(period["start"])
                        ).days
                        translated_case["period"] = dict(period)
                        for endpoint in ("start", "end"):
                            translated_case["period"][endpoint] = (
                                date.fromisoformat(period[endpoint])
                                + timedelta(days=delta)
                            ).isoformat()
                        if variant == "current":
                            # Determine calendar length from the original seed,
                            # including Feb-29 fiscal anniversaries, before the
                            # translator builds fixtures and parameter values.
                            translated_case["period"] = current_request(
                                {
                                    "queries": [{"period": period}],
                                    "dataset": {"inputs": [], "relations": []},
                                },
                                args.as_of,
                            )["queries"][0]["period"]
                        request = build_request(
                            test_file, translated_case, artifacts[side], root, roots
                        )
                        parameters = request.pop("_companion_parameters", {})
                        saved_requests[side] = copy.deepcopy(request)
                    except (ValueError, KeyError, OSError) as error:
                        preparation_error = {"error": str(error)}
                        report["incomplete"].append(
                            dict(
                                module=name,
                                case=case.get("name"),
                                engine=side,
                                error=str(error),
                            )
                        )
                    if preparation_error:
                        outcomes[side] = {mode: preparation_error for mode in modes}
                    else:
                        outcomes[side] = run_modes(
                            Path(engines[side]["binary"]),
                            workspace / "artifacts" / side / f"{module_index}.json",
                            request,
                            env,
                            modes,
                            executor=executor,
                        )
                        if parameters:
                            outcomes[side] = {
                                mode: dict(outcome, companion_parameters=parameters)
                                for mode, outcome in outcomes[side].items()
                            }
                        for mode, outcome in outcomes[side].items():
                            if missing_input(outcome):
                                report["incomplete"].append(
                                    dict(
                                        module=name,
                                        case=case.get("name"),
                                        kind=variant,
                                        engine=side,
                                        mode=mode,
                                        error=outcome["error"],
                                    )
                                )
                context = (
                    inventory_context(artifacts["new"], classification_inventory)
                    if "new" in artifacts
                    else []
                )
                rows = comparisons(outcomes, context, start)
                report["cases"].append(
                    dict(
                        module=name,
                        name=f"{variant}: {case.get('name', 'case')}",
                        kind=variant,
                        output=", ".join(case.get("output", {})),
                        period_start=start,
                        no_case=bool(no_case),
                        requests=saved_requests,
                        outcomes=outcomes,
                        comparisons=rows,
                        needs_encoder_repair=companion_needs_repair(
                            variant, rows, outcomes["old"]["explain"]
                        ),
                    )
                )
                report["cases"][-1]["source_case"] = copy.deepcopy(case)
        write_report(report, confined_output)

    for module_index, module in enumerate(modules):
        compare_module(module_index, module)
    counts = write_report(report, confined_output)
    print(json.dumps(dict(counts=report["counts"], report=str(out_dir / "report.md"))))
    return int(bool(counts["DIFF"] or counts["PARITY_BREAK"] or report["incomplete"]))
