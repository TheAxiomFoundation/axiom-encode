"""Scaffold a migrated RuleSpec jurisdiction lane with oracle readiness gating."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

READINESS_RELATIVE_PATH = Path("axiom_oracles/data/euromod_country_readiness.json")
CC_RE = re.compile(r"^[a-z]{2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
ZERO_MINOR_UNIT_CURRENCIES = {
    "BIF",
    "CLP",
    "DJF",
    "GNF",
    "ISK",
    "JPY",
    "KMF",
    "KRW",
    "PYG",
    "RWF",
    "UGX",
    "UYI",
    "VND",
    "VUV",
    "XAF",
    "XOF",
    "XPF",
}
THREE_MINOR_UNIT_CURRENCIES = {"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"}
FOUR_MINOR_UNIT_CURRENCIES = {"CLF", "UYW"}
SOUTHMOD_SCAN_MAX_BYTES = 1_000_000
SOUTHMOD_SCAN_DIRS = (
    Path("axiom_oracles/bridges/mappings"),
    Path("axiom_oracles/data"),
    Path("axiom_oracles/adapters"),
)


@dataclass(frozen=True, slots=True)
class OracleEvidence:
    name: str
    status: str
    matrix_file: str
    probed: str | None = None


def register_new_jurisdiction_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "new-jurisdiction",
        help="Scaffold a migrated RuleSpec jurisdiction lane",
        description=(
            "Scaffold a migrated RuleSpec jurisdiction lane with scaffold-adapted "
            "layout tests; the lane grows into the full migrated-lane suite as it "
            "fills in."
        ),
    )
    parser.add_argument("cc", help="lowercase ISO 3166-1 alpha-2 country code")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axiom-oracles-path", type=Path, required=True)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--jurisdiction-name", required=True)
    parser.add_argument("--currency", required=True, help="ISO 4217 currency code")
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument(
        "--oracle-plan",
        help="oracle plan text, or a path to a UTF-8 plan file",
    )
    gate.add_argument(
        "--record-oracle-override",
        metavar="REASON",
        help="record why an available oracle will not be used",
    )
    parser.add_argument(
        "--force", action="store_true", help="allow a non-empty output directory"
    )


def run_new_jurisdiction(args: argparse.Namespace) -> int:
    try:
        cc = _country_code(args.cc)
        currency = _currency_code(args.currency)
        output = args.output.expanduser().resolve()
        oracles = args.axiom_oracles_path.expanduser().resolve()
        engine = args.engine_path.expanduser().resolve()
        _require_checkout(oracles, "axiom-oracles")
        _require_checkout(engine, "axiom-rules-engine")
        evidence, checks = discover_oracles(cc, oracles)
        oracle_plan = _validated_oracle_plan(args.oracle_plan)
        override = _validated_override(args.record_oracle_override)
        if evidence and oracle_plan is None and override is None:
            descriptions = ", ".join(
                f"{item.name} (status: {item.status})" for item in evidence
            )
            raise ValueError(
                f"Refusing to scaffold {cc}: an oracle is available: {descriptions}. "
                "Provide --oracle-plan <file|text> or "
                "--record-oracle-override <reason>."
            )
        if output.exists() and any(output.iterdir()) and not args.force:
            raise ValueError(
                f"Refusing to scaffold into non-empty output directory {output}; "
                "pass --force to proceed."
            )
        files = _render_scaffold(
            cc=cc,
            jurisdiction_name=args.jurisdiction_name,
            evidence=evidence,
            checks=checks,
            oracle_plan=oracle_plan,
            override=override,
        )
        currency_seeded = _currency_is_seeded(engine, currency)
        if not currency_seeded:
            files["engine-currency-seed.diff"] = _currency_diff(engine, currency)

        # Phase B starts only after every input has been validated and every output
        # has been rendered successfully in memory.
        output.mkdir(parents=True, exist_ok=True)
        _write_files(output, files)
        print(f"Scaffolded rulespec-{cc} at {output}")
        print(
            "Oracle evidence: " + (", ".join(e.name for e in evidence) or "unavailable")
        )
        if currency_seeded:
            print(f"Engine currency seed: {currency} is already registered")
        else:
            print(
                f"Engine currency seed: {currency} is absent; wrote "
                "engine-currency-seed.diff (the engine checkout was not modified)"
            )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"new-jurisdiction: {exc}", file=sys.stderr)
        return 1


def discover_oracles(
    cc: str, oracles_path: Path
) -> tuple[list[OracleEvidence], list[str]]:
    matrix = oracles_path / READINESS_RELATIVE_PATH
    if not matrix.is_file():
        raise ValueError(f"EUROMOD readiness matrix not found: {matrix}")
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"malformed EUROMOD readiness matrix (root must be an object): {matrix}"
        )
    countries = payload.get("countries")
    if not isinstance(countries, dict):
        raise ValueError(
            f"malformed EUROMOD readiness matrix "
            f"(countries must be an object): {matrix}"
        )
    if any(
        not isinstance(country_code, str) or not isinstance(entry, dict)
        for country_code, entry in countries.items()
    ):
        raise ValueError(
            f"malformed EUROMOD readiness matrix "
            f"(country entries must be objects): {matrix}"
        )
    evidence: list[OracleEvidence] = []
    country = countries.get(cc.upper()) or countries.get(cc)
    if isinstance(country, dict):
        status = country.get("status")
    else:
        status = country
    if isinstance(status, str) and status.startswith("ok"):
        evidence.append(
            OracleEvidence(
                name="EUROMOD",
                status=status,
                matrix_file=READINESS_RELATIVE_PATH.as_posix(),
                probed=_probe_date(payload, country),
            )
        )
    southmod_files = _southmod_adapter_files(cc, oracles_path)
    if southmod_files:
        evidence.append(
            OracleEvidence(
                name="SOUTHMOD",
                status="adapter_available",
                matrix_file=", ".join(
                    path.relative_to(oracles_path).as_posix() for path in southmod_files
                ),
            )
        )
    checks = [
        f"EUROMOD readiness matrix: {READINESS_RELATIVE_PATH.as_posix()} "
        f"(country entry: {'present' if country is not None else 'absent'}"
        f"{f', status: {status}' if status is not None else ''})",
        "SOUTHMOD adapter inventory: "
        + (
            ", ".join(p.relative_to(oracles_path).as_posix() for p in southmod_files)
            or "no adapter found"
        ),
    ]
    return evidence, checks


def _probe_date(payload: dict[str, Any], country: Any) -> str | None:
    if isinstance(country, dict):
        for key in ("probed", "probe_date", "checked_at"):
            if isinstance(country.get(key), str):
                return country[key]
    for key in ("probed", "probe_date", "checked_at"):
        if isinstance(payload.get(key), str):
            return payload[key]
    return None


def _southmod_adapter_files(cc: str, root: Path) -> list[Path]:
    """Find bounded mapping/data records tying a country to SOUTHMOD."""
    candidates: list[Path] = []
    for relative_root in SOUTHMOD_SCAN_DIRS:
        scan_root = root / relative_root
        if not scan_root.is_dir():
            continue
        for path in sorted(scan_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".json",
                ".py",
                ".toml",
                ".yaml",
                ".yml",
            }:
                continue
            if path.stat().st_size > SOUTHMOD_SCAN_MAX_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "southmod" not in text.lower():
                continue
            country_tied = path.stem.lower() == cc or re.search(
                rf"(?im)^\s*(?:country|jurisdiction|code)\s*:\s*[\"']?{re.escape(cc)}[\"']?\s*$",
                text,
            )
            if country_tied:
                candidates.append(path)
    return candidates


def _render_scaffold(
    *,
    cc: str,
    jurisdiction_name: str,
    evidence: list[OracleEvidence],
    checks: list[str],
    oracle_plan: str | None,
    override: str | None,
) -> dict[str, str]:
    files = _base_files(cc, jurisdiction_name)
    if oracle_plan is not None:
        files[".axiom/oracle-plan.md"] = _oracle_plan(oracle_plan, evidence)
    elif override is not None:
        files[".axiom/oracle-override.md"] = _oracle_override(override, evidence)
    else:
        files[".axiom/oracle-status.md"] = _oracle_status(checks)
    return files


def _write_files(output: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _base_files(cc: str, name: str) -> dict[str, str]:
    return {
        ".axiom/registry.toml": _registry(cc),
        ".axiom/repository-structure.yaml": _repository_structure(cc),
        ".axiom/toolchain.toml": _toolchain(cc),
        ".github/workflows/repository-checks.yml": _workflow(),
        ".gitignore": ".venv/\n__pycache__/\n.pytest_cache/\n.ruff_cache/\n*.pyc\n",
        "CLAUDE.md": _bringup_document(cc, name, agent_notes=True),
        "README.md": _bringup_document(cc, name, agent_notes=False),
        "corpus-manifest-skeleton.yaml": _corpus_manifest(cc, name),
        "known-missing-money-atoms.yaml": _money_atoms(cc),
        "known-validation-gaps.yaml": _validation_gaps(cc),
        "oracle-coverage-pending.yaml": _oracle_pending(cc),
        "data/coverage/tax-benefit-source-map.json": _source_map(cc),
        "data/oracles/oracle-index.json": _oracle_index(cc),
        f"{cc}/policies/.gitkeep": "",
        f"{cc}/regulations/.gitkeep": "",
        f"{cc}/statutes/.gitkeep": "",
        "tests/test_repository_layout.py": _layout_tests(cc),
    }


def _registry(cc: str) -> str:
    return f"""# App-surface visibility for this repo's encodings on axiom-foundation.org.
# TODO({cc}): flip to "public" only after the listing gates in README.md hold.
[registry]
app_visibility = "experimental"
"""


def _repository_structure(cc: str) -> str:
    return f'''version: 1
allowed_root_directories:
  - .axiom
  - .github
  - bulk
  - data
  - programs
  - tests
  - {cc}
allowed_root_files:
  - .gitignore
  - CLAUDE.md
  - README.md
  - corpus-manifest-skeleton.yaml
  - engine-currency-seed.diff
  - known-missing-money-atoms.yaml
  - known-validation-gaps.yaml
  - oracle-coverage-pending.yaml
  - variables.toml
path_rules:
  - patterns: [".axiom/**"]
    allow_extensions: [".json", ".md", ".txt", ".toml", ".yaml"]
  - patterns: [".github/**"]
    allow_extensions: [".yml", ".yaml", ".md"]
  - patterns: ["bulk/**"]
    allow_extensions: [".md", ".py", ".yaml"]
  - patterns: ["data/**"]
    allow_extensions: [".json", ".jsonl", ".yaml", ".yml"]
  - patterns: ["programs/**"]
    allow_extensions: [".yaml"]
  - patterns: ["tests/**"]
    allow_extensions: [".py"]
  - patterns: ["{cc}/**"]
    allow_extensions: [".yaml"]
    allow_filenames: [".gitkeep"]
'''


def _toolchain(cc: str) -> str:
    return f'''# NON-VALIDATING PLACEHOLDER — rulespec-{cc} has no signed corpus release yet.
# Under TheAxiomFoundation/.github#39 rule 1, replace this file in a dedicated
# gated toolchain PR after the first release cut. Do not invent values here.
[toolchain]
# Required key 1: axiom_corpus_release = "{cc}-rulespec-YYYY-MM-DD"
# Required key 2: axiom_corpus_release_content_sha256 = "<64 lowercase hex>"
# Required key 3: validation_waiver_set_sha256 = "<64 lowercase hex>"
'''


def _workflow() -> str:
    return """name: Repository Checks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  validate:
    # <pin-me>: copy the reviewed validate-rulespec workflow SHA from TheAxiomFoundation/.github.
    uses: TheAxiomFoundation/.github/.github/workflows/validate-rulespec.yml@<pin-me>
    secrets: inherit
    with:
      # <pin-me>: use the commit of the dedicated toolchain/pin PR for each checkout.
      axiom-encode-ref: <pin-me>
      axiom-rules-engine-ref: <pin-me>
      axiom-corpus-ref: <pin-me>
      rulespec-us-ref: <pin-me>
      validate-roots: auto
      run-generated-guard: true
      guard-programs-root: false
"""


def _bringup_document(cc: str, name: str, *, agent_notes: bool) -> str:
    title = f"rulespec-{cc} — agent notes" if agent_notes else f"rulespec-{cc}"
    intro = f"RuleSpec lane scaffold for {name}. TODO({cc}): replace jurisdiction substance placeholders."
    return f"""# {title}

{intro}

This scaffold emits scaffold-adapted layout tests; the lane grows into the full
migrated-lane suite as it fills in.

## Lane bring-up sequence

1. Complete `corpus-manifest-skeleton.yaml`, move it to axiom-corpus `manifests/`, and open the corpus manifest PR.
2. Ingest the captured sources through the corpus pipeline.
3. Cut and sign the first immutable `{cc}` corpus release.
4. Land the dedicated gated `.axiom/toolchain.toml` PR with the real three-key binding.
5. Replace every workflow `<pin-me>` with reviewed commit SHAs.
6. Encode the first module through the supervised runtime; do not hand-author RuleSpec.
7. Run `axiom-encode ci` locally with the explicit dependency checkouts and release public key.

## Listing gates

Keep `.axiom/registry.toml` experimental until the signed release binding, pinned CI,
first supervised encoding, and oracle validation or recorded disposition are complete.
"""


def _validation_gaps(cc: str) -> str:
    return f"""# Known validation gaps ratchet for rulespec-{cc}.
# TODO({cc}): keep this decrement-only; the fresh lane has no waived failures.
validate_failures: {{}}
"""


def _money_atoms(cc: str) -> str:
    return f"""# Money-atom proof-obligation ratchet for rulespec-{cc}.
# TODO({cc}): every monetary value must ship with a provision-citing proof atom.
total_allowed: 0
"""


def _oracle_pending(cc: str) -> str:
    return f"""# Declared oracle-coverage debt for rulespec-{cc}; ratchets both ways.
version: 1
issue: "TODO({cc}): add the jurisdiction tracking issue URL"
ceiling: 0
entries: []
"""


def _source_map(cc: str) -> str:
    return (
        json.dumps(
            {
                "generated_at": f"TODO({cc})",
                "jurisdiction": cc,
                "validation_year": f"TODO({cc})",
                "note": f"TODO({cc}): map the tax-benefit surface to governing law.",
                "instruments": [],
            },
            indent=2,
        )
        + "\n"
    )


def _oracle_index(cc: str) -> str:
    return (
        json.dumps(
            {
                "generated_at": f"TODO({cc})",
                "jurisdiction": cc,
                "focus": f"TODO({cc}): define the validation surface.",
                "purpose": "Executable comparison-oracle inventory; oracles are not legal authority.",
                "oracles": [],
            },
            indent=2,
        )
        + "\n"
    )


def _corpus_manifest(cc: str, name: str) -> str:
    return f"""# Scaffold artifact: move the completed manifest to axiom-corpus/manifests/.
documents:
  - source_id: TODO-{cc}-official-source
    jurisdiction: {cc}
    document_class: statute
    citation_path: {cc}/statute/TODO
    title: "TODO({cc}): official {name} instrument title"
    source_url: "TODO({cc}): stable official source URL"
    source_format: pdf
    source_as_of: "TODO({cc}): YYYY-MM-DD"
    expression_date: "TODO({cc}): YYYY-MM-DD"
    language: "TODO({cc}): ISO 639-1 code"
    extraction:
      segmentation: single_block
    metadata:
      primary_source: true
      source_authority: "TODO({cc}): publishing authority"
      document_type: "TODO({cc}): instrument type"
      source_note: "TODO({cc}): provenance and amendment-diligence notes"
"""


def _layout_tests(cc: str) -> str:
    return f'''from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIRS = ("statutes", "regulations", "policies", "legislation")
IGNORED_DIRS = {{".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}}
ALLOWED_ROOT_DIRS = {{".axiom", ".github", "bulk", "data", "programs", "tests", "{cc}"}}
ALLOWED_ROOT_FILES = {{
    ".gitignore", "CLAUDE.md", "README.md", "corpus-manifest-skeleton.yaml",
    "engine-currency-seed.diff", "known-missing-money-atoms.yaml",
    "known-validation-gaps.yaml", "oracle-coverage-pending.yaml", "variables.toml",
}}


def rulespec_files() -> list[Path]:
    return sorted(path for bucket in CONTENT_DIRS for path in (ROOT / "{cc}" / bucket).rglob("*.yaml") if not path.name.endswith(".test.yaml"))


def test_only_{cc}_namespace_present() -> None:
    names = {{child.name for child in ROOT.iterdir() if child.is_dir() and re.fullmatch(r"[a-z]{{2}}(?:-[a-z0-9-]+)*", child.name) and any((child / marker).is_dir() for marker in CONTENT_DIRS)}}
    assert names <= {{"{cc}"}}


def test_{cc}_content_buckets_exist() -> None:
    for marker in ("statutes", "regulations", "policies"):
        assert (ROOT / "{cc}" / marker).is_dir()


def test_root_inventory_is_allowed() -> None:
    directories = {{child.name for child in ROOT.iterdir() if child.is_dir() and child.name not in IGNORED_DIRS and not child.name.startswith(("_", "."))}}
    files = {{child.name for child in ROOT.iterdir() if child.is_file() and child.name != ".git"}}
    assert not directories - ALLOWED_ROOT_DIRS
    assert not files - ALLOWED_ROOT_FILES


def test_every_rulespec_has_companion_test() -> None:
    for path in rulespec_files():
        assert path.with_name(path.stem + ".test.yaml").exists()


def test_empty_ratchets_have_current_shapes() -> None:
    assert yaml.safe_load((ROOT / "known-validation-gaps.yaml").read_text()) == {{"validate_failures": {{}}}}
    assert yaml.safe_load((ROOT / "known-missing-money-atoms.yaml").read_text()) == {{"total_allowed": 0}}
    pending = yaml.safe_load((ROOT / "oracle-coverage-pending.yaml").read_text())
    assert pending["ceiling"] == 0 and pending["entries"] == []


def test_scoped_indexes() -> None:
    for relative in ("data/oracles/oracle-index.json", "data/coverage/tax-benefit-source-map.json"):
        assert json.loads((ROOT / relative).read_text())["jurisdiction"] == "{cc}"


def test_toolchain_declares_non_validating_placeholder() -> None:
    text = (ROOT / ".axiom/toolchain.toml").read_text()
    payload = tomllib.loads(text)
    assert payload == {{"toolchain": {{}}}}
    assert "NON-VALIDATING PLACEHOLDER" in text
    for key in ("axiom_corpus_release", "axiom_corpus_release_content_sha256", "validation_waiver_set_sha256"):
        assert key in text
'''


def _oracle_plan(plan: str, evidence: list[OracleEvidence]) -> str:
    return (
        "# Oracle plan\n\n"
        + _evidence_markdown(evidence)
        + "\n## Plan\n\n"
        + plan.rstrip()
        + "\n"
    )


def _oracle_override(reason: str, evidence: list[OracleEvidence]) -> str:
    return (
        "# Oracle readiness override\n\nRECORDED: <fill on commit>\n\n"
        + _evidence_markdown(evidence)
        + "\n## Reason (recorded verbatim)\n\n"
        + reason
        + "\n"
    )


def _oracle_status(checks: list[str]) -> str:
    return (
        "# Oracle status\n\nNo runnable oracle was found.\n\n## Checks\n\n"
        + "".join(f"- {check}\n" for check in checks)
    )


def _evidence_markdown(evidence: list[OracleEvidence]) -> str:
    lines = ["## Readiness evidence", ""]
    for item in evidence:
        lines.extend(
            [
                f"- Oracle: {item.name}",
                f"  - Status: {item.status}",
                f"  - Matrix/inventory file: `{item.matrix_file}`",
                f"  - Probe date: {item.probed or 'not recorded'}",
            ]
        )
    return "\n".join(lines) + "\n"


def _currency_is_seeded(engine: Path, currency: str) -> bool:
    registry = engine / "src/formula.rs"
    if not registry.is_file():
        raise ValueError(f"engine currency registry not found: {registry}")
    return (
        re.search(
            rf'\("{re.escape(currency)}",\s*UnitKindSpec::Currency\s*\{{',
            registry.read_text(encoding="utf-8"),
        )
        is not None
    )


def _currency_diff(engine: Path, currency: str) -> str:
    registry = engine / "src/formula.rs"
    original = registry.read_text(encoding="utf-8")
    marker = '        ("count", UnitKindSpec::Count),'
    if marker not in original:
        raise ValueError(f"could not locate currency insertion point in {registry}")
    minor_units = _currency_minor_units(currency)
    addition = (
        f"        // {currency}: ISO 4217 currency exponent {minor_units}.\n"
        f'        ("{currency}", UnitKindSpec::Currency {{ minor_units: {minor_units} }}),\n'
    )
    updated = original.replace(marker, addition + marker, 1)
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile="a/src/formula.rs",
        tofile="b/src/formula.rs",
    )
    return "".join(diff)


def _validated_oracle_plan(value: str | None) -> str | None:
    if value is None:
        return None
    plan_path = Path(value).expanduser()
    try:
        is_plan_file = plan_path.is_file()
    except OSError:
        is_plan_file = False
    plan = plan_path.read_text(encoding="utf-8") if is_plan_file else value
    if not plan.strip():
        raise ValueError("--oracle-plan must not be empty or whitespace-only")
    return plan


def _validated_override(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError(
            "--record-oracle-override must not be empty or whitespace-only"
        )
    return value


def _currency_minor_units(currency: str) -> int:
    if currency in ZERO_MINOR_UNIT_CURRENCIES:
        return 0
    if currency in THREE_MINOR_UNIT_CURRENCIES:
        return 3
    if currency in FOUR_MINOR_UNIT_CURRENCIES:
        return 4
    return 2


def _country_code(value: str) -> str:
    if CC_RE.fullmatch(value) is None:
        raise ValueError(
            f"country code must be exactly two lowercase letters: {value!r}"
        )
    return value


def _currency_code(value: str) -> str:
    if CURRENCY_RE.fullmatch(value) is None:
        raise ValueError(
            f"currency must be a three-letter uppercase ISO code: {value!r}"
        )
    return value


def _require_checkout(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{label} checkout not found: {path}")
