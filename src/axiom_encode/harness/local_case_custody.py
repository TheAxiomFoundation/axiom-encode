"""Fresh companion-process provenance; not a source-validation capability."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import yaml

from . import local_compile_custody as compile_custody
from .same_case_execution import strict_explain_metadata


def _typed(value: Any) -> tuple:
    if value is None or type(value) in (str, bool, int):
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ("float", value.hex())
    if type(value) is date:
        return ("date", value.isoformat())
    if type(value) is list:
        return ("list", tuple(_typed(item) for item in value))
    if type(value) is dict and all(type(k) is str for k in value):
        return ("dict", tuple((k, _typed(v)) for k, v in value.items()))
    raise ValueError("unsupported typed companion value")


def _request(
    pipeline: Any, rules_file: Path, program: dict, case: dict, ordinal: int
) -> dict | None:
    from axiom_encode.companion_relations import executable_relation_directions

    from . import validator_pipeline as v

    if not isinstance(case, dict) or "tables" in case:
        raise ValueError("unsupported companion transport")
    derived, parameters = pipeline._rulespec_program_maps(program)
    output = case.get("output")
    if (
        not isinstance(output, dict)
        or not output
        or any(isinstance(x, (list, dict)) for x in output.values())
    ):
        raise ValueError("scalar public outputs required")
    if any(key not in derived and key not in parameters for key in output):
        raise ValueError("canonical output names required")
    outputs = [key for key in output if key in derived]
    if not outputs:
        return None
    entities = {derived[key].get("entity") for key in outputs}
    if len(entities) != 1 or not all(isinstance(x, str) and x for x in entities):
        raise ValueError("single declared root required")
    entity = next(iter(entities))
    period = pipeline._coerce_rulespec_period(case["period"])
    root = pipeline._rulespec_case_query_entity_id(case, entity, ordinal)
    slots = v._rulespec_declared_relation_slots(program)
    directions = executable_relation_directions(
        derived,
        outputs,
        period,
        entity,
        slots,
        program.get("program", {}).get("relations", []),
    )
    dataset = pipeline._build_rulespec_dataset(
        case.get("input", {}),
        case_tables=None,
        period=period,
        query_entity=entity,
        query_entity_id=root,
        require_legal_input_keys=v._rulespec_program_has_legal_ids(program),
        legal_ids_by_friendly_name=pipeline._rulespec_legal_ids_by_friendly_output_name(
            program
        ),
        module_target=v._rulespec_module_target(program),
        declared_relation_names=v._rulespec_declared_relation_names(program),
        declared_relation_slots=slots,
        executable_directions=directions,
        input_entities=pipeline._rulespec_companion_input_entities(
            rules_file, case.get("input", {})
        ),
        table_entity_ids={},
    )
    return {
        "mode": "explain",
        "dataset": dataset,
        "queries": [{"entity_id": root, "period": period, "outputs": outputs}],
    }


@dataclass(frozen=True)
class _Record:
    ordinal: int
    request_json: str
    response_json: str


@dataclass
class _Suite:
    owner: Any
    generation: int
    pipeline: Any
    artifact: Path
    raw_artifact: bytes
    expected: dict[int, tuple[tuple, tuple]]
    cwd: str | None
    env_digest: str
    records: dict[int, _Record] = field(default_factory=dict)
    failed: bool = False
    finished: bool = False
    cases_input: Any = None
    compiled_input: Any = None
    cases_identity: tuple | None = None
    compiled_identity: tuple | None = None

    def inputs_match(self) -> bool:
        try:
            return (
                self.cases_identity is None
                or _typed(self.cases_input) == self.cases_identity
            ) and (
                self.compiled_identity is None
                or _typed(self.compiled_input) == self.compiled_identity
            )
        except ValueError:
            return False

    def fresh(self) -> bool:
        try:
            return (
                self.inputs_match()
                and compile_custody._owner(self.owner.capture) is self.owner
                and self.owner.generation == self.generation
                and compile_custody._fresh(self.owner)
                and not self.artifact.is_symlink()
                and self.artifact.read_bytes() == self.raw_artifact
            )
        except OSError:
            return False

    def invalidate(self) -> None:
        self.failed = True
        self.owner.case_records = None
        _forget_seal(self.owner)

    def finish(self, issues: list[str]) -> None:
        if self.finished:
            self.invalidate()
            return
        self.finished = True
        if (
            not issues
            and not self.failed
            and set(self.records) == set(self.expected)
            and self.fresh()
            and self.inputs_match()
        ):
            self.owner.case_records = tuple(
                self.records[k] for k in sorted(self.records)
            )
            if _current.get() is self:
                _sealed[id(self.owner)] = _SealedSuite(
                    self,
                    self.owner.receipt,
                    self.owner.case_records,
                    _record_digest(self.owner.case_records),
                    tuple(sorted(self.expected.items())),
                )
        else:
            self.invalidate()


_current: ContextVar[_Suite | None] = ContextVar("local_case_suite", default=None)


@dataclass(frozen=True)
class _SealedSuite:
    suite: _Suite
    receipt: Any
    records: tuple[_Record, ...]
    digest: str
    expected: tuple


_sealed: dict[int, _SealedSuite] = {}


def _record_digest(records: tuple[_Record, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            [(r.ordinal, r.request_json, r.response_json) for r in records],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _forget_seal(owner: Any) -> None:
    _sealed.pop(id(owner), None)


def _live_seal(owner: Any, pipeline: Any) -> _SealedSuite | None:
    seal = _sealed.get(id(owner))
    if seal is None:
        return None
    suite = seal.suite
    try:
        if (
            suite.owner is owner
            and suite.pipeline is pipeline
            and suite.finished
            and not suite.failed
            and suite.fresh()
            and owner.receipt is seal.receipt
            and owner.case_records is seal.records
            and _record_digest(seal.records) == seal.digest
            and tuple(sorted(suite.expected.items())) == seal.expected
        ):
            return seal
    except (AttributeError, TypeError, ValueError, OSError):
        pass
    _forget_seal(owner)
    return None


@contextmanager
def suite_scope(
    capture: Any,
    pipeline: Any,
    rules_file: Path,
    artifact: Path,
    compiled: dict,
    cases: list,
) -> Iterator[_Suite | None]:
    from .validator_pipeline import _safe_load_unique_keys

    suite = None
    owner = compile_custody._owner(capture)
    if owner is not None:
        owner.case_records = None
        raw = (
            compile_custody.consume_compilation(capture, artifact)
            if owner.receipt is not None and owner.receipt.pipeline is pipeline
            else None
        )
        try:
            if raw is not None:
                parsed = json.loads(
                    raw, object_pairs_hook=compile_custody._unique_object
                )
                companion = pipeline._rulespec_test_path(rules_file)
                relative = companion.relative_to(capture.snapshot_root).as_posix()
                matches = [f for f in owner.binding[3] if f.relative == relative]
                if (
                    len(matches) != 1
                    or rules_file != owner.binding[1] / owner.binding[2]
                ):
                    raise ValueError("unbound source/companion")
                saved_cases = _safe_load_unique_keys(matches[0].raw.decode())
                if isinstance(saved_cases, dict):
                    saved_cases = saved_cases.get("cases")
                if _typed(saved_cases) != _typed(cases) or _typed(parsed) != _typed(
                    compiled
                ):
                    raise ValueError("different companion or program")
                expected = {}
                for ordinal, case in enumerate(saved_cases, 1):
                    request = _request(pipeline, rules_file, parsed, case, ordinal)
                    if request is not None:
                        expected[ordinal] = (_typed(case), _typed(request))
                environment = pipeline._rulespec_engine_env()
                environment_digest = compile_custody._environment_digest(environment)
                if environment_digest is None:
                    raise ValueError("unsupported environment")
                suite = _Suite(
                    owner,
                    owner.generation,
                    pipeline,
                    artifact,
                    raw,
                    expected,
                    str(pipeline.axiom_rules_path)
                    if pipeline.axiom_rules_path.exists()
                    else None,
                    environment_digest,
                    cases_input=cases,
                    compiled_input=compiled,
                    cases_identity=_typed(saved_cases),
                    compiled_identity=_typed(parsed),
                )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            yaml.YAMLError,
        ):
            suite = None
    token = _current.set(suite)
    try:
        yield suite
    except BaseException:
        if suite is not None:
            suite.invalidate()
        raise
    finally:
        if suite is not None and (
            not suite.finished or suite.failed or not suite.fresh()
        ):
            suite.owner.case_records = None
        _current.reset(token)


def run_observed_case(
    command: list[str], *, pipeline: Any, case: dict, ordinal: int, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    suite = _current.get()
    eligible = False
    request = None
    if suite is not None:
        try:
            request = json.loads(
                kwargs.get("input", ""),
                object_pairs_hook=compile_custody._unique_object,
            )
            eligible = (
                suite.pipeline is pipeline
                and not suite.finished
                and not suite.failed
                and suite.fresh()
                and type(ordinal) is int
                and ordinal in suite.expected
                and ordinal not in suite.records
                and suite.expected[ordinal] == (_typed(case), _typed(request))
                and command
                == [
                    str(suite.owner.binding[4]),
                    "run-compiled",
                    "--artifact",
                    str(suite.artifact),
                ]
                and set(kwargs)
                == {"input", "capture_output", "text", "timeout", "cwd", "env"}
                and kwargs.get("capture_output") is True
                and kwargs.get("text") is True
                and type(kwargs.get("timeout")) is int
                and kwargs.get("timeout") == 60
                and kwargs.get("cwd") == suite.cwd
                and compile_custody._environment_digest(kwargs.get("env"))
                == suite.env_digest
            )
        except (ValueError, TypeError):
            eligible = False
        if not eligible:
            suite.invalidate()
    try:
        result = subprocess.run(command, **kwargs)
    except BaseException:
        if suite is not None:
            suite.invalidate()
        raise
    if suite is not None and eligible:
        try:
            response = json.loads(
                result.stdout, object_pairs_hook=compile_custody._unique_object
            )
            rows = response.get("results")
            if (
                result.returncode != 0
                or not suite.fresh()
                or not strict_explain_metadata(request, response)
                or not isinstance(rows, list)
                or len(rows) != 1
            ):
                raise ValueError("unsuccessful strict single-root response")
            if any(
                rows[0].get(k) != request["queries"][0].get(k)
                for k in ("entity_id", "period", "assessment_date")
            ):
                raise ValueError("response identity mismatch")
            suite.records[ordinal] = _Record(ordinal, kwargs["input"], result.stdout)
        except (ValueError, TypeError, AttributeError, KeyError):
            suite.invalidate()
    return result
