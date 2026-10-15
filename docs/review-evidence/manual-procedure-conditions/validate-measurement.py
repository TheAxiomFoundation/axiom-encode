"""Validate the corpus-only Boolean reordering against pinned exact extraction."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import importlib.util
import json
import re
import signal
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_validate_latest_main_measurement", HERE / "reproduce-latest-main.py"
)
latest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latest)
shared = latest.shared


def timed_out(signum, frame):
    raise TimeoutError("Measurement validation exceeded its bounded time budget")


def isolated_helper(node: ast.FunctionDef, joined: bool):
    namespace = {"re": re, "_source_has_joined_operative_segment": lambda text: joined}
    tree = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    )
    exec(compile(tree, "<isolated-guidance>", "exec"), namespace)
    return namespace["_source_is_nonoperative_guidance_instruction"]


def authenticated_row(corpus_root: Path, inventory: dict, name: str, index: int):
    content = (corpus_root / name).read_bytes()
    entry = inventory["files"][name]
    assert len(content) == entry["bytes"] and shared.digest(content) == entry["sha256"]
    return json.loads(content.decode().splitlines()[index])


def predicate(module, text: str, citation: str, *, current: bool):
    scope = getattr(module, "_us_manual_prose_scope", None)
    with scope(citation) if scope else contextlib.nullcontext():
        options = {"branch_path": ()}
        if not current:
            options.update(corpus_citation_path=citation, source_text=text)
        return module._source_exception_requires_paired_witness(text, **options)


def differential(before: dict, after: dict) -> set:
    return {("removed", identity) for identity in before.keys() - after.keys()} | {
        ("added", identity) for identity in after.keys() - before.keys()
    }


def validate_clause_cache(module, citation: str) -> dict:
    text = "If income exceeds $100, the applicant is ineligible."
    branches = list(
        module.recognize_source_structure(text, corpus_citation_path=citation)
    )
    original = module._source_clause_spans
    calls = []

    def counted(source_text, *, branches):
        calls.append((source_text, branches))
        return original(source_text, branches=branches)

    scope = getattr(module, "_us_manual_prose_scope", None)
    try:
        module._source_clause_spans = counted
        with scope(citation) if scope else contextlib.nullcontext():
            with latest.clause_span_cache(module, enabled=True):
                first = module._source_clause_spans(text, branches=branches)
                second = module._source_clause_spans(text, branches=branches)
                assert first is not second
                expected = tuple(first)
                assert tuple(second) == expected and len(calls) == 1
                copied_text = (text + "\0")[:-1]
                assert copied_text == text and copied_text is not text
                assert (
                    tuple(module._source_clause_spans(copied_text, branches=branches))
                    == expected
                )
                assert len(calls) == 2
                assert (
                    tuple(module._source_clause_spans(text, branches=list(branches)))
                    == expected
                )
                assert len(calls) == 3
            assert module._source_clause_spans is counted
            with latest.clause_span_cache(module, enabled=True):
                assert (
                    tuple(module._source_clause_spans(text, branches=branches))
                    == expected
                )
            assert len(calls) == 4 and module._source_clause_spans is counted
            try:
                with latest.clause_span_cache(module, enabled=True):
                    raise RuntimeError("cache restoration control")
            except RuntimeError:
                pass
            assert module._source_clause_spans is counted
    finally:
        module._source_clause_spans = original
    return {
        "exact_argument_reuse": True,
        "fresh_iterators": True,
        "equal_text_with_different_identity_recomputed": True,
        "equal_branches_with_different_identity_recomputed": True,
        "separate_call_recomputed": True,
        "error_restores_original": True,
        "original_materializations": len(calls),
    }


def validate_clause_boundaries(
    originals: dict, accelerated: dict, citation: str
) -> dict:
    comparisons = {"clause_sequences": 0, "state_code_booleans": 0}

    def clauses(module, text):
        scope = getattr(module, "_us_manual_prose_scope", None)
        with scope(citation) if scope else contextlib.nullcontext():
            return tuple(module._source_clause_spans(text, branches=()))

    def check(text):
        for key in originals:
            assert clauses(originals[key], text) == clauses(accelerated[key], text)
            comparisons["clause_sequences"] += 1
            for boundary in re.finditer(r"[.;!?]", text):
                before = originals[
                    key
                ]._source_clause_boundary_splits_state_code_citation(text, boundary)
                after = accelerated[
                    key
                ]._source_clause_boundary_splits_state_code_citation(text, boundary)
                assert before == after
                comparisons["state_code_booleans"] += 1

    state_code_controls = (
        ("R.S.\n47:2", True),
        ("R.S.47:2", True),
        ("R.S. 47A:2b", True),
        ("R.S.\n\u0664\u0667:\u0662", True),
        ("xR.S. 47:2", False),
        ("_R.S. 47:2", False),
        ("\u03b1R.S. 47:2", False),
        ("R.\nS. 47:2", False),
        ("R.S. 47 :2", False),
        ("R.S. 47: 2", False),
        ("R.S.\n-47:2", False),
    )
    explicit = [
        "",
        "1. Lebensjahren",
        "1. Lebensjahrx",
        "1.\n2. Lebensjahr",
        "Before. 1. Lebensjahr. After.",
        *(
            prefix + digits + suffix
            for prefix in ("", "a", "_", "\u03b1")
            for digits in ("1", "\u0661", "\U0001d7d9")
            for suffix in (". Lebensjahr", ".\nLebensjahres", ".\u00a0Lebensjahre")
        ),
        *(text for text, expected in state_code_controls),
        "9" * 8192 + ". Lebensjahr",
        "\u0669" * 8192 + ".\nLebensjahr",
        "\U0001d7d9" * 8192 + ".\u00a0Lebensjahres",
        "\u03b1" * 8192 + "R.S.\n47:2",
        "9" * 8192 + "R.S.\n47:2",
    ]
    for text in explicit:
        check(text)
    generated_examples = 0

    @settings(max_examples=512, deadline=None, derandomize=True, database=None)
    @given(
        st.one_of(
            st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
            st.tuples(
                st.sampled_from(("", " ", ". ", "a", "_", "\u03b1", "\u0661")),
                st.text(alphabet=st.characters(categories=("Nd",))),
                st.sampled_from(
                    (
                        " Lebensjahr",
                        "\nLebensjahres",
                        "\u00a0Lebensjahre",
                        " Lebensjahrx",
                        " unrelated",
                    )
                ),
            ).map(lambda parts: parts[0] + parts[1] + "." + parts[2]),
            st.tuples(
                st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
                st.sampled_from(
                    (
                        "\n47:2",
                        " 47A:2b",
                        "\n\u0664\u0667:\u0662",
                        " 47 :2",
                        " 47: 2",
                        " text",
                    )
                ),
            ).map(lambda parts: parts[0] + "R.S." + parts[1]),
        )
    )
    def generated(text):
        nonlocal generated_examples
        generated_examples += 1
        check(text)

    generated()
    for text, expected in state_code_controls:
        boundary = list(re.finditer(r"\.", text))[-1]
        for key in originals:
            assert (
                originals[key]._source_clause_boundary_splits_state_code_citation(
                    text, boundary
                )
                is expected
            )
            assert (
                accelerated[key]._source_clause_boundary_splits_state_code_citation(
                    text, boundary
                )
                is expected
            )
    return {
        "explicit_source_examples": len(explicit),
        "generated_source_examples": generated_examples,
        "generated_text_and_digit_strategy_has_no_length_cap": True,
        "longest_explicit_digit_run": 8192,
        "expected_state_code_controls": len(state_code_controls) * len(originals),
        **comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, 525)
    originals, original_loaders = latest.load_revisions(accelerate=False)
    accelerated, accelerated_loaders = latest.load_revisions(accelerate=True)
    source = original_loaders["latest_main"].read(latest.SOURCE_MODULE)
    execution_source, proof = shared.accelerated_guidance_source(source)
    original_helper = shared.bindings(ast.parse(source))[
        "_source_is_nonoperative_guidance_instruction"
    ]
    execution_helper = shared.bindings(ast.parse(execution_source))[
        "_source_is_nonoperative_guidance_instruction"
    ]
    original_stubs = {
        joined: isolated_helper(original_helper, joined) for joined in (False, True)
    }
    execution_stubs = {
        joined: isolated_helper(execution_helper, joined) for joined in (False, True)
    }
    explicit = [
        "State agencies must verify income.",
        "STATE AGENCIES MUST REVIEW RECORDS.",
        "State\u00a0agencies\u2003must\u202ffollow procedures.",
        "ſtate agencies must verify documents.",
        "State agencies must provide notice.",
        "State agencies must provide documentation.",
        "State agencies may verify income.",
        "Applicants must verify income.",
        " State agencies must verify income.",
        "State agencies must determine eligibility.",
        "State agencies must provide payments.",
        "If the applicant is disabled, the applicant is eligible.",
        "When income exceeds the limit, the applicant is ineligible.",
    ]
    truth_checks = 0
    for text in explicit:
        for joined in (False, True):
            assert original_stubs[joined](text) == execution_stubs[joined](text)
            truth_checks += 1
    seen = 0

    @settings(max_examples=2000, deadline=None, derandomize=True, database=None)
    @given(
        st.one_of(
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200),
            st.sampled_from(explicit).flatmap(
                lambda prefix: st.text(
                    alphabet=st.characters(blacklist_categories=("Cs",)), max_size=100
                ).map(lambda suffix: prefix + " " + suffix)
            ),
        )
    )
    def generated(text):
        nonlocal seen
        seen += 1
        for joined in (False, True):
            assert original_stubs[joined](text) == execution_stubs[joined](text)

    generated()
    boundary_controls = validate_clause_boundaries(
        originals, accelerated, accelerated["final"]._MANUAL_PROCEDURE_CITATION
    )
    cache_controls = {
        key: validate_clause_cache(
            module, accelerated["final"]._MANUAL_PROCEDURE_CITATION
        )
        for key, module in accelerated.items()
    }
    fixtures = explicit + [
        "State agencies must verify records, if the applicant is disabled, the applicant is eligible.",
        "State agencies must follow procedures, and if income exceeds the threshold, the applicant is ineligible.",
        "If income is greater than or equal to the income threshold, the applicant is ineligible.",
        "Unless the applicant is disabled, the applicant is eligible.",
        "The applicant is eligible only when income is below the limit.",
        "If income exceeds $500, the applicant is ineligible.",
        "Unless age is below 18, the applicant is eligible.",
        "State agencies must verify income when income exceeds $500.",
        "If income exceeds $500 and the applicant is disabled, the applicant is ineligible.",
        "If monthly income / family size exceeds $500, the applicant is ineligible.",
        "Guidance documents lack the force and effect of law, unless expressly authorized by statute or incorporated into a contract.",
    ]
    citation = accelerated["final"]._MANUAL_PROCEDURE_CITATION
    controls = [(text, citation) for text in fixtures]
    controls += [
        (text, citation)
        for text in sorted(accelerated["final"]._MANUAL_PROCEDURE_CLAUSES)
    ]
    inventory = json.loads((HERE / "corpus-source-binding.json").read_bytes())
    florida_name = (
        "us-fl/manual/2026-05-27-fl-ess-manual-r2026-07-15-self-contained.jsonl"
    )
    florida_row = authenticated_row(options.corpus_root, inventory, florida_name, 253)
    florida_text = florida_row.get("body") or florida_row.get("text")
    expected_florida_final = {
        (path, start, end, shared.digest(florida_text[start:end].encode()))
        for path, start, end in (
            ((), 488, 682),
            (("1",), 1262, 1564),
            (("1",), 1676, 1902),
            (("1",), 1903, 2036),
            (("1",), 2427, 2529),
        )
    }
    expected_florida_latest = {
        identity for identity in expected_florida_final if identity[1] in {1262, 1676}
    }
    controls.append((florida_text, florida_row["citation_path"]))
    # Nonmanual controls execute the same source grammar with manual scope off.
    controls += [(text, "us/statute/7/2014") for text in fixtures[-6:]]
    expected_count_controls = {}
    for text, manual_counts in (
        ("If income exceeds $100, the applicant is ineligible.", (1, 1)),
        ("If the applicant is disabled, the applicant is ineligible.", (0, 1)),
        (
            "If both income and assets are greater than or equal to the income threshold, the applicant is ineligible.",
            (0, 1),
        ),
        ("State agencies must verify income.", (0, 0)),
        ("Add the income to $100.", (0, 0)),
        ("An unrelated sentence.", (0, 0)),
    ):
        for control_citation, expected in (
            (citation, manual_counts),
            ("de/statute/estg/32a", (manual_counts[1], manual_counts[1])),
        ):
            controls.append((text, control_citation))
            expected_count_controls[(text, control_citation)] = expected
    predicate_checks = 0
    extraction_checks = 0
    control_receipts = []
    for text, citation in controls:
        original_obligations = {}
        accelerated_obligations = {}
        for key in originals:
            current = key == "latest_main"
            assert predicate(
                originals[key], text, citation, current=current
            ) == predicate(accelerated[key], text, citation, current=current)
            predicate_checks += 1
            before = latest.obligations(
                originals[key], text, citation, latest=current, clause_cache=False
            )
            after = latest.obligations(accelerated[key], text, citation, latest=current)
            assert before.keys() == after.keys(), (
                key,
                shared.digest(text.encode()),
                citation,
            )
            original_obligations[key] = before
            accelerated_obligations[key] = after
            extraction_checks += 1
        expected = differential(
            original_obligations["latest_main"], original_obligations["final"]
        )
        observed = differential(
            accelerated_obligations["latest_main"], accelerated_obligations["final"]
        )
        assert expected == observed
        if (text, citation) in expected_count_controls:
            assert (
                len(original_obligations["latest_main"]),
                len(original_obligations["final"]),
            ) == expected_count_controls[(text, citation)]
        if text == florida_text and citation == florida_row["citation_path"]:
            assert original_obligations["final"].keys() == expected_florida_final
            assert original_obligations["latest_main"].keys() == expected_florida_latest
            assert sum(action == "removed" for action, _ in expected) == 0
            assert sum(action == "added" for action, _ in expected) == 3
        control_receipts.append(
            {
                "source_sha256": shared.digest(text.encode()),
                "citation": citation,
                "latest_main_obligations": len(original_obligations["latest_main"]),
                "final_obligations": len(original_obligations["final"]),
                "removed": sum(action == "removed" for action, _ in expected),
                "added": sum(action == "added" for action, _ in expected),
            }
        )
    for clause in accelerated["final"]._MANUAL_PROCEDURE_CLAUSES:
        assert (
            latest.obligations(
                accelerated["final"],
                clause,
                accelerated["final"]._MANUAL_PROCEDURE_CITATION,
                latest=False,
            )
            == {}
        )
    report = {
        "latest_main_revision": latest.LATEST_MAIN,
        "audit_sha256": {
            name: shared.digest((HERE / name).read_bytes())
            for name in (
                "reproduce-corpus.py",
                "reproduce-latest-main.py",
                "validate-measurement.py",
                "clause-boundary-measurement.py",
            )
        },
        "inventory_sha256": shared.digest(
            (HERE / "corpus-source-binding.json").read_bytes()
        ),
        "original_execution": {
            key: loader.receipt() for key, loader in original_loaders.items()
        },
        "accelerated_execution": {
            key: loader.receipt() for key, loader in accelerated_loaders.items()
        },
        "clause_cache": latest.clause_cache_receipt(accelerated_loaders),
        "clause_cache_controls": cache_controls,
        "clause_boundary_controls": boundary_controls,
        "explicit_truth_checks": truth_checks,
        "generated_examples": seen,
        "generated_truth_checks": seen * 2,
        "full_predicate_controls": predicate_checks,
        "full_extraction_controls": extraction_checks,
        "complete_extraction_differential_controls": len(controls),
        "expected_count_controls": len(expected_count_controls),
        "exact_procedure_exemptions": 2,
        "controls": control_receipts,
        "proof": proof,
        "scope": "Boolean classification results for valid strings, with exact original matcher and joined call; exact complete clause/citation guards permute unchanged pure operands, with direct ordered-clause and Boolean controls including Unicode digit and arbitrary-prefix strategies without a length cap; source-only pinned uncached original and accelerated memoized extraction use each revision's own manual scope, numeric filter, predicates and complete formula suppression; clause triples cache only identical text and branch objects within one fixed-scope call and restore the original generator afterwards; exception, call-count and resource-timeout parity are not asserted",
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "explicit_truth_checks",
                    "generated_examples",
                    "generated_truth_checks",
                    "full_predicate_controls",
                    "full_extraction_controls",
                    "complete_extraction_differential_controls",
                    "clause_boundary_controls",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
