"""Replay the complete-source analyzer over saved A/B candidates.

Usage (run once per checkout, with that checkout's interpreter):

    <checkout>/.venv/bin/python -I replay_completeness.py <checkout> <run dir> <out.json>
    python3 -I replay_completeness.py --compare a.json b.json

For every ``final-rejected-candidate`` in ``<run dir>/runs/*/r*/*``, the script
runs ``analyze_complete_source_unit`` with that run's workspace ``source.txt``
and companion tests, and writes the issues per candidate. Comparing two
outputs shows whether a change moved validator behaviour. A generator-only
change must compare identical. This check is necessary, not sufficient: unlike
the production call in validator_pipeline.py, the replay passes no proof source
texts (``source_context``), imported symbol contents, resolved dependency
outputs or authenticated same-Act aliases, and it reads the workspace
``source.txt`` and ``source-metadata.json`` instead of the resolver. A
validator change that acts only through those inputs still compares
identical. Absolute counts can exceed production; compare runs, not
absolutes.
"""

import functools
import glob
import json
import sys


def replay(checkout: str, run_dir: str, out_path: str) -> None:
    sys.path.insert(0, f"{checkout}/src")
    import yaml

    from axiom_encode.harness.source_completeness import (
        analyze_complete_source_unit,
        collect_artifact_numeric_bindings,
    )
    from axiom_encode.harness.validator_pipeline import (
        _numeric_profile_for_citation_path,
        extract_named_scalar_occurrences,
        extract_typed_numeric_inventory_occurrences_from_text,
        extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded,
    )

    out = {}
    pattern = f"{run_dir}/runs/*/r*/*/final-rejected-candidate/**/*.yaml"
    for cand in sorted(glob.glob(pattern, recursive=True)):
        if cand.endswith(".test.yaml"):
            continue
        run = cand.split("/final-rejected-candidate/")[0]
        sources = glob.glob(
            f"{run}/generated/_eval_workspaces/*/*/workspace/source.txt"
        )
        if not sources:
            continue
        source_text = open(sources[0]).read()
        try:
            meta = json.load(
                open(sources[0].replace("source.txt", "source-metadata.json"))
            )
        except (OSError, ValueError) as exc:
            out[cand.split("/runs/", 1)[1]] = [f"EXC metadata: {exc}"[:300]]
            continue
        citation = (meta.get("source_attestation") or {}).get(
            "requested_corpus_citation_path"
        ) or meta.get("citation_path")
        content = open(cand).read()
        tests = None
        test_path = cand[:-5] + ".test.yaml"
        if glob.glob(test_path):
            try:
                tests = yaml.safe_load(open(test_path))
                if isinstance(tests, dict):
                    tests = tests.get("cases")
            except yaml.YAMLError:
                tests = None
        try:
            profile = _numeric_profile_for_citation_path(citation)
            bindings = collect_artifact_numeric_bindings(
                content, extract_named_scalars=extract_named_scalar_occurrences
            )
            analysis = analyze_complete_source_unit(
                content,
                source_text,
                corpus_citation_path=citation,
                test_cases=tests,
                extract_numeric_occurrences=functools.partial(
                    extract_typed_numeric_inventory_occurrences_from_text,
                    profile=profile,
                ),
                extract_numeric_grounding_occurrences=functools.partial(
                    extract_typed_numeric_occurrences_from_text, profile=profile
                ),
                extract_named_scalars=extract_named_scalar_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
                artifact_numeric_bindings=bindings,
                artifact_numeric_values=tuple(value for _name, value in bindings),
            )
            issues = list(analysis.issues)
        except Exception as exc:  # record, do not stop the replay
            issues = [f"EXC {type(exc).__name__}: {exc}"[:300]]
        out[cand.split("/runs/", 1)[1]] = issues
    json.dump(out, open(out_path, "w"), indent=0, sort_keys=True)
    print(f"{len(out)} candidates, {sum(len(v) for v in out.values())} issues")


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print(f"{len(set(a) | set(b))} candidates, {len(changed)} changed")
    for key in changed[:50]:
        print(f"- {key}")
    return 1 if changed else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    replay(*sys.argv[1:])
