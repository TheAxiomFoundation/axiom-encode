"""Offline command surface for production admission scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def register_admission_score_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "admission-score",
        help="Score a candidate with signed-apply validation without installing it",
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--citation", required=True)
    parser.add_argument("--tests", type=Path, help="Companion test file")
    parser.add_argument(
        "--policy-repo",
        "--policy-repo-path",
        dest="policy_repo_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument(
        "--corpus-release-public-key",
        required=True,
        help="Canonical base64 public key for offline corpus verification",
    )
    parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--axiom-compose-path", type=Path)
    parser.add_argument(
        "--rulespec-dependency-root",
        dest="rulespec_dependency_roots",
        type=Path,
        action="append",
        default=[],
        help="Additional packaged RuleSpec dependency checkout (repeatable)",
    )
    parser.add_argument("--expected-source-body-sha256")
    parser.add_argument("--expected-policy-repo-commit")
    parser.add_argument(
        "--expected-dependency-content-sha256",
        action="append",
        help="Frozen dependency content digest, in dependency-root order (repeatable)",
    )
    parser.add_argument("--expected-corpus-release-content-sha256")


def run_admission_score(args: argparse.Namespace) -> int:
    if not os.path.lexists(args.candidate):
        print(
            f"admission-score: error: candidate path does not exist: {args.candidate}",
            file=sys.stderr,
        )
        return 2

    from axiom_encode.corpus_resolver import resolve_local_corpus_source
    from axiom_encode.harness.admission import (
        admission_identity,
        prerequisite_admission_result,
        score_admission,
    )
    from axiom_encode.toolchain import (
        load_rulespec_local_corpus_release,
        local_corpus_release_verification,
    )

    try:
        with local_corpus_release_verification(args.corpus_release_public_key):
            release = load_rulespec_local_corpus_release(
                args.policy_repo_path,
                args.corpus_path,
            )
        source = resolve_local_corpus_source(args.citation, release)
        with tempfile.TemporaryDirectory(
            prefix="axiom-admission-context-"
        ) as directory:
            generation_input = source.body.replace("\r\n", "\n").replace("\r", "\n")
            generation_input_bytes = generation_input.encode("utf-8")
            generation_input_sha256 = hashlib.sha256(generation_input_bytes).hexdigest()
            source_text_file = Path(directory) / "source.txt"
            source_text_file.write_bytes(generation_input_bytes)
            source_attestation = {
                **source.to_attestation(),
                "generation_input_sha256": generation_input_sha256,
            }
            context_manifest = Path(directory) / "context-manifest.json"
            context_manifest.write_text(
                json.dumps(
                    {
                        "source_text_file": source_text_file.name,
                        "source_text_sha256": generation_input_sha256,
                        "source_metadata": {"source_attestation": source_attestation},
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            candidate = argparse.Namespace(
                citation=args.citation,
                runner="admission-score",
                backend="",
                output_file=str(args.candidate),
                context_manifest_file=str(context_manifest),
                context_manifest_sha256=hashlib.sha256(
                    context_manifest.read_bytes()
                ).hexdigest(),
                source_attestation=source_attestation,
            )
            expected_context = {
                key: value
                for key, value in (
                    (
                        "source_body_sha256",
                        getattr(args, "expected_source_body_sha256", None),
                    ),
                    (
                        "policy_repo_commit",
                        getattr(args, "expected_policy_repo_commit", None),
                    ),
                    (
                        "dependency_content_sha256",
                        getattr(args, "expected_dependency_content_sha256", None),
                    ),
                    (
                        "corpus_release_content_sha256",
                        getattr(args, "expected_corpus_release_content_sha256", None),
                    ),
                )
                if value is not None
            }
            result = score_admission(
                candidate,
                citation=args.citation,
                companion_test_path=args.tests,
                policy_repo_path=args.policy_repo_path,
                axiom_rules_path=args.axiom_rules_path,
                axiom_compose_path=args.axiom_compose_path,
                local_corpus_release=release,
                rulespec_dependency_roots=tuple(args.rulespec_dependency_roots),
                expected_context=expected_context or None,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        result = prerequisite_admission_result(
            [str(exc)],
            identity=admission_identity(
                policy_repo_path=args.policy_repo_path,
                axiom_rules_path=args.axiom_rules_path,
                axiom_compose_path=args.axiom_compose_path,
                rulespec_dependency_roots=tuple(args.rulespec_dependency_roots),
            ),
        )
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["prerequisite_failure"]:
        return 2
    return 0 if result.admitted else 1
