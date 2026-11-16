"""Prepare selectors, never cached approval authority, on the main workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def load(name):
    path = Path(__file__).resolve().parents[1] / "src" / "axiom_encode" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("_reviewed_pr_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify-publication"])
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--attestation", type=Path)
    args = parser.parse_args()
    admission = load("reviewed_pr_admission")
    if args.command == "verify-publication":
        payload = json.loads(args.attestation.read_text())
        admission.revalidate_admission(payload["reviewed_pr_admission"])
        return
    helper = load("prepare_signed_backfill")
    dispatch = json.loads(os.environ["DISPATCH_JSON"])
    raw = dispatch.get("source_bundle_json", "[]")
    outer = json.loads(raw)
    enabled = (
        isinstance(outer, dict)
        and outer.get("schema") == "axiom-encode/reviewed-pr-promotion/v1"
    )
    if enabled:
        transaction = helper.split_atomic_source_input(raw)
        allowed = {
            "citation",
            "country",
            "rulespec_ref",
            "pr_base_branch",
            "corpus_ref",
            "rules_engine_ref",
            "source_bundle_json",
            "replace_rulespec_path",
            "open_pr",
        }
        if any(
            value not in ("", "[]", False, None)
            for key, value in dispatch.items()
            if key not in allowed
        ):
            raise ValueError(
                "live PR promotion cannot mix generation, repair, imports or queue inputs"
            )
        if dispatch.get("open_pr") not in (True, "true"):
            raise ValueError("live PR promotion requires protected PR publication")
        if (
            os.environ.get("GITHUB_REPOSITORY") != "TheAxiomFoundation/axiom-encode"
            or os.environ.get("GITHUB_REF") != "refs/heads/main"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        ):
            raise ValueError(
                "live PR promotion must run on the central main dispatch workflow"
            )
        selector = admission.PromotionSelector(
            country=dispatch["country"],
            pr=transaction["rulespec_pr"],
            head=dispatch["rulespec_ref"],
            citation=dispatch["citation"],
            rulespec_path=dispatch["replace_rulespec_path"],
            branch=dispatch["pr_base_branch"],
            actor=os.environ["DISPATCH_ACTOR"],
            run_id=os.environ["GITHUB_RUN_ID"],
        )
        if (
            str(helper.citation_rulespec_path(selector.citation))
            != selector.rulespec_path
        ):
            raise ValueError("live PR promotion requires its canonical source path")
        if (
            subprocess.check_output(
                ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
            ).strip()
            != selector.head
        ):
            raise ValueError(
                "live PR promotion checkout differs from exact reviewed head"
            )
        admission.verify_reviewed_pr(selector)
        with args.selector.open("x") as stream:
            json.dump(selector.__dict__, stream, sort_keys=True)
            stream.write("\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"enabled={'true' if enabled else 'false'}\n")


if __name__ == "__main__":
    main()
