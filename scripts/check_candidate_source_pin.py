#!/usr/bin/env python3
"""Refuse a reviewed-candidate promotion whose primary has no source pin.

A promotion signs the candidate bytes as they are. The protected job refuses a
primary whose ``module.source_verification`` does not carry the resolver's
``source_sha256``, but only after the production-signing approval (signing run
45328443323). This guard runs in the unprotected ``attempt_budget`` job, before
that approval and before any protected step. For a dispatch that asks for a
reviewed-candidate promotion it reads the candidate primary at the exact
``rulespec_ref`` and refuses when the pin is missing or malformed. Every other
dispatch passes untouched.

It is an early refusal, not the binding check. The read is the strict line read
of ``axiom_encode.reviewed_pr_admission`` (plain block form only; any other
shape refuses), and it knows neither the corpus nor the right digest. The
protected job still parses the module and compares the pin with the resolver's
attestation before it signs. Nothing here can admit a candidate.

It fails closed: when the candidate cannot be read (bad inputs, GitHub API
error, rate limit), the dispatch is refused and can be repeated; the budget
guard treats a run refused here as neutral, because the encode job never ran.

Standard library only: the job has no virtual environment. The read-scoped
workflow token, when present, is used only for the one GitHub read; the
repository is public, so the read is repeated without it if the token is not
accepted.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

PROMOTION_SCHEMA = "axiom-encode/reviewed-pr-promotion/v1"
MAX_MODULE_BYTES = 10 * 1024 * 1024
_SELECTOR_TEXT = re.compile(r"[A-Za-z0-9._/-]{1,512}")


class CandidateReadError(Exception):
    """The candidate primary could not be read at the exact ref."""


def _admission_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "axiom_encode"
        / "reviewed_pr_admission.py"
    )
    spec = importlib.util.spec_from_file_location("_source_pin_admission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load axiom_encode.reviewed_pr_admission")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def is_promotion_request(source_bundle_json: str) -> bool:
    """Whether the dispatch asks for a reviewed-candidate promotion.

    The same routing test as ``scripts/prepare_reviewed_pr_promotion.py``: the
    source input is a JSON object with the promotion wrapper schema. The
    protected job validates every other field of the wrapper.
    """
    try:
        outer = json.loads(source_bundle_json)
    except (ValueError, RecursionError):
        return False
    return isinstance(outer, dict) and outer.get("schema") == PROMOTION_SCHEMA


def _exact_path(value: str) -> bool:
    return _SELECTOR_TEXT.fullmatch(value) is not None and not any(
        part in {"", ".", ".."} for part in value.split("/")
    )


def candidate_selector(environment: Mapping[str, str]) -> tuple[str, str, str, str]:
    """``(repository, ref, primary path, citation)`` of the dispatch, or refuse."""
    country = environment.get("COUNTRY", "")
    ref = environment.get("RULESPEC_REF", "")
    path = environment.get("REPLACE_RULESPEC_PATH", "")
    citation = environment.get("CITATION", "")
    if (
        re.fullmatch(r"[a-z]{2}", country) is None
        or re.fullmatch(r"[0-9a-f]{40}", ref) is None
        or not _exact_path(path)
        or not _exact_path(citation)
        or not path.endswith(".yaml")
        or path.endswith(".test.yaml")
    ):
        raise CandidateReadError(
            "the dispatch inputs do not name one exact candidate primary "
            "(country, 40-character rulespec_ref, replace_rulespec_path, citation)"
        )
    return f"TheAxiomFoundation/rulespec-{country}", ref, path, citation


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CandidateReadError("the GitHub read must not redirect")


def fetch_candidate_primary(repository: str, ref: str, path: str, token: str) -> bytes:
    """The raw bytes of ``path`` at commit ``ref`` from the fixed GitHub API."""
    url = (
        f"https://api.github.com/repos/{repository}/contents/"
        f"{urllib.parse.quote(path, safe='/')}?ref={ref}"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    last = "no attempt"
    # The repository is public: a token that is not accepted is dropped.
    for authorization in dict.fromkeys((token, "")):
        headers = {
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "axiom-encode-candidate-source-pin",
        }
        if authorization:
            headers["Authorization"] = f"Bearer {authorization}"
        try:
            with opener.open(
                urllib.request.Request(url, headers=headers), timeout=30
            ) as response:
                if response.status != 200:
                    raise CandidateReadError(f"HTTP {response.status}")
                raw = response.read(MAX_MODULE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = type(exc).__name__
            continue
        if len(raw) > MAX_MODULE_BYTES:
            raise CandidateReadError("the candidate primary exceeds its size bound")
        return raw
    raise CandidateReadError(f"GitHub did not return the file ({last})")


TITLE_NO_PIN = "Candidate has no source pin"
TITLE_NOT_READ = "Could not read the candidate"


def evaluate(
    environment: Mapping[str, str],
    *,
    fetch: Callable[[str, str, str, str], bytes] = fetch_candidate_primary,
    admission=None,
) -> tuple[int, str]:
    """``(exit status, message)`` for one dispatch. Status 0 lets the run go on."""
    if not is_promotion_request(environment.get("SOURCE_BUNDLE_JSON", "")):
        return 0, "source-pin: not a reviewed-candidate promotion; nothing to check."
    admission = admission or _admission_module()
    try:
        repository, ref, path, citation = candidate_selector(environment)
        raw = fetch(repository, ref, path, environment.get("GH_TOKEN", ""))
    except CandidateReadError as exc:
        return 1, (
            "source-pin: REFUSED before the signing approval: could not read the "
            f"candidate primary at the exact rulespec_ref: {exc}. Nothing was "
            "approved or signed. Check rulespec_ref and replace_rulespec_path, "
            "then dispatch again."
        )
    try:
        admission.require_candidate_source_pin(raw, citation)
    except admission.AdmissionError as exc:
        return 1, (
            f"source-pin: REFUSED before the signing approval: {exc}. Nothing "
            "was approved or signed."
        )
    return 0, (
        f"source-pin: {path} at {ref} names the promotion citation and carries a "
        "source_sha256. The protected job still binds it to the corpus."
    )


def main() -> int:
    status, message = evaluate(os.environ)
    if status:
        print(message, file=sys.stderr)
        title = TITLE_NOT_READ if "could not read" in message else TITLE_NO_PIN
        print(f"::error title={title}::{message}")
    else:
        print(message)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
