#!/usr/bin/env python3
"""Refuse a targeted re-encode whose encoder the RuleSpec pin cannot admit.

A signed RuleSpec pull request only lands when the target repository's CI,
running the encoder pinned in ``.axiom/workflow-toolchain.toml``
(``axiom_encode_ref``), admits the producer encoder that signed the manifest.
The pinned guard admits the exact pin. From axiom-encode #1859 on, it also
admits a producer that descends from the pin and lies on the first-parent
history of encoder ``main`` (docs/signed-manifest-encoder-admission.md).

Before this check, a run from an inadmissible encoder spent model money and
reviewer approval, opened a draft PR, and only then failed RuleSpec CI. This
script makes the same decision up front: it runs in an unprotected job before
the production-signing environment asks for approval and before any model
call.

It is a cost damper, not a security gate. RuleSpec CI still enforces
admission. ``ENCODER_PIN_ADMISSION_OVERRIDE=true`` (a repository variable)
reports without blocking, for example to pre-generate encodes while a pin
bump is in review. A target repository that declares no ``axiom_encode_ref``
is reported and not blocked. Stdlib-only: it runs on a bare runner before any
toolchain setup.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
# The axiom-encode main commit (#1859) whose generated-change guard first
# admits a first-parent main descendant of the pin. A pin older than this
# admits only its exact commit.
PRODUCER_ANCESTRY_ADMISSION_COMMIT = "630aa73931e5aa87c51d25fe448e38dbcfaed50f"
DEFAULT_MAIN_REF = "refs/remotes/origin/main"


class AdmissionCheckError(RuntimeError):
    """The decision could not be made from the provided inputs."""


@dataclass(frozen=True)
class EncoderPin:
    ref: str | None
    version: str | None


@dataclass(frozen=True)
class Admission:
    """Whether the RuleSpec pin admits the encoder commit of this run.

    ``admissible`` is ``None`` when the target declares no encoder pin.
    """

    admissible: bool | None
    reason: str


def read_encoder_pin(toolchain_text: str) -> EncoderPin:
    """Read ``axiom_encode_ref`` and ``axiom_encode_version`` from the toolchain."""

    try:
        payload = tomllib.loads(toolchain_text)
    except tomllib.TOMLDecodeError as exc:
        raise AdmissionCheckError(f"workflow toolchain is not valid TOML: {exc}")
    section = payload.get("workflow_toolchain")
    if not isinstance(section, dict):
        raise AdmissionCheckError("workflow toolchain has no [workflow_toolchain]")
    ref = section.get("axiom_encode_ref")
    version = section.get("axiom_encode_version")
    if ref is not None and (
        not isinstance(ref, str) or SHA_PATTERN.fullmatch(ref) is None
    ):
        raise AdmissionCheckError(
            "workflow toolchain axiom_encode_ref must be a full lowercase commit SHA"
        )
    if version is not None and not isinstance(version, str):
        raise AdmissionCheckError(
            "workflow toolchain axiom_encode_version is malformed"
        )
    return EncoderPin(ref=ref, version=version)


def evaluate_admission(
    *,
    pin: EncoderPin,
    encoder_commit: str,
    is_ancestor: Callable[[str, str], bool],
    on_first_parent_main: Callable[[str], bool],
    ancestry_admission_commit: str | None = None,
) -> Admission:
    """Mirror the pinned guard's producer admission for this run's encoder.

    ``is_ancestor(a, b)`` is true when ``a`` is ``b`` or an ancestor of ``b``.
    Both callables raise :class:`AdmissionCheckError` when history cannot
    answer, which refuses the run: RuleSpec CI could not admit it either.
    """

    if SHA_PATTERN.fullmatch(encoder_commit) is None:
        raise AdmissionCheckError("encoder commit must be a full lowercase commit SHA")
    if ancestry_admission_commit is None:
        ancestry_admission_commit = PRODUCER_ANCESTRY_ADMISSION_COMMIT
    if pin.ref is None:
        return Admission(
            None, "the RuleSpec workflow toolchain declares no axiom_encode_ref"
        )
    pinned = f"{pin.ref} ({pin.version or 'version not declared'})"
    if encoder_commit == pin.ref:
        return Admission(True, f"this run's encoder is the exact pin {pinned}")
    if not is_ancestor(ancestry_admission_commit, pin.ref):
        return Admission(
            False,
            f"the RuleSpec repository pins axiom-encode {pinned}, which predates "
            "producer-ancestry admission (axiom-encode #1859); its CI admits "
            f"only that exact commit, and this run's encoder is {encoder_commit}",
        )
    if not is_ancestor(pin.ref, encoder_commit):
        return Admission(
            False,
            f"this run's encoder {encoder_commit} does not descend from the "
            f"pinned axiom-encode {pinned}",
        )
    if not on_first_parent_main(encoder_commit):
        return Admission(
            False,
            f"this run's encoder {encoder_commit} is not on the first-parent "
            "history of axiom-encode main",
        )
    return Admission(
        True,
        f"this run's encoder {encoder_commit} descends from the pin {pinned} "
        "on first-parent main",
    )


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )


def git_is_ancestor(checkout: Path) -> Callable[[str, str], bool]:
    def is_ancestor(ancestor: str, descendant: str) -> bool:
        for commit in (ancestor, descendant):
            verified = _git(checkout, "rev-parse", "--verify", f"{commit}^{{commit}}")
            if verified.returncode != 0 or verified.stdout.strip() != commit:
                raise AdmissionCheckError(
                    f"axiom-encode history does not contain commit {commit}"
                )
        result = _git(checkout, "merge-base", "--is-ancestor", ancestor, descendant)
        if result.returncode in (0, 1):
            return result.returncode == 0
        raise AdmissionCheckError(
            f"git merge-base --is-ancestor exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )

    return is_ancestor


def git_on_first_parent_main(checkout: Path, main_ref: str) -> Callable[[str], bool]:
    def on_first_parent_main(commit: str) -> bool:
        shallow = _git(checkout, "rev-parse", "--is-shallow-repository")
        if shallow.returncode != 0 or shallow.stdout.strip() != "false":
            raise AdmissionCheckError(
                "axiom-encode checkout is shallow; fetch complete main history"
            )
        result = _git(checkout, "rev-list", "--first-parent", main_ref)
        if result.returncode != 0:
            raise AdmissionCheckError(
                f"cannot list first-parent history of {main_ref}: "
                f"{result.stderr.strip()}"
            )
        return commit in set(result.stdout.split())

    return on_first_parent_main


def _append_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--encoder-checkout", type=Path, required=True)
    parser.add_argument("--encoder-commit", required=True)
    parser.add_argument("--main-ref", default=DEFAULT_MAIN_REF)
    args = parser.parse_args(argv)
    override = (
        os.environ.get("ENCODER_PIN_ADMISSION_OVERRIDE", "").strip().lower() == "true"
    )

    try:
        pin = read_encoder_pin(args.toolchain.read_text(encoding="utf-8"))
        admission = evaluate_admission(
            pin=pin,
            encoder_commit=args.encoder_commit,
            is_ancestor=git_is_ancestor(args.encoder_checkout),
            on_first_parent_main=git_on_first_parent_main(
                args.encoder_checkout, args.main_ref
            ),
        )
    except (AdmissionCheckError, OSError, subprocess.SubprocessError) as exc:
        admission = Admission(False, f"admission could not be established: {exc}")

    print(
        f"encoder-pin-admission: admissible={admission.admissible}: {admission.reason}"
    )
    if admission.admissible is not False:
        return 0
    lines = [
        "### RuleSpec encoder pin admission",
        "",
        f"- {admission.reason[0].upper()}{admission.reason[1:]}.",
    ]
    if override:
        lines.append(
            "- Report-only: `ENCODER_PIN_ADMISSION_OVERRIDE=true`. A signed PR "
            "from this run will fail RuleSpec CI until the pin admits it."
        )
        _append_summary(lines)
        print("encoder-pin-admission: override set; not blocking.")
        return 0
    lines.append(
        "- **Blocked before the signing environment and any model call.** A "
        "signed PR from this encoder would fail RuleSpec CI. Move the target "
        "repository's `.axiom/workflow-toolchain.toml` `axiom_encode_ref` to "
        "an admissible encoder first (see "
        "`docs/targeted-reencode-attempt-budget.md`). To generate anyway, set "
        "the repository variable `ENCODER_PIN_ADMISSION_OVERRIDE=true` and "
        "remove it afterwards."
    )
    _append_summary(lines)
    print(
        f"::error title=Encoder not admissible under the RuleSpec pin::"
        f"{admission.reason}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
