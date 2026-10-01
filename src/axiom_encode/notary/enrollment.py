"""Read-only eligibility for team-writer enrollment; never grants enrollment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass

from ._schema import lane_name
from .canonical import strict_parse


@dataclass(frozen=True, slots=True)
class EnrollmentEligibility:
    repository: str
    operator: str
    eligible_to_request_enrollment: bool
    reason: str
    enrolled: bool = False


def check_write_access(repository: str, operator: str) -> EnrollmentEligibility:
    """Query GitHub's permission authority, not PR/commit author metadata.

    Production callers must authenticate the operator independently and bind
    this check to the enrollment/intake session. A local command can query
    another user's access; it does not authenticate its invoker as that user.
    """

    def result(eligible: bool, reason: str) -> EnrollmentEligibility:
        return EnrollmentEligibility(repository, operator, eligible, reason)

    if not lane_name(repository) or not re.fullmatch(r"[A-Za-z0-9-]+", operator):
        return result(False, "invalid-identity")
    try:
        completed = subprocess.run(
            [
                "gh",
                "api",
                "--hostname",
                "github.com",
                "--method",
                "GET",
                f"repos/{repository}/collaborators/{operator}/permission",
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result(False, "permission-unavailable")
    if completed.returncode != 0:
        # Never echo gh stderr: it can contain credential or transport details.
        return result(False, "permission-unavailable")
    response = strict_parse(completed.stdout)
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("user"), dict)
        or not isinstance(response["user"].get("login"), str)
        or response["user"]["login"].lower() != operator.lower()
    ):
        return result(False, "permission-unavailable")
    if response.get("permission") not in ("write", "admin"):
        return result(False, "repository-write-required")
    return result(True, "custodian-authorization-required")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args(argv)
    result = check_write_access(args.repository, args.operator)
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.eligible_to_request_enrollment else 1


if __name__ == "__main__":
    raise SystemExit(main())
