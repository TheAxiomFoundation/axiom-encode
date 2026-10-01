"""Public hardware sidecars deposited by the custodian, never signing keys.

The signer's isolated host owns this directory. A reviewed forced SSH command
may expose `deposit` to reviewers. It accepts one bounded public sidecar on
stdin, not a path or command. Invalid signatures have no signing authority:
the typed signer always verifies the sidecar against its authenticated chain.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import sys
from pathlib import Path

from ._schema import canonical_object, decode_base64, digest, fields
from .identity import IdentityRefusal


class ApprovalInbox:
    def __init__(self, directory: Path):
        from .deployment import custodian_parent

        custodian_parent(directory / "entry")
        info = directory.stat()
        if (
            directory.is_symlink()
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise IdentityRefusal("approval_inbox_custody")
        self.directory = directory

    def read(self, address):
        from .deployment import custodian_file

        if not digest(address):
            raise IdentityRefusal("approval_inbox_address")
        path = self.directory / (address + ".sig")
        try:
            return custodian_file(str(path), private=True, limit=8192)
        except FileNotFoundError:
            return None

    def deposit(self, raw):
        body = canonical_object(raw)
        if (
            len(raw) > 8192
            or not fields(
                body,
                {
                    "schema",
                    "body_sha256",
                    "scope",
                    "signer_spki_sha256",
                    "signature_base64",
                },
            )
            or body["schema"] != "axiom/detached-signature/v1"
            or body["scope"]
            not in {"axiom/notary-approval/v1", "axiom/notary-admin-approval/v1"}
            or not digest(body["body_sha256"])
            or not digest(body["signer_spki_sha256"])
            or len(decode_base64(body["signature_base64"]) or b"") != 64
        ):
            raise IdentityRefusal("approval_inbox_sidecar")
        address = body["body_sha256"]
        from .producer_host import write_private

        fd = os.open(
            self.directory / ".deposit.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            existing = self.read(address)
            if existing is not None:
                if existing != raw:
                    raise IdentityRefusal("approval_inbox_conflict")
                return address
            write_private(self.directory / (address + ".sig"), raw)
            return address
        finally:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(
        description="Custodian-only bounded public approval deposit"
    )
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(8193).rstrip(b"\n")
        print(ApprovalInbox(args.directory).deposit(raw))
    except Exception:
        raise SystemExit("Approval deposit refused.") from None


if __name__ == "__main__":
    main()
