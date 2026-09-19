"""Lane consumer contract and byte-preserving genesis template binding."""

from ._schema import digest, fields, lane_name
from .canonical import strict_parse


def parse_consumer(raw: bytes) -> dict | None:
    # This is a lane trust-file preimage, not a JCS chain body. Preserve its
    # formatting while rejecting duplicate members and malformed JSON values.
    body = strict_parse(raw)
    if not fields(
        body,
        {"schema", "lane", "epoch_sha256", "notary_repository", "notary_spki_sha256"},
    ):
        return None
    if (
        body["schema"] != "axiom/notary-consumer/v1"
        or not lane_name(body["lane"])
        or not lane_name(body["notary_repository"])
        or body["notary_repository"] != body["lane"] + "-notary"
        or not digest(body["epoch_sha256"])
        or not digest(body["notary_spki_sha256"])
    ):
        return None
    return body


def epoch_template(raw: bytes, epoch: str) -> bytes | None:
    body = parse_consumer(raw)
    if body is None or body["epoch_sha256"] != epoch:
        return None
    literal = epoch.encode("ascii")
    # A duplicate textual occurrence would make which span is the epoch
    # ambiguous. The bootstrap package uses one literal epoch member.
    if raw.count(literal) != 1:
        return None
    return raw.replace(literal, b"0" * 64, 1)
