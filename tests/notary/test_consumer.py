import json

from axiom_encode.notary.canonical import strict_parse
from axiom_encode.notary.consumer import epoch_template, parse_consumer

from .chain_fixtures import Epoch


def test_template_preserves_preimage_whitespace():
    epoch = Epoch.create()
    raw = epoch.active.blobs[".axiom/notary/consumer.json"]
    body = strict_parse(raw)
    pretty = (json.dumps(body, indent=3) + "\n").encode()
    assert parse_consumer(pretty) == body
    assert epoch_template(pretty, epoch.anchor.epoch_sha256) == pretty.replace(
        epoch.anchor.epoch_sha256.encode(), b"0" * 64
    )


def test_duplicate_epoch_and_wrong_epoch_refuse():
    epoch = Epoch.create()
    raw = epoch.active.blobs[".axiom/notary/consumer.json"]
    assert epoch_template(raw, "f" * 64) is None
    duplicate = (
        raw[:-1] + b',"epoch_sha256":"' + epoch.anchor.epoch_sha256.encode() + b'"}'
    )
    assert parse_consumer(duplicate) is None
