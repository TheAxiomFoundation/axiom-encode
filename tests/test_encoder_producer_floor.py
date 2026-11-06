# ruff: noqa: F811 - tests take the imported graph fixtures
"""Floor-only producer admission and the encoder-main verifier (#2093, #2095).

With an explicit producer floor, the running verifier can be newer than the
producer. A producer is admitted when it is the floor or a first-parent main
descendant of it. Without a floor the running encoder stays the floor.
"""

from __future__ import annotations

import argparse
import json

import pytest

from axiom_encode import cli
from tests.test_encoder_manifest_ancestry import (
    _expected,
    _identity,
    graph,  # noqa: F401 - fixture
    identity_checkout,  # noqa: F401 - fixture
)


@pytest.fixture(autouse=True)
def _running_code_is_the_verifier(monkeypatch):
    """Graph fixtures are not the running package; one test checks this."""

    monkeypatch.setattr(cli, "_encoder_floor_running_issue", lambda _commit: None)


# graph: 0 base(main) <- 1 pin(main) <- 2 side; 3 merge(1, 2)(main) <- 4 head(main)
# <- 5 feature. Index 4 plays the encoder-main verifier below.
VERIFIER = 4
FLOOR = 1


def _verifier(graph, floor: int | str | None):
    identity = _expected(graph, pin=VERIFIER)
    if floor is None:
        return identity
    identity.producer_floor = graph.commits[floor] if isinstance(floor, int) else floor
    return identity


@pytest.mark.parametrize(
    ("producer", "accepted"),
    [(0, False), (1, True), (2, False), (3, True), (4, True), (5, False)],
    ids=["below-floor", "floor", "second-parent", "merge", "verifier", "feature"],
)
def test_floor_admits_floor_and_its_first_parent_main_descendants(
    graph, producer, accepted
):
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[producer]), _verifier(graph, FLOOR)
    )
    assert (issue is None) is accepted, issue


def test_without_a_floor_an_older_producer_is_rejected_by_a_newer_verifier(graph):
    """This is the coupling that #2095 removes for encoder-main verifiers."""

    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[FLOOR]), _verifier(graph, None)
    )
    assert issue is not None
    assert "not an ancestor" in issue


def test_floor_bump_revokes_producers_below_it(graph):
    raised = _verifier(graph, 3)
    assert cli._encoder_producer_admission_issue(_identity(graph.commits[1]), raised)
    assert (
        cli._encoder_producer_admission_issue(_identity(graph.commits[3]), raised)
        is None
    )


@pytest.mark.parametrize("floor", ["", "A" * 40, "a" * 39, "main"])
def test_malformed_floor_fails_closed(graph, floor):
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[3]), _verifier(graph, floor)
    )
    assert issue is not None
    assert "producer floor" in issue


def test_floor_argument_sets_the_floor_of_new_guard_identities(graph):
    token = cli._ENCODER_PRODUCER_FLOOR.set(None)
    try:
        cli._apply_encoder_producer_floor_argument(
            argparse.Namespace(encoder_producer_floor=graph.commits[FLOOR])
        )
        identity = _expected(graph, pin=VERIFIER)
        assert identity.producer_floor == graph.commits[FLOOR]
        assert (
            cli._encoder_producer_admission_issue(
                _identity(graph.commits[FLOOR]), identity
            )
            is None
        )
        # The floor never enters the serialized identity record.
        assert "producer_floor" not in json.loads(json.dumps(identity))
        with pytest.raises(ValueError, match="full lowercase commit"):
            cli._apply_encoder_producer_floor_argument(
                argparse.Namespace(encoder_producer_floor="main")
            )
    finally:
        cli._ENCODER_PRODUCER_FLOOR.reset(token)


@pytest.fixture
def running_as(monkeypatch):
    def install(commit: str) -> None:
        monkeypatch.setattr(
            cli,
            "_current_guard_encoder_execution_identity",
            lambda: _identity(commit, cli.__version__),
        )

    return install


def test_a_verifier_below_the_floor_admits_nothing(graph):
    """Even an exact match fails when the verifier is older than the floor."""

    old_verifier = _expected(graph, pin=1)
    old_verifier.producer_floor = graph.commits[3]
    for producer in (1, 3, 4):
        assert cli._encoder_producer_admission_issue(
            _identity(graph.commits[producer]), old_verifier
        ), producer


def test_a_verifier_off_first_parent_main_admits_nothing(graph):
    side_verifier = _expected(graph, pin=2)
    side_verifier.producer_floor = graph.commits[1]
    assert cli._encoder_producer_admission_issue(
        _identity(graph.commits[2]), side_verifier
    )


def test_a_command_without_the_flag_clears_an_inherited_floor(graph):
    token = cli._ENCODER_PRODUCER_FLOOR.set(graph.commits[FLOOR])
    try:
        cli._apply_encoder_producer_floor_argument(argparse.Namespace())
        assert _expected(graph, pin=VERIFIER).producer_floor is None
    finally:
        cli._ENCODER_PRODUCER_FLOOR.reset(token)


def test_verifier_identity_rejects_other_running_code(identity_checkout, running_as):
    checkout, pin, producer = identity_checkout
    running_as(producer)
    with pytest.raises(RuntimeError, match="running encoder commit"):
        cli.verifier_identity_record(checkout, producer_floor=pin)


def test_verifier_identity_records_an_encoder_main_verifier(
    identity_checkout, running_as
):
    checkout, pin, _producer = identity_checkout
    running_as(pin)
    record = cli.verifier_identity_record(checkout, producer_floor=pin)
    assert record == {
        "schema": cli.VERIFIER_IDENTITY_SCHEMA,
        "mode": cli.VERIFIER_MODE_ENCODER_MAIN,
        "verifier": _identity(pin, cli.__version__),
        "producer_floor": pin,
    }


def test_verifier_identity_rejects_a_floor_newer_than_the_verifier(
    identity_checkout, running_as
):
    checkout, pin, producer = identity_checkout
    running_as(pin)
    with pytest.raises(RuntimeError, match="not an admitted verifier"):
        cli.verifier_identity_record(checkout, producer_floor=producer)


def test_verifier_identity_cli_writes_the_record(
    identity_checkout, tmp_path, running_as
):
    checkout, pin, _producer = identity_checkout
    running_as(pin)
    output = tmp_path / "verifier.json"
    args = argparse.Namespace(
        expected_encoder_checkout=checkout,
        encoder_producer_floor=pin,
        output=output,
    )
    assert cli.cmd_verifier_identity(args) == 0
    assert json.loads(output.read_text())["verifier"]["commit"] == pin
    args.encoder_producer_floor = "not-a-commit"
    assert cli.cmd_verifier_identity(args) == 1


def test_floored_admission_requires_the_running_code_to_be_the_verifier(
    graph, monkeypatch
):
    monkeypatch.undo()
    monkeypatch.setattr(
        cli,
        "_current_guard_encoder_execution_identity",
        lambda: _identity(graph.commits[3]),
    )
    verifier = _verifier(graph, FLOOR)
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[FLOOR]), verifier
    )
    assert issue and "running encoder commit" in issue
    monkeypatch.setattr(
        cli,
        "_current_guard_encoder_execution_identity",
        lambda: _identity(graph.commits[VERIFIER]),
    )
    assert (
        cli._encoder_producer_admission_issue(
            _identity(graph.commits[FLOOR]), _verifier(graph, FLOOR)
        )
        is None
    )


def test_a_floor_without_history_fails_closed(graph):
    identity = cli._GuardEncoderIdentity(
        _identity(graph.commits[VERIFIER]),
        history_checkout=None,
        producer_floor=graph.commits[FLOOR],
    )
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[FLOOR]), identity
    )
    assert issue and "--expected-encoder-checkout" in issue


def test_malformed_floor_flag_is_an_argument_error(capsys):
    parser = argparse.ArgumentParser()
    cli._add_encoder_producer_floor_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--encoder-producer-floor", "main"])
    assert "full lowercase commit" in capsys.readouterr().err
