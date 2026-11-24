"""Each unit of a multi-unit promotion is validated twice (#2301).

Pass 1, isolated: the candidate head without the files of the other units.
Nothing a unit needs may come from another, still unsigned, unit.

Pass 2, batch: the whole candidate head, the tree that exists after the batch
lands. No unit may be refused because another unit of the batch is present.

A unit passes only if both passes pass; the batch signs all units or none.

The tests run the real overlay staging with a probe in place of the
engine-backed validator (see ``test_multi_unit_promotion_isolation``). The
probe judges a unit on the tree it is given, so a landing order can also be
simulated on plain directory trees with the same judge.
"""

import itertools
import json
import random
from pathlib import Path

import pytest
import yaml

import axiom_encode.cli as cli
from axiom_encode import prepare_signed_backfill as helper
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.harness.validator_pipeline import (
    _parse_rulespec_target,
    _resolve_rulespec_target_file,
    find_missing_child_exception_import_issues,
)
from tests import test_multi_unit_promotion_isolation as iso
from tests.test_multi_unit_promotion_isolation import (
    BASE,
    Promotion,
    candidate,
    citation,
    companion,
    primary,
    probe_issues,
    target,
    tree_of,
    unit_files,
)


@pytest.fixture(autouse=True)
def switch_on(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)


def run_batch(root, units, monkeypatch, judge=None, *, refuse=True):
    repo, head = candidate(root, units)
    run = Promotion(root, repo, head, units, batch=True)
    run.refuse = refuse
    if judge is not None:
        run.judge = judge
    error = None
    try:
        run(monkeypatch)
    except (RuntimeError, ValueError) as exc:
        error = exc
    return run, error


def run_alone(root, name, files, monkeypatch, judge=None):
    repo, head = candidate(root, {name: files})
    run = Promotion(root, repo, head, [name], batch=False)
    if judge is not None:
        run.judge = judge
    error = None
    try:
        run(monkeypatch)
    except (RuntimeError, ValueError) as exc:
        error = exc
    return run, error


# --- The two passes ------------------------------------------------------------


def test_every_unit_of_a_batch_has_an_isolated_pass_then_a_batch_pass(
    tmp_path, monkeypatch
):
    units = {name: () for name in "abe"}
    run, error = run_batch(tmp_path, units, monkeypatch)
    assert error is None
    assert run.passes == [
        (name, which) for name in "abe" for which in ("first", "batch")
    ]
    checkout = {
        path: raw
        for path, raw in tree_of(run.repo).items()
        if not path.startswith((".git/", ".axiom/"))
        and path not in {".git/", ".axiom/"}
    }
    for name in units:
        # Pass 2 reads the whole candidate head.
        assert run.full_trees[name] == checkout
        # Pass 1 reads it without the other units.
        hidden = {
            path
            for other in units
            if other != name
            for path in (primary(other), companion(other))
        }
        assert hidden.isdisjoint(run.trees[name])
        # Besides the hidden files only emptied directories are absent.
        assert all(
            path.endswith("/") for path in set(checkout) - set(run.trees[name]) - hidden
        )
        assert set(run.trees[name]) <= set(checkout)
    assert sorted(run.signed) == sorted(units)


def test_a_single_unit_promotion_and_a_batch_of_one_have_one_pass(
    tmp_path, monkeypatch
):
    alone, error = run_alone(tmp_path / "alone", "a", (), monkeypatch)
    assert error is None and alone.passes == [("a", "first")]
    one, error = run_batch(tmp_path / "one", {"a": ()}, monkeypatch)
    assert error is None and one.passes == [("a", "first")]
    assert one.full_trees == {}


def test_the_manifest_records_the_isolated_pass(tmp_path, monkeypatch):
    """The second pass is a check; it does not replace what is signed."""
    snapshots = []

    repo, head = candidate(tmp_path, {"a": (), "b": ()})
    run = Promotion(tmp_path, repo, head, ["a", "b"], batch=True)
    real_validate = Promotion._validate

    def tagging_validate(self, result, **kwargs):
        outcome = real_validate(self, result, **kwargs)
        which = self.passes[-1]
        setattr(
            result,
            cli._APPLY_VALIDATION_SNAPSHOT_ATTR,
            {"manifest_validation_execution": {"pass": list(which)}},
        )
        snapshots.append(which)
        return outcome

    monkeypatch.setattr(Promotion, "_validate", tagging_validate)
    run(monkeypatch)
    assert snapshots == [("a", "first"), ("a", "batch"), ("b", "first"), ("b", "batch")]
    for name in "ab":
        payload = json.loads(
            (
                repo / cli._applied_encoding_manifest_path(Path(primary(name)))
            ).read_text()
        )
        assert payload["validation_execution"] == {"pass": [name, "first"]}


# --- The reviewer's case: a check that is stricter when a module exists -------


def parent_files() -> dict[str, str]:
    """A unit whose source opens an exception list and that imports no child."""
    module = {
        "format": "rulespec/v1",
        "module": {
            "summary": "The term gift shall not include the following:",
            "source_verification": {
                "corpus_citation_path": citation("a"),
                "source_sha256": iso.SOURCE_SHA256,
            },
        },
        "rules": [
            {
                "name": "is_gift",
                "kind": "derived",
                "versions": [{"formula": "transfer_made and not_for_value"}],
            }
        ],
    }
    return {
        primary("a"): yaml.safe_dump(module, sort_keys=False),
        companion("a"): "[]\n",
    }


def child_files() -> dict[str, str]:
    """A child of ``a`` that exports an exception output."""
    module = {
        "format": "rulespec/v1",
        "module": {
            "summary": "A gift of small value.",
            "source_verification": {
                "corpus_citation_path": citation("1"),
                "source_sha256": iso.SOURCE_SHA256,
            },
        },
        "rules": [
            {
                "name": "gift_exception_applies",
                "kind": "derived",
                "versions": [{"formula": "is_small"}],
            }
        ],
    }
    return {
        primary("1"): yaml.safe_dump(module, sort_keys=False),
        companion("1"): "[]\n",
    }


def exception_judge(overlay_target: Path, content_root: Path) -> list[str]:
    """The encoder's own child-exception check on the tree validation reads."""
    return probe_issues(
        overlay_target, content_root
    ) + find_missing_child_exception_import_issues(
        overlay_target.read_text(),
        rules_file=overlay_target,
        policy_repo_path=content_root,
    )


def test_the_exception_check_is_stricter_with_the_child_present(tmp_path):
    """The premise, on plain trees, with the real check."""
    root = tmp_path / "rulespec-us"
    for relative, text in {**parent_files(), **child_files()}.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    with_child = exception_judge(root / primary("a"), root / "us")
    assert len(with_child) == 1
    assert with_child[0].startswith("Parent exception-list child import missing")
    (root / primary("1")).unlink()
    assert exception_judge(root / primary("a"), root / "us") == []


def test_a_parent_that_is_refused_beside_its_child_stops_the_batch(
    tmp_path, monkeypatch
):
    """Isolated, the parent passes; with the child present it is refused."""
    units = {"a": parent_files(), "1": child_files()}
    run, error = run_batch(tmp_path, units, monkeypatch, exception_judge)
    assert run.verdicts == {"a": []}
    assert len(run.full_verdicts["a"]) == 1
    assert isinstance(error, RuntimeError)
    assert str(error).startswith(
        f"unit {citation('a')} ({primary('a')}): reviewed candidate failed overlay "
        "validation with the other units of the batch present: "
        "Parent exception-list child import missing"
    )
    # The child was never reached; nothing is signed.
    assert run.passes == [("a", "first"), ("a", "batch")]
    assert run.signed == [] and run.manifests() == []


def test_the_same_pair_is_refused_in_either_single_unit_order_of_the_parent(
    tmp_path, monkeypatch
):
    """Parent last: refused at its turn. Parent first: it lands, and the tree
    that then holds both is the one the batch pass refuses. So the batch
    refuses exactly the state that no order reaches with every unit valid."""
    parent_alone, error = run_alone(
        tmp_path / "parent-first", "a", parent_files(), monkeypatch, exception_judge
    )
    assert error is None and parent_alone.signed == ["a"]
    # Parent after the child has landed: main holds the child.
    root = tmp_path / "child-on-main" / "rulespec-us"
    for relative, text in {**parent_files(), **child_files()}.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    assert exception_judge(root / primary("a"), root / "us") != []
    # The final tree of the order parent-then-child: the parent is not valid
    # in it, which is what pass 2 reports.
    assert exception_judge(root / primary("1"), root / "us") == []


def test_a_parent_that_imports_and_uses_its_child_is_another_refusal(
    tmp_path, monkeypatch
):
    """Fixing the parent by importing the child makes the pair an import
    between units of one batch: promote the child first."""
    files = parent_files()
    module = yaml.safe_load(files[primary("a")])
    module["imports"] = [f"{target('1')}#gift_exception_applies"]
    module["rules"][0]["versions"][0]["formula"] = (
        "transfer_made and not gift_exception_applies"
    )
    files[primary("a")] = yaml.safe_dump(module, sort_keys=False)
    run, error = run_batch(
        tmp_path, {"a": files, "1": child_files()}, monkeypatch, exception_judge
    )
    assert isinstance(error, ValueError)
    assert str(error) == (
        f"{primary('a')} imports {primary('1')}, another unit of the same batch; "
        "promote the imported unit first"
    )
    assert run.passes == [] and run.signed == []


# --- What the two-pass rule guarantees ------------------------------------------
#
# A judge that is a conjunction of two kinds of check:
#   needs:     a reference must resolve (passes more often as modules appear);
#   conflicts: a named module must be absent (passes less often as they appear).
# Every check the encoder runs on other modules that was found in review is of
# one of these two kinds.

NEEDS, CONFLICTS = "needs", "conflicts_with"


def ordered_unit(name: str, needs, conflicts) -> dict[str, str]:
    files = unit_files(name, tuple(("relation-target", f"{ref}#x") for ref in needs))
    module = yaml.safe_load(files[primary(name)])
    module[CONFLICTS] = list(conflicts)
    files[primary(name)] = yaml.safe_dump(module, sort_keys=False)
    return files


def order_judge(overlay_target: Path, content_root: Path) -> list[str]:
    issues = probe_issues(overlay_target, content_root)
    for reference in yaml.safe_load(overlay_target.read_text()).get(CONFLICTS, []):
        parsed = _parse_rulespec_target(reference)
        if _resolve_rulespec_target_file(parsed, content_root) is not None:
            issues.append(f"{overlay_target.name}: conflicts with `{reference}`")
    return issues


def passes_in_tree(root: Path, units, present, name) -> bool:
    """The judge's verdict on ``name`` in ``main`` plus the ``present`` units."""
    tree = root / "rulespec-us"
    files = {
        "us/statutes/7/2014/base.yaml": "format: rulespec/v1\nrules:\n- name: x\n",
        "us/statutes/7/2014/base.test.yaml": "[]\n",
    }
    for other in present:
        files.update(units[other])
    for relative, text in files.items():
        (tree / relative).parent.mkdir(parents=True, exist_ok=True)
        (tree / relative).write_text(text)
    return order_judge(tree / primary(name), tree / "us") == []


def order_lands(root: Path, units, order) -> bool:
    """Every unit passes at its turn: ``main`` plus the units landed so far."""
    return all(
        passes_in_tree(root / f"turn-{index}", units, order[: index + 1], name)
        for index, name in enumerate(order)
    )


def random_units(rng: random.Random, names) -> dict:
    units = {}
    for name in names:
        others = [other for other in names if other != name]
        needs, conflicts = [], []
        for other in others:
            roll = rng.random()
            if roll < 0.07:
                needs.append(target(other))
            elif roll < 0.14:
                conflicts.append(target(other))
        if rng.random() < 0.15:
            needs.append(BASE)
        if rng.random() < 0.03:
            conflicts.append(BASE)
        units[name] = ordered_unit(name, needs, conflicts)
    return units


@pytest.mark.parametrize("seed", range(16))
def test_the_two_pass_rule_on_random_batches(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    names = sorted(rng.sample("abcde", rng.randint(2, 4)), key=primary)
    units = random_units(rng, names)
    run, error = run_batch(tmp_path / "batch", units, monkeypatch, order_judge)
    signed = error is None
    assert signed is (sorted(run.signed) == sorted(names) and bool(run.signed))
    if not signed:
        assert run.signed == [] and run.manifests() == []

    alone = {
        name: passes_in_tree(tmp_path / f"alone-{name}", units, [name], name)
        for name in names
    }
    beside = {
        name: passes_in_tree(tmp_path / f"full-{name}", units, names, name)
        for name in names
    }
    # 1. The two stated properties, and nothing else, decide the batch.
    assert signed is (all(alone.values()) and all(beside.values()))
    orders = list(itertools.permutations(names))
    landing = {
        order: order_lands(tmp_path / ("order-" + "".join(order)), units, order)
        for order in orders
    }
    # 2. Sound: a signed batch can land one unit at a time in EVERY order.
    #    (Every check here is of the needs or the conflicts kind.)
    if signed:
        assert all(landing.values())
    # 3. Complete for these checks: if every order lands, the batch signs.
    assert signed is all(landing.values())
    # 4. It is stricter than "some order lands": the first unit of an order
    #    is judged alone and the last one beside all others, for every order.
    if any(landing.values()) and not signed:
        assert not all(landing.values())


def test_the_random_batches_cover_every_outcome(tmp_path):
    """The seeds above hold signed batches, refused ones, and refused ones
    with a landing order, so the properties are not vacuous."""
    signed = refused = refused_with_an_order = 0
    for seed in range(16):
        rng = random.Random(seed)
        names = sorted(rng.sample("abcde", rng.randint(2, 4)), key=primary)
        units = random_units(rng, names)
        root = tmp_path / str(seed)
        rule = all(
            passes_in_tree(root / f"alone-{name}", units, [name], name)
            and passes_in_tree(root / f"full-{name}", units, names, name)
            for name in names
        )
        some = any(
            order_lands(root / ("o-" + "".join(order)), units, order)
            for order in itertools.permutations(names)
        )
        signed += rule
        refused += not rule
        refused_with_an_order += some and not rule
    assert signed >= 3 and refused >= 3 and refused_with_an_order >= 1


def test_some_order_can_land_while_the_batch_is_refused(tmp_path, monkeypatch):
    """Not "some order exists": a unit that needs another unit of the batch
    lands after it, one at a time, but the batch is refused (pass 1)."""
    units = {
        "a": ordered_unit("a", [target("b")], []),
        "b": ordered_unit("b", [], []),
    }
    assert order_lands(tmp_path / "ba", units, ("b", "a")) is True
    assert order_lands(tmp_path / "ab", units, ("a", "b")) is False
    run, error = run_batch(tmp_path / "batch", units, monkeypatch, order_judge)
    assert "failed current overlay validation" in str(error)
    assert run.signed == []


def test_a_unit_that_conflicts_with_another_is_refused_by_the_batch_pass(
    tmp_path, monkeypatch
):
    units = {
        "a": ordered_unit("a", [], []),
        "b": ordered_unit("b", [], [target("a")]),
    }
    assert order_lands(tmp_path / "ba", units, ("b", "a")) is True
    assert order_lands(tmp_path / "ab", units, ("a", "b")) is False
    run, error = run_batch(tmp_path / "batch", units, monkeypatch, order_judge)
    assert str(error).startswith(
        f"unit {citation('b')} ({primary('b')}): reviewed candidate failed overlay "
        "validation with the other units of the batch present: "
    )
    # ``a`` passed both passes before ``b`` was refused; still nothing signed.
    assert run.passes == [
        ("a", "first"),
        ("a", "batch"),
        ("b", "first"),
        ("b", "batch"),
    ]
    assert run.signed == [] and run.manifests() == []


def test_for_two_units_the_rule_is_every_order_for_any_judge(tmp_path, monkeypatch):
    """With two units the two passes are exactly the four turns of the two
    orders, whatever the judge: each unit alone, each unit beside the other."""
    for seed in range(6):
        rng = random.Random(100 + seed)
        units = random_units(rng, ["a", "b"])
        _run, error = run_batch(
            tmp_path / f"batch-{seed}", units, monkeypatch, order_judge
        )
        both = all(
            order_lands(tmp_path / f"o{seed}-{''.join(order)}", units, order)
            for order in (("a", "b"), ("b", "a"))
        )
        assert (error is None) is both


# --- The record of what validation read ---------------------------------------


def test_the_record_names_what_each_unit_was_isolated_from(tmp_path, monkeypatch):
    """The command hides exactly the paths the record names."""
    names = ["a", "b", "e"]
    hidden_by_command = {}
    real = cli._isolate_promotion_batch_overlay

    def spy(overlay_repo, paths):
        present = {
            path.relative_to(overlay_repo).as_posix()
            for path in overlay_repo.rglob("*.yaml")
        }
        unit = next(
            name
            for name in names
            if primary(name) not in {path.as_posix() for path in paths}
        )
        hidden_by_command[unit] = sorted(path.as_posix() for path in paths)
        assert set(hidden_by_command[unit]) <= present
        return real(overlay_repo, paths)

    monkeypatch.setattr(cli, "_isolate_promotion_batch_overlay", spy)
    run, error = run_batch(tmp_path, {name: () for name in names}, monkeypatch)
    assert error is None
    record = helper.promotion_validation_record(
        [citation(name) for name in reversed(names)]
    )
    assert record["schema"] == "axiom-encode/promotion-batch-validation/v1"
    assert record["passes"] == ["isolated", "batch"]
    assert [unit["citation"] for unit in record["units"]] == [
        citation(name) for name in names
    ]
    for unit, name in zip(record["units"], names):
        assert unit["rulespec_path"] == primary(name)
        assert unit["companion_path"] == companion(name)
        assert unit["hidden_paths"] == hidden_by_command[name]
        # The validated tree can be rebuilt: the head without the hidden paths.
        rebuilt = {
            path: raw
            for path, raw in run.full_trees[name].items()
            if path not in unit["hidden_paths"] and not path.endswith("/")
        }
        assert rebuilt == {
            path: raw for path, raw in run.trees[name].items() if not path.endswith("/")
        }


def test_the_record_of_a_batch_of_one_has_one_pass_and_hides_nothing():
    record = helper.promotion_validation_record([citation("a")])
    assert record["passes"] == ["batch"]
    assert record["units"] == [
        {
            "citation": citation("a"),
            "rulespec_path": primary("a"),
            "companion_path": companion("a"),
            "hidden_paths": [],
        }
    ]


def test_the_record_refuses_two_citations_of_one_module(monkeypatch):
    monkeypatch.setattr(helper, "citation_rulespec_path", lambda _c: Path(primary("a")))
    with pytest.raises(ValueError, match="distinct modules"):
        helper.promotion_validation_record([citation("a"), citation("b")])


# --- The isolation stays inside the overlay ------------------------------------


def test_a_linked_parent_directory_refuses_before_any_file_is_removed(tmp_path):
    """A component that is a link would carry the removal out of the overlay."""
    outside = tmp_path / "outside"
    (outside / "2015").mkdir(parents=True)
    victim = outside / "2015" / "b.yaml"
    victim.write_text("outside the overlay")
    root = tmp_path / "overlay"
    (root / "us/statutes/7").mkdir(parents=True)
    (root / "us/statutes/7/2014").mkdir()
    inside = root / "us/statutes/7/2014/a.yaml"
    inside.write_text("inside the overlay")
    (root / "us/statutes/7/2015").symlink_to(outside / "2015", target_is_directory=True)
    with pytest.raises(ValueError, match="outside the validation overlay"):
        cli._isolate_promotion_batch_overlay(
            root,
            (Path("us/statutes/7/2014/a.yaml"), Path("us/statutes/7/2015/b.yaml")),
        )
    # Neither the file outside nor the valid file listed before it is removed.
    assert victim.read_text() == "outside the overlay"
    assert inside.read_text() == "inside the overlay"


def test_a_link_to_another_place_inside_the_overlay_also_refuses(tmp_path):
    """The path must be the file itself, not another name for a file."""
    root = tmp_path / "overlay"
    (root / "us/real").mkdir(parents=True)
    kept = root / "us/real/b.yaml"
    kept.write_text("kept")
    (root / "us/alias").symlink_to(root / "us/real", target_is_directory=True)
    with pytest.raises(ValueError, match="outside the validation overlay"):
        cli._isolate_promotion_batch_overlay(root, (Path("us/alias/b.yaml"),))
    assert kept.read_text() == "kept"


def test_a_linked_overlay_root_is_resolved_and_contained(tmp_path):
    real = tmp_path / "real"
    (real / "us").mkdir(parents=True)
    (real / "us/a.yaml").write_text("a")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    cli._isolate_promotion_batch_overlay(alias, (Path("us/a.yaml"),))
    assert not (real / "us/a.yaml").exists() and real.is_dir()
