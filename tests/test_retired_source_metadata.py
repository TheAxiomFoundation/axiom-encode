"""Invariant tests for the deterministic retired-metadata transaction."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness.validator_pipeline import (
    find_proof_import_hash_consistency_issues,
)
from axiom_encode.retired_source_metadata import (
    PLAN_SCHEMA,
    RECEIPT_DIR,
    RetiredSourceMetadataError,
    build_migration,
    canonical_json_bytes,
    load_plan_bytes,
    receipt_identity_sha256,
    rewrite_source_metadata,
    verify_migration_replay,
)

BASE = "a" * 40
TREE = "b" * 40
PRIMARY = "us/statutes/26/example.yaml"
FIXTURES = Path(__file__).parent / "fixtures" / "retired_source_metadata"


def _plan(*modules: str, **extra: object):
    return load_plan_bytes(
        json.dumps(
            {
                "schema_version": PLAN_SCHEMA,
                "base_commit": BASE,
                "modules": list(modules or (PRIMARY,)),
                **extra,
            }
        ).encode()
    )


def _module(block: str, *, before: str = "", after: str = "") -> bytes:
    return (
        "format: rulespec/v1\nmodule:\n"
        + before
        + "  source_verification:\n"
        + block
        + after
        + "  summary: Keep this exact string.\nrules: []\n"
    ).encode()


def _values_module(values: dict[str, int]) -> bytes:
    return _module(
        "    corpus_citation_path: us/statute/26/1\n    values:\n"
        + "".join(f"      {key}: {value}\n" for key, value in values.items()),
        before="  description: 'Spacing and quotes stay the same.' # retained\n",
        after="\n    source_sha256: 'unchanged' # retained\n",
    )


def _plural_module(paths: list[str], *, upstream: list[str] | None = None) -> bytes:
    block = "    corpus_citation_paths:\n" + "".join(
        f"      - {path}\n" for path in paths
    )
    if upstream is not None:
        block += "    upstream_source_check:\n      checked_paths:\n" + "".join(
            f"        - {path}\n" for path in upstream
        )
    return _module(block)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _importer(target: str, target_bytes: bytes, *, quoted: bool = False) -> bytes:
    pinned = "sha256:" + _hash(target_bytes)
    if quoted:
        pinned = "'" + pinned + "'"
    return f"""format: rulespec/v1
imports:
  - {target}#amount
rules:
  - name: result
    metadata:
      proof:
        atoms:
          - kind: import
            import:
              target: {target}#amount
              hash: {pinned} # keep this comment
    versions: []
""".encode()


class TestExactPlan:
    def test_binds_full_base_and_canonical_json(self):
        plan = _plan()
        assert plan.base_commit == BASE
        assert plan.modules == (Path(PRIMARY),)
        assert plan.sha256 == _hash(plan.canonical_bytes)
        assert load_plan_bytes(plan.canonical_bytes) == plan

    @pytest.mark.parametrize(
        "extra",
        [
            {"unexpected": True},
            {"base_commit": "a123"},
            {"modules": []},
            {"modules": [PRIMARY, PRIMARY]},
            {"modules": ["../us/statutes/26/1.yaml"]},
            {"modules": ["us/statutes/26/1.test.yaml"]},
            {"modules": ["us//statutes/26/1.yaml"]},
            {"modules": ["tools/file.yaml"]},
        ],
    )
    def test_refuses_overbroad_or_noncanonical_authority(self, extra):
        with pytest.raises(RetiredSourceMetadataError):
            _plan(**extra)

    def test_refuses_duplicate_json_keys(self):
        with pytest.raises(RetiredSourceMetadataError, match="duplicate key"):
            load_plan_bytes(b'{"schema_version":"x","schema_version":"y"}')


class TestStructureAndLocality:
    # I1 and I2: only values is removed; prefix, suffix, comments, quotes, and
    # the original scalar spellings outside the deleted span remain exact.
    @settings(max_examples=50, deadline=None)
    @given(
        st.dictionaries(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8).map(
                lambda value: "value_" + value
            ),
            st.integers(-10000, 10000),
            min_size=1,
            max_size=8,
        )
    )
    def test_i1_i2_drop_values_preserves_structure_and_all_other_bytes(self, values):
        raw = _values_module(values)
        rewrite = rewrite_source_metadata(raw)
        expected = copy.deepcopy(yaml.safe_load(raw))
        del expected["module"]["source_verification"]["values"]
        assert yaml.safe_load(rewrite.after) == expected
        block = b"    values:\n" + b"".join(
            f"      {key}: {value}\n".encode() for key, value in values.items()
        )
        assert rewrite.after == raw.replace(block, b"", 1)
        assert rewrite.removed_values == values

    @settings(max_examples=50, deadline=None)
    @given(
        st.permutations(
            ["us/statute/26/63/c/6", "us/statute/26/63/c/6/A", "us/statute/26/63/c/6/B"]
        )
    )
    def test_i1_i2_unique_ancestor_need_not_be_first(self, paths):
        raw = _plural_module(list(paths))
        rewrite = rewrite_source_metadata(raw)
        expected = yaml.safe_load(raw)
        del expected["module"]["source_verification"]["corpus_citation_paths"]
        expected["module"]["source_verification"]["corpus_citation_path"] = (
            "us/statute/26/63/c/6"
        )
        assert yaml.safe_load(rewrite.after) == expected
        block = b"    corpus_citation_paths:\n" + b"".join(
            f"      - {item}\n".encode() for item in paths
        )
        assert rewrite.after == raw.replace(
            block, b"    corpus_citation_path: us/statute/26/63/c/6\n", 1
        )
        assert rewrite.removed_corpus_citation_paths == tuple(paths)

    def test_r2_normalizes_entries_before_finding_the_ancestor(self):
        raw = _plural_module(["us:statutes/26/63/c/6/A", "26 USC 63(c)(6)"])
        rewrite = rewrite_source_metadata(raw)
        assert rewrite.corpus_citation_path == "us/statute/26/63/c/6"
        assert rewrite.removed_corpus_citation_paths == (
            "us:statutes/26/63/c/6/A",
            "26 USC 63(c)(6)",
        )

    def test_both_edits_and_consistent_upstream_preserve_history(self):
        paths = ["us/statute/26/1", "us/statute/26/1/a"]
        raw = _plural_module(paths, upstream=paths).replace(
            b"  summary:", b"    values:\n      amount: 42\n  summary:"
        )
        rewrite = rewrite_source_metadata(raw)
        expected = yaml.safe_load(raw)
        verification = expected["module"]["source_verification"]
        del verification["values"]
        del verification["corpus_citation_paths"]
        verification["corpus_citation_path"] = paths[0]
        assert yaml.safe_load(rewrite.after) == expected
        assert rewrite.removed_values == {"amount": 42}

    def test_upstream_audit_may_preserve_additional_parent_or_sibling_paths(self):
        paths = ["us/statute/26/1", "us/statute/26/1/a"]
        checked = ["us/statute/26", *paths, "us/statute/26/2"]
        raw = _plural_module(paths, upstream=checked)
        rewrite = rewrite_source_metadata(raw)
        verification = yaml.safe_load(rewrite.after)["module"]["source_verification"]
        assert verification["upstream_source_check"]["checked_paths"] == checked
        assert verification["corpus_citation_path"] == paths[0]


class TestRefusal:
    @pytest.mark.parametrize("indent", [5, 7, 8])
    def test_values_children_must_use_exactly_six_spaces(self, indent):
        raw = _module(
            "    corpus_citation_path: us/statute/26/1\n    values:\n"
            + " " * indent
            + "amount: 1\n"
        )
        with pytest.raises(
            RetiredSourceMetadataError, match="canonical block YAML indentation"
        ):
            rewrite_source_metadata(raw)

    @pytest.mark.parametrize("indent", [7, 9, 10])
    def test_nested_values_mapping_requires_two_space_indentation(self, indent):
        raw = _module(
            "    corpus_citation_path: us/statute/26/1\n    values:\n"
            "      household_size:\n" + " " * indent + "1: 100\n"
        )
        with pytest.raises(
            RetiredSourceMetadataError, match="canonical block YAML indentation"
        ):
            rewrite_source_metadata(raw)

    # I4: failure is pure; even a values deletion cannot escape an R2 refusal.
    @settings(max_examples=50, deadline=None)
    @given(st.integers(1, 100000), st.integers(1, 100000))
    def test_i4_no_unique_ancestor_is_refused_without_changing_input(
        self, first, second
    ):
        paths = [f"us/statute/26/1/a/{first}", f"us/statute/26/1/b/{second}"]
        raw = _plural_module(paths).replace(
            b"  summary:", b"    values:\n      amount: 1\n  summary:"
        )
        before = bytes(raw)
        with pytest.raises(
            RetiredSourceMetadataError, match="pending source shape decision"
        ):
            rewrite_source_metadata(raw)
        assert raw == before

    @pytest.mark.parametrize(
        "raw,match",
        [
            (
                _plural_module(["us/statute/26/1", "us/statute/26/1"]),
                "exactly one ancestor",
            ),
            (
                _plural_module(["us/statute/26/1"]).replace(
                    b"  summary:",
                    b"    corpus_citation_path: us/statute/26/1\n  summary:",
                ),
                "mixes singular and plural",
            ),
            (
                _plural_module(["us/statute/26/1"], upstream=["us/statute/26/2"]),
                "inconsistent",
            ),
            (
                _plural_module(["us/statute/26/1"]).replace(
                    b"  summary:", b"    source_sha256: aggregate\n  summary:"
                ),
                "ambiguous aggregate",
            ),
            (
                _module(
                    "    corpus_citation_path: us/statute/26/1\n    values: {amount: 1}\n"
                ),
                "canonical block",
            ),
            (
                _module("    corpus_citation_paths: [us/statute/26/1]\n"),
                "canonical block",
            ),
            (
                _module("    corpus_citation_paths:\n      - 'us/statute/26/1'\n"),
                "exact canonical",
            ),
            (
                _values_module({"amount": 1}).replace(
                    b"  summary:",
                    b"  retained: &anchor hello\n  alias: *anchor\n  summary:",
                ),
                "anchors or aliases",
            ),
            (
                _values_module({"amount": 1}).replace(
                    b"      amount: 1", b"      amount: 1\n      amount: 2"
                ),
                "duplicate key",
            ),
        ],
    )
    def test_ambiguous_noncanonical_or_alias_shapes_refuse(self, raw, match):
        with pytest.raises(RetiredSourceMetadataError, match=match):
            rewrite_source_metadata(raw)


class TestReplay:
    def test_canonical_nested_values_lists_and_multiline_history_stay_exact(self):
        block = (
            "    values:\n"
            "      household_size:\n"
            "        1: 100\n"
            "      rates:\n"
            "        - 0.25\n"
            "        - amount: 42\n"
            "          explanation: |-\n"
            "            first line\n"
            "            second line\n"
            "      description: >-\n"
            "        wrapped first line\n"
            "        wrapped second line\n"
        )
        raw = _module("    corpus_citation_path: us/statute/26/1\n" + block)
        migration = build_migration(_plan(), base_tree=TREE, base_files={PRIMARY: raw})
        assert migration.receipt["primaries"][0]["removed_values_yaml"] == block
        assert migration.files[0].after == raw.replace(block.encode(), b"", 1)
        assert (
            verify_migration_replay(migration.receipt_bytes, base_files={PRIMARY: raw})
            == migration
        )

    # I3: a migrated primary is refused; authenticated base replay produces
    # exactly the same transaction and preserves removed history in its receipt.
    def test_i3_idempotent_refusal_and_exact_base_replay(self):
        base = {PRIMARY: _values_module({"amount": 42})}
        migration = build_migration(_plan(), base_tree=TREE, base_files=base)
        replay = verify_migration_replay(migration.receipt_bytes, base_files=base)
        assert replay == migration
        assert migration.receipt["primaries"][0]["removed_values"] == {"amount": 42}
        assert (
            migration.receipt_relative
            == RECEIPT_DIR / f"{receipt_identity_sha256(migration.receipt)}.json"
        )
        with pytest.raises(RetiredSourceMetadataError, match="nothing-to-migrate"):
            rewrite_source_metadata(migration.files[0].after)
        with pytest.raises(RetiredSourceMetadataError, match="nothing-to-migrate"):
            build_migration(
                _plan(), base_tree=TREE, base_files={PRIMARY: migration.files[0].after}
            )

    @pytest.mark.parametrize(
        "field",
        [
            "base_commit",
            "base_tree",
            "files",
            "primaries",
            "cascade_rewrites",
            "plan_sha256",
            "identity_sha256",
        ],
    )
    def test_receipt_identity_binds_every_replay_record(self, field):
        base = {PRIMARY: _values_module({"amount": 42})}
        migration = build_migration(_plan(), base_tree=TREE, base_files=base)
        receipt = copy.deepcopy(migration.receipt)
        receipt[field] = (
            "c" * 40 if field in {"base_commit", "base_tree"} else [{"tampered": True}]
        )
        with pytest.raises(RetiredSourceMetadataError):
            verify_migration_replay(
                canonical_json_bytes(receipt) + b"\n", base_files=base
            )

    def test_missing_base_primary_and_invalid_tree_refuse(self):
        with pytest.raises(RetiredSourceMetadataError, match="absent"):
            build_migration(_plan(), base_tree=TREE, base_files={})
        with pytest.raises(RetiredSourceMetadataError, match="tree object"):
            build_migration(_plan(), base_tree="short", base_files={})

    def test_history_preserves_nested_integer_and_string_mapping_keys(self):
        raw = _module(
            "    corpus_citation_path: us/statute/26/1\n"
            "    values:\n"
            "      household_size:\n"
            "        1: 100\n"
            "        '1': 200\n"
        )
        migration = build_migration(_plan(), base_tree=TREE, base_files={PRIMARY: raw})
        record = migration.receipt["primaries"][0]
        assert record["removed_values"]["household_size"] == {
            "yaml_mapping_entries": [
                {"key": 1, "value": 100},
                {"key": "1", "value": 200},
            ]
        }
        preserved = yaml.safe_load(record["removed_values_yaml"])["values"]
        assert preserved == {"household_size": {1: 100, "1": 200}}
        assert record["removed_values_yaml"].encode() in raw
        assert (
            verify_migration_replay(migration.receipt_bytes, base_files={PRIMARY: raw})
            == migration
        )

    def test_history_preserves_top_level_numeric_values_keys(self):
        raw = _module(
            "    corpus_citation_path: us/statute/26/1\n    values:\n      1: 100\n"
        )
        migration = build_migration(_plan(), base_tree=TREE, base_files={PRIMARY: raw})
        assert migration.receipt["primaries"][0]["removed_values"] == {
            "yaml_mapping_entries": [{"key": 1, "value": 100}]
        }


class TestCascade:
    # I5: cascade reaches importers of importers and alters only exact pinned
    # hash spans. The ordinary validator sees no migration-induced stale pins.
    def test_i5_transitive_cascade_only_changes_pin_values(self, tmp_path):
        leaf = _values_module({"amount": 42})
        first = _importer("us:statutes/26/example", leaf, quoted=True)
        second = _importer("us:statutes/26/importer", first)
        base = {
            PRIMARY: leaf,
            "us/statutes/26/importer.yaml": first,
            "us/statutes/26/root.yaml": second,
        }
        migration = build_migration(_plan(), base_tree=TREE, base_files=base)
        post = {
            **base,
            **{item.path.as_posix(): item.after for item in migration.files},
        }
        assert len(migration.cascade_rewrites) == 2
        assert len(migration.files) == 3
        assert post["us/statutes/26/importer.yaml"] == first.replace(
            _hash(leaf).encode(), _hash(post[PRIMARY]).encode()
        )
        assert post["us/statutes/26/root.yaml"] == second.replace(
            _hash(first).encode(), _hash(post["us/statutes/26/importer.yaml"]).encode()
        )
        repo = tmp_path / "rulespec-us"
        for path, raw in post.items():
            file = repo / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(raw)
        for path, raw in post.items():
            assert (
                find_proof_import_hash_consistency_issues(
                    raw.decode(), rules_file=repo / path, policy_repo_path=repo / "us"
                )
                == []
            )
        assert (
            verify_migration_replay(migration.receipt_bytes, base_files=base)
            == migration
        )

    def test_preexisting_wrong_pin_stays_unchanged(self):
        leaf = _values_module({"amount": 1})
        importer = _importer("us:statutes/26/example", leaf).replace(
            _hash(leaf).encode(), b"0" * 64
        )
        base = {PRIMARY: leaf, "us/statutes/26/importer.yaml": importer}
        migration = build_migration(_plan(), base_tree=TREE, base_files=base)
        assert migration.cascade_rewrites == ()
        assert [item.path.as_posix() for item in migration.files] == [PRIMARY]

    def test_i5_legislation_importer_is_included_in_transitive_inventory(self):
        leaf = _values_module({"amount": 42})
        legislation = _importer("us:statutes/26/example", leaf)
        root = _importer("us:legislation/federal/bill", legislation)
        base = {
            PRIMARY: leaf,
            "us/legislation/federal/bill.yaml": legislation,
            "us/policies/aaa-root.yaml": root,
        }
        migration = build_migration(_plan(), base_tree=TREE, base_files=base)
        assert [item.path.as_posix() for item in migration.files] == [
            PRIMARY,
            "us/legislation/federal/bill.yaml",
            "us/policies/aaa-root.yaml",
        ]
        assert len(migration.cascade_rewrites) == 2

    def test_escaped_hashes_take_the_full_scanner_path(self):
        leaf = _values_module({"amount": 1})
        importer = (
            _importer("us:statutes/26/example", leaf)
            .replace(b"hash: sha256:", b'hash: "\\x73ha256:')
            .replace(b" # keep", b'" # keep')
        )
        migration = build_migration(
            _plan(),
            base_tree=TREE,
            base_files={PRIMARY: leaf, "us/statutes/26/importer.yaml": importer},
        )
        assert len(migration.cascade_rewrites) == 1

    def test_unrelated_huge_or_noncanonical_hash_free_module_is_skipped(self):
        base = {
            PRIMARY: _values_module({"amount": 1}),
            "us/policies/unrelated.yaml": b"rules: This historical module has no proof pins.\n",
        }
        migration = build_migration(_plan(), base_tree=TREE, base_files=base)
        assert [item.path.as_posix() for item in migration.files] == [PRIMARY]


@pytest.mark.parametrize("fixture", ["capital-gains.yaml", "25A.yaml", "6.yaml"])
def test_real_rulespec_us_base_modules_migrate_and_replay(fixture):
    raw = (FIXTURES / fixture).read_bytes()
    rewrite = rewrite_source_metadata(raw)
    verification = yaml.safe_load(rewrite.after)["module"]["source_verification"]
    assert "values" not in verification
    assert "corpus_citation_paths" not in verification
    assert isinstance(verification["corpus_citation_path"], str)
    migration = build_migration(_plan(), base_tree=TREE, base_files={PRIMARY: raw})
    assert migration.files[0].after == rewrite.after
    assert (
        verify_migration_replay(migration.receipt_bytes, base_files={PRIMARY: raw})
        == migration
    )
