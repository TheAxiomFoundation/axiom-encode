from pathlib import Path

import pytest
import yaml

import axiom_encode.program_scope as program_scope
from axiom_encode.program_scope import ProgramScopeError, sync_program_scope


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "rulespec-us"
    spec = repo / "programs/us-sc/snap/fy-2026.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        "# preserved header\n\n"
        "program: us-sc/snap\n"
        "period: 2026-01\n\n"
        "scope:\n"
        "  federal:\n"
        "    - regulations/7-cfr/273/9\n\n"
        "  state:\n"
        "    - policies/dss/snap/page-163\n"
        "    - policies/dss/snap/page-369\n\n"
        "outputs:\n"
        "  - snap_benefit\n"
    )
    for path in ("page-159", "page-163", "page-369"):
        module = repo / f"us-sc/policies/dss/snap/{path}.yaml"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("format: rulespec/v1\n")
    return repo, spec


def test_sync_program_scope_changes_only_selected_sequence(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    before = spec.read_text()

    result = sync_program_scope(
        repo=repo,
        program_spec=Path("programs/us-sc/snap/fy-2026.yaml"),
        scope="state",
        add=["policies/dss/snap/page-159"],
        remove=["policies/dss/snap/page-369"],
    )

    assert result.changed is True
    assert result.added == ("policies/dss/snap/page-159",)
    assert result.removed == ("policies/dss/snap/page-369",)
    after = spec.read_text()
    assert after.startswith("# preserved header\n\n")
    assert after.split("scope:\n", 1)[0] == before.split("scope:\n", 1)[0]
    assert after.split("outputs:\n", 1)[1] == before.split("outputs:\n", 1)[1]
    assert "page-163\n\noutputs:" in after
    assert yaml.safe_load(after)["scope"]["state"] == [
        "policies/dss/snap/page-159",
        "policies/dss/snap/page-163",
    ]


def test_sync_program_scope_check_mode_does_not_write(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    before = spec.read_text()

    result = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
        write=False,
    )

    assert result.changed is True
    assert spec.read_text() == before


def test_sync_program_scope_is_idempotent(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
    )

    result = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
    )

    assert result.changed is False


def test_sync_program_scope_rejects_missing_module(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)

    with pytest.raises(ProgramScopeError, match="do not resolve"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-999"],
        )


def test_sync_program_scope_rejects_module_symlink_outside_scope(
    tmp_path: Path,
) -> None:
    repo, spec = _repo(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_text("format: rulespec/v1\n")
    module = repo / "us-sc/policies/dss/snap/page-999.yaml"
    module.symlink_to(outside)

    with pytest.raises(ProgramScopeError, match="do not resolve"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-999"],
        )


def test_sync_program_scope_rejects_program_spec_symlink(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    real_spec = spec.with_name("real.yaml")
    spec.rename(real_spec)
    spec.symlink_to(real_spec)

    with pytest.raises(ProgramScopeError, match="must not contain symlinks"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
        )


@pytest.mark.parametrize("path", ["../outside", "/absolute", "page.yaml"])
def test_sync_program_scope_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    repo, spec = _repo(tmp_path)

    with pytest.raises(ProgramScopeError, match="safe repo-relative"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=[path],
        )


def test_sync_program_scope_rejects_colon_bearing_module_path(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    module = repo / "us-sc/statutes/47:294.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")

    with pytest.raises(ProgramScopeError, match="safe repo-relative"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["statutes/47:294"],
        )


def test_sync_program_scope_allows_already_absent_removal(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)

    result = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        remove=["policies/dss/snap/page-999"],
    )

    assert result.changed is False
    assert result.removed == ()


def test_sync_program_scope_preserves_unsorted_existing_order(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    text = spec.read_text().replace(
        "    - regulations/7-cfr/273/9\n",
        "    - statutes/7/2014/a\n    - regulations/7-cfr/273/9\n",
    )
    spec.write_text(text)
    module = repo / "us/regulations/7-cfr/273/10.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="federal",
        add=["regulations/7-cfr/273/10"],
    )

    assert yaml.safe_load(spec.read_text())["scope"]["federal"] == [
        "statutes/7/2014/a",
        "regulations/7-cfr/273/9",
        "regulations/7-cfr/273/10",
    ]


def test_sync_program_scope_uses_original_order_after_removal(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        spec.read_text().replace(
            "    - policies/dss/snap/page-163\n    - policies/dss/snap/page-369\n",
            "    - policies/dss/snap/page-369\n    - policies/dss/snap/page-163\n",
        )
    )

    result = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
        remove=["policies/dss/snap/page-369"],
    )

    assert result.changed is True
    assert yaml.safe_load(spec.read_text())["scope"]["state"] == [
        "policies/dss/snap/page-163",
        "policies/dss/snap/page-159",
    ]


def test_sync_program_scope_resolves_explicit_scope_to_named_jurisdiction(
    tmp_path: Path,
) -> None:
    repo, spec = _repo(tmp_path)
    text = spec.read_text().replace("  state:\n", "  us-ny:\n")
    spec.write_text(text)
    module = repo / "us-ny/policies/dss/snap/page-159.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")

    result = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="us-ny",
        add=["policies/dss/snap/page-159"],
    )

    assert result.changed is True
    assert yaml.safe_load(spec.read_text())["scope"]["us-ny"][0] == (
        "policies/dss/snap/page-159"
    )


def test_sync_program_scope_derives_non_us_federal_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-ca"
    spec = repo / "programs/ca-on/snap/fy-2026.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        "program: ca-on/snap\n"
        "period: 2026-01\n"
        "scope:\n"
        "  federal: []\n"
        "outputs: [snap_benefit]\n"
    )
    module = repo / "ca/statutes/example/1.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="federal",
        add=["statutes/example/1"],
    )

    assert yaml.safe_load(spec.read_text())["scope"]["federal"] == [
        "statutes/example/1"
    ]


def test_sync_program_scope_preserves_scope_comments(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    comment = "    # Page 369 remains documented for later composition.\n"
    spec.write_text(
        spec.read_text().replace(
            "    - policies/dss/snap/page-369\n",
            comment + "    - policies/dss/snap/page-369\n",
        )
    )

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
        remove=["policies/dss/snap/page-369"],
    )

    assert comment in spec.read_text()


def test_sync_program_scope_inserts_before_following_item_comments(
    tmp_path: Path,
) -> None:
    repo, spec = _repo(tmp_path)
    module = repo / "us-sc/policies/dss/snap/page-200.yaml"
    module.write_text("format: rulespec/v1\n")
    comment = "    # Page 369 has separate composition requirements.\n"
    spec.write_text(
        spec.read_text().replace(
            "    - policies/dss/snap/page-369\n",
            comment + "    - policies/dss/snap/page-369\n",
        )
    )

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-200"],
    )

    updated = spec.read_text()
    assert updated.index("page-200") < updated.index(comment.strip())
    assert updated.index(comment.strip()) < updated.index("page-369")


def test_sync_program_scope_rejects_aliased_sequence(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        spec.read_text()
        .replace(
            "scope:\n",
            "state_imports: &state_imports\n"
            "  - policies/dss/snap/page-163\n"
            "  - policies/dss/snap/page-369\n\n"
            "scope:\n",
        )
        .replace(
            "  state:\n"
            "    - policies/dss/snap/page-163\n"
            "    - policies/dss/snap/page-369\n",
            "  state: *state_imports\n",
        )
    )

    with pytest.raises(ProgramScopeError, match="must not use a YAML alias"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
        )


def test_sync_program_scope_rejects_aliased_sequence_entry(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        "program: us-sc/snap\n"
        "period: 2026-01\n"
        "shared_import: &shared_import policies/dss/snap/page-163\n"
        "scope:\n"
        "  state:\n"
        "    - *shared_import\n"
        "    - policies/dss/snap/page-369\n"
        "outputs: [snap_benefit]\n"
    )

    with pytest.raises(ProgramScopeError, match="entries must not use YAML aliases"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
        )


@pytest.mark.parametrize("decoration", ["&state_imports", "!!seq"])
def test_sync_program_scope_rejects_decorated_sequence(
    tmp_path: Path,
    decoration: str,
) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        spec.read_text().replace(
            "  state:\n",
            f"  state: {decoration}\n",
        )
    )

    with pytest.raises(ProgramScopeError, match="anchor or explicit tag"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
            write=False,
        )


def test_sync_program_scope_rejects_aliased_scope_mapping(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        "program: us-sc/snap\n"
        "period: 2026-01\n"
        "scope_defaults: &scope_defaults\n"
        "  federal:\n"
        "    - regulations/7-cfr/273/9\n"
        "  state:\n"
        "    - policies/dss/snap/page-163\n"
        "    - policies/dss/snap/page-369\n"
        "scope: *scope_defaults\n"
        "unused_scope: *scope_defaults\n"
        "outputs: [snap_benefit]\n"
    )

    with pytest.raises(ProgramScopeError, match="mapping must not use a YAML alias"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
        )


def test_sync_program_scope_rejects_flow_style_scope_mapping(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        "program: us-sc/snap\n"
        "period: 2026-01\n"
        "scope: {state: []}\n"
        "outputs: [snap_benefit]\n"
    )

    with pytest.raises(ProgramScopeError, match="mapping must use block style"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
            write=False,
        )


def test_sync_program_scope_rejects_scope_root_symlink(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    state_root = repo / "us-sc"
    other_root = repo / "us-ny"
    state_root.rename(other_root)
    state_root.symlink_to(other_root, target_is_directory=True)

    with pytest.raises(
        ProgramScopeError, match="scope root path must not contain symlinks"
    ):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
        )


def test_sync_program_scope_rejects_unsafe_scope_key(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)

    with pytest.raises(ProgramScopeError, match="lowercase identifier"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="../state",
            add=["policies/dss/snap/page-159"],
        )


def test_sync_program_scope_supports_empty_block_sequence(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)

    result = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        remove=[
            "policies/dss/snap/page-163",
            "policies/dss/snap/page-369",
        ],
    )

    assert result.changed is True
    assert yaml.safe_load(spec.read_text())["scope"]["state"] == []

    repopulated = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
    )

    assert repopulated.changed is True
    assert yaml.safe_load(spec.read_text())["scope"]["state"] == [
        "policies/dss/snap/page-159"
    ]


def test_sync_program_scope_supports_inline_empty_sequence(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        spec.read_text().replace(
            "  state:\n"
            "    - policies/dss/snap/page-163\n"
            "    - policies/dss/snap/page-369\n",
            "  state: []\n",
        )
    )

    result = sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
    )

    assert result.changed is True
    assert yaml.safe_load(spec.read_text())["scope"]["state"] == [
        "policies/dss/snap/page-159"
    ]


def test_sync_program_scope_preserves_crlf(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    original = spec.read_text().replace("\n", "\r\n")
    spec.write_bytes(original.encode())

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
    )

    updated = spec.read_bytes()
    assert updated.count(b"\r\n") == updated.count(b"\n")
    assert updated.count(b"\r\n") > 0


def test_sync_program_scope_disables_write_newline_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec = _repo(tmp_path)
    calls: list[dict[str, object]] = []
    original = program_scope.tempfile.NamedTemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        program_scope.tempfile,
        "NamedTemporaryFile",
        tracked_temporary_file,
    )

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
    )

    assert calls[0]["newline"] == ""


def test_sync_program_scope_empties_indentationless_sequence(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        spec.read_text().replace(
            "  state:\n"
            "    - policies/dss/snap/page-163\n"
            "    - policies/dss/snap/page-369\n",
            "  state:\n"
            "  - policies/dss/snap/page-163\n"
            "  - policies/dss/snap/page-369\n",
        )
    )

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        remove=[
            "policies/dss/snap/page-163",
            "policies/dss/snap/page-369",
        ],
    )

    assert yaml.safe_load(spec.read_text())["scope"]["state"] == []


def test_sync_program_scope_rejects_non_atomic_addition(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    module = repo / "us-sc/programs/other.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("program: us-sc/other\n")

    with pytest.raises(ProgramScopeError, match="do not resolve"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["programs/other"],
        )


def test_sync_program_scope_rejects_companion_test_addition(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    companion = repo / "us-sc/policies/dss/snap/page-999.test.yaml"
    companion.write_text("format: rulespec-test/v1\n")

    with pytest.raises(ProgramScopeError, match="safe repo-relative"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-999.test"],
        )


def test_sync_program_scope_appends_after_unterminated_final_line(
    tmp_path: Path,
) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(spec.read_text().split("\n\noutputs:", 1)[0].rstrip("\n"))

    sync_program_scope(
        repo=repo,
        program_spec=spec.relative_to(repo),
        scope="state",
        add=["policies/dss/snap/page-159"],
    )

    assert yaml.safe_load(spec.read_text())["scope"]["state"] == [
        "policies/dss/snap/page-159",
        "policies/dss/snap/page-163",
        "policies/dss/snap/page-369",
    ]


def test_sync_program_scope_rejects_absolute_program_spec(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)

    with pytest.raises(ProgramScopeError, match="relative to --repo"):
        sync_program_scope(
            repo=repo,
            program_spec=spec,
            scope="state",
            add=["policies/dss/snap/page-159"],
        )


def test_sync_program_scope_rejects_noncanonical_checkout(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    unrelated = tmp_path / "workspace"
    repo.rename(unrelated)

    with pytest.raises(ProgramScopeError, match="exact canonical"):
        sync_program_scope(
            repo=unrelated,
            program_spec=spec.relative_to(repo),
            scope="state",
            add=["policies/dss/snap/page-159"],
        )


def test_sync_program_scope_rejects_discovery_scope(tmp_path: Path) -> None:
    repo, spec = _repo(tmp_path)
    spec.write_text(
        spec.read_text().replace(
            "  state:\n",
            "  jurisdictions:\n",
        )
    )

    with pytest.raises(ProgramScopeError, match="does not contain RuleSpec imports"):
        sync_program_scope(
            repo=repo,
            program_spec=spec.relative_to(repo),
            scope="jurisdictions",
            add=["us-sc"],
        )
