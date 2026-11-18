"""Exercise the actual emitted broker, without a signer or fake Git."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.provision_verification_supervisor import _install_trusted_git_wrapper


@pytest.fixture
def broker(tmp_path):
    root = tmp_path.resolve()
    tools = root / "tools"
    tools.mkdir()
    native = Path(shutil.which("git")).resolve()
    wrapper = _install_trusted_git_wrapper(
        tools, Path(sys.executable).resolve(), native
    )
    source = root / "source"
    subprocess.run([str(native), "init", "-q", str(source)], check=True)
    (source / "policy.yaml").write_text("actual staged content\n")
    subprocess.run([str(native), "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            str(native),
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    snapshot_parent = root / "axiom-execution-snapshot-test"
    snapshot_parent.mkdir(mode=0o700)
    return wrapper, native, source, snapshot_parent / "rulespec-ca"


def invoke(broker, *args, env=None):
    return subprocess.run(
        [str(broker[0]), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def clone_args(source, destination):
    return [
        "clone",
        "--no-local",
        "--no-hardlinks",
        "--no-checkout",
        str(source),
        str(destination),
    ]


def test_exact_compound_identity_and_data_only_clone(broker):
    _, native, source, destination = broker
    identity = invoke(broker, "-C", source, "rev-parse", "--show-toplevel", "HEAD")
    assert identity.returncode == 0, identity.stderr
    top, head = identity.stdout.splitlines()
    assert top == str(source)
    result = invoke(broker, *clone_args(source, destination))
    assert result.returncode == 0, result.stderr
    actual = subprocess.check_output(
        [str(native), "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == head
    assert {p.name for p in destination.iterdir()} == {".git"}
    assert not (destination / ".git/objects/info/alternates").exists()
    assert not (destination / ".git/commondir").exists()
    assert not any(p.is_symlink() for p in destination.rglob("*"))
    assert not list(destination.parent.glob("axiom-sanitized-git-*"))


@pytest.mark.parametrize(
    "change",
    [
        "remote",
        "scp",
        "relative",
        "extra",
        "missing",
        "checkout",
        "uploadpack",
        "existing",
        "symlink_source",
        "symlink_destination_parent",
        "gitfile",
        "alternates",
        "commondir",
        "metadata_symlink",
        "format1",
        "extensions",
    ],
)
def test_snapshot_clone_refuses_unsupported_paths_flags_and_formats(broker, change):
    _, _, source, destination = broker
    args = clone_args(source, destination)
    if change == "remote":
        args[-2] = "https://example.invalid/repo.git"
    elif change == "scp":
        args[-2] = "user@example.invalid:repo"
    elif change == "relative":
        args[-2] = "source"
    elif change == "extra":
        args.insert(1, "--bare")
    elif change == "missing":
        args.remove("--no-hardlinks")
    elif change == "checkout":
        args.remove("--no-checkout")
    elif change == "uploadpack":
        args.insert(1, "--upload-pack=evil")
    elif change == "existing":
        destination.mkdir()
    elif change == "symlink_source":
        link = source.parent / "alias"
        link.symlink_to(source, target_is_directory=True)
        args[-2] = str(link)
    elif change == "symlink_destination_parent":
        link = source.parent / "alias"
        link.symlink_to(source.parent, target_is_directory=True)
        args[-1] = str(link / "snapshot")
    elif change == "gitfile":
        shutil.rmtree(source / ".git")
        (source / ".git").write_text("gitdir: /tmp/other\n")
    elif change == "alternates":
        (source / ".git/objects/info/alternates").write_text("/tmp/other\n")
    elif change == "commondir":
        (source / ".git/commondir").write_text("/tmp/other\n")
    elif change == "metadata_symlink":
        (source / ".git/hooks/link").symlink_to(source / "policy.yaml")
    elif change == "format1":
        p = source / ".git/config"
        p.write_text(
            p.read_text().replace(
                "repositoryformatversion = 0", "repositoryformatversion = 1"
            )
        )
    elif change == "extensions":
        with (source / ".git/config").open("a") as f:
            f.write("\n[extensions]\nobjectFormat = sha256\n")
    result = invoke(broker, *args)
    assert result.returncode == 126, (change, result.stdout, result.stderr)


@pytest.mark.parametrize(
    "args",
    [
        ["--show-toplevel", "HEAD", "--git-dir"],
        ["HEAD", "--show-toplevel"],
        ["--show-toplevel", "main"],
    ],
)
def test_compound_query_does_not_allow_other_combinations(broker, args):
    result = invoke(broker, "-C", broker[2], "rev-parse", *args)
    assert result.returncode == 126


def test_hostile_config_includes_hooks_templates_and_environment_are_inert(broker):
    _, native, source, destination = broker
    marker = source.parent / "EXECUTED"
    hook = source.parent / "hook"
    hook.write_text("#!/bin/sh\ntouch " + str(marker) + "\nexit 99\n")
    hook.chmod(0o755)
    template = source.parent / "template"
    (template / "hooks").mkdir(parents=True)
    shutil.copyfile(hook, template / "hooks/post-checkout")
    (template / "hooks/post-checkout").chmod(0o755)
    include = source.parent / "evil.config"
    include.write_text(
        "[uploadpack]\npackObjectsHook = "
        + str(hook)
        + "\n[core]\nfsmonitor = "
        + str(hook)
        + "\n"
    )
    with (source / ".git/config").open("a") as f:
        f.write(
            f"\n[include]\npath = {include}\n[uploadpack]\npackObjectsHook = {hook}\n[core]\nhooksPath = {template / 'hooks'}\n"
        )
    # ConfigParser deliberately rejects duplicate sections rather than accidentally
    # interpreting a Git format it cannot safely classify. Keep this positive valid.
    config = source / ".git/config"
    config.write_text(
        config.read_text().replace(
            "\n[core]\nhooksPath = " + str(template / "hooks"),
            "\n[init]\ntemplateDir = " + str(template),
        )
    )
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(include),
        "GIT_TEMPLATE_DIR": str(template),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "uploadpack.packObjectsHook",
        "GIT_CONFIG_VALUE_0": str(hook),
        "GIT_SSH_COMMAND": str(hook),
    }
    result = invoke(broker, *clone_args(source, destination), env=env)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert not (destination / ".git/hooks/post-checkout").exists()
    assert not list(destination.parent.glob("axiom-sanitized-git-*"))
    assert subprocess.check_output(
        [str(native), "-C", str(destination), "rev-parse", "HEAD"]
    ) == subprocess.check_output([str(native), "-C", str(source), "rev-parse", "HEAD"])


@pytest.mark.parametrize(
    "change",
    [
        "cwd_prefix",
        "double_slash",
        "dot_segment",
        "wrong_parent",
        "public_parent",
        "nonempty_parent",
        "source_git_symlink",
        "object_tree_symlink",
        "http_alternates",
        "missing_config",
        "malformed_config",
    ],
)
def test_closed_snapshot_parent_and_metadata_contract(broker, change):
    _, _, source, destination = broker
    args = clone_args(source, destination)
    if change == "cwd_prefix":
        args = ["-C", str(source), *args]
    elif change == "double_slash":
        args[-2] = str(source.parent) + "//" + source.name
    elif change == "dot_segment":
        args[-2] = str(source.parent) + "/./" + source.name
    elif change == "wrong_parent":
        args[-1] = str(source.parent / "destination")
    elif change == "public_parent":
        destination.parent.chmod(0o755)
    elif change == "nonempty_parent":
        (destination.parent / "existing").write_text("occupied")
    elif change == "source_git_symlink":
        moved = source.parent / "git-data"
        (source / ".git").rename(moved)
        (source / ".git").symlink_to(moved, target_is_directory=True)
    elif change == "object_tree_symlink":
        moved = source.parent / "objects"
        (source / ".git/objects").rename(moved)
        (source / ".git/objects").symlink_to(moved, target_is_directory=True)
    elif change == "http_alternates":
        (source / ".git/objects/info/http-alternates").write_text(
            "https://example.invalid/objects"
        )
    elif change == "missing_config":
        (source / ".git/config").unlink()
    elif change == "malformed_config":
        (source / ".git/config").write_text("not a config")
    result = invoke(broker, *args)
    assert result.returncode == 126, result.stderr
    assert not destination.exists()
    assert not list(destination.parent.glob("axiom-sanitized-git-*"))


def test_failed_native_clone_cleans_owned_destination(broker):
    source, destination = broker[2:]
    (source / ".git/HEAD").write_text("1" * 40 + "\n")
    result = invoke(broker, *clone_args(source, destination))
    assert result.returncode == 126, result.stderr
    assert not destination.exists()
    assert not list(destination.parent.glob("axiom-sanitized-git-*"))


def test_directory_swap_to_symlink_refuses_before_metadata_copy(broker, monkeypatch):
    wrapper, _, source, destination = broker
    native_open = os.open
    swapped = False

    def swap(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "objects" and flags & os.O_DIRECTORY and not swapped:
            swapped = True
            moved = source.parent / "moved-objects"
            (source / ".git/objects").rename(moved)
            (source / ".git/objects").symlink_to(moved, target_is_directory=True)
        return native_open(path, flags, *args, **kwargs)

    # Execute the actual emitted wrapper with only this deterministic filesystem
    # race injection. Git is not mocked; no capture or validation is fabricated.
    monkeypatch.setattr(os, "open", swap)
    monkeypatch.setattr(sys, "argv", [str(wrapper), *clone_args(source, destination)])
    original_environment = dict(os.environ)
    try:
        with pytest.raises(SystemExit) as stopped:
            exec(
                compile(wrapper.read_text(), str(wrapper), "exec"),
                {"__name__": "__main__"},
            )
        assert stopped.value.code == 126
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
    assert swapped
    assert not destination.exists()
    assert not list(destination.parent.glob("axiom-sanitized-git-*"))


@pytest.mark.parametrize("filename", ["alternates", "http-alternates"])
def test_late_alternate_object_store_refused_during_copy(broker, monkeypatch, filename):
    wrapper, _, source, destination = broker
    native_walk = os.walk
    injected = False

    def walk(path, *args, **kwargs):
        nonlocal injected
        if Path(path) == source / ".git/objects":
            injected = True
            (source / ".git/objects/info" / filename).write_text("/external/store\n")
        return native_walk(path, *args, **kwargs)

    monkeypatch.setattr(os, "walk", walk)
    monkeypatch.setattr(sys, "argv", [str(wrapper), *clone_args(source, destination)])
    original_environment = dict(os.environ)
    try:
        with pytest.raises(SystemExit) as stopped:
            exec(
                compile(wrapper.read_text(), str(wrapper), "exec"),
                {"__name__": "__main__"},
            )
        assert stopped.value.code == 126
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
    assert injected
    assert not destination.exists()
    assert not list(destination.parent.glob("axiom-sanitized-git-*"))


def test_snapshot_never_reads_source_config_include(broker):
    _, native, source, destination = broker
    expected = subprocess.check_output(
        [str(native), "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    include = source.parent / "unparseable-included-config"
    include.write_text("[this is not valid git config\n")
    with (source / ".git/config").open("a") as stream:
        stream.write(f"\n[include]\npath = {include}\n")
    # Non-vacuity: native Git reading this source actually refuses its include.
    refused = subprocess.run(
        [str(native), "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "bad config" in refused.stderr
    result = invoke(broker, *clone_args(source, destination))
    assert result.returncode == 0, result.stderr
    assert (
        subprocess.check_output(
            [str(native), "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        == expected
    )


def test_real_engine_overlay_snapshot_through_actual_broker(
    tmp_path, monkeypatch, real_engine
):
    from axiom_encode.harness import evals
    from tests.test_execution_overlay_origin import (
        test_real_engine_overlay_executes_only_import_closure as exercise_overlay,
    )

    tools = tmp_path.resolve() / "actual-broker"
    tools.mkdir()
    _install_trusted_git_wrapper(
        tools, Path(sys.executable).resolve(), Path(shutil.which("git")).resolve()
    )
    evaluate = evals.evaluate_artifact

    def through_broker(*args, **kwargs):
        with monkeypatch.context() as context:
            context.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
            return evaluate(*args, **kwargs)

    monkeypatch.setattr(evals, "evaluate_artifact", through_broker)
    # Reuse the actual changed-overlay vs base fixture: expected12, exact three
    # imported closure files, unrelated valid YAML excluded, no compiler/CI mocks.
    exercise_overlay(tmp_path, monkeypatch, real_engine)
