from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4


def load_queue_runner_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_snap_queue_until_idle.py"
    )
    module_name = f"run_snap_queue_until_idle_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_sha256_paths_ignores_file_names(tmp_path):
    module = load_queue_runner_module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first_left = left / "alpha.txt"
    second_left = left / "beta.txt"
    first_right = right / "renamed-one.txt"
    second_right = right / "renamed-two.txt"
    first_left.write_text("alpha")
    second_left.write_text("beta")
    first_right.write_text("alpha")
    second_right.write_text("beta")

    left_digest = module.sha256_paths([first_left, second_left])
    right_digest = module.sha256_paths([first_right, second_right])

    assert left_digest == right_digest


def test_sync_queue_with_manifests_skips_requeue_on_tracking_version_migration(
    monkeypatch,
):
    module = load_queue_runner_module()
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "done",
                "manifest": "/tmp/demo.yaml",
                "source_file": "/tmp/demo.txt",
                "source_inputs": ["/tmp/demo.txt"],
                "manifest_sha256": "manifest-sha",
                "source_sha256": "old-source-sha",
                "note": "closed fully ready",
            }
        ]
    }
    monkeypatch.setattr(
        module,
        "iter_manifest_queue_candidates",
        lambda: [
            {
                "name": "snap_demo",
                "manifest": "/tmp/demo.yaml",
                "source_file": "/tmp/demo.txt",
                "source_inputs": ["/tmp/demo.txt", "/tmp/demo.meta.yaml"],
                "source_repo": "rac-us-demo",
            }
        ],
    )
    monkeypatch.setattr(module, "sha256_file", lambda path: "manifest-sha")
    monkeypatch.setattr(module, "sha256_paths", lambda paths: "new-source-sha")

    changed, added, retired, refreshed = module.sync_queue_with_manifests(data)

    assert changed is True
    assert added == []
    assert retired == []
    assert refreshed == []
    assert data["items"][0]["status"] == "done"
    assert data["items"][0]["source_tracking_version"] == module.SOURCE_TRACKING_VERSION
    assert data["items"][0]["note"] == "closed fully ready"


def test_sync_queue_with_manifests_requeues_on_source_change_after_tracking_set(
    monkeypatch,
):
    module = load_queue_runner_module()
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "done",
                "manifest": "/tmp/demo.yaml",
                "source_file": "/tmp/demo.txt",
                "source_inputs": ["/tmp/demo.txt"],
                "manifest_sha256": "manifest-sha",
                "source_sha256": "old-source-sha",
                "source_tracking_version": module.SOURCE_TRACKING_VERSION,
                "output_dir": "/tmp/out",
                "archive_path": "/tmp/archive",
                "finished_at": "2026-04-15T00:00:00Z",
                "note": "closed fully ready",
            }
        ]
    }
    monkeypatch.setattr(
        module,
        "iter_manifest_queue_candidates",
        lambda: [
            {
                "name": "snap_demo",
                "manifest": "/tmp/demo.yaml",
                "source_file": "/tmp/demo.txt",
                "source_inputs": ["/tmp/demo.txt", "/tmp/demo.meta.yaml"],
                "source_repo": "rac-us-demo",
            }
        ],
    )
    monkeypatch.setattr(module, "sha256_file", lambda path: "manifest-sha")
    monkeypatch.setattr(module, "sha256_paths", lambda paths: "new-source-sha")

    changed, added, retired, refreshed = module.sync_queue_with_manifests(data)

    assert changed is True
    assert added == []
    assert retired == []
    assert refreshed == ["snap_demo"]
    assert data["items"][0]["status"] == "queued"
    assert data["items"][0]["output_dir"] is None
    assert data["items"][0]["archive_path"] is None
    assert data["items"][0]["finished_at"] is None
    assert data["items"][0]["note"] == "requeued after manifest/source change"


def test_reconcile_ready_output_items_marks_revalidated_blocked_item_done(tmp_path):
    module = load_queue_runner_module()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text('{"all_ready": true}')
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "blocked",
                "output_dir": str(output_dir),
                "note": "eval-suite exited with code 1",
            }
        ]
    }

    changed, reconciled = module.reconcile_ready_output_items(data)

    assert changed is True
    assert reconciled == ["snap_demo"]
    assert data["items"][0]["status"] == "done"
    assert data["items"][0]["source_tracking_version"] == module.SOURCE_TRACKING_VERSION
    assert data["items"][0]["note"] == "closed fully ready after revalidation"


def test_reconcile_stale_running_items_marks_ready_orphan_done(tmp_path):
    module = load_queue_runner_module()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text('{"all_ready": true}')
    data = {
        "items": [
            {
                "name": "snap_demo",
                "status": "running",
                "output_dir": str(output_dir),
                "note": "started with backend `codex`",
            }
        ]
    }

    changed = module.reconcile_stale_running_items(data, [])

    assert changed is True
    assert data["items"][0]["status"] == "done"
    assert data["items"][0]["source_tracking_version"] == module.SOURCE_TRACKING_VERSION
    assert data["items"][0]["note"] == "closed fully ready after orphaned eval completed"
    assert data["event_log"][0]["message"] == (
        "snap_demo was left in `running`, but its output is ready; marked done."
    )
