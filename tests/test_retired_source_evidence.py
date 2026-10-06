"""Admission diagnostics survive artifact packaging without following symlinks."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.retired_source_evidence import (
    REPORT_NAME,
    package_retired_source_evidence,
    write_retired_source_evidence,
)


def test_report_survives_success_packaging_with_exact_checksum(tmp_path):
    generated = tmp_path / "generated"
    artifact = tmp_path / "artifact"
    evidence = {
        "contract": "retired-source-containment/v1",
        "sources": [{"row": "child"}],
    }
    write_retired_source_evidence(generated / "target", evidence)
    inventory = package_retired_source_evidence(generated, "target", artifact)
    raw = (artifact / inventory["path"]).read_bytes()
    assert json.loads(raw) == evidence
    assert hashlib.sha256(raw).hexdigest() == inventory["sha256"]
    workflow = yaml.safe_load(
        Path(".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    package = next(
        step["run"]
        for step in steps
        if step.get("name") == "Package exact generated changes"
    )
    assert "package_retired_source_evidence(" in package
    assert '"source_admission": source_admission' in package


def test_no_report_for_normal_replacements(tmp_path):
    assert (
        package_retired_source_evidence(tmp_path, "target", tmp_path / "artifact")
        is None
    )


def test_report_symlink_is_rejected(tmp_path):
    lane = tmp_path / "generated" / "target"
    lane.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (lane / REPORT_NAME).symlink_to(outside)
    with pytest.raises(ValueError):
        package_retired_source_evidence(lane.parent, "target", tmp_path / "artifact")
