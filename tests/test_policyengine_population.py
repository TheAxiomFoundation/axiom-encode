from __future__ import annotations

from types import SimpleNamespace

import pytest

from axiom_encode.oracles.policyengine import population


def test_local_populace_artifact_prefers_specific_env(monkeypatch, tmp_path):
    artifact = tmp_path / "populace_us_2024.h5"
    artifact.write_text("")
    monkeypatch.setenv("AXIOM_POPULACE_US_2024_H5", str(artifact))

    assert population.local_populace_artifact_path("US", year=2024) == artifact


def test_local_populace_artifact_rejects_missing_env_override(monkeypatch, tmp_path):
    missing = tmp_path / "missing.h5"
    monkeypatch.setenv("AXIOM_POPULACE_US_H5", str(missing))

    with pytest.raises(FileNotFoundError, match="AXIOM_POPULACE_US_H5"):
        population.local_populace_artifact_path("us", year=2024)


def test_load_populace_dataset_exits_for_missing_env_override(monkeypatch, tmp_path):
    missing = tmp_path / "missing.h5"
    monkeypatch.setenv("AXIOM_POPULACE_US_H5", str(missing))

    with pytest.raises(SystemExit, match="AXIOM_POPULACE_US_H5"):
        population.load_populace_dataset(
            "us",
            year=2024,
            command="tax-populace-compare",
        )


def test_local_populace_artifact_reads_huggingface_cache_ref(tmp_path):
    cache_root = tmp_path / "hub"
    repo = cache_root / "datasets--policyengine--populace-us"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    artifact = snapshot / "populace_us_2024.h5"
    artifact.write_text("")
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123\n")

    assert (
        population.local_populace_artifact_path(
            "us",
            year=2024,
            cache_root=cache_root,
        )
        == artifact
    )


def test_load_populace_dataset_uses_local_engine_dataset_without_populace_data(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "populace_us_2024.h5"
    artifact.write_text("")
    monkeypatch.setenv("AXIOM_POPULACE_US_2024_H5", str(artifact))
    imports: list[str] = []

    class FakeDataset:
        def __init__(self, *, file_path: str):
            self.file_path = file_path

    def fake_import_module(name: str) -> SimpleNamespace:
        imports.append(name)
        return SimpleNamespace(USSingleYearDataset=FakeDataset)

    monkeypatch.setattr(population, "import_module", fake_import_module)

    dataset = population.load_populace_dataset(
        "us",
        year=2024,
        command="tax-populace-compare",
    )

    assert dataset.file_path == str(artifact)
    assert imports == ["policyengine_us.data"]
