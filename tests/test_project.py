from __future__ import annotations

from pathlib import Path
import json
import tarfile
import pytest
import studio.core.project as project_module

from studio.core.project import ProjectDocument, TimelineLabel
from studio.io.bundle import pack_project, verify_project_portability


def test_project_roundtrip_and_relative_assets(tmp_path) -> None:
    source_dir = tmp_path / "source_model"
    source_dir.mkdir()
    model = source_dir / "robot.xml"
    model.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    trajectory = tmp_path / "source.csv"
    trajectory.write_text("time,q0\n0,0\n.1,.1\n", encoding="utf-8")
    project_dir = tmp_path / "portable"
    project_dir.mkdir()
    project_path = project_dir / "demo.motionproj"
    project = ProjectDocument(
        "demo",
        model_adjustments=[
            {"type": "joint_center_distance", "distance_m": 0.422}
        ],
        joint_mapping={"q0": "joint0"},
        mapping_transforms={
            "q0": {"scale": -1.0, "offset": 0.5, "minimum": 0.0, "maximum": 1.0}
        },
        labels=[TimelineLabel("T001", 0.1, "marker")],
        operations=[{"type": "transition", "start": 0, "end": 0.1, "order": 5}],
    )
    project.save(project_path)
    project.model_path = project.import_model_bundle(model)
    project.trajectory_path = project.import_trajectory(trajectory)
    project.save()

    loaded = ProjectDocument.load(project_path)
    assert loaded.resolve(loaded.model_path).is_file()
    assert loaded.resolve(loaded.trajectory_path).is_file()
    assert loaded.labels[0].id == "T001"
    assert loaded.model_adjustments[0]["distance_m"] == 0.422
    assert loaded.mapping_transforms["q0"]["maximum"] == 1.0
    assert not verify_project_portability(project_path)


def test_pack_project(tmp_path) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "assets").mkdir(parents=True)
    (project_dir / "media").mkdir()
    model = project_dir / "assets" / "model.xml"
    trajectory = project_dir / "media" / "motion.csv"
    model.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    trajectory.write_text("time,q0\n0,0\n.1,.1\n", encoding="utf-8")
    project = ProjectDocument(
        "demo", model_path="assets/model.xml", trajectory_path="media/motion.csv"
    )
    project_path = project.save(project_dir / "demo.motionproj")
    archive_path = pack_project(project_path, tmp_path / "bundle.tar.gz")
    assert archive_path.is_file()
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        restored = tmp_path / "restored"
        archive.extractall(restored, filter="data")
    assert any(name.endswith("demo.motionproj") for name in names)
    assert any(name.endswith("bundle-manifest.json") for name in names)
    restored_project = next(restored.rglob("demo.motionproj"))
    assert not verify_project_portability(restored_project)


def test_legacy_output_duration_is_migrated(tmp_path) -> None:
    path = tmp_path / "legacy.motionproj"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "legacy",
                "operations": [
                    {
                        "type": "transition",
                        "start": 1.0,
                        "end": 2.0,
                        "output_duration": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    project = ProjectDocument.load(path)
    assert project.operations[0]["segment_duration"] == 0.5
    assert "output_duration" not in project.operations[0]


def test_failed_portable_directory_copy_keeps_existing_asset(
    tmp_path, monkeypatch
) -> None:
    project_path = tmp_path / "portable" / "demo.motionproj"
    project = ProjectDocument("demo")
    project.save(project_path)
    source = tmp_path / "source_model"
    source.mkdir()
    (source / "robot.xml").write_text("new", encoding="utf-8")
    existing = project_path.parent / "assets" / "models" / source.name
    existing.mkdir(parents=True)
    (existing / "robot.xml").write_text("old", encoding="utf-8")

    def fail_copytree(_source, destination) -> None:
        destination.mkdir()
        (destination / "partial").write_text("partial", encoding="utf-8")
        raise OSError("copy failed")

    monkeypatch.setattr(project_module.shutil, "copytree", fail_copytree)
    with pytest.raises(OSError, match="copy failed"):
        project.import_model_bundle(source / "robot.xml")
    assert (existing / "robot.xml").read_text(encoding="utf-8") == "old"
    assert not list(existing.parent.glob(".*.tmp"))
