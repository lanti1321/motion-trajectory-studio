from __future__ import annotations

from studio.cli import main
from studio.core.project import ProjectDocument
from studio.io.trajectory_io import load_trajectory


def test_cli_export_runs_validation_and_atomic_export(
    tmp_path, capsys
) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "time,q0\n0,0\n.01,.01\n.02,.02\n.03,.03\n",
        encoding="utf-8",
    )
    project = ProjectDocument(
        "demo", trajectory_path=str(source.resolve())
    )
    project_path = project.save(tmp_path / "demo.motionproj")
    output = tmp_path / "output.csv"

    assert main(["export", str(project_path), str(output)]) == 0
    exported = load_trajectory(output)
    assert exported.frame_count == 4
    assert str(output.resolve()) in capsys.readouterr().out


def test_cli_export_refuses_source_overwrite(tmp_path, capsys) -> None:
    source = tmp_path / "source.csv"
    original = "time,q0\n0,0\n.01,.01\n.02,.02\n"
    source.write_text(original, encoding="utf-8")
    project = ProjectDocument(
        "demo", trajectory_path=str(source.resolve())
    )
    project_path = project.save(tmp_path / "demo.motionproj")

    assert main(["export", str(project_path), str(source)]) == 1
    assert source.read_text(encoding="utf-8") == original
    assert "source trajectory" in capsys.readouterr().err
