from __future__ import annotations

import json

from studio.models.library import ModelLibrary


def test_library_uses_folder_name_and_discovers_new_models(tmp_path) -> None:
    library = ModelLibrary(tmp_path / "models")
    assert library.scan() == []

    alpha = library.root / "Alpha Robot"
    alpha.mkdir()
    (alpha / "Alpha Robot.xml").write_text(
        "<mujoco><worldbody/></mujoco>", encoding="utf-8"
    )
    entries = library.scan()
    assert [entry.name for entry in entries] == ["Alpha Robot"]
    assert entries[0].entry_file == alpha / "Alpha Robot.xml"
    assert entries[0].loadable

    beta = library.root / "Beta"
    (beta / "description").mkdir(parents=True)
    (beta / "description" / "main.urdf").write_text(
        "<robot name='beta'/>", encoding="utf-8"
    )
    (beta / "preview.xml").write_text(
        "<mujoco><worldbody/></mujoco>", encoding="utf-8"
    )
    (beta / "model.json").write_text(
        json.dumps(
            {
                "entry": "description/main.urdf",
                "default_mapping": {"q0": {"joint": "elbow", "scale": -1}},
            }
        ),
        encoding="utf-8",
    )
    entries = library.scan()
    beta_entry = next(entry for entry in entries if entry.name == "Beta")
    assert beta_entry.entry_file == beta / "description" / "main.urdf"
    assert beta_entry.format == "urdf"
    assert "default_mapping" in beta_entry.metadata


def test_library_reports_invalid_or_empty_folders(tmp_path) -> None:
    library = ModelLibrary(tmp_path / "models")
    empty = library.root / "Empty"
    empty.mkdir()
    escaped = library.root / "Escaped"
    escaped.mkdir()
    (escaped / "model.json").write_text(
        json.dumps({"entry": "../outside.xml"}), encoding="utf-8"
    )
    entries = {entry.name: entry for entry in library.scan()}
    assert not entries["Empty"].loadable
    assert "no .xml" in str(entries["Empty"].error)
    assert not entries["Escaped"].loadable
    assert "invalid model.json" in str(entries["Escaped"].error)
