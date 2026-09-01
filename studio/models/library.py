from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import xml.etree.ElementTree as ET


MODEL_SUFFIXES = {".xml", ".mjcf", ".urdf"}


@dataclass(frozen=True)
class ModelLibraryEntry:
    name: str
    folder: Path
    entry_file: Path | None
    candidates: tuple[Path, ...] = ()
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def loadable(self) -> bool:
        return self.entry_file is not None and self.error is None

    @property
    def format(self) -> str | None:
        if self.entry_file is None:
            return None
        return "urdf" if self.entry_file.suffix.lower() == ".urdf" else "mjcf"


class ModelLibrary:
    """Discovers one robot model per immediate child directory."""

    def __init__(self, root: str | Path | None = None) -> None:
        default_root = Path(__file__).resolve().parents[2] / "model_library"
        self.root = Path(root).expanduser().resolve() if root else default_root
        self.root.mkdir(parents=True, exist_ok=True)

    def scan(self) -> list[ModelLibraryEntry]:
        return [
            self._inspect_folder(folder)
            for folder in sorted(self.root.iterdir(), key=lambda item: item.name.casefold())
            if folder.is_dir() and not folder.name.startswith(".")
        ]

    def _inspect_folder(self, folder: Path) -> ModelLibraryEntry:
        candidates = tuple(
            sorted(
                (
                    path
                    for path in folder.rglob("*")
                    if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
                ),
                key=lambda path: path.relative_to(folder).as_posix().casefold(),
            )
        )
        manifest = folder / "model.json"
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                entry_value = payload.get("entry")
                if not isinstance(entry_value, str) or not entry_value.strip():
                    raise ValueError("model.json requires a non-empty string field: entry")
                entry = (folder / entry_value).resolve()
                if folder not in entry.parents or not entry.is_file():
                    raise ValueError(f"entry does not exist inside model folder: {entry_value}")
                if entry.suffix.lower() not in MODEL_SUFFIXES:
                    raise ValueError("entry must end in .xml, .mjcf, or .urdf")
                metadata = {
                    str(key): value
                    for key, value in payload.items()
                    if key != "entry"
                }
                return ModelLibraryEntry(
                    folder.name, folder, entry, candidates, metadata=metadata
                )
            except Exception as error:
                return ModelLibraryEntry(
                    folder.name, folder, None, candidates, f"invalid model.json: {error}"
                )
        if not candidates:
            return ModelLibraryEntry(
                folder.name,
                folder,
                None,
                (),
                "no .xml, .mjcf, or .urdf model file found",
            )
        return ModelLibraryEntry(
            folder.name,
            folder,
            self._choose_entry(folder, candidates),
            candidates,
        )

    @staticmethod
    def _choose_entry(folder: Path, candidates: tuple[Path, ...]) -> Path:
        stem = folder.name.casefold()
        priorities = (
            f"{stem}.xml",
            f"{stem}.mjcf",
            f"{stem}.urdf",
            "model.xml",
            "model.mjcf",
            "model.urdf",
            "scene.xml",
            "scene.mjcf",
            "robot.urdf",
        )
        by_name: dict[str, list[Path]] = {}
        for candidate in candidates:
            by_name.setdefault(candidate.name.casefold(), []).append(candidate)
        for preferred in priorities:
            matches = by_name.get(preferred, [])
            if len(matches) == 1:
                return matches[0]
        if len(candidates) == 1:
            return candidates[0]
        top_level = [path for path in candidates if path.parent == folder]
        if len(top_level) == 1:
            return top_level[0]
        valid_roots = [path for path in candidates if _looks_like_model(path)]
        if len(valid_roots) == 1:
            return valid_roots[0]
        return (top_level or valid_roots or list(candidates))[0]


def _looks_like_model(path: Path) -> bool:
    if path.suffix.lower() == ".urdf":
        return True
    try:
        _event, root = next(iter(ET.iterparse(path, events=("start",))))
        return root.tag.rsplit("}", 1)[-1] == "mujoco"
    except (ET.ParseError, OSError, StopIteration):
        return False
