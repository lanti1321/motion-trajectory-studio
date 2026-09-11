from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import json
import os
import shutil
import uuid


@dataclass
class TimelineLabel:
    id: str
    time: float
    title: str = ""
    note: str = ""


@dataclass
class TimelineSegment:
    """Named, colored action span on the timeline."""

    id: str
    start: float
    end: float
    title: str = ""
    color: str = "#4fc3f7"


@dataclass
class ProjectDocument:
    name: str
    model_path: str | None = None
    trajectory_path: str | None = None
    model_adjustments: list[dict[str, object]] = field(default_factory=list)
    joint_mapping: dict[str, str] = field(default_factory=dict)
    mapping_transforms: dict[str, dict[str, float]] = field(default_factory=dict)
    joint_groups: dict[str, list[str]] = field(default_factory=dict)
    custom_part_modules: list[dict[str, object]] = field(default_factory=list)
    labels: list[TimelineLabel] = field(default_factory=list)
    segments: list[TimelineSegment] = field(default_factory=list)
    operations: list[dict[str, object]] = field(default_factory=list)
    blocks: list[dict[str, object]] = field(default_factory=list)
    preview_hz: float = 60.0
    version: int = 1
    path: Path | None = field(default=None, repr=False)

    def resolve(self, value: str | None) -> Path | None:
        if not value:
            return None
        candidate = Path(value)
        if candidate.is_absolute() or self.path is None:
            return candidate.expanduser().resolve()
        return (self.path.parent / candidate).resolve()

    def save(self, path: str | Path | None = None) -> Path:
        destination = Path(path).expanduser().resolve() if path else self.path
        if destination is None:
            raise ValueError("project has no save path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "name": self.name,
            "model_path": self.model_path,
            "trajectory_path": self.trajectory_path,
            "model_adjustments": self.model_adjustments,
            "joint_mapping": self.joint_mapping,
            "mapping_transforms": self.mapping_transforms,
            "joint_groups": self.joint_groups,
            "custom_part_modules": self.custom_part_modules,
            "labels": [asdict(label) for label in self.labels],
            "segments": [asdict(segment) for segment in self.segments],
            "operations": self.operations,
            "blocks": self.blocks,
            "preview_hz": self.preview_hz,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self.path = destination
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ProjectDocument":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported project version")
        document = cls(
            name=str(payload.get("name", source.stem)),
            model_path=payload.get("model_path"),
            trajectory_path=payload.get("trajectory_path"),
            model_adjustments=list(payload.get("model_adjustments", [])),
            joint_mapping=dict(payload.get("joint_mapping", {})),
            mapping_transforms={
                str(name): {
                    key: float(transform[key])
                    for key in ("scale", "offset", "minimum", "maximum")
                    if key in transform
                }
                for name, transform in payload.get("mapping_transforms", {}).items()
            },
            joint_groups={
                str(name): [str(value) for value in values]
                for name, values in payload.get("joint_groups", {}).items()
            },
            custom_part_modules=[
                {
                    "id": str(module["id"]),
                    "title": str(module["title"]),
                    "joint_names": [
                        str(value)
                        for value in module.get("joint_names", [])
                    ],
                }
                for module in payload.get("custom_part_modules", [])
                if (
                    isinstance(module, dict)
                    and module.get("id")
                    and module.get("title")
                )
            ],
            labels=[TimelineLabel(**label) for label in payload.get("labels", [])],
            segments=[
                TimelineSegment(**segment)
                for segment in payload.get("segments", [])
            ],
            operations=[
                _migrate_operation(dict(operation))
                for operation in payload.get("operations", [])
            ],
            blocks=[
                dict(block)
                for block in payload.get("blocks", [])
                if isinstance(block, dict)
            ],
            preview_hz=float(payload.get("preview_hz", 60.0)),
        )
        document.path = source
        return document

    def import_portable_asset(self, source: str | Path, category: str) -> str:
        if self.path is None:
            raise ValueError("save the project before importing portable assets")
        source_path = Path(source).expanduser().resolve()
        destination_root = self.path.parent / category
        destination_root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        if source_path.is_dir():
            destination = destination_root / source_path.name
            if source_path == destination.resolve():
                return destination.relative_to(self.path.parent).as_posix()
            temporary = destination.with_name(
                f".{destination.name}.{token}.tmp"
            )
            backup = destination.with_name(
                f".{destination.name}.{token}.backup"
            )
            try:
                shutil.copytree(source_path, temporary)
                if destination.exists():
                    os.replace(destination, backup)
                try:
                    os.replace(temporary, destination)
                except Exception:
                    if backup.exists() and not destination.exists():
                        os.replace(backup, destination)
                    raise
                if backup.exists():
                    shutil.rmtree(backup)
            except Exception:
                if temporary.exists():
                    shutil.rmtree(temporary)
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
        else:
            destination = destination_root / source_path.name
            if source_path == destination.resolve():
                return destination.relative_to(self.path.parent).as_posix()
            temporary = destination.with_name(
                f".{destination.name}.{token}.tmp"
            )
            try:
                shutil.copy2(source_path, temporary)
                os.replace(temporary, destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        return destination.relative_to(self.path.parent).as_posix()

    def import_model_bundle(self, model_path: str | Path) -> str:
        model = Path(model_path).expanduser().resolve()
        # Copy the complete containing directory so relative mesh/include references remain valid.
        relative_directory = self.import_portable_asset(model.parent, "assets/models")
        return (Path(relative_directory) / model.name).as_posix()

    def import_trajectory(self, trajectory_path: str | Path) -> str:
        return self.import_portable_asset(trajectory_path, "media")


def _migrate_operation(
    operation: dict[str, object],
) -> dict[str, object]:
    if (
        "output_duration" in operation
        and "segment_duration" not in operation
    ):
        operation["segment_duration"] = operation.pop("output_duration")
    return operation
