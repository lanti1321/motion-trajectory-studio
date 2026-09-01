from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tarfile

from studio.core.project import ProjectDocument


def verify_project_portability(project_path: str | Path) -> list[str]:
    project = ProjectDocument.load(project_path)
    problems: list[str] = []
    for label, value in (
        ("model", project.model_path),
        ("trajectory", project.trajectory_path),
    ):
        if not value:
            problems.append(f"{label} path is not configured")
            continue
        raw = Path(value)
        if raw.is_absolute():
            problems.append(f"{label} path is absolute: {raw}")
        resolved = project.resolve(value)
        if resolved is None or not resolved.exists():
            problems.append(f"{label} asset is missing: {value}")
    for index, operation in enumerate(project.operations):
        if operation.get("type") not in {"splice", "splice_at"}:
            continue
        value = operation.get("path")
        if not value:
            problems.append(f"splice operation {index} has no trajectory path")
            continue
        raw = Path(str(value))
        if raw.is_absolute():
            problems.append(f"splice operation {index} path is absolute: {raw}")
        resolved = project.resolve(str(value))
        if resolved is None or not resolved.exists():
            problems.append(f"splice operation {index} asset is missing: {value}")
    return problems


def pack_project(project_path: str | Path, output_path: str | Path) -> Path:
    source = Path(project_path).expanduser().resolve()
    problems = verify_project_portability(source)
    if problems:
        raise ValueError("project is not portable:\n" + "\n".join(problems))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = source.parent
    manifest = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path != destination
    }
    manifest_path = root / "bundle-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        with tarfile.open(destination, "w:gz") as archive:
            for path in root.rglob("*"):
                if path == destination or ".venv" in path.parts or "__pycache__" in path.parts:
                    continue
                archive.add(path, arcname=Path(root.name) / path.relative_to(root))
    finally:
        manifest_path.unlink(missing_ok=True)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
