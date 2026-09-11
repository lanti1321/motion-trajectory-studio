from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from studio.core.project import ProjectDocument
from studio.core.timeline import TimelineEngine
from studio.diagnostics import environment_report
from studio.io.bundle import pack_project, verify_project_portability
from studio.io.trajectory_io import export_trajectory, load_trajectory
from studio.models.adapter import MuJoCoModelAdapter, auto_map_channels
from studio.validation.checks import (
    filter_new_joint_limit_issues,
    sample_collisions,
    validate_model_mapping,
    validate_rendered_operations,
    validate_trajectory,
    validate_velocity_channels,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motion-trajectory-studio")
    subparsers = parser.add_subparsers(dest="command")
    inspect_parser = subparsers.add_parser("inspect", help="inspect a model or trajectory")
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--hz", type=float)
    validate_parser = subparsers.add_parser("validate", help="validate a trajectory/project")
    validate_parser.add_argument("path")
    validate_parser.add_argument("--model")
    validate_parser.add_argument("--hz", type=float)
    export_parser = subparsers.add_parser("export", help="render and export a project")
    export_parser.add_argument("project")
    export_parser.add_argument("output")
    export_parser.add_argument("--hz", type=float)
    export_parser.add_argument("--recompute-velocities", action="store_true")
    export_parser.add_argument("--allow-warnings", action="store_true")
    pack_parser = subparsers.add_parser("pack", help="create a portable project archive")
    pack_parser.add_argument("project")
    pack_parser.add_argument("output")
    subparsers.add_parser("doctor", help="check the local runtime environment")
    subparsers.add_parser("gui", help="launch the desktop application")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {None, "gui"}:
        from studio.ui.main_window import run_gui

        return run_gui()
    if args.command == "inspect":
        return _inspect(args.path, args.hz)
    if args.command == "validate":
        return _validate(args.path, args.model, args.hz)
    if args.command == "export":
        try:
            return _export_project(
                args.project,
                args.output,
                args.hz,
                args.allow_warnings,
            )
        except Exception as error:
            print(f"[error] {error}", file=sys.stderr)
            return 1
    if args.command == "pack":
        print(pack_project(args.project, args.output))
        return 0
    if args.command == "doctor":
        report = environment_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    return 2


def _inspect(path: str, fallback_hz: float | None) -> int:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() in {".xml", ".mjcf", ".urdf"}:
        adapter = MuJoCoModelAdapter(source)
        payload = {
            "format": adapter.info.format,
            "joints": [
                {
                    "name": joint.name,
                    "type": joint.joint_type,
                    "range": [joint.lower, joint.upper],
                }
                for joint in adapter.info.joints
            ],
            "bodies": list(adapter.info.body_names),
        }
    else:
        trajectory = load_trajectory(source, fallback_hz)
        payload = {
            "frames": trajectory.frame_count,
            "duration": trajectory.duration,
            "frequency": trajectory.frequency.__dict__,
            "position_channels": trajectory.position_channels,
            "all_channels": list(trajectory.channels),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _validate(path: str, model_path: str | None, fallback_hz: float | None) -> int:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".motionproj":
        problems = verify_project_portability(source)
        project = ProjectDocument.load(source)
        trajectory_path = project.resolve(project.trajectory_path)
        if trajectory_path is None:
            problems.append("project has no trajectory")
            print("\n".join(problems))
            return 1
        source_trajectory = load_trajectory(trajectory_path, fallback_hz)
        operations = _resolve_project_operations(project)
        trajectory = TimelineEngine(
            source_trajectory,
            operations,
            project.labels,
        ).render()
        model = project.resolve(project.model_path)
        mapping = project.joint_mapping
        transforms = project.mapping_transforms
    else:
        problems = []
        project = None
        trajectory = load_trajectory(source, fallback_hz)
        source_trajectory = trajectory
        operations = []
        model = Path(model_path).resolve() if model_path else None
        mapping = {}
        transforms = {}
    report = validate_trajectory(trajectory)
    velocity_report = validate_velocity_channels(trajectory)
    report.issues.extend(velocity_report.issues)
    report.metrics.update(velocity_report.metrics)
    if project is not None:
        continuity_report = validate_rendered_operations(
            source_trajectory, operations
        )
        report.issues.extend(continuity_report.issues)
        report.metrics.update(continuity_report.metrics)
    if model:
        adapter = MuJoCoModelAdapter(model)
        if project is not None:
            _apply_model_adjustments(adapter, project.model_adjustments)
        if not mapping:
            mapping = auto_map_channels(trajectory.position_channels, adapter.info)
        mapping_report = validate_model_mapping(
            trajectory, adapter, mapping, transforms
        )
        if project is not None:
            source_mapping = validate_model_mapping(
                source_trajectory,
                adapter,
                mapping,
                transforms,
            )
            report.issues.extend(
                filter_new_joint_limit_issues(source_mapping, mapping_report)
            )
        else:
            report.issues.extend(mapping_report.issues)
    for problem in problems:
        print(f"[error] {problem}")
    for issue in report.issues:
        location = f" frame={issue.frame}" if issue.frame is not None else ""
        print(f"[{issue.severity}] {issue.code}: {issue.message}{location}")
    print(
        f"frames={trajectory.frame_count} duration={trajectory.duration:.6f}s "
        f"hz={trajectory.frequency.hz:.6f}"
    )
    return 0 if report.ok and not problems else 1


def _export_project(
    project_path: str,
    output_path: str,
    target_hz: float | None,
    allow_warnings: bool,
) -> int:
    project = ProjectDocument.load(project_path)
    trajectory_path = project.resolve(project.trajectory_path)
    if trajectory_path is None:
        raise ValueError("project has no trajectory")
    source = load_trajectory(trajectory_path)
    operations = _resolve_project_operations(project)
    rendered = TimelineEngine(
        source,
        operations,
        project.labels,
    ).render()
    rendered.recompute_velocities()
    report = validate_trajectory(rendered)
    velocity_report = validate_velocity_channels(rendered)
    report.issues.extend(velocity_report.issues)
    continuity_report = validate_rendered_operations(source, operations)
    report.issues.extend(continuity_report.issues)
    model_path = project.resolve(project.model_path)
    if model_path is not None:
        adapter = MuJoCoModelAdapter(model_path)
        _apply_model_adjustments(adapter, project.model_adjustments)
        source_mapping = validate_model_mapping(
            source,
            adapter,
            project.joint_mapping,
            project.mapping_transforms,
        )
        mapping_report = validate_model_mapping(
            rendered,
            adapter,
            project.joint_mapping,
            project.mapping_transforms,
        )
        report.issues.extend(
            filter_new_joint_limit_issues(source_mapping, mapping_report)
        )
        collision_report = sample_collisions(
            rendered,
            adapter,
            project.joint_mapping,
            transforms=project.mapping_transforms,
        )
        report.issues.extend(collision_report.issues)
    errors = [
        issue for issue in report.issues if issue.severity == "error"
    ]
    warnings = [
        issue for issue in report.issues if issue.severity == "warning"
    ]
    if errors:
        raise ValueError(
            "export validation failed: "
            + "; ".join(issue.message for issue in errors[:20])
        )
    if warnings and not allow_warnings:
        raise ValueError(
            "export validation has warnings (pass --allow-warnings): "
            + "; ".join(issue.message for issue in warnings[:20])
        )
    destination = export_trajectory(
        rendered,
        output_path,
        target_hz,
        recompute_velocities=True,
    )
    print(destination)
    return 0


def _resolve_project_operations(project: ProjectDocument) -> list[dict[str, object]]:
    operations: list[dict[str, object]] = []
    for operation in project.operations:
        item = dict(operation)
        if item.get("type") in {"splice", "splice_at", "insert_module"} and item.get("path"):
            item["path"] = str(project.resolve(str(item["path"])))
        operations.append(item)
    return operations


def _apply_model_adjustments(
    adapter: MuJoCoModelAdapter, adjustments: list[dict[str, object]]
) -> None:
    for adjustment in adjustments:
        if adjustment.get("type") == "joint_center_distance":
            adapter.set_joint_center_distance(
                str(adjustment["joint_a"]),
                str(adjustment["joint_b"]),
                str(adjustment["body_a"]),
                str(adjustment["body_b"]),
                float(adjustment["distance_m"]),
            )
