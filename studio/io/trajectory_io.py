from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from studio.core.trajectory import TrajectoryData, infer_frequency

TIME_NAMES = {"time", "times", "t", "timestamp", "seconds", "sec"}
NPZ_RESERVED_NAMES = TIME_NAMES | {
    "positions",
    "q",
    "joint_names",
    "hz",
    "frequency_hz",
}
VELOCITY_PATTERN = re.compile(r"^(?:dq|d_|velocity[_:]?|vel[_:]?)(.+)$", re.IGNORECASE)
EFFORT_PATTERN = re.compile(r"^(?:tau|effort[_:]?|torque[_:]?)(.+)$", re.IGNORECASE)
POSITION_PATTERN = re.compile(
    r"^(?:q\d+|position[_:]?.+|joint[_:]?.+|lh_.+|rh_.+)$",
    re.IGNORECASE,
)


def load_trajectory(path: str | Path, fallback_hz: float | None = None) -> TrajectoryData:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return load_csv(source, fallback_hz)
    if suffix == ".npz":
        return load_npz(source, fallback_hz)
    raise ValueError(f"unsupported trajectory format: {source.suffix}")


def load_csv(path: str | Path, fallback_hz: float | None = None) -> TrajectoryData:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError("CSV has no header")
    names = _unique_names([name.strip() or f"column_{index}" for index, name in enumerate(header)])
    values = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if values.shape[1] != len(names):
        raise ValueError("CSV header and data column counts do not match")
    time_index = next((i for i, name in enumerate(names) if name.lower() in TIME_NAMES), None)
    if time_index is None:
        frequency = infer_frequency(None, fallback_hz)
        times = np.arange(len(values), dtype=np.float64) / frequency.hz
    else:
        times = values[:, time_index]
        frequency = infer_frequency(times)
    channels = {
        name: values[:, index].copy()
        for index, name in enumerate(names)
        if index != time_index
    }
    positions, velocities, efforts = classify_channels(list(channels))
    metadata = {
        "source_path": str(source),
        "format": "csv",
        "frequency_hz": frequency.hz,
        "frequency_irregular": frequency.irregular,
        "frequency_max_jitter": frequency.max_jitter,
    }
    return TrajectoryData(times, channels, positions, velocities, efforts, metadata)


def load_npz(path: str | Path, fallback_hz: float | None = None) -> TrajectoryData:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        keys = list(payload.keys())
        time_key = next((key for key in keys if key.lower() in TIME_NAMES), None)
        matrix_key = next((key for key in ("positions", "q") if key in payload), None)
        channels: dict[str, np.ndarray] = {}
        position_names: list[str] = []
        if matrix_key:
            matrix = np.asarray(payload[matrix_key], dtype=np.float64)
            if matrix.ndim != 2:
                raise ValueError(f"NPZ key {matrix_key!r} must be a 2-D array")
            if "joint_names" in payload:
                joint_names = [str(name) for name in payload["joint_names"].tolist()]
            else:
                joint_names = [f"q{index}" for index in range(matrix.shape[1])]
            if len(joint_names) != matrix.shape[1]:
                raise ValueError("joint_names length does not match position matrix width")
            for index, name in enumerate(joint_names):
                channels[name] = matrix[:, index]
            position_names = joint_names
        for key in keys:
            if key in {time_key, matrix_key, "joint_names", "hz", "frequency_hz"}:
                continue
            value = np.asarray(payload[key])
            if value.ndim == 1 and np.issubdtype(value.dtype, np.number):
                channels[key] = value.astype(np.float64)
        if not channels:
            raise ValueError("NPZ contains no usable trajectory channels")
        frame_count = len(next(iter(channels.values())))
        if time_key:
            times = np.asarray(payload[time_key], dtype=np.float64)
            frequency = infer_frequency(times)
        else:
            metadata_hz = None
            for key in ("hz", "frequency_hz"):
                if key in payload:
                    metadata_hz = float(np.asarray(payload[key]).reshape(-1)[0])
                    break
            frequency = infer_frequency(None, metadata_hz or fallback_hz, source="npz metadata")
            times = np.arange(frame_count, dtype=np.float64) / frequency.hz
    classified, velocities, efforts = classify_channels(list(channels))
    positions = position_names or classified
    return TrajectoryData(
        times,
        channels,
        positions,
        velocities,
        efforts,
        {
            "source_path": str(source),
            "format": "npz",
            "frequency_hz": frequency.hz,
            "frequency_irregular": frequency.irregular,
        },
    )


def classify_channels(
    channel_names: list[str],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    positions: list[str] = []
    velocity_candidates: dict[str, str] = {}
    effort_candidates: dict[str, str] = {}
    for name in channel_names:
        velocity = VELOCITY_PATTERN.match(name)
        effort = EFFORT_PATTERN.match(name)
        if velocity:
            velocity_candidates[_normalize_base(velocity.group(1))] = name
        elif effort:
            effort_candidates[_normalize_base(effort.group(1))] = name
        elif POSITION_PATTERN.match(name):
            positions.append(name)
    if not positions:
        excluded = set(velocity_candidates.values()) | set(effort_candidates.values())
        positions = [name for name in channel_names if name not in excluded]
    velocities: dict[str, str] = {}
    efforts: dict[str, str] = {}
    for position in positions:
        base = _normalize_base(position)
        if base in velocity_candidates:
            velocities[position] = velocity_candidates[base]
        if base in effort_candidates:
            efforts[position] = effort_candidates[base]
    return positions, velocities, efforts


def export_trajectory(
    trajectory: TrajectoryData,
    path: str | Path,
    target_hz: float | None = None,
    recompute_velocities: bool = False,
    allow_source_overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    source_path = trajectory.metadata.get("source_path")
    if (
        not allow_source_overwrite
        and source_path
        and destination == Path(str(source_path)).expanduser().resolve()
    ):
        raise ValueError("refusing to overwrite the current source trajectory")
    data = trajectory.resample(target_hz) if target_hz else trajectory.copy()
    if recompute_velocities:
        data.recompute_velocities()
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix not in {".csv", ".npz"}:
        raise ValueError("export path must end in .csv or .npz")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            if suffix == ".csv":
                names = list(data.channels)
                matrix = np.column_stack(
                    [data.times]
                    + [data.channels[name] for name in names]
                )
                np.savetxt(
                    handle,
                    matrix,
                    delimiter=",",
                    header=",".join(["time"] + names),
                    comments="",
                    fmt="%.17g",
                )
            else:
                reserved = sorted(
                    set(data.channels) & NPZ_RESERVED_NAMES
                )
                if reserved:
                    raise ValueError(
                        "NPZ channel names conflict with reserved metadata: "
                        + ", ".join(reserved)
                    )
                payload: dict[str, object] = {
                    "times": data.times,
                    "frequency_hz": data.frequency.hz,
                }
                payload.update(data.channels)
                np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _normalize_base(name: str) -> str:
    return re.sub(r"^(?:q|joint[_:]?)", "", name.strip().lower())


def _unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        count = counts.get(name, 0)
        counts[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count + 1}")
    return result
