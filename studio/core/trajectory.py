from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class FrequencyInfo:
    hz: float
    median_dt: float
    max_jitter: float
    irregular: bool
    source: str


@dataclass
class TrajectoryData:
    times: np.ndarray
    channels: dict[str, np.ndarray]
    position_channels: list[str]
    velocity_channels: dict[str, str] = field(default_factory=dict)
    effort_channels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        if self.times.ndim != 1 or not len(self.times):
            raise ValueError("trajectory times must be a non-empty 1-D array")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("trajectory times contain NaN or infinity")
        if len(self.times) > 1 and np.any(np.diff(self.times) <= 0):
            raise ValueError("trajectory times must be strictly increasing")
        self.times = self.times - self.times[0]
        normalized: dict[str, np.ndarray] = {}
        for name, values in self.channels.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != self.times.shape:
                raise ValueError(f"channel {name!r} has {len(array)} values, expected {len(self.times)}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"channel {name!r} contains NaN or infinity")
            normalized[str(name)] = array
        self.channels = normalized
        missing = [name for name in self.position_channels if name not in self.channels]
        if missing:
            raise ValueError(f"position channels are missing: {missing}")

    @property
    def frame_count(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def frequency(self) -> FrequencyInfo:
        return infer_frequency(self.times, source="time")

    def copy(self) -> "TrajectoryData":
        return TrajectoryData(
            self.times.copy(),
            {name: values.copy() for name, values in self.channels.items()},
            list(self.position_channels),
            dict(self.velocity_channels),
            dict(self.effort_channels),
            dict(self.metadata),
        )

    def matrix(self, channel_names: Iterable[str]) -> np.ndarray:
        names = list(channel_names)
        if not names:
            return np.empty((self.frame_count, 0), dtype=np.float64)
        return np.column_stack([self.channels[name] for name in names])

    def nearest_frame(self, seconds: float) -> int:
        index = int(np.searchsorted(self.times, seconds, side="left"))
        if index <= 0:
            return 0
        if index >= self.frame_count:
            return self.frame_count - 1
        return index - 1 if seconds - self.times[index - 1] <= self.times[index] - seconds else index

    def resample(self, target_hz: float, method: str = "cubic") -> "TrajectoryData":
        if target_hz <= 0:
            raise ValueError("target_hz must be positive")
        count = max(1, int(round(self.duration * target_hz)) + 1)
        new_times = np.arange(count, dtype=np.float64) / target_hz
        new_times[-1] = self.duration
        channels: dict[str, np.ndarray] = {}
        for name, values in self.channels.items():
            if self.frame_count == 1:
                channels[name] = np.full(count, values[0])
            elif method == "linear" or self.frame_count < 4:
                channels[name] = np.interp(new_times, self.times, values)
            elif method == "cubic":
                channels[name] = CubicSpline(self.times, values)(new_times)
            else:
                raise ValueError(f"unsupported interpolation method: {method}")
        result = TrajectoryData(
            new_times,
            channels,
            list(self.position_channels),
            dict(self.velocity_channels),
            dict(self.effort_channels),
            dict(self.metadata),
        )
        result.metadata["resampled_hz"] = target_hz
        return result

    def recompute_velocities(self) -> None:
        for position_name, velocity_name in self.velocity_channels.items():
            if position_name in self.channels and velocity_name in self.channels:
                if self.frame_count < 2:
                    self.channels[velocity_name] = np.zeros(self.frame_count)
                else:
                    self.channels[velocity_name] = np.gradient(
                        self.channels[position_name],
                        self.times,
                        edge_order=min(2, self.frame_count - 1),
                    )


def infer_frequency(
    times: np.ndarray | None,
    fallback_hz: float | None = None,
    source: str = "time",
    jitter_tolerance: float = 0.02,
) -> FrequencyInfo:
    if times is None or len(times) < 2:
        if fallback_hz is None or fallback_hz <= 0:
            raise ValueError("trajectory has no usable time information; provide a control frequency")
        dt = 1.0 / fallback_hz
        return FrequencyInfo(float(fallback_hz), dt, 0.0, False, "manual")
    differences = np.diff(np.asarray(times, dtype=np.float64))
    if np.any(differences <= 0):
        raise ValueError("time values must be strictly increasing")
    median_dt = float(np.median(differences))
    max_jitter = float(np.max(np.abs(differences - median_dt)))
    irregular = max_jitter > jitter_tolerance * median_dt
    return FrequencyInfo(1.0 / median_dt, median_dt, max_jitter, irregular, source)
