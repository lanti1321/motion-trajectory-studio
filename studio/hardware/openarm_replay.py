from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import socket
import threading
import numpy as np

from studio.core.trajectory import TrajectoryData


@dataclass(frozen=True)
class MotorTemperatureSample:
    time: float
    mos: tuple[int, ...]
    rotor: tuple[int, ...]
    errors: tuple[int, ...]
    positions: tuple[float, ...] = ()


def is_motor_fault_code(code: int) -> bool:
    """Return whether a Damiao status nibble represents a drive fault."""
    return 8 <= int(code) <= 14


def motor_fault_text(code: int) -> str:
    return {
        8: "过压",
        9: "欠压",
        10: "过流",
        11: "MOS过温",
        12: "转子过温",
        13: "通信丢失",
        14: "过载",
    }.get(int(code), "正常")


def build_load_command(csv_path: str | Path, telemetry_port: int = 0) -> str:
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return f"load|{path}|{int(telemetry_port)}"


def build_dual_replay_clip(
    trajectory: TrajectoryData, start_frame: int, include_effort: bool = False
) -> TrajectoryData:
    """Create the exact interleaved CSV shape expected by CsvReplay (33/49 cells)."""
    if not 0 <= start_frame < trajectory.frame_count:
        raise IndexError("start_frame is outside the trajectory")
    required = [f"q{index}" for index in range(16)]
    missing = [name for name in required if name not in trajectory.channels]
    if missing:
        raise ValueError("dual replay is missing channels: " + ", ".join(missing))
    include_effort = include_effort and all(
        f"tau{index}" in trajectory.channels for index in range(16)
    )
    output: dict[str, np.ndarray] = {}
    velocity_channels: dict[str, str] = {}
    effort_channels: dict[str, str] = {}
    for index, position_name in enumerate(required):
        values = trajectory.channels[position_name]
        velocity_name = f"dq{index}"
        source_velocity_name = trajectory.velocity_channels.get(position_name)
        if (
            source_velocity_name is not None
            and source_velocity_name in trajectory.channels
        ):
            velocity = trajectory.channels[source_velocity_name]
        elif trajectory.frame_count > 1:
            velocity = np.gradient(
                values,
                trajectory.times,
                edge_order=min(2, trajectory.frame_count - 1),
            )
        else:
            velocity = np.zeros(trajectory.frame_count)
        output[position_name] = values[start_frame:].copy()
        output[velocity_name] = velocity[start_frame:].copy()
        velocity_channels[position_name] = velocity_name
        if include_effort:
            effort_name = f"tau{index}"
            output[effort_name] = trajectory.channels[effort_name][start_frame:].copy()
            effort_channels[position_name] = effort_name
    return TrajectoryData(
        trajectory.times[start_frame:] - trajectory.times[start_frame],
        output,
        required,
        velocity_channels,
        effort_channels,
        dict(trajectory.metadata),
    )


class OpenArmReplayClient:
    """UDP client for Motion Studio's independent hardware service."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        command_port: int | None = None,
        status_port: int | None = None,
    ) -> None:
        self.host = host
        self.command_port = command_port or int(
            os.getenv("OPENARM_REPLAY_COMMAND_PORT", "47970")
        )
        self.status_port = status_port or int(
            os.getenv("OPENARM_REPLAY_STATUS_PORT", "47971")
        )
        self._command = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._status: socket.socket | None = None
        self._telemetry: socket.socket | None = None
        self.telemetry_port = 0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def connect_status(self) -> None:
        if self._status is not None:
            return
        status = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        status.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        status.bind((self.host, self.status_port))
        status.setblocking(False)
        self._status = status
        self._start_heartbeat_thread()

    def _start_heartbeat_thread(self) -> None:
        if self._heartbeat_thread is not None:
            return

        def run() -> None:
            while not self._heartbeat_stop.wait(0.25):
                try:
                    self.heartbeat()
                except OSError:
                    # The GUI status poll owns connection/error reporting.  A
                    # transient send failure must not kill the keepalive loop.
                    pass

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=run,
            name="openarm-replay-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def load(self, csv_path: str | Path, telemetry_port: int = 0) -> None:
        message = build_load_command(csv_path, telemetry_port)
        self._command.sendto(message.encode("utf-8"), (self.host, self.command_port))

    def prepare(self, frame: int) -> None:
        if frame < 0:
            raise ValueError("frame must be non-negative")
        self._command.sendto(
            f"prepare|{frame}".encode("utf-8"), (self.host, self.command_port)
        )

    def execute(self) -> None:
        self._command.sendto(b"execute", (self.host, self.command_port))

    def configure_motion_limits(
        self, velocity_rad_s: float, acceleration_rad_s2: float
    ) -> None:
        if velocity_rad_s <= 0 or acceleration_rad_s2 <= 0:
            raise ValueError("motion limits must be positive")
        message = f"limits|{velocity_rad_s:.6f}|{acceleration_rad_s2:.6f}"
        self._command.sendto(message.encode("utf-8"), (self.host, self.command_port))

    def stop(self) -> None:
        self._command.sendto(b"stop", (self.host, self.command_port))

    def open_telemetry(self) -> int:
        if self._telemetry is not None:
            return self.telemetry_port
        telemetry = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        telemetry.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        telemetry.bind((self.host, 0))
        telemetry.setblocking(False)
        self._telemetry = telemetry
        self.telemetry_port = int(telemetry.getsockname()[1])
        return self.telemetry_port

    def poll_temperatures(self) -> list[MotorTemperatureSample]:
        samples: list[MotorTemperatureSample] = []
        if self._telemetry is None:
            return samples
        while True:
            try:
                packet, _address = self._telemetry.recvfrom(4096)
            except BlockingIOError:
                break
            sample = parse_temperature_packet(
                packet.decode("utf-8", errors="replace").strip()
            )
            if sample is not None:
                samples.append(sample)
        return samples

    def shutdown(self) -> None:
        self._command.sendto(b"shutdown", (self.host, self.command_port))

    def heartbeat(self) -> None:
        """Keep an active hardware replay armed; loss stops robot motion."""
        self._command.sendto(b"heartbeat", (self.host, self.command_port))

    def poll_status(self) -> list[str]:
        messages: list[str] = []
        if self._status is None:
            return messages
        while True:
            try:
                packet, _address = self._status.recvfrom(4096)
            except BlockingIOError:
                break
            messages.append(packet.decode("utf-8", errors="replace").strip())
        return messages

    def close(self) -> None:
        self._heartbeat_stop.set()
        heartbeat_thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if heartbeat_thread is not None and heartbeat_thread is not threading.current_thread():
            heartbeat_thread.join(timeout=0.6)
        if self._status is not None:
            self._status.close()
            self._status = None
        if self._telemetry is not None:
            self._telemetry.close()
            self._telemetry = None
            self.telemetry_port = 0
        self._command.close()


def parse_temperature_packet(text: str) -> MotorTemperatureSample | None:
    """Parse the independent controller's extended telemetry packet."""
    parts = text.split(",")
    if len(parts) < 70:
        return None
    try:
        return MotorTemperatureSample(
            float(parts[0]),
            tuple(int(float(value)) for value in parts[38:54]),
            tuple(int(float(value)) for value in parts[54:70]),
            tuple(int(float(value)) for value in parts[22:38]),
            tuple(float(value) for value in parts[70:86])
            if len(parts) >= 86
            else (),
        )
    except ValueError:
        return None
