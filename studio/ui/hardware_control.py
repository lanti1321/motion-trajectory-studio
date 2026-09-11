from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QInputDialog, QLineEdit

from studio.hardware.openarm_replay import OpenArmReplayClient, build_dual_replay_clip
from studio.io.trajectory_io import export_trajectory
from studio.ui.temperature_monitor import TemperatureCurveDialog
from studio.ui.message_box import QMessageBox


class HardwareControlMixin:
    """Hardware service lifecycle, execution state machine, and telemetry UI."""

    def connect_hardware_service(self) -> None:
        if self.hardware_client is not None:
            self.statusBar().showMessage("真机回放客户端已启动")
            return
        try:
            client = OpenArmReplayClient()
            client.connect_status()
            telemetry_port = client.open_telemetry()
            self.hardware_client = client
            self._hardware_service_ready = False
            self._hardware_loaded_data_id = None
            self.hardware_timer.start()
            if self._hardware_command_port_in_use(client.command_port):
                if self._hardware_port_retry_count < 4:
                    client.close()
                    self.hardware_client = None
                    self.hardware_timer.stop()
                    self._hardware_port_retry_count += 1
                    self.statusBar().showMessage(
                        f"真机控制端口正在释放，自动重试 "
                        f"{self._hardware_port_retry_count}/4…"
                    )
                    QTimer.singleShot(500, self.connect_hardware_service)
                    return
                self._hardware_port_retry_count = 0
                answer = QMessageBox.warning(
                    self,
                    "真机控制端口被占用",
                    f"UDP {client.command_port} 已被其他控制进程占用。可以先向该端口发送"
                    "安全停止/阻尼释放命令，1.8 秒后再启动本项目的独立服务。\n\n"
                    "只有确认占用者是 OpenArm 控制服务时才继续。",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    client.shutdown()
                client.close()
                self.hardware_client = None
                self.hardware_timer.stop()
                if answer == QMessageBox.StandardButton.Yes:
                    self.statusBar().showMessage("已请求占用服务安全停止，等待端口释放…")
                    QTimer.singleShot(1800, self.connect_hardware_service)
                return
            self._hardware_port_retry_count = 0
            answer = QMessageBox.warning(
                self,
                "启动真机控制服务",
                "未检测到 Motion Studio 独立真机服务。软件将启动该服务，"
                "该过程会配置 CAN、使能电机并可能请求 sudo 密码。\n"
                "请确认没有其他程序占用 CAN，机械臂工作空间无人且急停可用。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                client.close()
                self.hardware_client = None
                self.hardware_timer.stop()
                return
            self._start_hardware_server()
        except InterruptedError as error:
            if self.hardware_client is not None:
                self.hardware_client.close()
                self.hardware_client = None
            self.hardware_timer.stop()
            self.statusBar().showMessage(str(error))
        except Exception as error:
            if self.hardware_client is not None:
                self.hardware_client.close()
                self.hardware_client = None
            self.hardware_timer.stop()
            QMessageBox.critical(self, "连接真机服务失败", str(error))

    def connect_real_camera(self) -> None:
        self.show_camera_monitor()
        self.statusBar().showMessage("已打开多相机监视器")

    @staticmethod
    def _hardware_command_port_in_use(port: int) -> bool:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True
        finally:
            probe.close()

    def _start_hardware_server(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        server_root = project_root / "hardware_controller"
        script = server_root / "scripts" / "run_service.sh"
        if not script.is_file():
            raise FileNotFoundError(f"真机服务脚本不存在：{script}")
        environment = os.environ.copy()
        if environment.get("OPENARM_AUTO_CONFIG_CAN", "1") == "1":
            self._configure_hardware_can(server_root, environment)
            # CAN has already been configured through the one-shot privileged helper.
            # Prevent the detached service from trying to authenticate without a TTY.
            environment["OPENARM_AUTO_CONFIG_CAN"] = "0"
        log_path = Path(tempfile.gettempdir()) / "motion_studio_hardware_server.log"
        # One launch, one diagnostic log. Appending mixed stale failures with a
        # new run made port/authentication diagnosis misleading.
        self._hardware_server_log_handle = log_path.open("wb", buffering=0)
        self.hardware_server_process = subprocess.Popen(
            ["bash", str(script)],
            cwd=server_root,
            env=environment,
            stdout=self._hardware_server_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._hardware_shutdown_requested = False
        self._hardware_service_ready = False
        self._hardware_loaded_data_id = None
        self.statusBar().showMessage(
            f"正在启动真机服务，日志：{log_path}；等待 ready 状态"
        )

    def _configure_hardware_can(
        self, server_root: Path, environment: dict[str, str]
    ) -> None:
        helper = server_root / "scripts" / "configure_can.sh"
        if not helper.is_file():
            raise FileNotFoundError(f"CAN 配置脚本不存在：{helper}")

        arguments = [
            environment.get("OPENARM_LEFT_FOLLOWER_CAN", "can1"),
            environment.get("OPENARM_RIGHT_FOLLOWER_CAN", "can0"),
            environment.get("OPENARM_CAN_BITRATE", "1000000"),
            environment.get("OPENARM_CAN_DBITRATE", "5000000"),
        ]
        cached = subprocess.run(
            ["sudo", "-n", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        password = ""
        if not cached:
            password, accepted = QInputDialog.getText(
                self,
                "配置真机 CAN",
                "请输入当前系统用户的 sudo 密码。\n"
                "密码仅传给 sudo 完成一次 CAN 配置，不会保存或写入日志：",
                QLineEdit.EchoMode.Password,
            )
            if not accepted:
                raise InterruptedError("已取消启动真机服务")
            if not password:
                raise ValueError("sudo 密码不能为空")

        command = ["sudo", "-n" if cached else "-S", "-p", ""]
        command.extend(["bash", str(helper), *arguments])
        try:
            result = subprocess.run(
                command,
                input=None if cached else password + "\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
        finally:
            password = ""
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            reason = detail[-1] if detail else f"sudo 返回 {result.returncode}"
            raise RuntimeError(f"CAN 配置失败：{reason}")

    def execute_from_current_frame(self) -> None:
        data = self._rendered()
        if data is None:
            QMessageBox.information(self, "真机执行", "请先加载轨迹")
            return
        detection_mode = self.temperature_monitor.detection_mode.isChecked()
        high_dynamics_mode = self.temperature_monitor.high_dynamics_mode.isChecked()
        self._hardware_motion_limits = (
            (3.5, 60.0) if high_dynamics_mode else (2.0, 8.0)
        )
        if not detection_mode and self.frame >= data.frame_count - 1:
            QMessageBox.information(self, "真机执行", "当前帧之后没有可执行的动作")
            return
        required = [f"q{index}" for index in range(16)]
        missing = [name for name in required if name not in data.channels]
        if missing:
            QMessageBox.critical(
                self,
                "真机执行不可用",
                "独立真机控制器需要 q0～q15，缺少：" + ", ".join(missing),
            )
            return
        execution_frame = 0 if detection_mode else self.frame
        start_time = float(data.times[execution_frame])
        remaining = float(data.times[-1] - start_time)
        playback_data, handoff_note = data, ""
        hardware_clip_key = (id(playback_data), execution_frame)
        answer = QMessageBox.warning(
            self,
            "确认真机执行",
            f"将从 {start_time:.3f}s 开始执行，动作时长 {remaining:.3f}s。\n"
            + (
                "温度检测模式已启用：强制完整轨迹、实时记录双温度曲线。\n"
                if detection_mode
                else ""
            )
            + handoff_note
            +
            "控制服务会先依据真机反馈生成至少 3 秒、必要时自动延长的五次多项式过渡。\n"
            + (
                "高动态模式：速度上限 3.5 rad/s，加速度上限 60 rad/s²。\n"
                if high_dynamics_mode
                else "普通模式：速度上限 2 rad/s，加速度上限 8 rad/s²。\n"
            )
            +
            "请确认机械臂工作空间无人、急停可用且 CAN 控制服务已独占。",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        try:
            if self.hardware_client is None:
                self.connect_hardware_service()
            if self.hardware_client is None:
                return
            if self._hardware_service_ready and self._hardware_loaded_data_id is not None:
                if self._hardware_loaded_data_id != hardware_clip_key:
                    QMessageBox.warning(
                        self,
                        "真机执行起点已改变",
                        "当前服务仍保持上一次裁剪后的真机轨迹。为避免保持期间加载大文件造成控制中断，"
                        "请先点击“真机停止/释放”，重新连接服务后再执行修改后的轨迹。",
                    )
                    return
                self._temperature_history = []
                self._last_temperature_record_time = -float("inf")
                self._temperature_recording = detection_mode
                self._hardware_source_start_time = start_time
                self._hardware_transition_duration = 3.0
                self._pending_hardware_frame = 0
                self.hardware_client.configure_motion_limits(
                    *self._hardware_motion_limits
                )
                self.hardware_client.prepare(0)
                self.statusBar().showMessage(
                    f"正在准备从编辑器第 {execution_frame} 帧裁剪的真机轨迹（尚未运动）"
                )
                return
            # Export only the part that will actually execute.  Loading the
            # complete editor timeline made the controller reject unrelated
            # acceleration spikes *before* the selected start frame.
            clip = build_dual_replay_clip(playback_data, execution_frame)
            export_root = Path(tempfile.mkdtemp(prefix="motion_studio_replay_"))
            export_path = export_trajectory(clip, export_root / "timeline_clip.csv")
            self._hardware_exports.append(export_path)
            self._temperature_history = []
            self._last_temperature_record_time = -float("inf")
            self._temperature_recording = detection_mode
            self._hardware_source_start_time = start_time
            self._hardware_transition_duration = 3.0
            self._pending_hardware_frame = 0
            self._pending_hardware_export_path = export_path
            self._pending_hardware_data_id = hardware_clip_key
            if self._hardware_service_ready:
                self.hardware_client.configure_motion_limits(
                    *self._hardware_motion_limits
                )
                self.hardware_client.load(
                    export_path, telemetry_port=self.hardware_client.telemetry_port
                )
                self.statusBar().showMessage(
                    f"正在加载从第 {execution_frame} 帧开始的真机轨迹"
                )
            else:
                self.statusBar().showMessage(
                    f"等待服务就绪，随后加载从第 {execution_frame} 帧开始的真机轨迹"
                )
        except Exception as error:
            QMessageBox.critical(self, "真机执行失败", str(error))

    def shutdown_hardware_service(self) -> None:
        if self.hardware_client is None:
            QMessageBox.information(self, "真机停止", "尚未连接真机回放服务")
            return
        answer = QMessageBox.warning(
            self,
            "停止并释放真机",
            "这会停止独立真机控制服务并进入阻尼释放。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._hardware_shutdown_requested = True
            self.hardware_client.shutdown()
            self.statusBar().showMessage("已发送真机停止/阻尼释放命令")

    def _poll_hardware_status(self) -> None:
        if self.hardware_client is None:
            return
        status_labels = {
            "ready": "独立真机控制服务就绪",
            "loaded": "真机时间轴已加载",
            "prepared": "真机执行起点已准备（尚未运动）",
            "prepare_failed": "真机执行起点准备失败",
            "transition_started": "正在进行自适应五次多项式过渡",
            "accepted": "真机已接受轨迹",
            "transition_added": "已生成 3 秒真机初始过渡",
            "started": "真机动作执行中",
            "finished": "真机动作完成，保持末帧",
            "busy": "真机服务忙，命令被拒绝",
            "load_failed": "真机轨迹加载失败",
            "transition_failed": "真机初始过渡生成失败",
            "limits": "真机动态限制已配置",
            "safety": "真机安全保护触发",
            "watchdog": "上位机心跳中断，真机已停止并阻尼释放",
            "shutdown_done": "真机已停止并完成阻尼释放",
        }
        try:
            self.hardware_client.heartbeat()
            if (
                self.hardware_server_process is not None
                and self.hardware_server_process.poll() is not None
            ):
                exit_code = self.hardware_server_process.returncode
                self.hardware_server_process = None
                self._hardware_service_ready = False
                self._hardware_loaded_data_id = None
                if self._hardware_server_log_handle is not None:
                    self._hardware_server_log_handle.close()
                    self._hardware_server_log_handle = None
                failed_client = self.hardware_client
                self.hardware_client = None
                if failed_client is not None:
                    failed_client.close()
                self._pending_hardware_frame = None
                self._pending_hardware_export_path = None
                self._pending_hardware_data_id = None
                self._temperature_recording = False
                self.hardware_timer.stop()
                if self._hardware_shutdown_requested:
                    self.statusBar().showMessage("真机服务已完成停止/阻尼释放并退出")
                    self._hardware_shutdown_requested = False
                else:
                    QMessageBox.critical(
                        self,
                        "真机服务异常退出",
                        f"独立真机控制服务已退出，退出码 {exit_code}。\n"
                        f"请查看日志：{Path(tempfile.gettempdir()) / 'motion_studio_hardware_server.log'}",
                    )
                return
            samples = self.hardware_client.poll_temperatures()
            if samples:
                latest = samples[-1]
                self.temperature_monitor.update_sample(latest)
                self._apply_hardware_feedback(latest)
                if self._temperature_recording:
                    # Temperature curves do not benefit from 1 kHz storage.
                    for sample in samples:
                        if sample.time - self._last_temperature_record_time >= 0.05:
                            self._temperature_history.append(sample)
                            self._last_temperature_record_time = sample.time
            for message in self.hardware_client.poll_status():
                state = message.split("|", 1)[0]
                if state == "ready":
                    self._hardware_service_ready = True
                    self.hardware_client.configure_motion_limits(
                        *self._hardware_motion_limits
                    )
                    if self._pending_hardware_export_path is not None:
                        self.hardware_client.load(
                            self._pending_hardware_export_path,
                            telemetry_port=self.hardware_client.telemetry_port,
                        )
                elif state == "loaded" and self._pending_hardware_frame is not None:
                    self._hardware_loaded_data_id = self._pending_hardware_data_id
                    self._pending_hardware_data_id = None
                    self._pending_hardware_export_path = None
                    self.hardware_client.prepare(self._pending_hardware_frame)
                elif state == "prepared" and self._pending_hardware_frame is not None:
                    self.hardware_client.execute()
                    self._pending_hardware_frame = None
                elif state == "transition_started":
                    self.playing = False
                    self._preview_stop_time = None
                    for field in message.split("|")[1:]:
                        if field.startswith("duration="):
                            try:
                                self._hardware_transition_duration = float(
                                    field.split("=", 1)[1]
                                )
                            except ValueError:
                                pass
                    data = self._rendered()
                    if data is not None:
                        self.frame = data.nearest_frame(
                            self._hardware_source_start_time
                        )
                        self._apply_current_frame()
                elif state == "started":
                    self._preview_stop_time = None
                    data = self._rendered()
                    if data is not None:
                        self.frame = data.nearest_frame(
                            self._hardware_source_start_time
                        )
                        self.play_started_time = float(data.times[self.frame])
                        self.play_started_at = time.monotonic()
                        self.playing = True
                elif state == "finished":
                    self.playing = False
                    self._preview_stop_time = None
                    data = self._rendered()
                    if data is not None:
                        self.frame = data.frame_count - 1
                        self._apply_current_frame()
                if state == "transition_skipped":
                    self._hardware_transition_duration = 0.0
                self.statusBar().showMessage(
                    f"{status_labels.get(state, state)}：{message}"
                )
                if state in {
                    "busy",
                    "safety",
                    "watchdog",
                    "load_failed",
                    "prepare_failed",
                    "transition_failed",
                    "shutdown_done",
                }:
                    if state in {"load_failed", "prepare_failed", "transition_failed"}:
                        self._pending_hardware_frame = None
                        self._pending_hardware_export_path = None
                        self._pending_hardware_data_id = None
                    self._temperature_recording = False
                    if state == "transition_failed":
                        self.hardware_client.shutdown()
                    if state in {"safety", "watchdog", "load_failed", "prepare_failed", "transition_failed"}:
                        detail = status_labels[state]
                        if state == "transition_failed":
                            detail += "，已立即发送停止/阻尼释放命令"
                        QMessageBox.critical(
                            self, "真机回放状态", detail + "\n" + message
                        )
                elif state == "finished" and self._temperature_recording:
                    self._temperature_recording = False
                    dialog = TemperatureCurveDialog(
                        list(self._temperature_history), self
                    )
                    dialog.finished.connect(
                        lambda _result, item=dialog: self._temperature_dialogs.remove(item)
                        if item in self._temperature_dialogs
                        else None
                    )
                    self._temperature_dialogs.append(dialog)
                    dialog.show()
        except Exception as error:
            self.hardware_timer.stop()
            failed_client = self.hardware_client
            self.hardware_client = None
            self._hardware_service_ready = False
            self._hardware_loaded_data_id = None
            if failed_client is not None:
                failed_client.close()
            QMessageBox.critical(self, "真机状态监听失败", str(error))

    def _apply_hardware_feedback(self, sample) -> None:  # type: ignore[no-untyped-def]
        # Deliberately monitoring-only. Simulation playback is driven from the
        # controller's started/finished status and the editor's monotonic clock.
        # Measured joints must never overwrite MuJoCo state.
        return
