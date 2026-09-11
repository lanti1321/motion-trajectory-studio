from __future__ import annotations

import ssl
import threading
import urllib.request

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import (
    QCamera,
    QMediaCaptureSession,
    QMediaDevices,
    QVideoFrame,
    QVideoFrameFormat,
    QVideoSink,
)
from PySide6.QtWidgets import QLabel, QSizePolicy


class RealCameraView(QLabel):
    frameReceived = Signal(bytes)
    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 270)
        self.setText("尚未连接真机相机")
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_image: QImage | None = None
        self._camera: QCamera | None = None
        self._capture_session: QMediaCaptureSession | None = None
        self._video_sink: QVideoSink | None = None
        self.frameReceived.connect(self._display_jpeg)
        self.statusChanged.connect(self.setText)

    def connect_stream(self, source: str) -> None:
        if source.startswith(("http://", "https://")):
            self._connect_mjpeg(source)
        else:
            self._connect_v4l2(source)

    def _connect_mjpeg(self, url: str) -> None:
        self.disconnect_stream()
        self._stop = threading.Event()
        self.setText(f"正在连接真机相机…\n{url}")
        self._thread = threading.Thread(
            target=self._stream_loop,
            args=(url, self._stop),
            name="motion-studio-real-camera",
            daemon=True,
        )
        self._thread.start()

    def _connect_v4l2(self, source: str) -> None:
        self.disconnect_stream()
        normalized = source.strip()
        if normalized.isdigit():
            normalized = f"/dev/video{normalized}"
        devices = QMediaDevices.videoInputs()
        device = next(
            (
                candidate
                for candidate in devices
                if bytes(candidate.id()).decode(errors="replace") == normalized
            ),
            None,
        )
        if device is None:
            available = ", ".join(
                bytes(candidate.id()).decode(errors="replace")
                for candidate in devices
            ) or "无"
            self.setText(f"找不到相机 {normalized}\n可用设备：{available}")
            return
        camera = QCamera(device, self)
        preferred_formats = {
            QVideoFrameFormat.PixelFormat.Format_YUYV,
            QVideoFrameFormat.PixelFormat.Format_UYVY,
        }
        formats = list(device.videoFormats())
        yuv_formats = [
            camera_format
            for camera_format in formats
            if camera_format.pixelFormat() in preferred_formats
        ]
        candidates = yuv_formats or formats
        if candidates:
            requested_width = 640
            requested_height = 360
            selected = min(
                candidates,
                key=lambda camera_format: (
                    abs(camera_format.resolution().width() - requested_width)
                    + abs(camera_format.resolution().height() - requested_height),
                    -camera_format.maxFrameRate(),
                ),
            )
            camera.setCameraFormat(selected)
            resolution = selected.resolution()
            pixel_name = selected.pixelFormat().name.replace("Format_", "")
            self.setText(
                f"正在打开 {normalized}：{pixel_name} "
                f"{resolution.width()}×{resolution.height()}"
            )
        sink = QVideoSink(self)
        sink.videoFrameChanged.connect(self._display_video_frame)
        session = QMediaCaptureSession(self)
        session.setCamera(camera)
        session.setVideoSink(sink)
        camera.errorOccurred.connect(
            lambda _error, message: self.statusChanged.emit(
                f"真机相机错误：{message}"
            )
        )
        self._camera = camera
        self._video_sink = sink
        self._capture_session = session
        camera.start()

    def disconnect_stream(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.deleteLater()
            self._camera = None
        if self._capture_session is not None:
            self._capture_session.deleteLater()
            self._capture_session = None
        if self._video_sink is not None:
            self._video_sink.deleteLater()
            self._video_sink = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
            self._thread = None
        self._last_image = None
        self.clear()
        self.setText("真机相机已断开")

    def _display_video_frame(self, frame: QVideoFrame) -> None:
        image = frame.toImage()
        if image.isNull():
            return
        self._last_image = image.copy()
        self._set_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._set_scaled_pixmap()

    def _display_jpeg(self, payload: bytes) -> None:
        image = QImage.fromData(payload, "JPG")
        if image.isNull():
            return
        self._last_image = image
        self._set_scaled_pixmap()

    def _set_scaled_pixmap(self) -> None:
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def _stream_loop(self, url: str, stop: threading.Event) -> None:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        while not stop.is_set():
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "MotionTrajectoryStudio/0.1"}
                )
                with urllib.request.urlopen(
                    request, timeout=5.0, context=ssl_context
                ) as response:
                    self.statusChanged.emit("真机相机已连接，等待首帧…")
                    buffer = bytearray()
                    while not stop.is_set():
                        chunk = response.read(16384)
                        if not chunk:
                            raise ConnectionError("相机流已关闭")
                        buffer.extend(chunk)
                        while True:
                            start = buffer.find(b"\xff\xd8")
                            end = buffer.find(b"\xff\xd9", start + 2)
                            if start < 0 or end < 0:
                                if len(buffer) > 4 * 1024 * 1024:
                                    del buffer[:-2]
                                break
                            jpeg = bytes(buffer[start : end + 2])
                            del buffer[: end + 2]
                            self.frameReceived.emit(jpeg)
            except Exception as error:
                if stop.is_set():
                    break
                self.statusChanged.emit(f"真机相机断线，1秒后重连：\n{error}")
                stop.wait(1.0)
