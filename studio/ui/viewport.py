from __future__ import annotations

import mujoco
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel, QSizePolicy

from studio.models.adapter import MuJoCoModelAdapter
from studio.ui.i18n import ui_text

_MAX_MOUSE_STEP = 0.04  # mjv_moveCamera: 1.0 == 180°, keep each event under ~7°
_DRAG_THRESHOLD_PX = 4.0


def apply_overview_camera(model: mujoco.MjModel, camera: mujoco.MjvCamera) -> None:
    """Place a free camera above the model, looking down at its center."""
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
    extent = max(float(model.stat.extent), 0.2)
    camera.lookat[:] = model.stat.center
    camera.distance = max(float(camera.distance), 1.6 * extent)
    # Negative elevation looks down. XML defaults around -10 sit at hip height,
    # and one oversized mouse jump can slam the camera through ±90°.
    camera.elevation = min(float(camera.elevation), -25.0)


def clamped_mouse_delta(delta_x: float, delta_y: float, height: float) -> tuple[float, float]:
    scale = max(120.0, float(height))
    rel_x = max(-_MAX_MOUSE_STEP, min(_MAX_MOUSE_STEP, delta_x / scale))
    rel_y = max(-_MAX_MOUSE_STEP, min(_MAX_MOUSE_STEP, delta_y / scale))
    return rel_x, rel_y


def rgb_frame_for_qt(rgb: np.ndarray) -> np.ndarray:
    """Return a C-contiguous RGB frame QImage can copy without stride traps."""
    frame = np.ascontiguousarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    return frame


class MuJoCoViewport(QLabel):
    endEffectorDrag = Signal(float, float, float, bool)
    endEffectorDragCanceled = Signal()

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 135)
        self.setText(ui_text("导入 MJCF/XML 或 URDF 模型"))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip(ui_text(
            "左键拖动：旋转｜Shift+左键/中键：平移｜右键拖动/滚轮：缩放｜双击：重置视角"
        ))
        self.adapter: MuJoCoModelAdapter | None = None
        self.renderer: mujoco.Renderer | None = None
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self._named_camera: str | None = None
        self._last_image: QImage | None = None
        self._render_size = (0, 0)
        self._drag_position: QPointF | None = None
        self._orbit_started = False
        self._end_effector_drag_enabled = False
        self._end_effector_adjusted = False
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._resize_renderer)

    def set_adapter(self, adapter: MuJoCoModelAdapter | None) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._last_image = None
        self.clear()
        self.adapter = adapter
        self._drag_position = None
        self._orbit_started = False
        if adapter is None:
            self.setText(ui_text("导入 MJCF/XML 或 URDF 模型"))
            return
        try:
            apply_overview_camera(adapter.model, self.camera)
            self._ensure_renderer(force=True)
            self.refresh()
        except Exception as error:
            self._close_renderer()
            self.setText(f"模型已加载，离屏渲染不可用：\n{error}")

    def refresh(self) -> None:
        if self.adapter is None:
            return
        try:
            self._ensure_renderer()
            if self.renderer is None:
                return
            camera: mujoco.MjvCamera | str = self._named_camera or self.camera
            self.renderer.update_scene(self.adapter.data, camera=camera)
            rgb = rgb_frame_for_qt(self.renderer.render())
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                int(rgb.strides[0]),
                QImage.Format.Format_RGB888,
            ).copy()
            image.setDevicePixelRatio(self.devicePixelRatioF())
            self._last_image = image
            self._set_scaled_pixmap()
        except Exception as error:
            self.setText(f"渲染失败：{error}")

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._set_scaled_pixmap()
        self._resize_timer.start()

    def reset_camera(self) -> None:
        if self.adapter is not None:
            apply_overview_camera(self.adapter.model, self.camera)
            self.refresh()

    def set_camera_view(self, camera_name: str | None) -> None:
        if camera_name and self.adapter is not None:
            if camera_name not in self.adapter.info.camera_names:
                raise ValueError(f"模型中没有摄像机 {camera_name!r}")
        self._named_camera = camera_name
        self.refresh()

    def camera_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return world-space right, up and forward axes for the active view."""
        if self._named_camera and self.adapter is not None:
            camera_id = mujoco.mj_name2id(
                self.adapter.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                self._named_camera,
            )
            rotation = self.adapter.data.cam_xmat[camera_id].reshape(3, 3)
            return rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        azimuth = np.deg2rad(self.camera.azimuth)
        elevation = np.deg2rad(self.camera.elevation)
        forward = np.asarray(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-9:
            right = np.asarray([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    def set_end_effector_drag_enabled(self, enabled: bool) -> None:
        self._end_effector_drag_enabled = enabled
        self._drag_position = None
        self._orbit_started = False
        self._end_effector_adjusted = False
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        if enabled:
            self.setToolTip(
                "末端增量拖拽：左键拖动逐步调整屏幕平面位置，滚轮逐步调整深度，松开左键提交，Esc取消"
            )
        else:
            self.setToolTip(
                "左键拖动：旋转｜Shift+左键/中键：平移｜右键拖动/滚轮：缩放｜双击：重置视角"
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._drag_position = event.position()
        self._orbit_started = False
        self.setFocus()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._end_effector_drag_enabled
            and event.button() == Qt.MouseButton.LeftButton
            and self._end_effector_adjusted
        ):
            self.endEffectorDrag.emit(0.0, 0.0, 0.0, True)
        self._drag_position = None
        self._orbit_started = False
        event.accept()

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._drag_position = None
        self._orbit_started = False
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_position is None
            or self.adapter is None
            or self.renderer is None
        ):
            return
        delta = event.position() - self._drag_position
        if not self._orbit_started:
            if (
                abs(float(delta.x())) < _DRAG_THRESHOLD_PX
                and abs(float(delta.y())) < _DRAG_THRESHOLD_PX
            ):
                return
            self._orbit_started = True
            self._drag_position = event.position()
            event.accept()
            return
        self._drag_position = event.position()
        buttons = event.buttons()
        rel_x, rel_y = clamped_mouse_delta(
            float(delta.x()), float(delta.y()), float(self.height())
        )
        if (
            self._end_effector_drag_enabled
            and buttons & Qt.MouseButton.LeftButton
        ):
            self._end_effector_adjusted = True
            self.endEffectorDrag.emit(rel_x, rel_y, 0.0, False)
            event.accept()
            return
        if buttons & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        elif buttons & Qt.MouseButton.MiddleButton or (
            buttons & Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif buttons & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        else:
            return
        mujoco.mjv_moveCamera(
            self.adapter.model,
            action,
            rel_x,
            rel_y,
            self.camera,
        )
        self.refresh()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.adapter is None or self.renderer is None:
            return
        steps = event.angleDelta().y() / 120.0
        if self._end_effector_drag_enabled:
            self._end_effector_adjusted = True
            self.endEffectorDrag.emit(
                0.0,
                0.0,
                0.01 * float(steps),
                False,
            )
            event.accept()
            return
        zoom = max(-_MAX_MOUSE_STEP, min(_MAX_MOUSE_STEP, 0.05 * float(steps)))
        mujoco.mjv_moveCamera(
            self.adapter.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            zoom,
            self.camera,
        )
        self.refresh()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.reset_camera()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            self._end_effector_drag_enabled
            and event.key() == Qt.Key.Key_Escape
        ):
            self.set_end_effector_drag_enabled(False)
            self.endEffectorDragCanceled.emit()
            event.accept()
            return
        if (
            self._end_effector_drag_enabled
            and self._end_effector_adjusted
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            self.endEffectorDrag.emit(0.0, 0.0, 0.0, True)
            event.accept()
            return
        super().keyPressEvent(event)

    def _desired_render_size(self) -> tuple[int, int]:
        pixel_ratio = self.devicePixelRatioF()
        width = int(round(max(2, self.width()) * pixel_ratio))
        height = int(round(max(2, self.height()) * pixel_ratio))
        width -= width % 2
        height -= height % 2
        return max(2, min(width, 3840)), max(2, min(height, 2160))

    def _ensure_renderer(self, force: bool = False) -> None:
        if self.adapter is None:
            return
        width, height = self._desired_render_size()
        if not force and self.renderer is not None and self._render_size == (width, height):
            return
        self._close_renderer()
        self.adapter.model.vis.global_.offwidth = max(
            int(self.adapter.model.vis.global_.offwidth), width
        )
        self.adapter.model.vis.global_.offheight = max(
            int(self.adapter.model.vis.global_.offheight), height
        )
        self.renderer = mujoco.Renderer(
            self.adapter.model, height=height, width=width
        )
        self._render_size = (width, height)

    def _resize_renderer(self) -> None:
        if self.adapter is None:
            return
        try:
            self._ensure_renderer()
            self.refresh()
        except Exception as error:
            self.setText(f"渲染尺寸调整失败：{error}")

    def _close_renderer(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._render_size = (0, 0)

    def _set_scaled_pixmap(self) -> None:
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image)
        pixel_ratio = self.devicePixelRatioF()
        target_width = max(1, int(round(self.width() * pixel_ratio)))
        target_height = max(1, int(round(self.height() * pixel_ratio)))
        if pixmap.width() != target_width or pixmap.height() != target_height:
            pixmap = pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        pixmap.setDevicePixelRatio(pixel_ratio)
        self.setPixmap(pixmap)
