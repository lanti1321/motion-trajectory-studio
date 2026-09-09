from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from studio.ui.viewport import apply_overview_camera, clamped_mouse_delta, rgb_frame_for_qt

mujoco = pytest.importorskip("mujoco")

O6_SCENE = Path(__file__).resolve().parents[1] / "model_library" / "OpenArm_O6" / "scene.xml"
V1_SCENE = Path(__file__).resolve().parents[1] / "model_library" / "OpenArm_V1" / "scene.xml"


def _camera_pose(model: mujoco.MjModel, camera: mujoco.MjvCamera) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=20000)
    option = mujoco.MjvOption()
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return np.asarray(scene.camera[0].pos), np.asarray(scene.camera[0].forward)


@pytest.mark.parametrize("scene_path", [O6_SCENE, V1_SCENE])
def test_overview_camera_looks_down_from_above(scene_path: Path) -> None:
    if not scene_path.is_file():
        pytest.skip(f"{scene_path.name} is unavailable")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    camera = mujoco.MjvCamera()
    apply_overview_camera(model, camera)
    assert camera.elevation <= -25.0
    position, forward = _camera_pose(model, camera)
    assert position[2] > camera.lookat[2]
    assert forward[2] < 0.0


def test_mouse_delta_cannot_flip_the_camera_in_one_event() -> None:
    rel_x, rel_y = clamped_mouse_delta(8000.0, -8000.0, height=1.0)
    assert abs(rel_x) <= 0.04
    assert abs(rel_y) <= 0.04
    camera = mujoco.MjvCamera()
    camera.azimuth = 160.0
    camera.elevation = -25.0
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    )
    mujoco.mjv_moveCamera(
        model, int(mujoco.mjtMouse.mjMOUSE_ROTATE_H), rel_x, rel_y, camera
    )
    assert camera.elevation > -90.0
    assert camera.elevation < 0.0


def test_rgb_frame_for_qt_is_contiguous() -> None:
    flipped = np.flipud(np.zeros((4, 6, 3), dtype=np.uint8))
    frame = rgb_frame_for_qt(flipped)
    assert frame.flags["C_CONTIGUOUS"]
    assert frame.strides[0] > 0


def test_viewport_pixmap_fills_widget_when_enlarged() -> None:
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from studio.ui.viewport import MuJoCoViewport

    application = QApplication.instance() or QApplication([])
    view = MuJoCoViewport()
    view.show()
    view.resize(480, 270)
    application.processEvents()
    image = QImage(80, 45, QImage.Format.Format_RGB888)
    image.fill(0x336699)
    view._last_image = image
    view._set_scaled_pixmap()
    pixmap = view.pixmap()
    assert pixmap is not None
    size = pixmap.deviceIndependentSize()
    assert size.width() == pytest.approx(view.width(), abs=1)
    assert size.height() == pytest.approx(view.height(), abs=1)
    view.close()
    application.processEvents()
