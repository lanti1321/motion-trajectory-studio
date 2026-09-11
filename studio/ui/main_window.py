from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
    QGuiApplication,
    QKeySequence,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QDockWidget,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolBar,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from studio.core.project import ProjectDocument
from studio.core.timeline import TimelineEngine
from studio.core.trajectory import TrajectoryData
from studio.io.trajectory_io import load_trajectory
from studio.hardware.openarm_replay import OpenArmReplayClient
from studio.models.adapter import MuJoCoModelAdapter, auto_map_channels
from studio.models.library import ModelLibrary
from studio.models.pose import (
    channel_mapping_for_joints,
    channel_value_from_joint,
    default_joint_groups,
    extend_trajectory_with_constants,
    hold_pose_trajectory,
    independent_joint_positions,
    independent_scalar_joints,
    joint_pose_patch_operation,
    joint_value_from_channel,
    scalar_joint_positions,
)
from studio.ui.mapping_dialog import JointMappingDialog
from studio.ui.jog_teach_dialog import JogTeachDialog
from studio.ui.part_modules import CustomPartModuleDialog, PartModuleHost
from studio.ui.timeline_widget import TimelineWidget
from studio.ui.viewport import MuJoCoViewport
from studio.ui.temperature_monitor import TemperatureCurveDialog, TemperatureMonitor
from studio.ui.multi_camera import MultiCameraWindow
from studio.ui.i18n import install_ui_language
from studio.ui.hardware_control import HardwareControlMixin
from studio.ui.project_actions import ProjectActionsMixin
from studio.ui.end_effector_control import EndEffectorControlMixin
from studio.ui.editing_actions import EditingActionsMixin
from studio.ui.timeline_actions import TimelineActionsMixin
from studio.ui.message_box import QMessageBox
from studio.validation.checks import (
    ValidationReport,
    filter_new_joint_limit_issues,
    validate_model_mapping,
    validate_operation_continuity,
)


def _scrollable_dock_content(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(widget)
    scroll.setMinimumSize(160, 80)
    scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
    return scroll


class MainWindow(
    HardwareControlMixin,
    ProjectActionsMixin,
    EndEffectorControlMixin,
    EditingActionsMixin,
    TimelineActionsMixin,
    QMainWindow,
):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motion Trajectory Studio")
        self.resize(1500, 940)
        self._did_fit_to_screen = False
        self.project = ProjectDocument("未命名工程")
        self.adapter: MuJoCoModelAdapter | None = None
        self.source: TrajectoryData | None = None
        self.timeline_engine: TimelineEngine | None = None
        self.frame = 0
        self.playing = False
        self.play_started_at = 0.0
        self.play_started_time = 0.0
        self._preview_stop_time: float | None = None
        self.model_library = ModelLibrary()
        self._model_library_signature: tuple[object, ...] = ()
        self._dirty = False
        self.hardware_client: OpenArmReplayClient | None = None
        self.hardware_server_process: subprocess.Popen[bytes] | None = None
        self._hardware_server_log_handle = None
        self._hardware_shutdown_requested = False
        self._hardware_exports: list[Path] = []
        self._temperature_history = []
        self._last_temperature_record_time = -float("inf")
        self._temperature_recording = False
        self._temperature_dialogs: list[TemperatureCurveDialog] = []
        self._hardware_source_start_time = 0.0
        self._hardware_transition_duration = 3.0
        self._hardware_motion_limits = (2.0, 8.0)
        self._pending_hardware_frame: int | None = None
        self._pending_hardware_export_path: Path | None = None
        self._hardware_service_ready = False
        self._hardware_loaded_data_id: tuple[int, int] | None = None
        self._pending_hardware_data_id: tuple[int, int] | None = None
        self._trajectory_clipboard: tuple[int, float, float, tuple[str, ...]] | None = None
        self._hardware_port_retry_count = 0
        configured_cameras = os.getenv(
            "OPENARM_REAL_CAMERA_SOURCES",
            os.getenv("OPENARM_REAL_CAMERA_SOURCE", "/dev/video16"),
        )
        camera_sources = [
            value.strip()
            for value in configured_cameras.replace(";", ",").split(",")
            if value.strip()
        ]
        self.camera_monitor = MultiCameraWindow(camera_sources, self)

        self.viewport = MuJoCoViewport()
        self.viewport.endEffectorDrag.connect(self._handle_end_effector_drag)
        self.viewport.endEffectorDragCanceled.connect(
            self._cancel_end_effector_drag
        )
        self.camera_preview = MuJoCoViewport()
        self.camera_preview.setMinimumSize(160, 90)
        self.camera_preview.hide()
        self._end_effector_drag_context: dict[str, object] | None = None
        self._jog_teach_dialog: JogTeachDialog | None = None
        self._joint_pose_dragging: str | None = None
        self._pending_drag_delta = np.zeros(3)
        self._last_drag_update_at = 0.0
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.setToolTip(
            "滚轮或触控板捏合以指针位置缩放时间轴；Shift+滚轮平移。"
            "在选区主体上拖动，可将所选关节片段移动到静止空白区。"
        )
        self.timeline_widget.seekRequested.connect(self.seek_time)
        self.timeline_widget.labelSelectionChanged.connect(
            self._on_label_selection_changed
        )
        self.timeline_widget.rangeSelectionChanged.connect(
            self._on_range_selection_changed
        )
        self.timeline_widget.visibleWindowChanged.connect(
            self._sync_timeline_slider
        )
        self.timeline_widget.clipMoveRequested.connect(
            self._move_selected_timeline_clip
        )
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidget(self.timeline_widget)
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.timeline_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.timeline_scroll.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self.timeline_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.timeline_scroll.setMinimumHeight(160)
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 10000)
        self.timeline_slider.setSingleStep(25)
        self.timeline_slider.setPageStep(500)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.setToolTip("缩放时间轴后的可见区间位置")
        self.timeline_slider.valueChanged.connect(
            lambda value: self.timeline_widget.set_view_fraction(value / 10000.0)
        )
        timeline_container = QWidget()
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(2)
        timeline_layout.addWidget(self.timeline_scroll, 1)
        timeline_navigation = QHBoxLayout()
        timeline_navigation.setContentsMargins(4, 0, 4, 0)
        self.timeline_first_button = QPushButton("⏮ 首帧")
        self.timeline_first_button.setToolTip("将播放头跳到整个轨迹的第一帧")
        self.timeline_first_button.clicked.connect(self.seek_first_frame)
        self.timeline_last_button = QPushButton("尾帧 ⏭")
        self.timeline_last_button.setToolTip("将播放头跳到整个轨迹的最后一帧")
        self.timeline_last_button.clicked.connect(self.seek_last_frame)
        timeline_navigation.addWidget(self.timeline_first_button)
        timeline_navigation.addWidget(self.timeline_slider, 1)
        timeline_navigation.addWidget(self.timeline_last_button)
        timeline_layout.addLayout(timeline_navigation)
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter.addWidget(self.viewport)
        self.preview_splitter.addWidget(self.camera_preview)
        self.preview_splitter.setStretchFactor(0, 2)
        self.preview_splitter.setStretchFactor(1, 1)
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.addWidget(self.preview_splitter)
        self.center_splitter.addWidget(timeline_container)
        self.center_splitter.setStretchFactor(0, 3)
        self.center_splitter.setStretchFactor(1, 2)
        self.center_splitter.setSizes([500, 420])
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(4, 4, 4, 4)
        center_layout.addWidget(self.center_splitter)
        self.setCentralWidget(center)

        self.camera_selector = QComboBox()
        self.camera_selector.addItem("不显示", None)
        self.camera_selector.setToolTip("在自由视角旁同步显示模型摄像机")
        self.camera_selector.currentIndexChanged.connect(
            self._on_camera_view_changed
        )

        self.model_library_list = QListWidget()
        self.model_library_list.itemDoubleClicked.connect(self._load_library_item)
        model_library_widget = QWidget()
        model_library_layout = QVBoxLayout(model_library_widget)
        model_library_layout.setContentsMargins(4, 4, 4, 4)
        library_path = QLabel(str(self.model_library.root))
        library_path.setWordWrap(True)
        library_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        model_library_layout.addWidget(library_path)
        model_library_layout.addWidget(self.model_library_list, 1)
        refresh_models_button = QPushButton("刷新模型库")
        refresh_models_button.clicked.connect(lambda: self.refresh_model_library(force=True))
        model_library_layout.addWidget(refresh_models_button)
        model_library_dock = QDockWidget("模型库（双击加载）", self)
        model_library_dock.setWidget(_scrollable_dock_content(model_library_widget))
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, model_library_dock)

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.channel_list.itemSelectionChanged.connect(
            self._on_channel_selection_changed
        )
        channel_widget = QWidget()
        channel_layout = QVBoxLayout(channel_widget)
        channel_layout.setContentsMargins(4, 4, 4, 4)
        self.motor_scope_status = QLabel("未单独选择电机：操作作用于全部位置通道")
        self.motor_scope_status.setWordWrap(True)
        channel_layout.addWidget(self.motor_scope_status)
        channel_layout.addWidget(self.channel_list, 1)
        self.joint_group_combo = QComboBox()
        self.joint_group_combo.setPlaceholderText("快速关节组（Alt+1…Alt+9）")
        self.joint_group_combo.setToolTip(
            "选择已保存的关节组；Alt+1 到 Alt+9 可直接切换"
        )
        self.joint_group_combo.activated.connect(
            self._select_joint_group_combo_index
        )
        channel_layout.addWidget(self.joint_group_combo)
        channel_buttons = QHBoxLayout()
        select_all_motors = QPushButton("选择全部")
        select_all_motors.clicked.connect(self.channel_list.selectAll)
        clear_motors = QPushButton("清除选择")
        clear_motors.clicked.connect(self.channel_list.clearSelection)
        channel_buttons.addWidget(select_all_motors)
        channel_buttons.addWidget(clear_motors)
        channel_layout.addLayout(channel_buttons)
        group_buttons = QHBoxLayout()
        save_group = QPushButton("保存/覆盖组")
        save_group.clicked.connect(self.save_joint_group)
        rename_group = QPushButton("重命名组")
        rename_group.clicked.connect(self.rename_joint_group)
        delete_group = QPushButton("删除组")
        delete_group.clicked.connect(self.delete_joint_group)
        group_buttons.addWidget(save_group)
        group_buttons.addWidget(rename_group)
        group_buttons.addWidget(delete_group)
        channel_layout.addLayout(group_buttons)
        channel_dock = QDockWidget("关节/通道", self)
        channel_dock.setWidget(_scrollable_dock_content(channel_widget))
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, channel_dock)
        self.splitDockWidget(model_library_dock, channel_dock, Qt.Orientation.Vertical)

        properties = QWidget()
        properties_layout = QVBoxLayout(properties)
        self.label_selection_status = QLabel(
            "框选时间段后可命名并换色；分段栏显示已保存的彩色区间"
        )
        self.label_selection_status.setWordWrap(True)
        properties_layout.addWidget(self.label_selection_status)
        for item in (
            ("添加标签", self.add_label),
            ("重命名所选标签", self.rename_selected_label),
            ("所选标签设为选区", self.select_range_from_labels),
            ("命名/换色当前选区", self.annotate_selected_segment),
            ("编辑所选区间名称/颜色", self.edit_selected_segment),
            ("删除动作分段", self.delete_selected_segment),
            ("裁剪保留", self.add_trim),
            ("删除片段", self.add_delete),
            ("复制片段", self.add_copy, "复制片段 (Ctrl+C)"),
            ("粘贴片段到播放头", self.paste_copy, "粘贴片段到播放头 (Ctrl+V)"),
            ("循环片段", self.add_loop),
            ("增加空白等待", self.add_hold),
            ("片段变速", self.add_speed),
            ("选区五次平滑", self.add_transition),
            ("所选标签间五次平滑", self.add_transition_between_labels),
            ("高速相位平滑（不减速）", self.add_high_speed_smooth),
            ("拼接另一轨迹", self.add_splice),
            ("末端拖拽调整", self.start_end_effector_drag),
            ("撤销", self.undo),
            ("重做", self.redo),
        ):
            text, handler, *rest = item
            button = QPushButton(text)
            button.clicked.connect(handler)
            if rest:
                button.setToolTip(rest[0])
            properties_layout.addWidget(button)
        properties_layout.addStretch(1)
        properties_dock = QDockWidget("编辑工具", self)
        properties_dock.setWidget(_scrollable_dock_content(properties))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, properties_dock)

        self.part_module_host = PartModuleHost(
            self, self._preview_joint_pose, self._commit_joint_pose
        )
        self.part_module_host.catalog.createRequested.connect(
            self.create_custom_part_module
        )
        self.part_module_host.catalog.deleteRequested.connect(
            self.delete_custom_part_module
        )
        self.part_module_dock = QDockWidget("部件模块", self)
        self.part_module_dock.setWidget(
            _scrollable_dock_content(self.part_module_host.catalog)
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.part_module_dock
        )

        self.temperature_monitor = TemperatureMonitor()
        temperature_dock = QDockWidget("真机关节温度", self)
        temperature_dock.setWidget(_scrollable_dock_content(self.temperature_monitor))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, temperature_dock)
        self.splitDockWidget(
            properties_dock, self.part_module_dock, Qt.Orientation.Vertical
        )
        self.splitDockWidget(
            self.part_module_dock, temperature_dock, Qt.Orientation.Vertical
        )

        self._build_actions()
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.library_timer = QTimer(self)
        self.library_timer.setInterval(1500)
        self.library_timer.timeout.connect(self.refresh_model_library)
        self.library_timer.start()
        self.hardware_timer = QTimer(self)
        # Controller telemetry is 1 kHz; the GUI only needs the latest sample.
        self.hardware_timer.setInterval(33)
        self.hardware_timer.timeout.connect(self._poll_hardware_status)
        self.refresh_model_library(force=True)
        self.statusBar().showMessage("导入模型和轨迹开始编辑")

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        actions = (
            ("新建工程", self.new_project),
            ("导入模型", self.import_model),
            ("导入模型文件夹到模型库", self.import_model_folder_to_library),
            ("刷新模型库", lambda: self.refresh_model_library(force=True)),
            ("打开模型库文件夹", self.open_model_library_folder),
            ("导入轨迹", self.import_trajectory),
            ("新建空白轨迹", self.new_blank_trajectory),
            ("打开工程", self.open_project),
            ("保存工程", self.save_project),
            ("工程另存为", self.save_project_as),
            ("保存可迁移工程", self.save_portable_project),
            ("导出轨迹", self.export),
            ("打包工程", self.pack),
        )
        for text, handler in actions:
            action = QAction(text, self)
            action.triggered.connect(handler)
            file_menu.addAction(action)
        edit_menu = self.menuBar().addMenu("编辑")
        for text, handler, shortcut in (
            ("复制时间片段", self.add_copy, QKeySequence.StandardKey.Copy),
            ("粘贴到播放头", self.paste_copy, QKeySequence.StandardKey.Paste),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(handler)
            edit_menu.addAction(action)
        for text, handler in (
            ("在播放头写入生成器关键点", self.generate_trajectory_from_playhead),
        ):
            action = QAction(text, self)
            action.triggered.connect(handler)
            edit_menu.addAction(action)
        self._install_copy_paste_shortcuts()
        self.part_module_menu = self.menuBar().addMenu("部件模块")
        self._refresh_part_module_menu()
        self.joint_group_menu = self.menuBar().addMenu("关节组")
        self._joint_group_actions: list[QAction] = []
        for slot in range(9):
            action = QAction(f"关节组 {slot + 1}：空", self)
            action.setShortcut(QKeySequence(f"Alt+{slot + 1}"))
            action.triggered.connect(
                lambda _checked=False, index=slot: self.select_joint_group_slot(index)
            )
            self.joint_group_menu.addAction(action)
            self._joint_group_actions.append(action)
        self._refresh_joint_group_controls()
        toolbar = QToolBar("播放")
        self.addToolBar(toolbar)
        for text, handler in (
            ("播放/暂停", self.toggle_play),
            ("上一帧", lambda: self.seek_frame(self.frame - 1)),
            ("下一帧", lambda: self.seek_frame(self.frame + 1)),
            ("时间线放大", lambda: self.timeline_widget.set_zoom(self.timeline_widget.zoom * 1.25)),
            ("时间线缩小", lambda: self.timeline_widget.set_zoom(self.timeline_widget.zoom / 1.25)),
            ("重置视角", self.viewport.reset_camera),
            ("关节映射", self.configure_mapping),
            ("验证", self.validate),
            ("启动/连接真机服务", self.connect_hardware_service),
            ("从当前帧执行", self.execute_from_current_frame),
            ("真机停止/释放", self.shutdown_hardware_service),
            ("多相机监视器", self.show_camera_monitor),
        ):
            action = QAction(text, self)
            action.triggered.connect(handler)
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("摄像机："))
        toolbar.addWidget(self.camera_selector)

    def show_camera_monitor(self) -> None:
        self.camera_monitor.show()
        self.camera_monitor.raise_()
        self.camera_monitor.activateWindow()

    def _refresh_camera_selector(self) -> None:
        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        self.camera_selector.addItem("不显示", None)
        if self.adapter is not None:
            for name in self.adapter.info.camera_names:
                self.camera_selector.addItem(name, name)
        self.camera_selector.setEnabled(self.camera_selector.count() > 1)
        self.camera_selector.blockSignals(False)
        self.camera_preview.hide()
        self.camera_preview.set_camera_view(None)

    def _on_camera_view_changed(self, index: int) -> None:
        camera_name = self.camera_selector.itemData(index)
        if camera_name is None:
            self.camera_preview.hide()
            return
        self.camera_preview.set_camera_view(str(camera_name))
        self.camera_preview.show()

    def new_project(self) -> None:
        if not self._confirm_discard_or_save():
            return
        default_name = "untitled"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "新建工程",
            os.path.join(os.path.expanduser("~"), f"{default_name}.motionproj"),
            "Motion project (*.motionproj)",
        )
        if not path:
            return
        destination = self._motion_project_path(path, default_name=default_name)
        project = ProjectDocument(destination.stem)
        try:
            project.save(destination)
        except Exception as error:
            QMessageBox.critical(self, "新建工程失败", str(error))
            return
        self.project = project
        self.adapter = None
        self.source = None
        self.timeline_engine = None
        self.frame = 0
        self.playing = False
        self._end_effector_drag_context = None
        self.viewport.set_adapter(None)
        self.camera_preview.set_adapter(None)
        self._refresh_camera_selector()
        self.channel_list.clear()
        self._update_motor_scope_status()
        self.part_module_host.clear()
        self._refresh_part_module_menu()
        self._joint_pose_dragging = None
        self.timeline_widget.clear_label_selection()
        self.timeline_widget.clear_selected_range(emit=False)
        self.timeline_widget.set_data(0.0, [], [])
        self._set_dirty(False)
        self.statusBar().showMessage(f"已新建工程：{destination}")

    def import_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入模型", "", "Robot models (*.xml *.mjcf *.urdf)"
        )
        if not path:
            return
        self._load_model_path(Path(path))

    def _load_model_path(
        self, path: str | Path, preset: dict[str, object] | None = None
    ) -> None:
        previous_adapter = self.adapter
        try:
            adapter = MuJoCoModelAdapter(path)
            adjustments = [
                dict(value)
                for value in (preset or {}).get("adjustments", [])
                if isinstance(value, dict)
            ]
            self._apply_adjustments_to(adapter, adjustments)
            mapping: dict[str, str] = {}
            transforms: dict[str, dict[str, float]] = {}
            default_mapping = (preset or {}).get("default_mapping")
            if isinstance(default_mapping, dict):
                for channel, mapping_value in default_mapping.items():
                    if isinstance(mapping_value, str):
                        mapping[str(channel)] = mapping_value
                    elif isinstance(mapping_value, dict) and mapping_value.get("joint"):
                        mapping[str(channel)] = str(mapping_value["joint"])
                        transforms[str(channel)] = {
                            key: float(mapping_value[key])
                            for key in ("scale", "offset", "minimum", "maximum")
                            if key in mapping_value
                        }
            self.viewport.set_adapter(adapter)
            self.camera_preview.set_adapter(adapter)
            self.adapter = adapter
            self._refresh_camera_selector()
            self.project.model_adjustments = adjustments
            self.project.joint_mapping = mapping
            self.project.mapping_transforms = transforms
            self.project.model_path = str(Path(path).expanduser().resolve())
            self._install_timeline_validator()
            self._set_dirty(True)
            self._maybe_configure_mapping()
            self._sync_model_pose_session()
            self._reload_part_modules()
            self._apply_current_frame()
            self.statusBar().showMessage(
                f"已加载 {Path(path).parent.name}。"
                "在右侧「部件模块」点「打开」调节关节；"
                "3D 窗口左键拖动只旋转视角。"
            )
        except Exception as error:
            self.adapter = previous_adapter
            if previous_adapter is not None:
                try:
                    self.viewport.set_adapter(previous_adapter)
                    self.camera_preview.set_adapter(previous_adapter)
                    self._refresh_camera_selector()
                except Exception:
                    pass
            QMessageBox.critical(self, "模型加载失败", str(error))

    def _apply_model_adjustments(
        self, adjustments: list[dict[str, object]]
    ) -> None:
        if self.adapter is None:
            return
        self._apply_adjustments_to(self.adapter, adjustments)

    @staticmethod
    def _apply_adjustments_to(
        adapter: MuJoCoModelAdapter,
        adjustments: list[dict[str, object]],
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

    def refresh_model_library(self, force: bool = False) -> None:
        entries = self.model_library.scan()
        signature = tuple(
            (
                entry.name,
                str(entry.entry_file) if entry.entry_file else None,
                entry.error,
                tuple(str(candidate) for candidate in entry.candidates),
                repr(entry.metadata),
            )
            for entry in entries
        )
        if not force and signature == self._model_library_signature:
            return
        self._model_library_signature = signature
        self.model_library_list.clear()
        for entry in entries:
            if entry.loadable:
                text = f"{entry.name}  [{entry.format}]"
            else:
                text = f"{entry.name}  [不可加载]"
            item = QListWidgetItem(text)
            item.setData(
                Qt.ItemDataRole.UserRole,
                str(entry.entry_file) if entry.entry_file else "",
            )
            item.setData(Qt.ItemDataRole.UserRole + 1, entry.metadata)
            detail = str(entry.entry_file) if entry.entry_file else str(entry.error)
            if len(entry.candidates) > 1:
                detail += f"\n检测到 {len(entry.candidates)} 个候选文件"
            item.setToolTip(detail)
            if not entry.loadable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_library_list.addItem(item)
        if entries:
            self.statusBar().showMessage(f"模型库已刷新：检测到 {len(entries)} 个模型目录")

    def _load_library_item(self, item: QListWidgetItem) -> None:
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if path:
            preset = item.data(Qt.ItemDataRole.UserRole + 1)
            self._load_model_path(
                path, preset if isinstance(preset, dict) else None
            )

    def open_model_library_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.model_library.root)))

    def import_model_folder_to_library(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择完整模型文件夹")
        if not selected:
            return
        source = Path(selected).expanduser().resolve()
        destination = self.model_library.root / source.name
        if source == destination.resolve():
            self.refresh_model_library(force=True)
            return
        if destination.exists():
            answer = QMessageBox.question(
                self,
                "覆盖模型",
                f"模型库中已存在“{source.name}”，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            shutil.rmtree(destination)
        try:
            shutil.copytree(source, destination)
            self.refresh_model_library(force=True)
            self.statusBar().showMessage(f"已导入模型库：{source.name}")
        except Exception as error:
            if destination.exists():
                shutil.rmtree(destination)
            QMessageBox.critical(self, "模型文件夹导入失败", str(error))

    def import_trajectory(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入轨迹", "", "Trajectories (*.csv *.npz)"
        )
        if not path:
            return
        self._import_trajectory_path(Path(path))

    def _import_trajectory_path(
        self, path: Path, *, preserve_project: bool = False, confirm: bool = True
    ) -> None:
        replacing_session = self.source is not None
        if confirm and replacing_session and not self._confirm_discard_or_save():
            return
        self.statusBar().showMessage(f"正在导入轨迹：{path}")
        QApplication.processEvents()
        try:
            try:
                source = load_trajectory(path)
            except ValueError as error:
                if "control frequency" not in str(error):
                    raise
                fallback_hz, ok = QInputDialog.getDouble(
                    self,
                    "控制频率",
                    "轨迹没有时间列，请输入 Hz",
                    100.0,
                    0.001,
                    100000.0,
                    3,
                )
                if not ok:
                    self.statusBar().showMessage("已取消轨迹导入")
                    return
                source = load_trajectory(path, fallback_hz)

            if replacing_session and not preserve_project:
                project = ProjectDocument(
                    Path(path).stem,
                    model_path=self.project.model_path,
                    model_adjustments=list(
                        self.project.model_adjustments
                    ),
                    joint_mapping=dict(self.project.joint_mapping),
                    mapping_transforms={
                        name: dict(transform)
                        for name, transform
                        in self.project.mapping_transforms.items()
                    },
                    joint_groups={
                        name: list(channels)
                        for name, channels
                        in self.project.joint_groups.items()
                    },
                    custom_part_modules=[
                        {
                            "id": str(module.get("id", "")),
                            "title": str(module.get("title", "")),
                            "joint_names": list(module.get("joint_names", [])),
                        }
                        for module in self.project.custom_part_modules
                    ],
                )
            else:
                project = self.project
                project.operations = []
                project.labels = []
                project.segments = []
                project.blocks = []
                if (
                    project.path is None
                    and project.name == "未命名工程"
                ):
                    project.name = Path(path).stem
            project.trajectory_path = str(Path(path).resolve())
            timeline_engine = TimelineEngine(
                source, validator=self._operation_validator
            )
            timeline_engine.render()
        except Exception as error:
            QMessageBox.critical(self, "轨迹加载失败", str(error))
            self.statusBar().showMessage("轨迹导入失败")
            return

        self.source = source
        self.project = project
        self.timeline_engine = timeline_engine
        self.frame = 0
        self.timeline_widget.clear_selected_range(emit=False)
        self.timeline_widget.clear_label_selection()
        self.channel_list.clear()
        self.channel_list.addItems(self.source.position_channels)
        self._update_motor_scope_status()
        self._refresh_timeline()
        frequency = self.source.frequency
        self.statusBar().showMessage(
            f"{self.source.frame_count} frames, {frequency.hz:.3f} Hz"
            + (" (非均匀)" if frequency.irregular else "")
        )
        self._maybe_configure_mapping()
        self._ensure_independent_joint_channels()
        self._reload_part_modules()
        self._apply_current_frame()
        self._set_dirty(True)

    def configure_mapping(self) -> None:
        if self.source is None or self.adapter is None:
            QMessageBox.information(self, "关节映射", "请先导入模型和轨迹")
            return
        dialog = JointMappingDialog(
            self.source.position_channels,
            self.adapter.info,
            self.project.joint_mapping,
            self.project.mapping_transforms,
            self,
        )
        if dialog.exec():
            self.project.joint_mapping = dialog.mapping()
            self.project.mapping_transforms = dialog.transforms()
            self._install_timeline_validator()
            self._set_dirty(True)
            self._apply_current_frame()

    def _maybe_configure_mapping(self) -> None:
        if self.source is None or self.adapter is None:
            return
        if not self.project.joint_mapping:
            self.project.joint_mapping = auto_map_channels(
                self.source.position_channels, self.adapter.info
            )
        mapped_source_channels = {
            channel
            for channel in self.project.joint_mapping
            if channel in self.source.position_channels
        }
        if len(mapped_source_channels) < len(self.source.position_channels):
            self.configure_mapping()

    def _sync_model_pose_session(self) -> None:
        if self.adapter is None:
            return
        generated = bool(
            self.source is not None
            and self.source.metadata.get("generated_hold_pose")
        )
        if self.source is None or (generated and not self.project.operations):
            self._install_rest_pose()
            return
        self._ensure_independent_joint_channels()

    def _install_rest_pose(self) -> None:
        if self.adapter is None:
            return
        joints = independent_scalar_joints(self.adapter.info)
        if not joints:
            return
        mapping = channel_mapping_for_joints(
            [joint.name for joint in joints],
            self.project.joint_mapping,
        )
        positions = independent_joint_positions(self.adapter)
        source = hold_pose_trajectory(
            {
                channel: positions.get(joint_name, 0.0)
                for channel, joint_name in mapping.items()
            }
        )
        source.metadata["generated_hold_pose"] = True
        self.source = source
        self.timeline_engine = TimelineEngine(
            source, validator=self._operation_validator
        )
        self.project.joint_mapping = mapping
        self.project.mapping_transforms = {
            channel: dict(transform)
            for channel, transform in self.project.mapping_transforms.items()
            if channel in mapping
        }
        self.project.trajectory_path = None
        if not self.project.joint_groups:
            self.project.joint_groups = default_joint_groups(mapping)
        self.frame = 0
        self.channel_list.clear()
        self.channel_list.addItems(source.position_channels)
        self._update_motor_scope_status()
        self._refresh_timeline()

    def _ensure_independent_joint_channels(self) -> None:
        if self.adapter is None or self.source is None:
            return
        positions = independent_joint_positions(self.adapter)
        mapped_joints = {
            joint
            for channel, joint in self.project.joint_mapping.items()
            if channel in self.source.channels
        }
        extras: dict[str, float] = {}
        for joint in independent_scalar_joints(self.adapter.info):
            if joint.name in mapped_joints:
                continue
            # A model preset may already reserve q8...q15 even when the
            # trajectory loaded first only contains q0...q7. Reuse that
            # declared channel name and materialize it as a constant track.
            reserved_channel = next(
                (
                    channel
                    for channel, mapped_joint
                    in self.project.joint_mapping.items()
                    if mapped_joint == joint.name
                ),
                None,
            )
            channel = reserved_channel or joint.name
            if channel in self.source.channels:
                existing_target = self.project.joint_mapping.get(channel)
                if existing_target in {None, joint.name}:
                    self.project.joint_mapping[channel] = joint.name
                    mapped_joints.add(joint.name)
                    continue
                # Avoid taking over an existing trajectory channel that is
                # already mapped to a different joint.
                base = joint.name
                channel = base
                suffix = 2
                while (
                    channel in self.source.channels
                    or channel in self.project.joint_mapping
                ):
                    channel = f"{base}_{suffix}"
                    suffix += 1
            self.project.joint_mapping[channel] = joint.name
            extras[channel] = positions.get(joint.name, 0.0)
        extended = extend_trajectory_with_constants(self.source, extras)
        if extended is not self.source:
            operations = (
                list(self.timeline_engine.operations)
                if self.timeline_engine is not None
                else []
            )
            labels = (
                list(self.timeline_engine.labels)
                if self.timeline_engine is not None
                else list(self.project.labels)
            )
            self.source = extended
            self.timeline_engine = TimelineEngine(
                extended,
                operations,
                labels,
                self._operation_validator,
            )
            self.channel_list.clear()
            self.channel_list.addItems(extended.position_channels)
            self._update_motor_scope_status()
            self._refresh_timeline()
        if not self.project.joint_groups:
            self.project.joint_groups = default_joint_groups(
                self.project.joint_mapping
            )
            self._refresh_joint_group_controls()

    def _reload_part_modules(self) -> None:
        if self.adapter is None:
            self.part_module_host.clear()
            self._refresh_part_module_menu()
            return
        self.part_module_host.set_joints(
            independent_scalar_joints(self.adapter.info),
            self.project.custom_part_modules,
        )
        self._refresh_part_module_menu()
        self._sync_part_module_windows_from_frame()

    def create_custom_part_module(self) -> None:
        if self.adapter is None:
            QMessageBox.information(
                self, "新建自定义模块", "请先加载模型"
            )
            return
        dialog = CustomPartModuleDialog(
            independent_scalar_joints(self.adapter.info), self
        )
        if not dialog.exec():
            return
        module = dialog.module_definition()
        self.project.custom_part_modules.append(module)
        self._set_dirty(True)
        self._reload_part_modules()
        module_id = str(module["id"])
        self.open_part_module(module_id)
        self.statusBar().showMessage(
            f"已创建自定义模块「{module['title']}」，"
            f"包含 {len(module['joint_names'])} 个电机"
        )

    def delete_custom_part_module(self, module_id: str) -> None:
        module = next(
            (
                item
                for item in self.project.custom_part_modules
                if str(item.get("id")) == module_id
            ),
            None,
        )
        if module is None:
            return
        title = str(module.get("title") or module_id)
        answer = QMessageBox.question(
            self,
            "删除自定义模块",
            f"确定删除自定义模块「{title}」吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.project.custom_part_modules = [
            item
            for item in self.project.custom_part_modules
            if str(item.get("id")) != module_id
        ]
        self._set_dirty(True)
        self._reload_part_modules()
        self.statusBar().showMessage(f"已删除自定义模块「{title}」")

    def open_part_module(self, module_id: str):
        window = self.part_module_host.open_module(module_id)
        if window is not None:
            self.part_module_dock.show()
            self._sync_part_module_windows_from_frame()
        return window

    def _refresh_part_module_menu(self) -> None:
        menu = getattr(self, "part_module_menu", None)
        if menu is None:
            return
        menu.clear()
        show_dock = QAction("显示部件模块", self)
        show_dock.triggered.connect(self.part_module_dock.show)
        menu.addAction(show_dock)
        create_module = QAction("新建自定义模块", self)
        create_module.triggered.connect(self.create_custom_part_module)
        menu.addAction(create_module)
        modules = self.part_module_host.modules
        if not modules:
            return
        menu.addSeparator()
        for module in modules:
            count = len(list(module["joint_names"]))  # type: ignore[arg-type]
            action = QAction(f"{module['title']} ({count})", self)
            action.triggered.connect(
                lambda _checked=False, mid=str(module["id"]): self.open_part_module(mid)
            )
            menu.addAction(action)

    def _mapped_joint_values_at_current_frame(self) -> dict[str, float]:
        if self.adapter is None:
            return {}
        values = scalar_joint_positions(self.adapter)
        data = self._rendered()
        if data is None:
            return values
        frame = int(np.clip(self.frame, 0, data.frame_count - 1))
        for channel, joint_name in self.project.joint_mapping.items():
            if channel not in data.channels:
                continue
            values[joint_name] = joint_value_from_channel(
                float(data.channels[channel][frame]),
                self.project.mapping_transforms.get(channel, {}),
            )
        return values

    def _sync_part_module_windows_from_frame(self) -> None:
        self.part_module_host.set_values(
            self._mapped_joint_values_at_current_frame(),
            skip=self._joint_pose_dragging,
        )

    def _channel_for_joint(self, joint_name: str) -> str | None:
        for channel, mapped in self.project.joint_mapping.items():
            if mapped == joint_name:
                return channel
        return None

    def _preview_joint_pose(self, joint_name: str, radians: float) -> None:
        if self.adapter is None:
            return
        self.playing = False
        self._joint_pose_dragging = joint_name
        values = self._mapped_joint_values_at_current_frame()
        values[joint_name] = radians
        try:
            self.adapter.apply_positions(values)
            self.viewport.refresh()
            if self.camera_preview.isVisible():
                self.camera_preview.refresh()
        except Exception as error:
            self.statusBar().showMessage(str(error))

    def _commit_joint_pose(self, joint_name: str, radians: float) -> None:
        self._joint_pose_dragging = None
        if self.adapter is None:
            return
        if self.source is None or self.timeline_engine is None:
            self._install_rest_pose()
        channel = self._channel_for_joint(joint_name)
        if channel is None:
            self.project.joint_mapping[joint_name] = joint_name
            if self.source is not None:
                extended = extend_trajectory_with_constants(
                    self.source, {joint_name: radians}
                )
                self.source = extended
                self.timeline_engine = TimelineEngine(
                    extended,
                    list(self.timeline_engine.operations)
                    if self.timeline_engine is not None
                    else [],
                    list(self.timeline_engine.labels)
                    if self.timeline_engine is not None
                    else [],
                    self._operation_validator,
                )
                self.channel_list.clear()
                self.channel_list.addItems(extended.position_channels)
            channel = joint_name
        named = {joint.name: joint for joint in self.adapter.info.joints}
        joint = named.get(joint_name)
        if joint is not None and joint.lower is not None:
            radians = max(radians, joint.lower)
        if joint is not None and joint.upper is not None:
            radians = min(radians, joint.upper)
        try:
            channel_value = channel_value_from_joint(
                radians,
                self.project.mapping_transforms.get(channel, {}),
            )
        except Exception as error:
            QMessageBox.critical(self, "部件模块", str(error))
            self._sync_part_module_windows_from_frame()
            return
        data = self._rendered()
        if data is None or channel not in data.channels:
            self._preview_joint_pose(joint_name, radians)
            self._joint_pose_dragging = None
            return
        operation = joint_pose_patch_operation(
            data,
            channel,
            channel_value,
            self.frame,
            self.timeline_widget.selected_range,
        )
        if self.timeline_widget.selected_range is not None:
            operation["direct_pose_override"] = True
        written_values = dict(operation.get("channel_values") or {}).get(
            channel, []
        )
        written_sample_count = len(written_values)  # type: ignore[arg-type]
        if self._add_operation(operation):
            selected = self.timeline_widget.selected_range
            if selected is None:
                message = f"已写入 {joint_name} = {radians:.4f} rad"
            else:
                message = (
                    f"已将 {joint_name} 在 {selected[0]:.3f}s → "
                    f"{selected[1]:.3f}s 直接设为 {radians:.4f} rad；"
                    f"共写入 {written_sample_count} 帧；"
                    "边界连续性请按需要手动平滑"
                )
            self.statusBar().showMessage(message)
        else:
            self._sync_part_module_windows_from_frame()

    def _rendered(self) -> TrajectoryData | None:
        return self.timeline_engine.render() if self.timeline_engine else None

    def _install_timeline_validator(self) -> None:
        if self.timeline_engine is not None:
            self.timeline_engine.set_validator(self._operation_validator)

    def _operation_validator(
        self,
        data: TrajectoryData,
        operation: dict[str, object],
    ) -> None:
        issues = []
        if self.adapter is not None and self.project.joint_mapping:
            before_report = ValidationReport()
            if self.timeline_engine is not None:
                before_report = validate_model_mapping(
                    self.timeline_engine.render(),
                    self.adapter,
                    self.project.joint_mapping,
                    self.project.mapping_transforms,
                )
            mapping_report = validate_model_mapping(
                data,
                self.adapter,
                self.project.joint_mapping,
                self.project.mapping_transforms,
            )
            issues.extend(
                filter_new_joint_limit_issues(before_report, mapping_report)
            )
        if bool(operation.get("strict_continuity")):
            channel_values = operation.get("channels")
            channels = (
                [str(value) for value in channel_values]  # type: ignore[union-attr]
                if channel_values
                else None
            )
            report = validate_operation_continuity(
                data, operation, channels
            )
            issues.extend(report.issues)
        if issues:
            detail = "\n".join(issue.message for issue in issues[:8])
            raise ValueError(detail)

    def _apply_current_frame(self) -> None:
        data = self._rendered()
        if data is None:
            return
        self.frame = int(np.clip(self.frame, 0, data.frame_count - 1))
        self.timeline_widget.set_current_time(float(data.times[self.frame]))
        if self.adapter and self.project.joint_mapping:
            values: dict[str, float] = {}
            for channel, joint in self.project.joint_mapping.items():
                if channel not in data.channels:
                    continue
                transform = self.project.mapping_transforms.get(channel, {})
                value = (
                    float(data.channels[channel][self.frame])
                    * transform.get("scale", 1.0)
                    + transform.get("offset", 0.0)
                )
                if "minimum" in transform:
                    value = max(value, transform["minimum"])
                if "maximum" in transform:
                    value = min(value, transform["maximum"])
                values[joint] = value
            try:
                self.adapter.apply_positions(values)
                self.viewport.refresh()
                if self.camera_preview.isVisible():
                    self.camera_preview.refresh()
            except Exception as error:
                self.statusBar().showMessage(str(error))
        self._sync_part_module_windows_from_frame()

    def seek_frame(self, frame: int) -> None:
        data = self._rendered()
        if data:
            self.frame = int(np.clip(frame, 0, data.frame_count - 1))
            self.playing = False
            self._apply_current_frame()

    def seek_time(self, seconds: float) -> None:
        data = self._rendered()
        if data:
            self.seek_frame(data.nearest_frame(seconds))

    def seek_first_frame(self) -> None:
        self.seek_frame(0)

    def seek_last_frame(self) -> None:
        data = self._rendered()
        if data is not None:
            self.seek_frame(data.frame_count - 1)

    def _sync_timeline_slider(
        self, visible_start: float, visible_end: float, duration: float
    ) -> None:
        visible_span = max(0.0, visible_end - visible_start)
        scrollable = duration > 0.0 and visible_span < duration - 1e-9
        self.timeline_slider.setEnabled(scrollable)
        maximum_start = max(0.0, duration - visible_span)
        value = (
            int(round(10000.0 * visible_start / maximum_start))
            if scrollable and maximum_start > 0.0
            else 0
        )
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(value)
        self.timeline_slider.setPageStep(
            max(1, int(round(10000.0 * visible_span / max(duration, 1e-9))))
        )
        self.timeline_slider.blockSignals(False)

    def toggle_play(self) -> None:
        data = self._rendered()
        if data is None:
            return
        self.playing = not self.playing
        self._preview_stop_time = None
        if self.playing:
            if self.frame >= data.frame_count - 1:
                self.frame = 0
            self.play_started_at = time.monotonic()
            self.play_started_time = float(data.times[self.frame])

    def _tick(self) -> None:
        data = self._rendered()
        if not self.playing or data is None:
            return
        seconds = self.play_started_time + time.monotonic() - self.play_started_at
        if self._preview_stop_time is not None and seconds >= self._preview_stop_time:
            seconds = self._preview_stop_time
            self.playing = False
            self._preview_stop_time = None
        self.frame = data.nearest_frame(seconds)
        if seconds >= data.duration:
            self.frame = data.frame_count - 1
            self.playing = False
        self._apply_current_frame()

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        title = "Motion Trajectory Studio"
        if self.project.path is not None:
            title += f" — {self.project.path.name}"
        self.setWindowTitle(title + (" *" if dirty else ""))

    def _confirm_discard_or_save(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.warning(
            self,
            "未保存修改",
            "当前工程有未保存修改。",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._did_fit_to_screen:
            self._did_fit_to_screen = True
            self._fit_to_available_screen()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            QTimer.singleShot(0, self._refresh_viewports_for_size)

    def _fit_to_available_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        chrome_w = max(0, frame.width() - self.width())
        chrome_h = max(0, frame.height() - self.height())
        width = min(self.width(), max(640, available.width() - chrome_w))
        height = min(self.height(), max(480, available.height() - chrome_h))
        x = available.x() + max(0, (available.width() - width - chrome_w) // 2)
        y = available.y() + max(0, (available.height() - height - chrome_h) // 2)
        self.setGeometry(x, y, width, height)

    def _refresh_viewports_for_size(self) -> None:
        for view in (self.viewport, self.camera_preview):
            view._resize_timer.stop()
            view._set_scaled_pixmap()
            view._resize_renderer()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard_or_save():
            event.ignore()
            return
        self.part_module_host.close_all()
        self._remove_copy_paste_shortcuts()
        self.viewport._close_renderer()
        self.camera_preview._close_renderer()
        self.camera_monitor.disconnect_all()
        self.camera_monitor.close()
        if self.hardware_client is not None:
            self.hardware_client.close()
            self.hardware_client = None
        if self._hardware_server_log_handle is not None:
            self._hardware_server_log_handle.close()
            self._hardware_server_log_handle = None
        event.accept()


def run_gui() -> int:
    application = QApplication.instance() or QApplication([])
    language_controller = install_ui_language(application)
    window = MainWindow()
    window.show()
    if language_controller is not None:
        language_controller.translate_all()
    return application.exec()
