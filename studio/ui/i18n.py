"""UI language at the presentation boundary.

Chinese is the default. Set MOTION_STUDIO_LANG=en to translate labels through
the existing phrase table. Editing and hardware paths stay language-agnostic.
"""

from __future__ import annotations

import os
import re

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QDockWidget, QGroupBox, QLabel,
    QLineEdit, QMainWindow, QMenu, QTabWidget, QTableWidget, QWidget,
)

from studio.ui.style import apply_application_style

_HAN = re.compile(r"[\u3400-\u9fff]")

# Exact labels and reusable phrases. Longest phrases are substituted first so
# formatted status messages and diagnostic details are translated as well.
_EN = {
    "未命名工程": "Untitled Project",
    "导入 MJCF/XML 或 URDF 模型": "Import an MJCF/XML or URDF model",
    "左键拖动：旋转｜Shift+左键/中键：平移｜右键拖动/滚轮：缩放｜双击：重置视角": "Left drag: rotate | Shift+left/middle drag: pan | Right drag/wheel: zoom | Double-click: reset view",
    "末端增量拖拽：左键拖动逐步调整屏幕平面位置，滚轮逐步调整深度，松开左键提交，Esc取消": "Incremental end-effector drag: left-drag in the screen plane, use the wheel for depth, release to apply, Esc to cancel",
    "框选时间段后可命名并换色；分段栏显示已保存的彩色区间": "Select a time range to name and color it; saved ranges appear in the segment track",
    "未单独选择电机：操作作用于全部位置通道": "No individual motors selected: operation applies to all position channels",
    "按一下单步点动；按住按钮连续点动。坐标相对所选基座。": "Click for one jog step; hold for continuous jog. Coordinates are relative to the selected base.",
    "将每个位置通道映射到模型关节。未映射通道不会驱动模型。": "Map each position channel to a model joint. Unmapped channels do not drive the model.",
    "温度检测模式（强制从头完整执行并记录曲线）": "Temperature test mode (run the complete trajectory from the start and record curves)",
    "高动态真机模式（摇酒等高速动作：3.5 rad/s，60 rad/s²）": "High-dynamics hardware mode (fast motions: 3.5 rad/s, 60 rad/s²)",
    "缩放时间轴后的可见区间位置": "Position of the visible range after timeline zoom",
    "在选区主体上拖动，可将所选关节片段移动到静止空白区": "Drag the selection body to move the selected joint clip into a stationary empty range",
    "将播放头跳到整个轨迹的第一帧": "Move the playhead to the first trajectory frame",
    "将播放头跳到整个轨迹的最后一帧": "Move the playhead to the last trajectory frame",
    "在自由视角旁同步显示模型摄像机": "Show a model camera next to the free view",
    "导入模型和轨迹开始编辑": "Import a model and trajectory to start editing",
    "/dev/videoN 或 HTTP(S) MJPEG 地址": "/dev/videoN or HTTP(S) MJPEG URL",
    "动作分段标注": "Action Segment Annotation",
    "轨迹通道映射": "Trajectory Channel Mapping",
    "末端 6D 按钮示教": "End-Effector 6D Button Teaching",
    "完整轨迹温度曲线（橙色 MOS / 蓝色转子）": "Full-Trajectory Temperature Curves (orange: MOS / blue: rotor)",
    "模型库（双击加载）": "Model Library (double-click to load)",
    "真机关节温度": "Hardware Joint Temperatures", "真机相机": "Hardware Camera",
    "关节/通道": "Joints / Channels", "关节位姿": "Joint Pose", "编辑工具": "Editing Tools",
    "部件模块": "Part Modules", "显示部件模块": "Show Part Modules", "打开": "Open",
    "左手": "Left Hand", "右手": "Right Hand", "左臂": "Left Arm", "右臂": "Right Arm",
    "其他关节": "Other Joints", "拇指": "Thumb", "食指": "Index", "中指": "Middle",
    "无名指": "Ring", "小指": "Pinky", "其他": "Other",
    "CMC 偏航": "CMC Yaw", "CMC 俯仰": "CMC Pitch",
    "MCP 俯仰": "MCP Pitch", "MCP 偏航": "MCP Yaw", "MCP": "MCP",
    "关节 1": "Joint 1", "关节 2": "Joint 2", "关节 3": "Joint 3",
    "关节 4": "Joint 4", "关节 5": "Joint 5", "关节 6": "Joint 6",
    "关节 7": "Joint 7", "夹爪 1": "Gripper 1", "夹爪 2": "Gripper 2",
    "拇指 CMC 偏航": "Thumb CMC Yaw", "拇指 CMC 俯仰": "Thumb CMC Pitch",
    "拇指 IP": "Thumb IP", "食指 MCP 俯仰": "Index MCP Pitch", "食指 DIP": "Index DIP",
    "中指 MCP 俯仰": "Middle MCP Pitch", "中指 DIP": "Middle DIP",
    "无名指 MCP 俯仰": "Ring MCP Pitch", "无名指 DIP": "Ring DIP",
    "小指 MCP 俯仰": "Pinky MCP Pitch", "小指 DIP": "Pinky DIP",
    "拖动滑条可预览灵巧手/手臂关节；松开后写入当前轨迹。": "Drag a slider to preview a hand or arm joint; release to write it into the current trajectory.",
    "拖动滑条可预览；松开后写入当前轨迹。": "Drag a slider to preview; release to write it into the current trajectory.",
    "在右侧「部件模块」点「打开」调节关节；": "On the right, click Open in Part Modules to pose joints; ",
    "3D 窗口左键拖动只旋转视角。": "left-drag in the 3D view only rotates the camera.",
    "请先加载模型": "Load a model first",
    "关节组": "Joint Groups", "空": "Empty", "快速关节组（Alt+1…Alt+9）": "Quick Joint Groups (Alt+1…Alt+9)",
    "保存/覆盖组": "Save / Overwrite Group", "重命名组": "Rename Group",
    "删除组": "Delete Group", "重命名关节组": "Rename Joint Group",
    "删除关节组": "Delete Joint Group", "保存关节组": "Save Joint Group",
    "选择已保存的关节组；Alt+1 到 Alt+9 可直接切换": "Select a saved joint group; use Alt+1 through Alt+9 to switch directly",
    "新建工程": "New Project", "导入模型文件夹到模型库": "Import Model Folder into Library",
    "打开模型库文件夹": "Open Model Library Folder", "导入模型": "Import Model",
    "刷新模型库": "Refresh Model Library", "导入轨迹": "Import Trajectory",
    "打开工程": "Open Project", "保存可迁移工程": "Save Portable Project",
    "工程另存为": "Save Project As", "保存工程": "Save Project",
    "导出轨迹": "Export Trajectory", "打包工程": "Package Project",
    "添加标签": "Add Label", "重命名所选标签": "Rename Selected Label",
    "所选标签设为选区": "Set Selected Labels as Range",
    "命名/换色当前选区": "Name / Color Current Range",
    "编辑所选区间名称/颜色": "Edit Selected Range Name / Color",
    "删除动作分段": "Delete Action Segment", "保存关节组": "Save Joint Group",
    "裁剪保留": "Trim to Selection", "删除片段": "Delete Segment",
    "复制时间片段": "Copy Time Segment", "复制片段": "Copy Segment",
    "粘贴到播放头": "Paste at Playhead", "粘贴片段到播放头": "Paste Segment at Playhead",
    "循环片段": "Loop Segment", "增加空白等待": "Add Hold",
    "片段变速": "Change Segment Speed", "选区五次平滑": "Quintic Smooth Selection",
    "所选标签间五次平滑": "Quintic Smooth Between Labels",
    "高速相位平滑（不减速）": "High-Speed Phase Smoothing (no slowdown)",
    "拼接另一轨迹": "Splice Another Trajectory", "末端拖拽调整": "End-Effector Drag Adjustment",
    "播放/暂停": "Play / Pause", "上一帧": "Previous Frame", "下一帧": "Next Frame",
    "时间线放大": "Zoom In Timeline", "时间线缩小": "Zoom Out Timeline",
    "重置视角": "Reset View", "关节映射": "Joint Mapping", "验证": "Validate",
    "启动/连接真机服务": "Start / Connect Hardware Service",
    "从当前帧执行": "Execute from Current Frame", "真机停止/释放": "Stop / Release Hardware",
    "连接真机相机": "Connect Hardware Camera", "断开": "Disconnect",
    "多相机监视器": "Multi-Camera Monitor", "添加相机": "Add Camera",
    "添加检测到的本机相机": "Add Detected Local Cameras",
    "全部连接": "Connect All", "全部断开": "Disconnect All",
    "连接": "Connect", "移除": "Remove", "未命名相机": "Unnamed Camera",
    "设备或流地址": "Device or Stream URL",
    "选择全部": "Select All", "清除选择": "Clear Selection", "提交示教": "Apply Teaching",
    "平移步长": "Translation Step", "旋转步长": "Rotation Step",
    "平移 XYZ": "Translation XYZ", "旋转 Rx / Ry / Rz": "Rotation Rx / Ry / Rz",
    "前 +X": "Forward +X", "后 −X": "Backward −X", "左 +Y": "Left +Y",
    "右 −Y": "Right −Y", "上 +Z": "Up +Z", "下 −Z": "Down −Z",
    "框选起点": "Selection Start", "框选终点": "Selection End",
    "选区原时长": "Original Selection Duration", "本次作用电机": "Affected Motors",
    "全部位置通道": "All Position Channels", "推荐最低时长": "Recommended Minimum Duration",
    "预计轨迹总时长": "Estimated Total Duration",
    "预计总轨迹时长变化": "Estimated Total Duration Change",
    "分段名称": "Segment Name", "选择自定义颜色…": "Choose Custom Color…",
    "选择区间颜色": "Choose Range Color", "例如：抓取、放置、回零": "e.g. Pick, Place, Home",
    "自定义": "Custom", "起点": "Start", "终点": "End", "时长": "Duration",
    "颜色": "Color", "未映射": "Unmapped",
    "等待真机温度遥测": "Waiting for hardware temperature telemetry",
    "没有足够的温度样本": "Not enough temperature samples",
    "整体/MOS ℃": "Overall/MOS °C", "转子 ℃": "Rotor °C", "关节": "Joint",
    "状态": "Status", "左夹爪": "Left Gripper", "右夹爪": "Right Gripper",
    "左臂": "Left Arm", "右臂": "Right Arm", "不显示": "Hidden",
    "尚未连接真机相机": "Hardware camera not connected",
    "真机相机已断开": "Hardware camera disconnected",
    "正在连接真机相机": "Connecting to hardware camera", "正在打开": "Opening",
    "找不到相机": "Camera not found", "可用设备": "Available devices",
    "未检测到本机相机": "No local cameras detected",
    "相机流已关闭": "Camera stream closed", "真机相机错误": "Hardware camera error",
    "真机相机已连接，等待首帧…": "Hardware camera connected; waiting for first frame…",
    "真机相机断线，1秒后重连": "Hardware camera disconnected; reconnecting in 1 second",
    "模型已加载，离屏渲染不可用": "Model loaded, but offscreen rendering is unavailable",
    "渲染尺寸调整失败": "Failed to resize renderer", "渲染失败": "Rendering failed",
    "模型中没有摄像机": "The model has no camera named",
    "时间": "Time", "分段": "Segments", "轨迹": "Trajectory",
    "⏮ 首帧": "⏮ First Frame", "尾帧 ⏭": "Last Frame ⏭",
    "文件": "File", "编辑": "Edit", "播放": "Play", "摄像机": "Camera",
    "撤销": "Undo", "重做": "Redo",
    "蓝": "Blue", "绿": "Green", "橙": "Orange", "紫": "Purple",
    "红": "Red", "青": "Cyan", "黄": "Yellow", "粉": "Pink",
    "正常": "Normal", "故障": "Fault", "过压": "Overvoltage", "欠压": "Undervoltage",
    "过流": "Overcurrent", "MOS过温": "MOS overtemperature",
    "转子过温": "Rotor overtemperature", "通信丢失": "Communication lost", "过载": "Overload",
    "未保存修改": "Unsaved Changes", "当前工程有未保存修改。": "The current project has unsaved changes.",
    "编辑失败": "Edit Failed", "末端拖拽失败": "End-Effector Drag Failed",
    "真机安全保护触发": "Hardware safety protection triggered",
    "真机轨迹加载失败": "Failed to load hardware trajectory",
    "真机初始过渡生成失败": "Failed to generate hardware initial transition",
    "真机执行失败": "Hardware Execution Failed", "连接真机服务失败": "Failed to Connect Hardware Service",
    "真机回放状态": "Hardware Playback Status", "真机执行": "Hardware Execution",
    "真机停止": "Hardware Stop", "停止并释放真机": "Stop and Release Hardware",
    "启动真机控制服务": "Start Hardware Control Service",
    "真机控制端口被占用": "Hardware Control Port Is in Use", "配置真机 CAN": "Configure Hardware CAN",
    "该数值仅表示框选平滑段编辑后的时长，不是轨迹总时长。改变时长会重采样选区内全部通道并移动后续标签。": "This value is the edited duration of the selected smoothing segment, not the total trajectory duration. Changing it resamples all channels in the selection and moves later labels.",
    "确认真机执行": "Confirm Hardware Execution",
    "这会停止独立真机控制服务并进入阻尼释放。是否继续？": "This stops the standalone hardware service and enters damped release. Continue?",
    "独立真机控制服务就绪": "Standalone hardware control service ready",
    "真机时间轴已加载": "Hardware timeline loaded",
    "真机执行起点已准备（尚未运动）": "Hardware start frame prepared (no motion yet)",
    "真机执行起点准备失败": "Failed to prepare hardware start frame",
    "正在进行自适应五次多项式过渡": "Running adaptive quintic transition",
    "真机已接受轨迹": "Hardware accepted the trajectory",
    "已生成 3 秒真机初始过渡": "Generated 3-second hardware initial transition",
    "真机动作执行中": "Hardware trajectory executing",
    "真机动作完成，保持末帧": "Hardware trajectory finished; holding final frame",
    "真机服务忙，命令被拒绝": "Hardware service busy; command rejected",
    "真机动态限制已配置": "Hardware dynamic limits configured",
    "上位机心跳中断，真机已停止并阻尼释放": "Host heartbeat lost; hardware stopped and released with damping",
    "真机已停止并完成阻尼释放": "Hardware stopped and damped release completed",
    "选择完整模型文件夹": "Select Complete Model Folder", "标签标题": "Label Title",
    "重命名标签": "Rename Label", "标签名称": "Label Name", "组名称": "Group Name",
    "平滑段新时长 (s)": "New Smoothing Segment Duration (s)", "循环": "Loop",
    "重复次数": "Repeat Count", "维持当前姿态的持续时间(s)": "Hold Current Pose Duration (s)",
    "变速": "Speed Change", "速度倍数": "Speed Factor",
    "标签间五次平滑": "Quintic Smoothing Between Labels", "五次平滑参数": "Quintic Smoothing Parameters",
    "高速相位平滑": "High-Speed Phase Smoothing", "选择要拼接的轨迹": "Select Trajectory to Splice",
    "轨迹拼接": "Trajectory Splice", "选择基座body": "Select Base Body",
    "选择要编辑的末端body": "Select End-Effector Body to Edit",
    "选择两个末端共同参考的基座 body": "Select the Base Body Shared by Both End Effectors",
    "选择主末端（完整保留其原始高频运动）": "Select Primary End Effector (fully preserve its original high-frequency motion)",
    "选择从末端（通过 IK 保持相对位姿）": "Select Secondary End Effector (preserve relative pose through IK)",
    "没有收到可定位的 IK 失败采样详情": "No locatable IK failure sample details were received",
    "末端拖拽": "End-Effector Drag", "末端示教模式": "End-Effector Teaching Mode",
    "选择操作方式": "Select Operation Mode", "末端拖拽已取消": "End-effector drag canceled",
    "验证通过": "Validation passed", "真机回放客户端已启动": "Hardware playback client started",
    "请输入 /dev/videoN 或 MJPEG 地址": "Enter a /dev/videoN device or MJPEG URL",
    "正在启动真机服务，日志：": "Starting hardware service; log: ", "；等待 ready 状态": "; waiting for ready state",
    "请输入当前系统用户的 sudo 密码。\n密码仅传给 sudo 完成一次 CAN 配置，不会保存或写入日志：": "Enter the current system user's sudo password.\nThe password is passed only to sudo for one CAN setup and is never saved or logged:",
    "请先加载轨迹": "Load a trajectory first", "当前帧之后没有可执行的动作": "No executable motion after the current frame",
    "真机执行不可用": "Hardware Execution Unavailable",
    "请确认机械臂工作空间无人、急停可用且 CAN 控制服务已独占。": "Confirm that the robot workspace is clear, the emergency stop is available, and the CAN control service has exclusive access.",
    "尚未连接真机回放服务": "Hardware playback service is not connected",
    "已发送真机停止/阻尼释放命令": "Hardware stop/damped-release command sent",
    "已新建工程：": "Created project: ", "覆盖模型": "Replace Model",
    "正在导入轨迹：": "Importing trajectory: ", "请先导入模型和轨迹": "Import a model and trajectory first",
    "高频相位拼接片段至少需要 12 帧": "High-frequency phase splice requires at least 12 frames",
    "高频相位拼接没有位置通道": "High-frequency phase splice has no position channels",
    "高速相位平滑选区至少需要 8 帧和一个位置通道": "High-speed phase smoothing requires at least 8 frames and one position channel",
    "选区前后没有足够上下文用于高速相位分析": "Insufficient context around the selection for high-speed phase analysis",
    "上下文过短，无法估计高速运动周期": "Context is too short to estimate the high-speed motion period",
    "五次时间映射输出采样数不一致": "Quintic time mapping produced an inconsistent sample count",
    "五次变速参数无效": "Invalid quintic speed-change parameters",
    "五次时间映射产生了无效速度倍率": "Quintic time mapping produced an invalid speed factor",
    "当前选区过短或倍速过高，无法生成单调五次时间映射": "The selection is too short or speed factor too high for a monotonic quintic time map",
    "变速选区或速度倍数无效": "Invalid speed-change selection or speed factor",
    "变速后的片段至少需要保留 7 帧才能维持 C2 连续；请降低速度倍数或扩大选区": "The retimed segment needs at least 7 frames to remain C2 continuous; reduce the speed factor or enlarge the selection",
    "平滑后的轨迹过大，可能耗尽内存：预计 ": "The smoothed trajectory is too large and may exhaust memory: estimated ",
    " 通道。当前工程建议该选区最长约 ": " channels. Recommended maximum selection duration for this project: ",
    "s；请缩短时长或先降低轨迹采样率。": "s; shorten the duration or reduce the sample rate first.",
    "末端偏移要求选区起点 < 红色播放头 <= 选区终点": "End-effector offset requires selection start < red playhead <= selection end",
    "末端约束区间过短": "End-effector constraint interval is too short",
    "双末端相对位姿约束区间过短": "Dual-end-effector relative-pose constraint interval is too short",
    "采样末端姿态": "Sampling end-effector poses", "批量逆运动学": "Batch inverse kinematics",
    "变速时间映射不是有限、严格单调且端点完整的映射": "Speed-change time mapping is not finite, strictly monotonic, and endpoint-complete",
    "详细诊断信息如下，可滚动查看并复制：": "Detailed diagnostics are shown below; scroll to review or copy:",
}

_REPLACEMENTS = sorted(
    ((source, target) for source, target in _EN.items()),
    key=lambda pair: len(pair[0]), reverse=True,
)


def language() -> str:
    value = os.environ.get("MOTION_STUDIO_LANG", "zh").strip().lower()
    return "en" if value == "en" else "zh"


def ui_text(value: str) -> str:
    """Return localized UI text; unknown technical detail is left intact."""
    if language() != "en" or not value or not _HAN.search(value):
        return value
    if value in _EN:
        return _EN[value]
    translated = value
    for source, target in _REPLACEMENTS:
        if source in translated:
            translated = translated.replace(source, target)
    return translated


class UiLanguageController(QObject):
    def __init__(self, application: QApplication) -> None:
        super().__init__(application)
        self._application = application
        application.installEventFilter(self)
        self._timer = QTimer(self)
        # Dynamic status labels need localization too. A modest polling rate
        # keeps those current without adding work to the rendering hot path.
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.translate_all)
        self._timer.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in (
            QEvent.Type.Show,
            QEvent.Type.Polish,
            QEvent.Type.ChildAdded,
        ):
            QTimer.singleShot(0, self.translate_all)
        return False

    def translate_all(self) -> None:
        for window in self._application.topLevelWidgets():
            if window.isVisible():
                self.translate_tree(window)

    def translate_tree(self, root: QObject) -> None:
        self._translate_object(root)
        for child in root.findChildren(QObject):
            self._translate_object(child)

    @staticmethod
    def _translate_object(obj: QObject) -> None:
        def replace(getter, setter) -> None:  # type: ignore[no-untyped-def]
            current = getter()
            translated = ui_text(current)
            if translated != current:
                setter(translated)

        if isinstance(obj, QWidget):
            replace(obj.toolTip, obj.setToolTip)
            replace(obj.statusTip, obj.setStatusTip)
            replace(obj.whatsThis, obj.setWhatsThis)
            if obj.windowTitle():
                replace(obj.windowTitle, obj.setWindowTitle)
        if isinstance(obj, (QLabel, QAbstractButton)):
            replace(obj.text, obj.setText)
        if isinstance(obj, QLineEdit):
            replace(obj.placeholderText, obj.setPlaceholderText)
        if isinstance(obj, QGroupBox):
            replace(obj.title, obj.setTitle)
        if isinstance(obj, QDockWidget):
            replace(obj.windowTitle, obj.setWindowTitle)
        if isinstance(obj, QAction):
            replace(obj.text, obj.setText)
            replace(obj.toolTip, obj.setToolTip)
            replace(obj.statusTip, obj.setStatusTip)
        if isinstance(obj, QMenu):
            replace(obj.title, obj.setTitle)
        if isinstance(obj, QComboBox):
            for index in range(obj.count()):
                current = obj.itemText(index)
                translated = ui_text(current)
                if translated != current:
                    obj.setItemText(index, translated)
        if isinstance(obj, QTabWidget):
            for index in range(obj.count()):
                current = obj.tabText(index)
                translated = ui_text(current)
                if translated != current:
                    obj.setTabText(index, translated)
        if isinstance(obj, QTableWidget):
            for column in range(obj.columnCount()):
                item = obj.horizontalHeaderItem(column)
                if item is not None:
                    current = item.text()
                    translated = ui_text(current)
                    if translated != current:
                        item.setText(translated)
            for row in range(obj.rowCount()):
                item = obj.verticalHeaderItem(row)
                if item is not None:
                    current = item.text()
                    translated = ui_text(current)
                    if translated != current:
                        item.setText(translated)
                for column in range(obj.columnCount()):
                    cell = obj.item(row, column)
                    if cell is not None:
                        current = cell.text()
                        translated = ui_text(current)
                        if translated != current:
                            cell.setText(translated)
        if isinstance(obj, QMainWindow):
            replace(obj.windowTitle, obj.setWindowTitle)


def install_ui_language(application: QApplication) -> UiLanguageController:
    """Install shared UI polish and optional English localization."""
    apply_application_style(application)
    controller = UiLanguageController(application)
    setattr(application, "_motion_studio_language_controller", controller)
    return controller
