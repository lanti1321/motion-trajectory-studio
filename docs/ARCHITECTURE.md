# Architecture / 架构说明

## English

Motion Trajectory Studio separates editing, rendering, hardware execution, and
device I/O so a UI change cannot silently change the real-time controller.

```text
app.py / studio.cli
├── studio.ui                 Qt interface, timeline, MuJoCo views, localization
├── studio.core               project document, sampled trajectory, time engine
├── studio.editing            non-destructive operations and kinematics
├── studio.validation         continuity, limits, dynamics, mapping, collision
├── studio.io                 CSV/NPZ/project import, export, portable bundles
├── studio.models             MuJoCo adapter and model library
├── studio.hardware           UDP client, clip export, telemetry decoding
└── hardware_controller       independent 1 kHz OpenArmCAN service
```

Key boundaries:

- Editing operations are serialized in `.motionproj`; source trajectories are
  never overwritten.
- MuJoCo preview follows the rendered editor timeline. Hardware feedback is
  monitoring data and does not rewrite the simulation pose.
- The hardware service owns CAN and real-time command output. The editor sends
  load/prepare/execute/stop commands over UDP.
- Execution starts with an adaptive, minimum-three-second, rest-to-rest quintic
  transition from measured joints to the selected editor frame. The service
  then follows exported position and velocity samples without online reshaping.
- Position, velocity, acceleration, heartbeat, motor-fault, and temperature
  checks remain active at the hardware boundary.

Language support lives in `studio/ui/i18n.py`. Chinese is the default;
`MOTION_STUDIO_LANG=en` activates English. Both languages share every editing
and hardware code path. Global font selection lives in `style.py`; all user
diagnostics use `message_box.py`, which provides wrapped, selectable messages
and a scrollable detail view for long validation reports.

Within `studio.ui`, `MainWindow` is now a composition root. Hardware lifecycle
and telemetry are isolated in `hardware_control.py`; project persistence,
export, packaging, and full validation live in `project_actions.py`; edit
commands live in `editing_actions.py`; labels, selections, undo/redo, and
timeline refresh live in `timeline_actions.py`; end-effector interaction lives
in `end_effector_control.py`.

The editing package keeps `operations.py` as a stable public facade and
dispatcher. Implementations are grouped into `basic_operations.py`,
`phase_operations.py`, `retiming.py`, `transitions.py`, and shared trajectory
rebuilding helpers in `operation_data.py`. Timeline painting and curve
envelopes are separated from mouse interaction in `timeline_paint.py`.

## 中文

Motion Trajectory Studio 将编辑、渲染、真机执行和设备 I/O 分层，避免界面修改
隐式改变实时控制器行为。

- `studio.ui`：Qt 界面、时间轴、MuJoCo 视图和语言层。
- `studio.core`：工程文档、采样轨迹和时间引擎。
- `studio.editing`：非破坏编辑操作和运动学。
- `studio.validation`：连续性、限位、动力学、映射和碰撞检查。
- `studio.io`：CSV/NPZ/工程导入导出及可迁移打包。
- `studio.models`：MuJoCo 适配器和模型库。
- `studio.hardware`：UDP 客户端、执行片段导出和遥测解析。
- `hardware_controller`：独立的 1 kHz OpenArmCAN 真机服务。

关键边界：源轨迹不被覆盖；仿真按编辑器渲染后的时间轴播放，真机反馈只用于
监测；CAN 和实时下发仅由独立服务负责。真机从选定帧执行前，先由实测关节位姿
生成不少于 3 秒的静止到静止自适应五次过渡，之后严格播放导出的位置/速度样本，
不进行在线轨迹变形。位置、速度、加速度、心跳、电机故障和温度保护均保留在真机
边界。

语言层位于 `studio/ui/i18n.py`。默认中文，设置 `MOTION_STUDIO_LANG=en` 后使用
英文界面；两种界面完全共用编辑和真机代码。全局字体策略位于 `style.py`；所有
用户可见诊断统一使用 `message_box.py`，短消息自动换行并可复制，长报告使用可
滚动的详情框，避免报错内容被窗口截断。

`studio.ui` 内部的 `MainWindow` 现在作为界面组合入口：真机生命周期和遥测位于
`hardware_control.py`，工程持久化、导出、打包和完整验证位于
`project_actions.py`，编辑命令位于 `editing_actions.py`，标签、选区、撤销/重做及
时间轴刷新位于 `timeline_actions.py`，末端交互位于
`end_effector_control.py`。

编辑包继续以 `operations.py` 作为稳定的公共入口和操作分发器；具体实现拆分到
`basic_operations.py`、`phase_operations.py`、`retiming.py`、
`transitions.py`，共享的轨迹重建工具位于 `operation_data.py`。时间轴的绘制和
曲线包络已迁到 `timeline_paint.py`，与鼠标选择交互分离。
