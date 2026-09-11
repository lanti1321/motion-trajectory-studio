# Release Test Matrix / 发布测试矩阵

## Automated / 自动测试

| Area / 范围 | Covered behavior / 已覆盖行为 |
|---|---|
| Trajectory I/O | CSV/NPZ import, frequency inference, reserved names, atomic export, source-overwrite protection |
| Project | save/load, relative assets, portable copy rollback, archive packaging, splice path rebasing |
| Basic editing | global and per-joint trim/delete/hold/loop/copy/paste, per-joint clip drag/move, occupied-target rejection, labels, undo/redo, transactional failure rollback |
| Retiming | constant speed, quintic speed ramps, partial-motor duration, nonlinear label remapping |
| Smoothing | quintic C1/C2 boundaries, high-frequency residual mode, phase smoothing, limit-aware duration, memory guard |
| Splicing | append, insert/replace, shared phase, transition recommendation and boundary limiting |
| Kinematics | batch IK, TCP offset, offset window, keyframe constraints, relative dual-end pose, branch/error protection |
| Timeline UI | range drag, zoom coordinates, right-click clear, label multi-select, first/last frame, quick joint groups/shortcuts, curve envelope and motion blocks |
| Validation | time irregularity, stalls, continuity, mapping limits, collision/contact details, velocity-channel consistency |
| Hardware protocol | clip layout, maintained velocity export, commands, telemetry parsing, motor faults, initial 3-second bridge |
| UI infrastructure | Chinese/English startup text, delayed global-name detection, readable/scrollable error dialogs, font setup |
| Multi-camera UI | independent source panels, duplicate prevention, grid placement, disconnect-all lifecycle |
| OpenArm O6 hands | rest-pose generation, part-module inspectors, independent finger channels, DIP/IP equalities |

## Safe build checks / 安全构建检查

- Import and byte-compile every Python module.
- Build the standalone C++ hardware service without running it.
- Run `app.py doctor` and shell syntax checks.
- Scan runtime source for sibling-project imports and developer absolute paths.

## Hardware-dependent manual checks / 依赖设备的人工检查

The following cannot be truthfully completed in an offscreen automated run:

- Opening the actual `/dev/videoN` YUYV/UYVY stream and judging image quality.
- CAN enable, measured joint/temperature feedback, emergency stop, watchdog and
  physical trajectory playback.
- Visual collision clearance on the real workcell.

这些项目需要真实相机、CAN 设备和已清空的机械臂工作区，不能在无设备自动测试中
假装完成。真机测试应先使用短、低能量轨迹，并保持急停可触达。
