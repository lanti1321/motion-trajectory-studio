# Motion Studio hardware controller

This is the trajectory editor's independent real-time hardware path. It vendors
only the Apache-2.0 OpenArmCAN transport/driver and does not build, link, launch,
or read any sibling project at runtime.

The UDP state machine is `load -> prepare(frame) -> execute`. `prepare` never
moves hardware. `execute` samples the measured pose, creates an adaptive
minimum-three-second quintic transition, then continues from the prepared
timeline frame. Loss of GUI
heartbeat, motor fault, excessive temperature/tracking error, or an explicit
stop causes damping release and service termination.

## 中文

这是轨迹编辑器自带的独立实时真机链路。工程内已经包含 Apache-2.0 许可的
OpenArmCAN 传输与驱动代码；运行时不会构建、链接、启动或读取任何同级项目。

UDP 状态机为 `load -> prepare(frame) -> execute`。`prepare` 只准备播放帧，不会
驱动真机；`execute` 读取实测位姿，生成至少 3 秒且可按动力学限制自动延长的五次
过渡，然后从准备好的时间轴帧继续执行。GUI 心跳丢失、电机故障、温度/跟踪误差
超限或显式停止都会触发阻尼释放并终止服务。
