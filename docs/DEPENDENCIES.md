# Dependency Audit / 依赖审计

## English

The repository is self-contained with respect to application source code.
It does not import, build, launch, or read `vr_control` or another sibling
repository.

Runtime inputs supplied by the machine are normal platform dependencies:

- Python 3.10+ and the packages pinned in `requirements-lock.txt`;
- MuJoCo model/trajectory files selected by the user;
- for real hardware only: CMake, a C++17 compiler, Linux SocketCAN, `sudo` for
  CAN interface configuration, and the configured CAN devices;
- camera support supplied by Qt Multimedia/V4L2.

The OpenArmCAN source used by the standalone controller is vendored under
`hardware_controller/vendor/openarm_can`. All launchers resolve paths relative
to their own repository location; no developer-machine absolute path is used.

## 中文

本仓库在应用源码层面是独立的，不会导入、构建、启动或读取 `vr_control` 以及其他
同级仓库。

仍需由运行机器提供的都是常规平台依赖：

- Python 3.10+ 和 `requirements-lock.txt` 中锁定的依赖；
- 用户选择的 MuJoCo 模型和轨迹素材；
- 仅真机模式需要：CMake、C++17 编译器、Linux SocketCAN、配置 CAN 所需的
  `sudo` 权限，以及对应 CAN 设备；
- 相机由 Qt Multimedia/V4L2 提供支持。

独立真机控制器使用的 OpenArmCAN 源码已包含在
`hardware_controller/vendor/openarm_can`。所有启动脚本都以自身所在目录解析路径，
不包含开发机器的绝对路径。
