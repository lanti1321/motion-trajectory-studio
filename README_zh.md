# Motion Trajectory Studio

一款仅提供英文界面的机器人采样动作轨迹桌面编辑器，集成 MuJoCo 预览，并可选
使用独立的 OpenArm 真机回放服务。模型、关节数量、通道名称和采样频率均未硬编码。

## 主要功能

- 导入 MJCF/XML/URDF 模型和 CSV/NPZ 轨迹。
- 自动或手动关节映射，支持比例、偏移和关节限位。
- 实时 MuJoCo 预览、模型摄像机、独立多相机监视器、逐帧定位，以及可缩放的
  多轨时间轴。
- 命名和彩色选区、支持 `Alt+1` 至 `Alt+9` 的关节编组、逐关节编辑范围、
  片段拖拽，以及撤销/重做。
- 裁剪、删除、复制/粘贴、循环、等待、变速、五次多项式平滑、高速相位平滑、
  轨迹拼接和 6D 末端编辑。
- 连续性、有限值、跳变、速度、加速度、关节限位、映射和采样碰撞验证。
- 可迁移的 `.motionproj` 工程和 `.tar.gz` 打包文件。
- 可选从当前播放头开始真机回放、MOS/转子温度监测、完整运行温度曲线，以及
  多路真实相机流。

## 安装与运行

### 前置条件

- 带 X11 或 Wayland 桌面会话的 Linux 系统；推荐 Ubuntu 24.04，Ubuntu 22.04
  需要另外安装 Python 3.12+。
- Python 3.12 或更高版本；锁定的 NumPy 和 SciPy 软件包要求 Python 3.12+。
- 安装期间需要联网，并预留约 3 GB 磁盘空间。
- 仅真机模式需要：CMake 3.16+、C++17 编译器、Linux SocketCAN、`iproute2`
  以及配置 CAN 接口的权限。

Ubuntu 上可安装以下常用系统软件包：

```bash
sudo apt update
sudo apt install python3 python3-venv build-essential cmake iproute2 libxcb-cursor0
```

### 克隆与安装

```bash
git clone <repository-url>
cd motion_trajectory_studio
bash install.sh
bash run.sh
```

`install.sh` 会在仓库内创建 `.venv`，安装 `requirements-lock.txt` 中锁定的依赖，
并以可编辑模式安装应用。它不会修改 `vr_control` 或其他同级仓库。需要 Python
3.12 或 3.13；conda `base` 若是 3.14，锁文件中的 wheel 装不上。脚本会优先使用
`python3.12` / `python3.13`。如需指定 Python：

```bash
MOTION_STUDIO_PYTHON=/usr/bin/python3.12 bash install.sh
```

应用程序只有一种界面语言：英文。`run_en.sh` 仅作为已有快捷方式的兼容别名保留，
启动的是完全相同的界面。

### 模型和轨迹属于用户数据

仓库内置 `model_library/OpenArm_V1`（达妙夹爪）和 `model_library/OpenArm_O6`
（灵心 O6 灵巧手）作为可直接加载的模型，包含其 MJCF 文件和 mesh。其他本机机器
人模型、轨迹、工程、相机录像及生成的打包文件不会上传。Git 仍会忽略
`model_library/*`，但会保留 README 以及内置的 `OpenArm_V1` 和 `OpenArm_O6`。
克隆后可以直接双击其中一个模型，也可以通过以下方式添加其他模型：

1. 使用 **File -> Import Model** 或 **Import Model Folder into Library**；或者
2. 将完整模型目录复制到 `model_library/MyRobot/`，保留全部 mesh、纹理和
   MJCF include 的相对路径。

随后导入 CSV/NPZ 轨迹并确认关节映射。如果无法自动选择主 MJCF/XML/URDF 文件，
可在模型目录中添加带 `entry` 路径的 `model.json`。编辑、MuJoCo 预览、验证和导出
均不要求连接真机。

### 可选真机功能

首次连接时会自动构建真机服务，也可以手动提前构建：

```bash
cmake -S hardware_controller -B hardware_controller/build
cmake --build hardware_controller/build --target motion_studio_hardware_service -j2
```

默认左臂使用 `can1`，右臂使用 `can0`：

```bash
export OPENARM_LEFT_FOLLOWER_CAN=can1
export OPENARM_RIGHT_FOLLOWER_CAN=can0
bash run.sh
```

服务使用 `pkexec` 或免密 `sudo` 配置 CAN。如果单独配置 CAN，请设置
`OPENARM_AUTO_CONFIG_CAN=0`。真机执行时必须清空工作区并确保急停可触达。

### 可选相机功能

本机相机使用 Qt Multimedia/V4L2。当前用户必须能够读取所选 `/dev/videoN` 设备，
通常需要加入 `video` 用户组。网络相机必须提供本机可访问的 HTTP(S) MJPEG 流。

### 验证安装

```bash
./.venv/bin/python app.py doctor
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen \
  ./.venv/bin/python -m pytest
```

CLI 命令：

```bash
./.venv/bin/python app.py doctor
./.venv/bin/python app.py inspect trajectory.csv --hz 250
./.venv/bin/python app.py validate project.motionproj
./.venv/bin/python app.py export project.motionproj result.csv --hz 100
./.venv/bin/python app.py pack project.motionproj project.tar.gz
```

## 典型工作流程

1. 导入机器人模型，或双击 `model_library/` 中的模型。
2. 导入 CSV/NPZ 动作数据并确认通道到关节的映射。
3. 选择关节，然后在时间轴上拖动以选取时间范围。
4. 应用编辑、检查 MuJoCo 预览并运行验证。
5. 保存 `.motionproj` 工程或导出 CSV/NPZ。
6. 真机运行时，将播放头置于所需起始帧，连接独立服务，确认工作区和急停后执行。

## 编辑范围与时序

关节/通道列表控制所有适用编辑操作的作用范围。不选择任何通道表示选择全部位置
关节。选择部分关节后，裁剪、删除、等待、循环、复制/粘贴、变速、平滑和片段
移动只作用于这些关节轨道。

选择一个时间范围，然后拖动橙色选区主体，即可移动所选关节片段。为进行小幅时序
调整，目标范围可以与源范围部分重叠；只有目标中新占用的部分必须保持静止。移动
不会改变总时长，也不会修改未选中的关节。

常用通道选择可以保存为关节组。按显示顺序，前九个编组使用 `Alt+1` 至 `Alt+9`
快捷键。

对于孤立抽动和普通过渡，请使用五次多项式平滑。对于必须保持相位和速度的周期性
运动，请使用高速相位平滑。

## 多相机监视器

从主工具栏打开 **Multi-Camera Monitor**。它支持多个 `/dev/videoN` 设备和/或
HTTP(S) MJPEG 视频源。每个面板独立连接；关闭监视器或应用程序时会释放全部视频流。

通过以下方式配置启动时的视频源：

```bash
export OPENARM_REAL_CAMERA_SOURCES="/dev/video0,/dev/video2,https://host/cam.mjpg"
```

## 真机安全

`hardware_controller/` 是独立服务，不会导入或启动同级的 `vr_control` 项目。
编辑器使用 `load -> prepare -> execute` 流程通信；`prepare` 只选择起始帧，不会
驱动真机运动。

执行时，系统使用不少于三秒的自适应静止到静止五次多项式过渡，将实测关节状态
连接到所选帧。随后，服务严格播放导出的位置和速度样本，不在线重塑轨迹。位置、
速度、加速度、心跳、电机故障和温度保护始终保持启用。

真机执行涉及安全风险。请从短时、低能量片段开始，确保急停始终可触达，并且绝不
绕过系统报告的安全停机。

## 开发

架构和安全边界见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。独立性审计见
[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)，自动与人工测试范围见
[docs/TEST_MATRIX.md](docs/TEST_MATRIX.md)。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen \
  ./.venv/bin/python -m pytest
```
