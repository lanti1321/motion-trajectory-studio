# Motion Trajectory Studio

An English-only desktop editor for sampled robot motion trajectories, with
MuJoCo preview and an optional standalone OpenArm hardware playback service.
Models, joint counts, channel names, and sample rates are not hard-coded.

## Highlights

- Import MJCF/XML/URDF models and CSV/NPZ trajectories.
- Automatic or manual joint mapping with scale, offset, and joint limits.
- Real-time MuJoCo preview, model cameras, an independent multi-camera monitor,
  frame stepping, and a zoomable multi-track timeline.
- Named and colored ranges, joint groups with `Alt+1` through `Alt+9`, per-joint
  editing scope, clip dragging, and undo/redo.
- Trim, delete, copy/paste, loop, hold, retiming, quintic smoothing, high-speed
  phase smoothing, splicing, and 6D end-effector editing.
- Continuity, finite-value, jump, velocity, acceleration, joint-limit, mapping,
  and sampled collision validation.
- Portable `.motionproj` projects and `.tar.gz` bundles.
- Optional hardware playback from the current playhead, MOS/rotor temperature
  monitoring, full-run temperature curves, and multiple real-camera streams.

## Install and run

### Prerequisites

- Linux desktop with an X11 or Wayland session (Ubuntu 24.04 recommended;
  Ubuntu 22.04 needs a separately installed Python 3.12+).
- Python 3.12 or newer. The pinned NumPy and SciPy packages require Python 3.12+.
- Internet access during installation and about 3 GB of free disk space.
- For real hardware only: CMake 3.16+, a C++17 compiler, Linux SocketCAN,
  `iproute2`, and permission to configure the CAN interfaces.

On Ubuntu, install the common system packages with:

```bash
sudo apt update
sudo apt install python3 python3-venv build-essential cmake iproute2 libxcb-cursor0
```

### Clone and install

```bash
git clone <repository-url>
cd motion_trajectory_studio
bash install.sh
bash run.sh
```

`install.sh` creates a repository-local `.venv`, installs the pinned packages
from `requirements-lock.txt`, and installs the application in editable mode.
It does not modify `vr_control` or another sibling repository. Python 3.12 or
3.13 is required; conda `base` on Python 3.14 cannot install the locked wheels.
The script prefers `python3.12` / `python3.13` over `python3`. To select a
specific interpreter, run:

```bash
MOTION_STUDIO_PYTHON=/usr/bin/python3.12 bash install.sh
```

The application has one interface language: English. `run_en.sh` remains only
as a compatibility alias for existing shortcuts and starts the same UI.

### Models and trajectories are user data

The repository includes `model_library/OpenArm_V1` (Damiao jaws) and
`model_library/OpenArm_O6` (Linker Hand O6) as ready-to-load models, including
MJCF files and meshes. Other local robot models, trajectories, projects, camera
recordings, and generated bundles are not published. `model_library/*` remains
ignored by Git except for its README and the bundled `OpenArm_V1` and
`OpenArm_O6` models. After cloning, users can immediately double-click either
model, or add another model by either:

1. Use **File -> Import Model** or **Import Model Folder into Library**; or
2. Copy a complete model folder into `model_library/MyRobot/`, preserving all
   relative mesh, texture, and MJCF include paths.

Then import a CSV/NPZ trajectory and confirm joint mapping. A model folder may
contain `model.json` with an `entry` path when its main MJCF/XML/URDF file cannot
be selected automatically. Editing, MuJoCo preview, validation, and export do
not require real hardware.

### Optional real hardware

The hardware service is built automatically on first connection, or manually:

```bash
cmake -S hardware_controller -B hardware_controller/build
cmake --build hardware_controller/build --target motion_studio_hardware_service -j2
```

The default CAN interfaces are `can1` for the left arm and `can0` for the right:

```bash
export OPENARM_LEFT_FOLLOWER_CAN=can1
export OPENARM_RIGHT_FOLLOWER_CAN=can0
bash run.sh
```

The service uses `pkexec` or passwordless `sudo` to configure CAN. Set
`OPENARM_AUTO_CONFIG_CAN=0` if CAN is configured separately. Always use a clear
workcell and keep the emergency stop reachable.

### Optional cameras

Local cameras use Qt Multimedia/V4L2. The current user must be able to read the
selected `/dev/videoN` device, commonly through the `video` group. Network
cameras must expose an HTTP(S) MJPEG stream reachable from the machine.

### Verify the installation

```bash
./.venv/bin/python app.py doctor
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen \
  ./.venv/bin/python -m pytest
```

CLI commands:

```bash
./.venv/bin/python app.py doctor
./.venv/bin/python app.py inspect trajectory.csv --hz 250
./.venv/bin/python app.py validate project.motionproj
./.venv/bin/python app.py export project.motionproj result.csv --hz 100
./.venv/bin/python app.py pack project.motionproj project.tar.gz
```

## Typical workflow

1. Import a robot model or double-click one in `model_library/`.
2. Import CSV/NPZ motion data and confirm channel-to-joint mapping.
3. Select joints, then drag the timeline to select a time range.
4. Apply edits, inspect the MuJoCo preview, and run validation.
5. Save a `.motionproj` or export CSV/NPZ.
6. For hardware, put the playhead at the required start frame, connect the
   standalone service, verify the workspace and emergency stop, then execute.

## Editing scope and timing

The joint/channel list controls every applicable edit. No selection means all
position joints. With a subset selected, trim, delete, hold, loop, copy/paste,
retiming, smoothing, and clip movement operate only on those joint lanes.

Select a time range and drag the orange range body to move the selected joint
clip. Partial overlap with the source is allowed for timing nudges; only the
newly occupied destination must be stationary. The move preserves total
duration and unselected joints.

Frequently used channel selections can be stored as joint groups. The first
nine groups use `Alt+1` through `Alt+9` in displayed order.

Use quintic smoothing for isolated jerks and ordinary transitions. Use
high-speed phase smoothing for periodic motion where phase and speed must be
preserved.

## Multi-camera monitor

Open **Multi-Camera Monitor** from the main toolbar. It accepts multiple
`/dev/videoN` devices and/or HTTP(S) MJPEG sources. Each panel connects
independently, and all streams are released when the monitor or application
closes.

Configure startup sources with:

```bash
export OPENARM_REAL_CAMERA_SOURCES="/dev/video0,/dev/video2,https://host/cam.mjpg"
```

## Hardware safety

`hardware_controller/` is standalone and does not import or start the sibling
`vr_control` project. The editor communicates using `load -> prepare ->
execute`; prepare selects the start frame without moving hardware.

At execution, measured joints are connected to the selected frame with an
adaptive rest-to-rest quintic transition of at least three seconds. The service
then plays the exported position and velocity samples without online trajectory
reshaping. Position, velocity, acceleration, heartbeat, motor-fault, and
temperature protection remain active.

Hardware execution is safety-critical. Begin with a short, low-energy segment,
keep the emergency stop reachable, and never bypass a reported safety stop.

## Development

Architecture and safety boundaries are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). See
[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) for the independence audit and
[docs/TEST_MATRIX.md](docs/TEST_MATRIX.md) for automated and manual coverage.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen \
  ./.venv/bin/python -m pytest
```
