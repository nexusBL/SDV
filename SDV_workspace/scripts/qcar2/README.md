# QCar2 Autonomous Lane Following & Obstacle Avoidance

Ported from [JoseBalbuena181096/QCarQuanser](https://github.com/JoseBalbuena181096/QCarQuanser) (QCar v1) to the **Quanser QCar 2** platform (NVIDIA Jetson Orin AGX).

---

## Architecture

```
qcar2/
├── config.py                    # All tunable parameters (HSV, PID, camera, safety)
├── main.py                      # Main orchestrator with state machine
├── capture_frame.py             # Calibration utility for perspective transform
├── hardware/
│   ├── camera_manager.py        # CSI camera with nvarguscamerasrc EGL fix
│   ├── car_controller.py        # Motor/steering/LED hardware abstraction
│   └── safety_monitor.py        # RPLidar A2M12 obstacle detection
├── perception/
│   └── lane_cv.py               # Full CV pipeline (BEV, HSV, RANSAC, HUD)
└── control/
    └── pid_controller.py        # Time-scaled PID with derivative low-pass filter
```

## Key Upgrades from QCar v1

| Feature | QCar v1 (Original) | QCar2 (This Port) |
|---------|--------------------|--------------------|
| **Camera API** | `Camera2D(cameraId='3')` | `QCarCameras(enableFront=True)` |
| **Car API** | `QCar().read_write_std()` | `QCar(readMode=1).write()` |
| **LiDAR API** | `Lidar(type='RPLidar')` | `QCarLidar()` |
| **EGL Fix** | None (crashes headless) | Auto-pops `DISPLAY`/`XAUTHORITY` |
| **Architecture** | Flat scripts | Modular packages + state machine |
| **Thresholding** | Mutates baselines permanently | Non-destructive adaptive |
| **PID D-term** | Raw derivative (jittery) | Low-pass filtered |
| **Obstacle Logic** | Magic pixel coordinates | Angular ROI cone (±15°) |
| **Shutdown** | Basic try/except | Guaranteed zero-command in finally |

## Quick Start

### 1. Deploy to QCar2
```bash
scp -r qcar2/ nvidia@192.168.41.88:~/Desktop/SDV_workspace/scripts/
```

### 2. SSH into the Jetson
```bash
ssh nvidia@192.168.41.88
```

### 3. Set up environment
```bash
source /opt/ros/humble/setup.bash
source ~/Documents/Quanser/5_research/sdcs/qcar2/ros2/install/setup.bash
export PYTHONPATH="/home/nvidia/Documents/Quanser/0_libraries/python:$PYTHONPATH"
```

### 4. Calibrate (first time only)
```bash
cd ~/Desktop/SDV_workspace/scripts/qcar2/
python3 capture_frame.py
# Open calibration_frame.png and pick your 4 perspective points
# Update config.py CVConfig.src_points
```

### 5. Run
```bash
# Preview mode (motors disabled, vision only):
python3 main.py --preview

# Full autonomous mode:
python3 main.py
```

### 6. Controls
| Key | Action |
|-----|--------|
| `a` | Start autonomous driving |
| `s` | Stop (manual override) |
| `q` | Quit & shutdown safely |

## Configuration

All parameters are in `config.py`. Key values to tune:

- **`CVConfig.src_points`** — Perspective transform corners (MUST calibrate per track)
- **`CVConfig.hsv_lower/upper`** — Yellow tape detection bounds
- **`ControlConfig.kp/ki/kd`** — PID steering gains
- **`ControlConfig.base_speed`** — Throttle (default 0.08, max safe 0.3)
- **`SafetyConfig.stop_distance_m`** — Obstacle stop distance (default 0.5m)
