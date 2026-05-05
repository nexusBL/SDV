#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║       SDV SENSOR HEALTH DIAGNOSTIC TOOL                      ║
║       Checks connectivity and status of all QCar2 sensors    ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python3 test_sensor_health.py
"""

import sys
import time


def _check(label, func):
    """Run a diagnostic check, print PASS/FAIL."""
    try:
        result = func()
        status = "✅ PASS"
        detail = result if result else ""
    except Exception as e:
        status = "❌ FAIL"
        detail = str(e)
    print(f"  {status}  {label:.<40s} {detail}")
    return "PASS" in status


def check_cuda():
    """Check PyTorch CUDA availability."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{name} ({mem:.1f} GB)"


def check_yolo():
    """Check YOLOv8 model loads and runs on GPU."""
    import torch
    import numpy as np
    from ultralytics import YOLO
    model_path = "/home/nvidia/Desktop/SDV_workspace/yolov8n.pt"
    model = YOLO(model_path)
    model.to('cuda')
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    with torch.no_grad():
        results = model.predict(source=dummy, device='cuda', verbose=False)
    return f"Loaded {model_path.split('/')[-1]} — inference OK"


def check_csi():
    """Check the 4 wide-angle CSI cameras (Right, Rear, Front, Left)."""
    from pal.products.qcar import QCarCameras
    import numpy as np
    
    cameras = QCarCameras(
        frameWidth=820, frameHeight=616, frameRate=80,
        enableBack=True, enableFront=True, enableLeft=True, enableRight=True
    )
    
    # Spin up
    time.sleep(1.0)
    flags = cameras.readAll()
    active_count = 0
    
    for i, c in enumerate(cameras.csi):
        if c is not None and c.imageData is not None and c.imageData.size > 0:
            active_count += 1
            
    cameras.terminate()
    
    if active_count == 0:
        raise RuntimeError("No CSI frames received. GStreamer or NVMem may be locked.")
    elif active_count < 4:
        raise RuntimeError(f"Only {active_count}/4 CSI streams active. (flags={flags})")
        
    return f"All 4 streams active @ 820x616"


def check_realsense_rgb():
    """Check RealSense RGB camera."""
    from pal.products.qcar import QCarRealSense
    rs = QCarRealSense(mode='RGB', frameWidthRGB=640, frameHeightRGB=480)
    time.sleep(0.5)
    rs.read_RGB()
    frame = rs.imageBufferRGB
    rs.terminate()
    if frame is None or frame.size == 0:
        raise RuntimeError("No RGB frame received")
    return f"RGB frame: {frame.shape}"


def check_realsense_depth():
    """Check RealSense depth camera."""
    from pal.products.qcar import QCarRealSense
    rs = QCarRealSense(mode='DEPTH', frameWidthRGB=640, frameHeightRGB=480)
    time.sleep(0.5)
    rs.read_depth(dataMode='PX')
    frame = rs.imageBufferDepthPX
    rs.terminate()
    if frame is None or frame.size == 0:
        raise RuntimeError("No depth frame received")
    return f"Depth frame: {frame.shape}"


def check_lidar():
    """Check RPLidar A2."""
    from pal.products.qcar import QCarLidar
    lidar = QCarLidar(numMeasurements=400, rangingDistanceMode=2, interpolationMode=0)
    
    # Needs to spin up
    t0 = time.time()
    distances = None
    while time.time() - t0 < 2.5:
        lidar.read()
        if lidar.distances is not None and len(lidar.distances) > 0 and sum(lidar.distances) > 0.0:
            distances = lidar.distances
            break
        time.sleep(0.05)
        
    lidar.terminate()
    if distances is None or len(distances) == 0:
        raise RuntimeError("No LiDAR data received after spinning up")
    
    import numpy as np
    valid = np.array(distances)
    valid = valid[(valid > 0.05) & (valid < 10.0)]
    if len(valid) == 0:
        raise RuntimeError("LiDAR running but only seeing out-of-bounds noise")
        
    return f"{len(distances)} pts, min={valid.min():.2f}m, max={valid.max():.2f}m"


def check_opencv():
    """Check OpenCV build info."""
    import cv2
    ver = cv2.__version__
    cuda_enabled = 'YES' in cv2.getBuildInformation().split('CUDA')[1][:20] \
        if 'CUDA' in cv2.getBuildInformation() else False
    return f"v{ver}, CUDA={'yes' if cuda_enabled else 'no'}"


def check_numpy():
    """Check NumPy."""
    import numpy as np
    return f"v{np.__version__}"


def check_ros2():
    """Check ROS2 Humble availability."""
    import subprocess
    result = subprocess.run(
        ['bash', '-c', 'source /opt/ros/humble/setup.bash && echo $ROS_DISTRO'],
        capture_output=True, text=True, timeout=5,
    )
    distro = result.stdout.strip()
    if not distro or distro != 'humble':
        raise RuntimeError(f"ROS2 Humble not found. Got: {distro}")
    return distro


def main():
    print("=" * 60)
    print("  SDV SENSOR & SYSTEM HEALTH DIAGNOSTIC")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    print()

    results = {}

    # Software checks
    print("── Software Environment ──")
    results['numpy'] = _check("NumPy", check_numpy)
    results['opencv'] = _check("OpenCV", check_opencv)
    results['cuda'] = _check("CUDA / GPU", check_cuda)
    results['yolo'] = _check("YOLOv8 Model (GPU)", check_yolo)
    results['ros2'] = _check("ROS2 Humble", check_ros2)
    print()

    # Hardware checks
    print("── Hardware Sensors ──")
    results['csi'] = _check("CSI Wide Cameras (4)", check_csi)
    results['realsense_rgb'] = _check("RealSense RGB Camera", check_realsense_rgb)
    results['realsense_depth'] = _check("RealSense Depth Camera", check_realsense_depth)
    results['lidar'] = _check("RPLidar A2", check_lidar)
    print()

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print("=" * 60)
    if passed == total:
        print(f"  🟢 ALL CHECKS PASSED ({passed}/{total})")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  🟡 {passed}/{total} PASSED — Failed: {', '.join(failed)}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
