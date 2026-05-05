# QCar2 Navigation Project - Handover Status Report

## What has been accomplished:
1. **Precise Stopping**: Implemented 5cm stop logic in `tessting.py` and `SafetyMonitor.py`.
2. **CM-Based Configuration**: Refactored `config.py` to use `stop_distance_cm` (5.0) and `side_clearance_target_cm` (3.0) to match user requirements.
3. **Automatic Maneuver**:
    - **Backing Up**: Added automatic reverse phase after obstacle detection.
    - **45-Degree Rotation**: Implemented `STATE_ROTATE` for a timed 45-degree pulse.
    - **3cm Bypass**: Implemented `STATE_AVOIDING` with LiDAR-based P-controller for side clearance.
    - **Lane Re-acquisition**: Added Side CSI camera detection and "Vision Pull" re-alignment in `STATE_SEARCHING`.
4. **Calibrated Control**: Corrected steering signs (Positive=Right, Negative=Left) across all avoidance states.
5. **Awareness**: Updated detection range to 0.75m per user feedback.

## What to do next:
1. **Physical Test**: Run `python3 tessting.py` on the track with a physical obstacle.
2. **Fine-tuning**:
    - Adjust `rotate_duration_frames` (currently 12) if the car rotates more or less than 45 degrees.
    - Adjust `side_clearance_gain` (currently 0.8) if the 3cm gap is not maintained strictly.
    - Tune `rotate_throttle` and `avoidance_throttle` for smoothness.
3. **Resuming Lane Follow**: Verify that once the car pulls back into the lane, it switches to `STATE_DRIVING` reliably.

## Key Files:
- [tessting.py](file:///home/nvidia/Desktop/SDV_workspace/scripts/qcar2/tessting.py) - Main logic and state machine.
- [config.py](file:///home/nvidia/Desktop/SDV_workspace/scripts/qcar2/config.py) - Tunable parameters (CM).
- [safety_monitor.py](file:///home/nvidia/Desktop/SDV_workspace/scripts/qcar2/hardware/safety_monitor.py) - LiDAR stop-check logic.
- [depth_monitor.py](file:///home/nvidia/Desktop/SDV_workspace/scripts/qcar2/hardware/depth_monitor.py) - RealSense depth sensor driver.

## Antigravity (AI) Updates (April 2026):
1. **Dynamic Stop Precision**: Implemented "Crawl Speed" logic (0.04) that activates at 50cm to ensure a more precise stop at the 35cm target.
2. **Refined Clearance**: Updated `side_clearance_target_cm` to **20.0cm** per user request to maintain a specific gap from obstacles.
3. **Smart Decision-Making**: Restored and refined LiDAR-based autonomous Left/Right bypass selection based on real-time space scanning.
4. **Lane Recovery**: Enhanced Side CSI camera integration to detect the yellow lane line and pull back into the lane automatically after the bypass is complete.
5. **Calibrated Parameters**: Updated `config.py` for 35cm stop and 50cm slow-down zones.
6. **RealSense Infrared Integration**: Enabled the IR stream (Y8 grayscale format) on the RealSense camera to provide superior obstacle detection in reflection conditions.
7. **IR HUD Overlay**: Added a dedicated IR grayscale view with distance annotations to the main `tessting.py` HUD. CSI camera logic remains completely untouched.
8. **LiDAR Stability**: Confirmed `safety_monitor.py` has robust pre-initialization cleanup (`pkill -f lidar`) to ensure LiDAR starts reliably even after script crashes.
