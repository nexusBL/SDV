# Implementation Plan - QCar2 Obstacle Avoidance Refinement

The goal is to implement a specific obstacle avoidance behavior: follow line -> stop at 5cm -> choose direction -> back up -> rotate 45 deg -> bypass with 3cm clearance -> re-align using side cameras.

## User Review Required

> [!IMPORTANT]
> The bypass maneuver requires precise lateral control. We will use LiDAR for the 3cm side clearance and Side CSI cameras for lane re-acquisition.

## Proposed Changes

### [Component Name] Configuration
#### [MODIFY] [config.py](file:///home/nvidia/Desktop/SDV_workspace/scripts/qcar2/config.py)
- Update `side_clearance_target_m` to 0.03 (3cm).
- Ensure `stop_distance_m` is 0.05 (5cm).
- Add `rotate_duration_frames` and `rotate_steering` if needed for the 45-deg turn.

### [Component Name] Navigation Logic
#### [MODIFY] [tessting.py](file:///home/nvidia/Desktop/SDV_workspace/scripts/qcar2/tessting.py)
- **STATE_OBSTACLE_STOP**: Automatically transition to `STATE_BACKING` after 1s stay and direction decision.
- **STATE_BACKING**: Move as currently implemented (15 frames).
- **[NEW] STATE_ROTATE**: Rotate the car by ~45 degrees using a fixed steering/throttle pulse for a specific duration.
- **STATE_AVOIDING**: 
    - Maintain **3cm** side clearance using LiDAR-based P-controller.
    - Monitor Side CSI camera for the yellow line.
    - Transition to `STATE_SEARCHING` once the yellow line is detected and the path ahead is clear.
- **STATE_SEARCHING**:
    - Re-align with the yellow line using side-camera offset to "pull" the car back into the lane.
    - Transition to `STATE_DRIVING` once the front camera confirms lane lock.

## Verification Plan

### Automated Tests
- Run `python3 tessting.py --preview` to verify state transitions in logs/HUD.

### Manual Verification
- Physical test on track with an obstacle:
    1. Confirm car stops ~5cm from obstacle.
    2. Confirm it backs up and rotates.
    3. Confirm it maintains 3cm side distance.
    4. Confirm it re-detects the yellow line and resumes perfectly.
