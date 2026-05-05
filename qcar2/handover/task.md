# Task: QCar2 Autonomous Obstacle Avoidance Refinement

- [x] Analyze `tessting.py` and existing perception/control modules
- [x] Create Implementation Plan
- [x] Implement stop-at-5cm logic (verified in CM)
- [x] Implement space-based decision (Left/Right)
- [x] Implement bypass maneuver:
    - [x] Reverse and Rotate 45 degrees
    - [x] Side-clearance control (3cm) using LiDAR
    - [x] Yellow line detection using side CSI cameras
    - [x] Re-alignment with lane
- [ ] Physical verification of full sequence [ ]
- [ ] Fine-tune rotation and bypass parameters [ ]

### Antigravity Updates (April 2026):
- [x] Implement 35cm Stop + 50cm Crawl logic (Dynamic Precision)
- [x] Implement 20cm Side Clearance (Autonomous Space Decision)
- [x] Implement 1-minute "Thinking" phase (1800 frames)
- [x] Restore dynamic Left/Right bypass selection
- [x] Verify Side CSI Lane Re-acquisition
- [x] Integrate precise LiDAR data for 3cm side clearance target
- [x] Add grayscale reflection suppression to lane tracking

