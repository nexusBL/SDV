"""
Autonomous SLAM - Integrated entry point.

Combines particle-filter SLAM, loop closure, A* path planning,
and pure-pursuit waypoint following into a single script.

Usage:
    Teleop mapping:   python3 autonomous_slam.py --mode teleop
    Autonomous nav:   python3 autonomous_slam.py --mode autonomous --goal 20 15
    Map + then nav:   python3 autonomous_slam.py --mode full --goal 20 15 --map-time 30
"""
import argparse
import sys
import time
import copy
import numpy as np
import cv2

from Quanser.q_essential import LIDAR
from Quanser.q_ui import gamepadViaTarget
from Quanser.product_QCar import QCar
from speedCalc import speedCalc
from GridMap import GridMap
from ParticleFilter import ParticleFilter
from loop_closure import LoopClosureDetector
from path_planner import AStarPlanner
from waypoint_follower import (
    PurePursuitController, ObstacleChecker, WaypointManager,
    IDLE, MAPPING, PLANNING, FOLLOWING, ESTOP, DONE
)
import utils


# ── Configuration ──────────────────────────────────────────────────────────────
NUM_MEASUREMENTS = 360
MAX_DISTANCE = 2          # meters
MAP_UNITS = 20            # conversion factor
NUM_PARTICLES = 5
SAMPLE_RATE = 50          # Hz
MAP_PARAMS = [0.4, -0.4, 5.0, -5.0]  # lo_occ, lo_free, lo_max, lo_min
SAVE_INTERVAL = 50        # save map image every N iterations


def SensorMapping(m, bot_pos, angles, dists):
    """Update occupancy grid from a LiDAR scan."""
    for i in range(NUM_MEASUREMENTS):
        if dists[i] >= MAX_DISTANCE * MAP_UNITS:
            continue
        if dists[i] < 0.05:
            continue
        theta = bot_pos[2] - angles[i]
        m.GridMapLine(
            int(bot_pos[0]),
            int(bot_pos[0] + dists[i] * np.cos(theta)),
            int(bot_pos[1]),
            int(bot_pos[1] + dists[i] * np.sin(theta))
        )


def AdaptiveGetMap(gmap):
    """Render the occupancy grid to an image."""
    mimg = gmap.GetMapProb(
        gmap.boundary[0] - 20, gmap.boundary[1] + 20,
        gmap.boundary[2] - 20, gmap.boundary[3] + 20
    )
    mimg = (255 * mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    return mimg


def DrawParticle(img, plist, scale=1.0):
    """Draw particle positions on the map image."""
    for p in plist:
        cv2.circle(
            img,
            (int(p.gmap.center[0] + scale * p.pos[0]),
             int(p.gmap.center[1] + scale * p.pos[1])),
            2, (0, 200, 0), -1
        )
    return img


def DrawPath(img, path, gmap, color=(0, 0, 255)):
    """Draw the planned path on the map image."""
    if path is None or len(path) < 2:
        return img
    for i in range(len(path) - 1):
        p1 = (int(gmap.center[0] + path[i][0]),
              int(gmap.center[1] + path[i][1]))
        p2 = (int(gmap.center[0] + path[i + 1][0]),
              int(gmap.center[1] + path[i + 1][1]))
        cv2.line(img, p1, p2, color, 2)
    return img


def DrawGoal(img, goal, gmap, color=(255, 0, 0)):
    """Draw the goal marker on the map image."""
    if goal is None:
        return img
    pt = (int(gmap.center[0] + goal[0]), int(gmap.center[1] + goal[1]))
    cv2.circle(img, pt, 5, color, -1)
    cv2.circle(img, pt, 8, color, 2)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous SLAM for QCar")
    parser.add_argument('--mode', choices=['teleop', 'autonomous', 'full'],
                        default='teleop',
                        help='teleop=gamepad SLAM, autonomous=nav to goal, '
                             'full=map then nav')
    parser.add_argument('--goal', nargs=2, type=float, default=None,
                        metavar=('X', 'Y'),
                        help='Goal position in grid coordinates (required for autonomous/full)')
    parser.add_argument('--map-time', type=float, default=30.0,
                        help='Mapping warmup time in seconds (for full mode)')
    parser.add_argument('--particles', type=int, default=NUM_PARTICLES,
                        help='Number of particles for the filter')
    parser.add_argument('--lookahead', type=float, default=3.0,
                        help='Pure-pursuit lookahead distance (map units)')
    parser.add_argument('--speed', type=float, default=0.05,
                        help='Base autonomous speed (throttle)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ('autonomous', 'full') and args.goal is None:
        print("ERROR: --goal X Y is required for autonomous/full mode")
        sys.exit(1)

    # ── Hardware init ──────────────────────────────────────────────────────
    print("[Init] Starting hardware...")
    robot_pos = np.array([0.0, 0.0, 0.0])
    myLidar = LIDAR(NUM_MEASUREMENTS, MAX_DISTANCE)
    myCar = QCar()
    gpad = gamepadViaTarget(1)
    mySpeed = speedCalc(robot_pos, myCar, time.time())

    # ── SLAM components ────────────────────────────────────────────────────
    m = GridMap(MAP_PARAMS, gsize=1)
    loop_detector = LoopClosureDetector(
        distance_threshold=5.0, min_scan_gap=20,
        icp_error_threshold=2.0, max_dist=MAX_DISTANCE, map_units=MAP_UNITS
    )

    # ── Autonomous components ──────────────────────────────────────────────
    planner = AStarPlanner(obstacle_threshold=0.65, inflation_radius=3)
    controller = PurePursuitController(
        wheelbase=0.26, lookahead_dist=args.lookahead,
        max_steering=0.35, base_speed=args.speed
    )
    obstacle_checker = ObstacleChecker(
        stop_distance=0.3, front_angle_range=30, map_units=MAP_UNITS
    )
    waypoint_mgr = WaypointManager(planner, controller, obstacle_checker, MAP_UNITS)

    # ── Timing ─────────────────────────────────────────────────────────────
    startTime = time.time()
    sampleTime = 1.0 / SAMPLE_RATE

    def elapsed():
        return time.time() - startTime

    # ── Initial map warmup ─────────────────────────────────────────────────
    print("[Init] Building initial map (5s warmup)...")
    myLidar.read()
    while elapsed() < 5.0:
        myLidar.read()

    SensorMapping(m, robot_pos, myLidar.angles, myLidar.distances * MAP_UNITS)
    pf = ParticleFilter(
        robot_pos.copy(), NUM_MEASUREMENTS, MAX_DISTANCE,
        MAP_UNITS, copy.deepcopy(m), args.particles
    )

    # ── State setup ────────────────────────────────────────────────────────
    if args.mode == 'teleop':
        waypoint_mgr.start_mapping()
        print("[Mode] TELEOP — drive with gamepad, press B to quit")
    elif args.mode == 'autonomous':
        goal = (int(args.goal[0]), int(args.goal[1]))
        waypoint_mgr.set_goal(goal)
        print(f"[Mode] AUTONOMOUS — navigating to {goal}")
    elif args.mode == 'full':
        waypoint_mgr.start_mapping()
        print(f"[Mode] FULL — mapping for {args.map_time}s, then navigating to goal")

    counter = 0
    full_mode_switched = False

    try:
        # ── Main loop ──────────────────────────────────────────────────────
        while gpad.B != 1:
            gpad.read()
            start = time.time()

            # ── Drive command ──────────────────────────────────────────────
            state = waypoint_mgr.get_state()

            if state == MAPPING:
                # Gamepad teleop
                mtr_cmd = np.array([0.07 * gpad.RT,
                                    (gpad.left - gpad.right) * 0.3])
            elif state in (FOLLOWING,):
                # Get autonomous command from waypoint manager
                best_pose = pf.get_best_pose()
                best_map = pf.get_best_particle().gmap
                auto_cmd = waypoint_mgr.get_motor_command(
                    best_pose, myLidar.angles,
                    myLidar.distances * MAP_UNITS, best_map
                )
                mtr_cmd = np.array([auto_cmd[0], auto_cmd[1]])
            elif state == PLANNING:
                # Planning phase — don't move, just compute
                best_pose = pf.get_best_pose()
                best_map = pf.get_best_particle().gmap
                waypoint_mgr.get_motor_command(
                    best_pose, myLidar.angles,
                    myLidar.distances * MAP_UNITS, best_map
                )
                mtr_cmd = np.array([0.0, 0.0])
            elif state == DONE:
                print("[Done] Goal reached! Stopping.")
                mtr_cmd = np.array([0.0, 0.0])
                LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
                myCar.read_write_std(mtr_cmd, LEDs)
                break
            else:
                mtr_cmd = np.array([0.0, 0.0])

            LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
            myCar.read_write_std(mtr_cmd, LEDs)

            # ── Odometry + LiDAR ──────────────────────────────────────────
            encoder_dist = mySpeed.encoder_dist()
            robot_pos = utils.posUpdate(robot_pos, mtr_cmd[1], MAP_UNITS, encoder_dist)
            myLidar.read()

            # ── Particle filter update ────────────────────────────────────
            if encoder_dist > 0:
                pf.Feed(robot_pos[2], mtr_cmd[1], encoder_dist,
                        myLidar.angles, myLidar.distances * MAP_UNITS)
                pf.Resampling(NUM_MEASUREMENTS, myLidar.angles,
                              myLidar.distances * MAP_UNITS)

                # ── Loop closure ──────────────────────────────────────────
                best_pose = pf.get_best_pose()
                loop_detector.add_scan(best_pose, myLidar.angles,
                                       myLidar.distances * MAP_UNITS)
                correction = loop_detector.detect(
                    best_pose, myLidar.angles, myLidar.distances * MAP_UNITS
                )
                if correction is not None:
                    loop_detector.apply_correction(pf, correction)

            # ── Status output ─────────────────────────────────────────────
            best = pf.get_best_particle()
            if counter % 10 == 0:
                print(f"[{elapsed():.1f}s] state={state} pos=({best.pos[0]:.1f}, "
                      f"{best.pos[1]:.1f}, {np.degrees(best.pos[2]):.0f}°)")

            # ── Map image save ────────────────────────────────────────────
            if counter % SAVE_INTERVAL == 0:
                imgp0 = AdaptiveGetMap(best.gmap)
                imgp0 = DrawParticle(imgp0, pf.particle_list)
                if waypoint_mgr.path:
                    imgp0 = DrawPath(imgp0, waypoint_mgr.path, best.gmap)
                if waypoint_mgr.goal:
                    imgp0 = DrawGoal(imgp0, waypoint_mgr.goal, best.gmap)
                cv2.imwrite(f'slam_map_{counter:05d}.jpg', imgp0)

            # ── Full mode: switch from mapping to autonomous ──────────────
            if args.mode == 'full' and not full_mode_switched:
                if elapsed() > args.map_time:
                    goal = (int(args.goal[0]), int(args.goal[1]))
                    waypoint_mgr.set_goal(goal)
                    full_mode_switched = True
                    print(f"[Full] Mapping done, switching to autonomous nav → {goal}")

            counter += 1
            end = time.time()

            # ── Timing ────────────────────────────────────────────────────
            computationTime = end - start
            sleepTime = sampleTime - (computationTime % sampleTime)
            msSleepTime = max(int(1000 * sleepTime), 1)
            time.sleep(msSleepTime / 1000.0)

    except KeyboardInterrupt:
        print("\n[Shutdown] User interrupted!")
    finally:
        # Save final map
        try:
            best = pf.get_best_particle()
            final_img = AdaptiveGetMap(best.gmap)
            final_img = DrawParticle(final_img, pf.particle_list)
            cv2.imwrite('slam_map_final.jpg', final_img)
            print("[Shutdown] Final map saved to slam_map_final.jpg")
        except Exception as e:
            print(f"[Shutdown] Could not save final map: {e}")

        myLidar.terminate()
        gpad.terminate()
        myCar.terminate()
        print("[Shutdown] Hardware terminated. Done.")


if __name__ == '__main__':
    main()
