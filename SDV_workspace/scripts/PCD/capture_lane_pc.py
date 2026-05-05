#!/usr/bin/env python3
"""
capture_lane_pc.py — Live RealSense point cloud capture on QCar2.
Shows live color+depth feed. Press 's' to save PLY, 'q' to quit.
"""
import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    width, height = 640, 480
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")

    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    print("Controls (click on the camera window first!):")
    print("  s : Save point cloud (.ply)")
    print("  q : Quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 3D Processing
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense (Color | Depth)', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = "lane_map.ply"
                print(f"Saving to {filename}...")
                points.export_to_ply(filename, color_frame)
                print("Done.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
