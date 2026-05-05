
import cv2

import numpy as np



frame = cv2.imread('/home/nvidia/Desktop/SDV_workspace/scripts/frame_1.jpg')

h, w = frame.shape[:2]



src = np.float32([

    [w * 0.38, h * 0.35],   # top-left — upar uthaya

    [w * 0.62, h * 0.35],   # top-right — upar uthaya

    [w * 1.00, h * 0.98],

    [w * 0.00, h * 0.98],

])



dst = np.float32([

    [w * 0.20, 0],

    [w * 0.80, 0],

    [w * 0.80, h],

    [w * 0.20, h],

])



M = cv2.getPerspectiveTransform(src, dst)

birdseye = cv2.warpPerspective(frame, M, (w, h))



vis = frame.copy()

for pt in src.astype(int):

    cv2.circle(vis, tuple(pt), 8, (0,255,0), -1)

cv2.polylines(vis, [src.astype(int)], True, (0,255,0), 2)



cv2.imwrite('/home/nvidia/Desktop/SDV_workspace/scripts/roi.jpg', vis)

cv2.imwrite('/home/nvidia/Desktop/SDV_workspace/scripts/birdseye.jpg', birdseye)

print("Done!")

