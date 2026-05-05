
import cv2

import numpy as np

import sys

sys.path.insert(0, '/home/nvidia/Desktop/SDV_workspace/install/sdv_perception/lib/python3.8/site-packages')

from sdv_perception.config import SDVConfig



SDVConfig.reset()

cfg = SDVConfig()



# Load frame

frame = cv2.imread('/home/nvidia/Desktop/SDV_workspace/scripts/frame_1.jpg')

if frame is None:

    print("ERROR: frame_1.jpg not found!")

    exit()



h, w = frame.shape[:2]

print(f"Frame size: {w}x{h}")



# Perspective transform points from config

p = cfg.lane['perspective']

src = np.float32([

    [w * p['src_bottom_left'][0],  h * p['src_bottom_left'][1]],

    [w * p['src_top_left'][0],     h * p['src_top_left'][1]],

    [w * p['src_top_right'][0],    h * p['src_top_right'][1]],

    [w * p['src_bottom_right'][0], h * p['src_bottom_right'][1]],

])

dst = np.float32([

    [w * p['dst_left'],  h],

    [w * p['dst_left'],  0],

    [w * p['dst_right'], 0],

    [w * p['dst_right'], h],

])

M     = cv2.getPerspectiveTransform(src, dst)

M_inv = cv2.getPerspectiveTransform(dst, src)



# Preprocess — grayscale + CLAHE + Canny (best for black lines on white)

gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

gray  = clahe.apply(gray)

blur  = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 20, 80)



# ROI mask

mask = np.zeros_like(edges)

roi  = np.array([[

    [int(w*0.00), int(h*0.98)],

    [int(w*0.38), int(h*0.35)],

    [int(w*0.62), int(h*0.35)],

    [int(w*1.00), int(h*0.98)],

]], dtype=np.int32)

cv2.fillPoly(mask, roi, 255)

masked = cv2.bitwise_and(edges, mask)



# Bird's eye

birdseye = cv2.warpPerspective(masked, M, (w, h))



# Sliding window

histogram  = np.sum(birdseye[h//2:, :], axis=0)

midpoint   = w // 2

left_base  = np.argmax(histogram[:midpoint])

right_base = np.argmax(histogram[midpoint:]) + midpoint



n_windows  = 9

margin     = 100

min_pixels = 15

win_height = h // n_windows



nonzero  = birdseye.nonzero()

nonzero_y = np.array(nonzero[0])

nonzero_x = np.array(nonzero[1])



left_x_curr  = left_base

right_x_curr = right_base

left_inds    = []

right_inds   = []



for window in range(n_windows):

    win_y_low  = h - (window+1) * win_height

    win_y_high = h - window * win_height

    good_left  = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &

                  (nonzero_x >= left_x_curr - margin) & (nonzero_x < left_x_curr + margin)).nonzero()[0]

    good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &

                  (nonzero_x >= right_x_curr - margin) & (nonzero_x < right_x_curr + margin)).nonzero()[0]

    left_inds.append(good_left)

    right_inds.append(good_right)

    if len(good_left)  > min_pixels: left_x_curr  = int(np.mean(nonzero_x[good_left]))

    if len(good_right) > min_pixels: right_x_curr = int(np.mean(nonzero_x[good_right]))



left_inds  = np.concatenate(left_inds)

right_inds = np.concatenate(right_inds)



left_x  = nonzero_x[left_inds]

left_y  = nonzero_y[left_inds]

right_x = nonzero_x[right_inds]

right_y = nonzero_y[right_inds]



left_ok  = len(left_x)  > 50

right_ok = len(right_x) > 50



print(f"Left pixels: {len(left_x)}, Right pixels: {len(right_x)}")

print(f"Left OK: {left_ok}, Right OK: {right_ok}")



# Fit polynomial

result = frame.copy()

offset = 0.0



if left_ok and right_ok:

    left_fit  = np.polyfit(left_y,  left_x,  2)

    right_fit = np.polyfit(right_y, right_x, 2)



    plot_y   = np.linspace(0, h-1, h)

    left_px  = left_fit[0]*plot_y**2  + left_fit[1]*plot_y  + left_fit[2]

    right_px = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]



    # Draw filled lane polygon on bird's eye

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    pts_left  = np.array([np.transpose(np.vstack([left_px,  plot_y]))])

    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_px, plot_y])))])

    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(overlay, np.int_([pts]), (0, 200, 0))



    # Warp back to original view

    unwarped = cv2.warpPerspective(overlay, M_inv, (w, h))

    result   = cv2.addWeighted(frame, 1.0, unwarped, 0.4, 0)



    # Draw lane lines — DISABLED

    for y, lx, rx in zip(plot_y[::20].astype(int), left_px[::20].astype(int), right_px[::20].astype(int)):

        pass  # if 0 <= lx < w: cv2.circle(result, (lx, y), 3, (255, 0, 0), -1)

        pass  # if 0 <= rx < w: cv2.circle(result, (rx, y), 3, (0, 0, 255), -1)



    # Compute offset

    lane_center  = (left_fit[0]*h**2 + left_fit[1]*h + left_fit[2] +

                    right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]) / 2

    offset = float((w/2 - lane_center) / (w/2))



# HUD overlay

status = "BOTH" if (left_ok and right_ok) else "LEFT" if left_ok else "RIGHT" if right_ok else "NONE"

color  = (0,255,0) if (left_ok and right_ok) else (0,0,255)

cv2.putText(result, f"LANE: {status}",      (30, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

cv2.putText(result, f"Offset: {offset:+.3f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

cv2.putText(result, f"Left:  {'OK' if left_ok  else 'X'}", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if left_ok  else (0,0,255), 2)

cv2.putText(result, f"Right: {'OK' if right_ok else 'X'}", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if right_ok else (0,0,255), 2)



# Save debug images too

cv2.imwrite('/home/nvidia/Desktop/SDV_workspace/scripts/lane_proof.jpg',    result)

cv2.imwrite('/home/nvidia/Desktop/SDV_workspace/scripts/lane_edges.jpg',    edges)

cv2.imwrite('/home/nvidia/Desktop/SDV_workspace/scripts/lane_birdseye.jpg', birdseye)

print(f"Saved! Status={status} Offset={offset:.3f}")

