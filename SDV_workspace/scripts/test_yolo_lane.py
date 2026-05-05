
import cv2

import numpy as np

from ultralytics import YOLO



model = YOLO('/home/nvidia/Desktop/SDV_workspace/models/yolov8n-seg.pt')



frame = cv2.imread('/home/nvidia/Desktop/SDV_workspace/scripts/frame_1.jpg')

if frame is None:

    print("ERROR: frame not found!")

    exit()



results = model(frame, conf=0.3, verbose=False, device=0)



annotated = frame.copy()

for result in results:

    if result.masks is not None:

        for mask, box in zip(result.masks.data, result.boxes):

            cls_id = int(box.cls[0])

            cls_name = model.names[cls_id]

            conf = float(box.conf[0])

            

            # Draw mask

            mask_np = mask.cpu().numpy()

            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

            colored_mask = np.zeros_like(frame)

            colored_mask[mask_resized > 0.5] = [0, 255, 0]

            annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.4, 0)

            

            # Label

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            cv2.putText(annotated, f'{cls_name} {conf:.2f}', (x1, y1-5),

                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            print(f'Detected: {cls_name} {conf:.2f}')



cv2.imwrite('/home/nvidia/Desktop/SDV_workspace/scripts/yolo_seg.jpg', annotated)

print("Saved! Check scripts/yolo_seg.jpg")

