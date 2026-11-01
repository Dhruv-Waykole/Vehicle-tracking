import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
capture = cv.VideoCapture('/home/dhruv/Downloads/vid5.mp4')
sift = cv.SIFT_create()

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    target_w = 640
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    frame = cv.resize(frame, (target_w, target_h), interpolation=cv.INTER_AREA)

    # YOLO finds the vehicles
    r = model(frame, classes=[2, 3, 5, 7], conf=0.3, verbose=False)[0]
    boxes = []
    for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
        x1, y1 = max(0, int(x1)), max(0, int(y1))                     # keep inside the frame
        x2, y2 = min(target_w, int(x2)), min(target_h, int(y2))
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    # SIFT only inside the boxes
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for (x, y, bw, bh) in boxes:
        mask[y:y + bh, x:x + bw] = 255
    kp, des = sift.detectAndCompute(gray, mask)

    vis = cv.drawKeypoints(frame, kp, None,
                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for (x, y, bw, bh) in boxes:
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    cv.imshow("Video", vis)
    if cv.waitKey(10) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()