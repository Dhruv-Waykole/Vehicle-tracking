import cv2 as cv
import numpy as np
from collections import deque


VIDEO = '/home/dhruv/Downloads/vid5.mp4'
TARGET_W = 640
GAP = 3                 
RATIO = 0.75            
MIN_MOVE = 1.5          
MAX_MOVE = 80           
JOIN_SIZE = 41          
MIN_BOX_AREA = 3000     

capture = cv.VideoCapture(VIDEO)
if not capture.isOpened():
    raise SystemExit("Could not open the video, check the path")

sift = cv.SIFT_create(nfeatures=2000)
bf = cv.BFMatcher(cv.NORM_L2)           
join_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (JOIN_SIZE, JOIN_SIZE))
history = deque(maxlen=GAP)           


while True:
    ok, frame = capture.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    frame = cv.resize(frame, (TARGET_W, int(h * TARGET_W / w)), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    
    kp, des = sift.detectAndCompute(gray, None)

    
    moving = []
    if len(history) == GAP and des is not None:
        kp_old, des_old = history[0]
        if des_old is not None and len(des_old) >= 2 and len(des) >= 2:
            pairs = bf.knnMatch(des_old, des, k=2)
            for p in pairs:
                if len(p) < 2:
                    continue
                m, n = p
                if m.distance < RATIO * n.distance:            
                    x0, y0 = kp_old[m.queryIdx].pt
                    x1, y1 = kp[m.trainIdx].pt
                    dist = np.hypot(x1 - x0, y1 - y0)            
                    if MIN_MOVE < dist < MAX_MOVE:
                        moving.append(kp[m.trainIdx])
    history.append((kp, des))

    
    blob = np.zeros(gray.shape, np.uint8)
    for k in moving:
        cv.circle(blob, (int(k.pt[0]), int(k.pt[1])), 3, 255, -1)
    blob = cv.dilate(blob, join_k)

    
    contours, _ = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    pad = JOIN_SIZE // 2                                  
    for c in contours:
        if cv.contourArea(c) >= MIN_BOX_AREA:
            x, y, bw, bh = cv.boundingRect(c)
            if bw > 2 * pad and bh > 2 * pad:
                boxes.append((x + pad, y + pad, bw - 2 * pad, bh - 2 * pad))   

    
    vis = cv.drawKeypoints(frame, moving, None,
                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for (x, y, bw, bh) in boxes:
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    cv.imshow("Video", vis)
    cv.imshow("Moving keypoints", blob)
    if cv.waitKey(10) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()