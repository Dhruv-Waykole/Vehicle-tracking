import cv2 as cv
import numpy as np
from collections import deque

capture = cv.VideoCapture('/home/dhruv/Downloads/vid3.mp4')
sift = cv.SIFT_create()

# ---------------- NEW: matching settings ----------------
bf = cv.BFMatcher(cv.NORM_L2)   # L2 = straight-line distance between two 128-number descriptors
RATIO = 0.75                    # Lowe's ratio test: best match must beat 2nd best by this factor
GAP = 3                         # compare with the frame GAP steps earlier
MIN_MOVE = 1.5                  # pixels: moved less than this = background (road, trees, signs)
MAX_MOVE = 60                   # pixels: moved more than this = wrong match
history = deque(maxlen=GAP)     # (keypoints, descriptors) of the last GAP frames

# ---------------- your existing settings ----------------
bg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
open_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
MIN_BOX_AREA = 4000     # a merged box smaller than this is not a vehicle

WARMUP = 30
frame_no = 0




while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_no += 1

    target_w = 640
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    frame = cv.resize(frame, (target_w, target_h), interpolation=cv.INTER_AREA)

    # ---------------- your existing box detection ----------------
    lr = -1 if frame_no <= WARMUP else 0.002          # learn fast at first, then slowly
    fg = bg.apply(frame, learningRate=lr)
    _, fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)       # drop shadows
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, open_k)            # remove specks
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, close_k)          # join pieces of one car
    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    if frame_no > WARMUP:
         for c in contours:
          x, y, bw, bh = cv.boundingRect(c)
          if bw * bh >= MIN_BOX_AREA:
            boxes.append((x, y, bw, bh))
        
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for (x, y, bw, bh) in boxes:
        mask[y:y + bh, x:x + bw] = 255
    kp, des = sift.detectAndCompute(gray, mask)

    # ---------------- NEW: L2 matching + Lowe's ratio test + movement filter ----------------
    good = []        # keypoints in the current frame that passed every test
    arrows = []      # (old position, new position) of each good match
    if len(history) == GAP and des is not None:
        kp_old, des_old = history[0]
        if des_old is not None and len(des_old) >= 2 and len(des) >= 2:
            pairs = bf.knnMatch(des_old, des, k=2)        # the 2 closest candidates (by L2)
            for p in pairs:
                if len(p) < 2:                            # sometimes only one candidate exists
                    continue
                m, n = p                                  # m = best, n = second best
                if m.distance < RATIO * n.distance:       # Lowe's ratio test
                    x0, y0 = kp_old[m.queryIdx].pt
                    x1, y1 = kp[m.trainIdx].pt
                    dist = np.hypot(x1 - x0, y1 - y0)     # how far the point moved
                    if MIN_MOVE < dist < MAX_MOVE:        # keep only points that really moved
                        good.append(kp[m.trainIdx])
                        arrows.append(((int(x0), int(y0)), (int(x1), int(y1))))
    history.append((kp, des))

    # ---------------- draw ----------------
    vis = frame.copy()
    for (p0, p1) in arrows:
        cv.line(vis, p0, p1, (0, 255, 0), 1)              # green line = how the point moved
    for k in good:
        cv.circle(vis, (int(k.pt[0]), int(k.pt[1])), 3, (0, 255, 255), -1)   # yellow dot = matched keypoint

    for (x, y, bw, bh) in boxes:
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)   # (0,0,255) is red in BGR

    cv.imshow("Video", vis)
    cv.imshow("Foreground", fg)
    if cv.waitKey(10) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()