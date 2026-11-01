import cv2 as cv
import numpy as np

capture = cv.VideoCapture('/home/dhruv/Downloads/vid5.mp4')
sift = cv.SIFT_create()


bg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
open_k  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
MIN_BOX_AREA = 4000     # a merged box smaller than this is not a vehicle
MERGE_GAP = 15
WARMUP = 30
frame_no = 0
def merge_boxes(boxes, gap=15):
    
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        while boxes:
            x, y, w, h = boxes.pop()
            i = 0
            while i < len(boxes):
                bx, by, bw, bh = boxes[i]
                if (x - gap < bx + bw and bx - gap < x + w and
                        y - gap < by + bh and by - gap < y + h):
                    nx, ny = min(x, bx), min(y, by)
                    x2, y2 = max(x + w, bx + bw), max(y + h, by + bh)
                    x, y, w, h = nx, ny, x2 - nx, y2 - ny
                    boxes.pop(i)
                    merged = True
                else:
                    i += 1
            out.append([x, y, w, h])
        boxes = out
    return [tuple(b) for b in boxes]

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_no += 1

    target_w = 640
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    frame = cv.resize(frame, (target_w, target_h), interpolation=cv.INTER_AREA)

    
    # inside the loop, replacing the old fg / dilate / findContours / boxes code
    lr = -1 if frame_no <= WARMUP else 0.002          # learn fast at first, then slowly
    fg = bg.apply(frame, learningRate=lr)
    _, fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)       # drop shadows
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, open_k)            # remove specks
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, close_k)          # join pieces of one car
    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    if frame_no > WARMUP:
        raw = [cv.boundingRect(c) for c in contours if cv.contourArea(c) >= 150]
        boxes = [b for b in merge_boxes(raw, MERGE_GAP) if b[2] * b[3] >= MIN_BOX_AREA]

    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for (x, y, bw, bh) in boxes:
        mask[y:y + bh, x:x + bw] = 255
    kp, des = sift.detectAndCompute(gray, mask)

    vis = cv.drawKeypoints(frame, kp, None,
                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    for (x, y, bw, bh) in boxes:
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)   # (0,0,255) is red in BGR

    cv.imshow("Video", vis)
    cv.imshow("Foreground", fg)
    if cv.waitKey(10) & 0xFF == ord('d'):
         break

capture.release()
cv.destroyAllWindows()