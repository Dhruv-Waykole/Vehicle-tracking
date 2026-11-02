import cv2 as cv
import numpy as np
from collections import deque
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")
capture = cv.VideoCapture('/home/dhruv/Downloads/vid3.mp4')
sift = cv.SIFT_create()


bf = cv.BFMatcher(cv.NORM_L2)   
RATIO = 0.75                    
GAP = 3                         
MIN_MOVE = 1.5                 
MAX_MOVE = 60                   
history = deque(maxlen=GAP) 

GIF_SECONDS = 8                 
GIF_FPS = 10                     
GIF_WIDTH = 480                 

video_fps = capture.get(cv.CAP_PROP_FPS) or 25
step = max(1, round(video_fps / GIF_FPS))     
max_frames = GIF_SECONDS * GIF_FPS
gif_frames = []
frame_no = 0

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    target_w = 640
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    frame = cv.resize(frame, (target_w, target_h), interpolation=cv.INTER_AREA)

   
    r = model(frame, classes=[2, 3, 5, 7], conf=0.3, verbose=False)[0]
    boxes = []
    for x1, y1, x2, y2 in r.boxes.xyxy.tolist():
        x1, y1 = max(0, int(x1)), max(0, int(y1))                   
        x2, y2 = min(target_w, int(x2)), min(target_h, int(y2))
        boxes.append((x1, y1, x2 - x1, y2 - y1))

  
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for (x, y, bw, bh) in boxes:
        mask[y:y + bh, x:x + bw] = 255
    kp, des = sift.detectAndCompute(gray, mask)

    good = []        
    arrows = []     
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
                        good.append(kp[m.trainIdx])
                        arrows.append(((int(x0), int(y0)), (int(x1), int(y1))))
    history.append((kp, des))

   
    vis = frame.copy()                       
    boxes_only = frame.copy()                

    for (p0, p1) in arrows:
        cv.line(vis, p0, p1, (0, 255, 0), 1)
    for k in good:
        cv.circle(vis, (int(k.pt[0]), int(k.pt[1])), 3, (0, 255, 255), -1)

    for (x, y, bw, bh) in boxes:
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv.rectangle(boxes_only, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv.putText(boxes_only, "vehicle", (x, y - 6),
           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv.imshow("Video", vis)
    frame_no += 1
    if frame_no % step == 0 and len(gif_frames) < max_frames:
     small = cv.resize(vis, (GIF_WIDTH, int(vis.shape[0] * GIF_WIDTH / vis.shape[1])))
     gif_frames.append(Image.fromarray(cv.cvtColor(small, cv.COLOR_BGR2RGB)))

    if len(gif_frames) >= max_frames:             
     break
    cv.moveWindow("Boxes only", 700, 0)
    cv.imshow("Boxes only", boxes_only)
    if cv.waitKey(10) & 0xFF == ord('d'):
        break
gif_frames[0].save("output1.gif", save_all=True, append_images=gif_frames[1:],
                   duration=int(1000 / GIF_FPS), loop=0, optimize=True)
capture.release()
cv.destroyAllWindows()