import cv2 as cv
import numpy as np

capture = cv.VideoCapture('/home/dhruv/Downloads/vid3.mp4')
sift = cv.SIFT_create()

bg = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
MIN_AREA = 800       
WARMUP = 30           
frame_no = 0



bf = cv.BFMatcher(cv.NORM_L2)        
RATIO = 0.75                          
MIN_MOVE = 1.5                        
MAX_MOVE = 60                        
prev_kp, prev_des = None, None

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_no += 1

    target_w = 640
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    frame = cv.resize(frame, (target_w, target_h), interpolation=cv.INTER_AREA)

  
    fg = bg.apply(frame)                                        
    _, fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)        
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel)             
    fg = cv.dilate(fg, kernel, iterations=2)                   

    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    if frame_no > WARMUP:
        for c in contours:
            if cv.contourArea(c) >= MIN_AREA:
                boxes.append(cv.boundingRect(c))              

    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for (x, y, bw, bh) in boxes:
        mask[y:y + bh, x:x + bw] = 255
    kp, des = sift.detectAndCompute(gray, mask)

    arrows = []                                   
    if prev_des is not None and des is not None and len(prev_des) >= 2 and len(des) >= 2:
      pairs = bf.knnMatch(prev_des, des, k=2)    
      for p in pairs:
        if len(p) < 2:                       
            continue
        m, n = p                               
        if m.distance < RATIO * n.distance:    
            p0 = prev_kp[m.queryIdx].pt
            p1 = kp[m.trainIdx].pt
            move = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if MIN_MOVE < move < MAX_MOVE:     
                arrows.append((p0, p1, m.distance))
    prev_kp, prev_des = kp, des                    

    vis = cv.drawKeypoints(frame, kp, None,
                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    for (x, y, bw, bh) in boxes:
        cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)   

    for (p0, p1, d) in arrows:
        cv.line(vis, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 1)
    cv.imshow("Video", vis)
    cv.imshow("Foreground", fg)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()