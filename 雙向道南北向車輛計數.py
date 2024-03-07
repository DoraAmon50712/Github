import torch
import numpy as np
import cv2
import time
from temp import isBetween

prev_time = 0
count = 0    # 北上
count2 = 0   # 南下
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
cap = cv2.VideoCapture("DJI_0008.MOV")

while cap.isOpened():
    success, frame = cap.read()
    cv2.line(frame, (609,1533), (3000,1533), (255, 255, 255), 20)     # 左下座標線
    # cv2.rectangle(frame,(1103,224),(609,224),(2,2,255),3,cv2.LINE_AA)
    if not success:
        print("Ignoring empty camera frame.")
        break
    frame = cv2.resize(frame, (1240, 810))
    results = model(frame)
    detections = results.xyxy[0]  # 獲取偵測到的物件和其邊界框座標

    for detection in detections:
        class_idx = int(detection[-1].item())  # 獲取物件的類別標籤
        if class_idx in [2, 7]:  # 只取汽車和卡車
            xmin, ymin, xmax, ymax = map(int, detection[:4].tolist())  # 解析邊界框座標
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 繪製邊界框

            # 計算中心點座標
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            # 計數車輛並繪製辨識框
            if(550 < center_y < 555):
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                if(center_x > 600):         # 北上數量+1
                    count += 1
                else:                       # 南下數量+1
                    count2 += 1
            else:
                cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)

    # 標記FPS及數量
    cv2.putText(frame, f'FPS: {int(1 / (time.time() - prev_time))}',
                (500 ,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(frame, f'North: {count}',
                (990, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(frame, f'South: {count2}',
                (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    prev_time = time.time()
    cv2.imshow('YOLO COCO 02', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
