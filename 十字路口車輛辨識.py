import torch
import numpy as np
import cv2
import time

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")

prev_time = 0
targets = {}
count1 = 0    # 左下
count2 = 0    # 右下
count3 = 0    # 右上
count4 = 0    # 左上
dist_threshold = 150


model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
cap = cv2.VideoCapture("DJI_0006 - Trim.mp4")
cv2.namedWindow("video")
cv2.setMouseCallback("video", mouse_callback)
while cap.isOpened():
    success, frame = cap.read()
    cv2.line(frame, (175, 1250), (1650, 1900), (0, 0, 255), 5)   # 左下座標線

    #cv2.line(frame, (200, 1200), (748, 1500), (0, 0, 255), 5)
    #cv2.line(frame, (1050, 1590), (1590, 1845), (0, 0, 255), 5)
    if not success:
        print("Ignoring empty camera frame.")
        continue
    frame = cv2.resize(frame, (1240, 810))
    results = model(frame)
    detections = results.xyxy[0]  # 獲取偵測到的物件和其邊界框座標

    for detection in detections:
        class_idx = int(detection[-1].item())  # 獲取物件的類別標籤
        if class_idx in [2, 7]:  # 只取汽車和卡車
            xmin, ymin, xmax, ymax = map(int, detection[:4].tolist())  # 解析邊界框座標
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 繪製邊界框
            # 計算中心點座標並繪製
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            # print(center_x,center_y)
            center = (center_x,center_y)

    cv2.putText(frame, f'FPS: {int(1 / (time.time() - prev_time))}',
                (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.putText(frame, f'South: {count1}',
                (3, 775), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.putText(frame, f'East: {count2}',
                (975, 775), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.putText(frame, f'North: {count3}',
                (975, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.putText(frame, f'West: {count4}',
                (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    prev_time = time.time()
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

