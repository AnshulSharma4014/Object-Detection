import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import Constants

classNames = Constants.CLASS_NAMES
cap = cv2.VideoCapture("../Videos/people.mp4")
model = YOLO("../Yolo_weights/yolov8n.pt")
mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits_up = [100, 190, 300, 150]
limits_down = [590, 570, 770, 500]
total_count_up = 0
total_count_down = 0
person_count = set()

while True:
    success, img = cap.read()
    graphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, graphics, (730, 50))
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = box.cls[0]

            currentClass = classNames[int(cls)]
            if currentClass == 'person' and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), color=(0, 0, 255), thickness=5)
    cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), color=(0, 0, 255), thickness=5)

    for results in resultsTracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w//2, y1 + h//2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
        cvzone.putTextRect(img, f'Id = {Id}', (max(0, x1), max(35, y1)), scale=1.5, thickness=3, offset=3)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        if limits_up[0] < cx < limits_up[2] and limits_up[1] - 25 < cy < limits_up[1] and Id not in person_count:
            total_count_up += 1
            person_count.add(Id)
            cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), color=(0, 255, 0), thickness=5)

        if limits_down[0] < cx < limits_down[2] and limits_down[1] - 50 < cy < limits_down[1] and Id not in person_count:
            total_count_down += 1
            person_count.add(Id)
            cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), color=(0, 255, 0), thickness=5)

    cv2.putText(img, str(total_count_up), (930, 135), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(total_count_down), (1190, 135), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("Image", img)
    cv2.waitKey(1)