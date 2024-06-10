from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Videos/ppe-1-1.mp4")
model = YOLO("ConstructionSafetyPPE.pt")
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)   # red

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Check for individual bounding boxes
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Classification Names
            cls = box.cls[0]

            currentClass = classNames[int(cls)]
            if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                myColor = (0, 255, 0)
            else:
                myColor = (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

            # Confidence Score
            conf = math.ceil(box.conf[0] * 100) / 100

            cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor, colorT=(255,255,255), colorR=myColor)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
