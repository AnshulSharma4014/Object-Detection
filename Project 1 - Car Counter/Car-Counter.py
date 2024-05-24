import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *  # Sorting is used for serialisation of frames

'''
# Used to camera access
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height
'''

'''
# Used for videos
'''
cap = cv2.VideoCapture("../Videos/cars.mp4")

# Creating MODAL
model = YOLO("../Yolo_weights/yolov8n.pt")

# For Classification
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

'''
To mask the video so that we only detect the motion when the object is in certain area 
'''
mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Crossing Line
# [0] = starting position on x-axis from left
# [1] = starting position on y-axis from top
# [2] = ending position on x-axis from left
# [3] = ending position on y-axis from top
limits = [400, 300, 700, 300]
total_count = 0

vehicle_count = set()

while True:
    success, img = cap.read()

    # Import graphics.png in while loop, else it will deteriorate the quality overtime
    graphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, graphics, (0, 0))

    # Now Overlay the mask on the img using bitwise AND
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    # Check for individual bounding boxes
    for r in results:
        boxes = r.boxes

        for box in boxes:

            print(box)

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence Score
            conf = math.ceil(box.conf[0] * 100) / 100  # Rounding to 2 decimal places
            print(conf)

            # Classification Names
            cls = box.cls[0]
            print(cls)

            currentClass = classNames[int(cls)]

            detection_list = ['car', 'bus', 'truck', 'motorbike']

            if (currentClass in detection_list) and conf > 0.3:
                # For Fancy Rectangle, use cvzone library
                cvzone.cornerRect(
                    img,
                    (x1, y1, w, h),
                    l=9,
                    rt=5
                )

                cvzone.putTextRect(
                    img,
                    f'{currentClass} {conf}',
                    (max(0, x1),
                     max(35, y1)),
                    scale=0.6,  # Scale of the text
                    thickness=1,  # Thickness of text
                    offset=3  # Removes the padding of the box in which the class is displayed
                )

                currentArray = np.array([x1, y1, x2, y2, conf])

                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 0, 255), thickness=5)

    for results in resultsTracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        print(results)

        w, h = x2 - x1, y2 - y1

        '''
        cvzone.cornerRect(
            img,
            (x1, y1, w, h),
            l=9,
            rt=2,
            colorR=(255, 0, 0)
        )

        cvzone.putTextRect(
            img,
            f'Id = {Id}',
            (max(0, x1),
             max(35, y1)),
            scale=2,  # Scale of the text
            thickness=3,  # Thickness of text
            offset=10  # Removes the padding of the box in which the class is displayed
        )
        '''

        # Find centre, if the object crossed line
        cx, cy = x1 + w//2, y1 + h//2

        # Test the centre
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Now finally count vehicles
        '''
        We added and subtracted 15 from the "Y" axis because giving single value
        is not recommended. It might miss the single value if the vehicle is 
        moving very fast. Hence, by adding and subtracting 15, we create a region
        for detection.
        '''
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and Id not in vehicle_count:
            total_count += 1
            vehicle_count.add(Id)

            # Change the line Colour if anything is detected
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=5)

    # cvzone.putTextRect(img, f'Count: {total_count}', (50, 50))
    cv2.putText(img, str(total_count), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 4)

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)
