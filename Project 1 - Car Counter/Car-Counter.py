from ultralytics import YOLO
import cv2
import cvzone
import math

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
model = YOLO("../Yolo_weights/yolov8l.pt")

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

while True:
    success, img = cap.read()

    # Now Overlay the mask on the img using bitwise AND
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

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
                cvzone.cornerRect(img, (x1, y1, w, h), l=8)

                cvzone.putTextRect(
                    img,
                    f'{currentClass} {conf}',
                    (max(0, x1),
                     max(35, y1)),
                    scale=0.6,  # Scale of the text
                    thickness=1,    # Thickness of text
                    offset=3    # Removes the padding of the box in which the class is displayed
                )

    cv2.imshow("Image", img)
    cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(0)
