from ultralytics import YOLO
import cv2

model = YOLO('../Yolo_weights/yolov8l.pt')  #yolo version 8 NANO (n for NANO)
results = model("Images/car-bus-building.jpeg", show=True)
cv2.waitKey(0)  # we need this to stop the image from automatically closing the file. 0 means unless user click close, do not close