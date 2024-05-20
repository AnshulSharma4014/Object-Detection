from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 1280)    #width
cap.set(4, 720)    #height

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)