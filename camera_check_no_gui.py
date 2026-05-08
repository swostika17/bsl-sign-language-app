import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
time.sleep(2)

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Camera opened")

ret, frame = cap.read()

if ret and frame is not None:
    print("Frame captured successfully")
    print("Frame shape:", frame.shape)
else:
    print("Failed to capture frame")

cap.release()
