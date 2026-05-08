import cv2
import time

# Use macOS native backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Give camera time to initialise (IMPORTANT on macOS)
time.sleep(2)

# Check if webcam opened
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to grab frame")
        continue   # don't break immediately

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
