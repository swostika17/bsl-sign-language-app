import csv
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

SIGN_LABEL = "peace"   # CHANGE THIS PER SIGN
SAMPLES = 200
CSV_FILE = "hand_landmarks.csv"

detector = HandDetector(maxHands=2, detectionCon=0.7)
cap = cv2.VideoCapture(0)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for h in range(2):
            for i in range(21):
                header += [f"h{h}_x{i}", f"h{h}_y{i}", f"h{h}_z{i}"]
        header.append("label")
        writer.writerow(header)

count = 0
print(f"Collecting data for sign: {SIGN_LABEL}")

while count < SAMPLES:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if hands:
        data = []

        hands = sorted(hands, key=lambda x: x["type"] == "Right")

        for hand in hands[:2]:
            lm = np.array(hand["lmList"])[:, :3]
            wrist = lm[0]
            lm = lm - wrist
            scale = np.linalg.norm(lm[9]) + 1e-6
            lm = lm / scale
            data.extend(lm.flatten())

        if len(hands) == 1:
            data.extend([0.0] * (21 * 3))

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data + [SIGN_LABEL])

        count += 1
        cv2.putText(img, f"{count}/{SAMPLES}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collecting Data", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
import cv2
