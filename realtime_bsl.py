import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model

# ---------- Load trained model and labels ----------
model = load_model("sign_model.h5")          # from your new train_model.py
labels = np.load("labels.npy", allow_pickle=True)

# ---------- Hand detector (up to 2 hands) ----------
detector = HandDetector(maxHands=2, detectionCon=0.7)
cap = cv2.VideoCapture(0)

last_label = ""
stable_count = 0
STABLE_FRAMES = 8         # how many frames in a row before showing text
CONF_THRESHOLD = 0.6      # minimum softmax confidence to consider it valid

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img, draw=True)

    if hands:
        features = []

        # --- build EXACTLY the same features as collect_data.py ---

        # 1) sort hands for consistency (left to right by x-center)
        hands_sorted = sorted(hands, key=lambda h: h["center"][0])

        # 2) process up to 2 hands
        for hand in hands_sorted[:2]:
            lm = np.array(hand["lmList"])[:, :3]   # shape (21, 3)

            # wrist-relative + scale normalisation
            wrist = lm[0]
            lm = lm - wrist
            scale = np.linalg.norm(lm[9]) + 1e-6   # use landmark 9 as reference
            lm = lm / scale

            features.extend(lm.flatten().tolist())  # 21*3 = 63 values

        # 3) pad if only one hand (second hand = zeros)
        if len(hands_sorted) == 1:
            features.extend([0.0] * (21 * 3))      # 63 zeros

        # Now features length should be 126 = 2 * 21 * 3
        if len(features) == 126:
            X = np.array(features, dtype=np.float32).reshape(1, -1)

            # ---------- Predict ----------
            probs = model.predict(X, verbose=0)[0]   # softmax probs
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = str(labels[idx])

            # ---------- Temporal smoothing ----------
            if conf >= CONF_THRESHOLD:
                if label == last_label:
                    stable_count += 1
                else:
                    last_label = label
                    stable_count = 0

                if stable_count >= STABLE_FRAMES:
                    cv2.putText(
                        img,
                        f"{label} ({conf:.2f})",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

    cv2.imshow("BSL Realtime Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
