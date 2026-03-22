import cv2
import numpy as np
import time
import os
import handTrackingModule as htm

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ── Model Training ────────────────────────────────────────────────────────────

df = pd.read_csv("digit-recognizer/train.csv")
train = df.iloc[0:int(df.shape[0] * 0.7), :]
test  = df.iloc[int(df.shape[0] * 0.7):, :]

y_train = train["label"]
x_train = train.drop("label", axis=1) / 255.0   # FIX 4: normalise at training time

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

x_test = test.drop("label", axis=1) / 255.0      # FIX 4: normalise consistently
y_test = test["label"]
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ── Helper: canvas → 28×28 MNIST image ───────────────────────────────────────

def canvas_to_mnist(imgCanvas):
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.bitwise_not(imgGray)
    _, imgThresh = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(imgThresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        imgCrop   = imgThresh[y:y + h, x:x + w]
        imgResize = cv2.resize(imgCrop, (28, 28), interpolation=cv2.INTER_AREA)
        return imgResize

    return None


# ── Load palette overlays ─────────────────────────────────────────────────────

folder_path = "virtualpainter"
# FIX 2: filter hidden files (e.g. .DS_Store) instead of blindly popping index 0
myList = sorted([f for f in os.listdir(folder_path) if not f.startswith('.')])
print(f"Loaded {len(myList)} overlay image(s)")

img_overlay = []
for impath in myList:
    image = cv2.imread(f'{folder_path}/{impath}')
    img_overlay.append(image)

default_header  = img_overlay[2]
draw_color      = (230, 216, 173)
brush_thickness = 15
index_x_p, index_y_p = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)


# ── Camera setup ──────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = htm.HandDetector(detection_confidence=0.85)


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    # 1) Read frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # 2) Detect hand landmarks
    frame   = detector.findHands(frame)
    lmkList = detector.findPosition(frame, 0, False)

    if len(lmkList) != 0:
        index_x,  index_y  = lmkList[8][1:]
        middle_x, middle_y = lmkList[12][1:]

        both_up  = (lmkList[8][2]  < lmkList[5][2]) and (lmkList[12][2] < lmkList[9][2])
        index_up = (lmkList[8][2]  < lmkList[5][2]) and (lmkList[12][2] > lmkList[9][2])

        # ── SELECTION MODE (both fingers up) ─────────────────────────────────
        if both_up:
            index_x_p, index_y_p = 0, 0   # FIX 3: reset pen position on mode switch

            cv2.putText(frame, "Both Fingers are up!", (500, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, draw_color, 3)

            # Blue
            if 0 < index_x < 200 and 0 < index_y < 180:
                default_header  = img_overlay[2]
                draw_color      = (230, 216, 173)
                brush_thickness = 20
            # Pink
            elif 0 < index_x < 200 and 180 < index_y < 360:
                default_header  = img_overlay[1]
                draw_color      = (193, 182, 255)
                brush_thickness = 20
            # Yellow
            elif 0 < index_x < 200 and 360 < index_y < 540:
                default_header  = img_overlay[0]
                draw_color      = (224, 255, 255)
                brush_thickness = 20
            # Eraser
            elif 0 < index_x < 200 and 540 < index_y < 720:
                default_header = img_overlay[3]
                draw_color     = (0, 0, 0)

            cv2.rectangle(frame,
                          (index_x, index_y),
                          (index_x + 30, index_y + 30),
                          draw_color, cv2.FILLED)

        # ── DRAWING MODE (index finger only) ─────────────────────────────────
        elif index_up:
            cv2.putText(frame, "Index Finger is up!", (500, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
            cv2.circle(frame, (index_x, index_y), 15, draw_color, cv2.FILLED)

            if index_x_p == 0 and index_y_p == 0:
                index_x_p, index_y_p = index_x, index_y

            if draw_color == (0, 0, 0):
                # Eraser — thick line, no jitter filter needed
                cv2.line(frame,     (index_x_p, index_y_p), (index_x, index_y), draw_color, 100)
                cv2.line(imgCanvas, (index_x_p, index_y_p), (index_x, index_y), draw_color, 100)
            else:
                # Reduce jitter with distance threshold
                dist = ((index_x - index_x_p) ** 2 + (index_y - index_y_p) ** 2) ** 0.5
                if dist > 5:
                    cv2.line(frame,     (index_x_p, index_y_p), (index_x, index_y), draw_color, brush_thickness)
                    cv2.line(imgCanvas, (index_x_p, index_y_p), (index_x, index_y), draw_color, brush_thickness)

            index_x_p, index_y_p = index_x, index_y

        else:
            # No recognised gesture — reset pen anchor
            index_x_p, index_y_p = 0, 0

    # 3) Composite canvas onto live frame
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame  = cv2.bitwise_and(frame, imgInv)
    frame  = cv2.bitwise_or(frame, imgCanvas)

    # 4) Overlay palette sidebar
    frame[0:720, 0:200] = default_header

    cv2.imshow("Camera", frame)

    # FIX 1: single waitKey call — handles both 's' and 'q' correctly
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # FIX 5: user feedback when canvas is blank
        img28 = canvas_to_mnist(imgCanvas)
        if img28 is not None:
            img_flat   = img28.reshape(1, 784) / 255.0
            prediction = model.predict(img_flat)
            print("Predicted digit:", prediction[0])
        else:
            print("Canvas is empty — draw something first!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# TODO: allow user to change brush size with pinch distance between thumb and index finger