import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import csv
import os

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
model = load_model("emotion_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

DURATION = 5
start_time = time.time()
predictions = []     
final_emotion = None
running = True

QUIT_TL = (20, 420)
QUIT_BR = (160, 470)
RESTART_TL = (180, 420)
RESTART_BR = (360, 470)

if not os.path.exists("results.csv"):
    with open("results.csv", "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Emotion"])

def mouse_click(event, x, y, flags, param):
    global running, start_time, predictions, final_emotion
    if event == cv2.EVENT_LBUTTONDOWN:
        if QUIT_TL[0] <= x <= QUIT_BR[0] and QUIT_TL[1] <= y <= QUIT_BR[1]:
            running = False
        if RESTART_TL[0] <= x <= RESTART_BR[0] and RESTART_TL[1] <= y <= RESTART_BR[1]:
            start_time = time.time()
            predictions = []
            final_emotion = None

cv2.namedWindow("Timed Emotion Detection")
cv2.setMouseCallback("Timed Emotion Detection", mouse_click)

while running:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        margin = 20
        y1 = max(0, y - margin)
        y2 = min(gray.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(gray.shape[1], x + w + margin)

        face = gray[y1:y2, x1:x2]
        face = cv2.GaussianBlur(face, (3,3), 0)
        face = cv2.equalizeHist(face)
        face = cv2.resize(face, (48,48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        pred = model.predict(face, verbose=0)[0]
        emotion = emotion_labels[np.argmax(pred)]
        confidence = np.max(pred)

        if final_emotion is None:
            predictions.append((emotion, confidence))

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    elapsed = time.time() - start_time
    remaining = int(DURATION - elapsed)

    if elapsed < DURATION:
        cv2.putText(frame, f"Detecting in: {remaining}s",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 2)

    else:
        if final_emotion is None:
            score = {}
            for emo, conf in predictions:
                score[emo] = score.get(emo, 0) + conf

            sorted_emotions = sorted(score.items(), key=lambda x: x[1], reverse=True)

            top, top_score = sorted_emotions[0]
            second, second_score = sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0)

            if top == "Happy" and second == "Sad":
                if (top_score - second_score) < 0.15 * top_score:
                    final_emotion = "Sad"
                else:
                    final_emotion = "Happy"
            else:
                final_emotion = top

            with open("results.csv", "a", newline="") as f:
                csv.writer(f).writerow([time.ctime(), final_emotion])

        cv2.putText(frame, f"Final Emotion: {final_emotion}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

        cv2.putText(frame, "Result locked",
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200,200,200), 1)

    cv2.rectangle(frame, QUIT_TL, QUIT_BR, (0,0,255), -1)
    cv2.putText(frame, "QUIT", (45,455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.rectangle(frame, RESTART_TL, RESTART_BR, (0,180,0), -1)
    cv2.putText(frame, "RESTART", (200,455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame, "Press Q to exit", (380,455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Timed Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
