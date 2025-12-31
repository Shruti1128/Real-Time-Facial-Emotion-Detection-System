import os
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# FER+ emotion mapping
emotion_map = {
    0: "neutral",
    1: "happy",
    2: "surprise",
    3: "sad",
    4: "angry",
    5: "disgust",
    6: "fear"
}

CSV_FILE = "fer2013new.csv"

# Output folders
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

images = []
labels = []

with open(CSV_FILE) as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header

    for row in reader:
        # pixels
        pixels = np.array(row[1].split(), dtype=np.uint8)
        img = pixels.reshape(48, 48)

        # FER+ voting columns (8 columns)
        votes = np.array(row[2:10], dtype=int)
        label = emotion_map[np.argmax(votes)]

        images.append(img)
        labels.append(label)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

def save_split(X, y, split):
    for i, (img, label) in enumerate(zip(X, y)):
        folder = f"dataset/{split}/{label}"
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f"{i}.png"), img)

save_split(X_train, y_train, "train")
save_split(X_test, y_test, "test")

print("✅ FER+ dataset prepared successfully")
