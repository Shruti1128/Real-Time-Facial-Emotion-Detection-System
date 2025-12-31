import csv, os, cv2, numpy as np

os.makedirs("fer_images", exist_ok=True)

with open("fer2013.csv") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        pixels = np.array(row["pixels"].split(), dtype=np.uint8)
        img = pixels.reshape(48,48)
        cv2.imwrite(f"fer_images/fer{i:07d}.png", img)

print("✅ FER images extracted")
