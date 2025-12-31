from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

train_dir = "dataset/train"
test_dir = "dataset/test"


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

print("📊 Class Weights:")
for k, v in class_weights.items():
    print(f"Class {k}: {v:.2f}")


model = build_model()

history = model.fit(
    train_data,
    epochs=30,
    validation_data=test_data,
    class_weight=class_weights
)

model.save("emotion_model.h5")
print("✅ Model saved as emotion_model.h5")
