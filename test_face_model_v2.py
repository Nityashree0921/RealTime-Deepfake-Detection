import os
import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/deepfake_face_model_v2.keras"

REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

IMG_SIZE = 224


print("=" * 60)
print("LOADING FACE MODEL V2")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print()


def predict_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Could not read:", image_path)
        return

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    # IMPORTANT:
    # Do NOT divide by 255.
    # MobileNetV2 preprocessing is already inside the model.
    img = img.astype(np.float32)

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(
        img,
        verbose=0
    )[0][0]

    if prediction >= 0.5:

        label = "FAKE"
        confidence = prediction * 100

    else:

        label = "REAL"
        confidence = (1 - prediction) * 100

    print(
        f"{os.path.basename(image_path):30s}"
        f" -> {label:5s}"
        f" ({confidence:.2f}%)"
        f" | raw={prediction:.4f}"
    )


# =========================================================
# REAL
# =========================================================

print("=" * 60)
print("TESTING REAL FACE IMAGES")
print("=" * 60)

real_images = [
    f for f in os.listdir(REAL_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

for filename in real_images[:20]:

    path = os.path.join(
        REAL_DIR,
        filename
    )

    predict_image(path)


# =========================================================
# FAKE
# =========================================================

print()
print("=" * 60)
print("TESTING FAKE FACE IMAGES")
print("=" * 60)

fake_images = [
    f for f in os.listdir(FAKE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

for filename in fake_images[:20]:

    path = os.path.join(
        FAKE_DIR,
        filename
    )

    predict_image(path)