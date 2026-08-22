import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/deepfake_face_model_v2.keras"
REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

IMG_SIZE = 224

print("=" * 60)
print("LOADING FACE MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print()


def predict_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Could not read:", image_path)
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32)

img = tf.keras.applications.mobilenet_v2.preprocess_input(
    img
)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "FAKE"
        confidence = prediction * 100
    else:
        label = "REAL"
        confidence = (1 - prediction) * 100

    print("Image:", os.path.basename(image_path))
    print("Prediction:", label)
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 60)


print("=" * 60)
print("TESTING REAL FACE IMAGES")
print("=" * 60)

real_images = os.listdir(REAL_DIR)

for filename in real_images[:10]:
    path = os.path.join(REAL_DIR, filename)
    predict_image(path)


print("=" * 60)
print("TESTING FAKE FACE IMAGES")
print("=" * 60)

fake_images = os.listdir(FAKE_DIR)

for filename in fake_images[:10]:
    path = os.path.join(FAKE_DIR, filename)
    predict_image(path)