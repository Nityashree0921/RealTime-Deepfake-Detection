import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/deepfake_face_model_v5.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # IMPORTANT:
    # Do NOT call preprocess_input().
    # The model already contains Rescaling.

    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    prediction = float(model.predict(img, verbose=0)[0][0])

    return prediction


print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

print("\nREAL IMAGES")
print("-" * 70)

real_dir = "face_frames/real"

real_files = [
    f for f in os.listdir(real_dir)
    if f.lower().endswith(".jpg")
][:10]

for filename in real_files:

    path = os.path.join(real_dir, filename)

    p = predict_image(path)

    print(
        f"{filename:25s} "
        f"FAKE={p*100:6.2f}% "
        f"REAL={(1-p)*100:6.2f}%"
    )


print("\nFAKE IMAGES")
print("-" * 70)

fake_dir = "face_frames/fake"

fake_files = [
    f for f in os.listdir(fake_dir)
    if f.lower().endswith(".jpg")
][:10]

for filename in fake_files:

    path = os.path.join(fake_dir, filename)

    p = predict_image(path)

    print(
        f"{filename:25s} "
        f"FAKE={p*100:6.2f}% "
        f"REAL={(1-p)*100:6.2f}%"
    )