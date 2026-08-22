import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v5.keras"
REAL_DIR = "webcam_test/real"

IMG_SIZE = 224

# Best threshold from previous analysis
THRESHOLD = 0.60

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("WEBCAM REAL-WORLD MODEL TEST")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =========================================================
# LOAD WEBCAM IMAGES
# =========================================================

files = [
    f for f in os.listdir(REAL_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("\nWebcam images:", len(files))

if len(files) == 0:
    print("ERROR: No webcam images found!")
    exit()

# =========================================================
# PREPARE IMAGES
# =========================================================

images = []
valid_files = []

print("\nLoading images...")

for filename in files:

    path = os.path.join(REAL_DIR, filename)

    img = cv2.imread(path)

    if img is None:
        print("Could not read:", filename)
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    images.append(img)
    valid_files.append(filename)

X = np.array(images, dtype=np.float32)

print("Images loaded:", len(X))

# =========================================================
# PREDICTIONS
# =========================================================

print("\n" + "=" * 60)
print("RUNNING PREDICTIONS")
print("=" * 60)

predictions = model.predict(
    X,
    batch_size=16,
    verbose=1
).flatten()

# =========================================================
# CLASSIFICATION
# =========================================================

# All webcam images are REAL
true_labels = np.zeros(
    len(predictions),
    dtype=int
)

predicted_labels = (
    predictions >= THRESHOLD
).astype(int)

# 0 = REAL
# 1 = FAKE

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

# =========================================================
# RESULTS
# =========================================================

real_count = np.sum(
    predicted_labels == 0
)

fake_count = np.sum(
    predicted_labels == 1
)

print("\n")
print("=" * 60)
print("WEBCAM TEST RESULTS")
print("=" * 60)

print(f"\nTotal webcam images : {len(predictions)}")

print(f"Predicted REAL      : {real_count}")
print(f"Predicted FAKE      : {fake_count}")

print(f"\nThreshold            : {THRESHOLD:.2f}")

print(
    f"\nReal detection rate : "
    f"{real_count / len(predictions) * 100:.2f}%"
)

print(
    f"False FAKE rate     : "
    f"{fake_count / len(predictions) * 100:.2f}%"
)

print("\n")
print("=" * 60)
print("METRICS")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

# =========================================================
# CONFIDENCE ANALYSIS
# =========================================================

real_probabilities = 1 - predictions

print("\n")
print("=" * 60)
print("CONFIDENCE ANALYSIS")
print("=" * 60)

print(
    f"\nAverage REAL confidence: "
    f"{np.mean(real_probabilities) * 100:.2f}%"
)

print(
    f"Minimum REAL confidence: "
    f"{np.min(real_probabilities) * 100:.2f}%"
)

print(
    f"Maximum REAL confidence: "
    f"{np.max(real_probabilities) * 100:.2f}%"
)

# =========================================================
# SHOW FALSE FAKE PREDICTIONS
# =========================================================

print("\n")
print("=" * 60)
print("FALSE FAKE PREDICTIONS")
print("=" * 60)

false_fake_indices = np.where(
    predicted_labels == 1
)[0]

if len(false_fake_indices) == 0:

    print("\nExcellent! No real webcam images were")
    print("classified as FAKE.")

else:

    print(
        f"\n{len(false_fake_indices)} real webcam "
        "images were classified as FAKE:\n"
    )

    for index in false_fake_indices[:30]:

        fake_probability = predictions[index]

        print(
            f"{valid_files[index]:30s} "
            f"FAKE probability: "
            f"{fake_probability * 100:.2f}%"
        )

print("\n")
print("=" * 60)
print("WEBCAM TEST COMPLETED")
print("=" * 60)
