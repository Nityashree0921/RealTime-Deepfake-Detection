import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v5.keras"

REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

IMG_SIZE = 224
THRESHOLD = 0.50

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 70)
print("V5 CORRECT LABEL EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print("Model input :", model.input_shape)
print("Model output:", model.output_shape)

# =========================================================
# LOAD IMAGES
# =========================================================

def load_images(folder):

    images = []
    filenames = []

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for filename in files:

        path = os.path.join(folder, filename)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        # IMPORTANT:
        # Do NOT use mobilenet_v2.preprocess_input().
        # The model already contains Rescaling.

        img = img.astype(np.float32)

        images.append(img)
        filenames.append(filename)

    return np.array(images), filenames


print("\nLoading REAL images...")

real_images, real_files = load_images(REAL_DIR)

print("REAL:", len(real_images))

print("\nLoading FAKE images...")

fake_images, fake_files = load_images(FAKE_DIR)

print("FAKE:", len(fake_images))

# =========================================================
# COMBINE
# =========================================================

X = np.concatenate(
    [real_images, fake_images],
    axis=0
)

# Our evaluation labels:
# 0 = FAKE
# 1 = REAL

y_true = np.concatenate([
    np.ones(len(real_images), dtype=int),
    np.zeros(len(fake_images), dtype=int)
])

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print("REAL :", len(real_images))
print("FAKE :", len(fake_images))
print("TOTAL:", len(X))

print("\nInput shape:", X.shape)
print("Input min :", X.min())
print("Input max :", X.max())

# =========================================================
# PREDICTIONS
# =========================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

# Model output = P(REAL)
p_real = model.predict(
    X,
    batch_size=16,
    verbose=1
).flatten()

p_fake = 1.0 - p_real

# =========================================================
# CLASSIFICATION
# =========================================================

# Model:
# p_real >= threshold -> REAL
# p_real < threshold  -> FAKE

y_pred = (
    p_real >= THRESHOLD
).astype(int)

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

# FAKE = positive class (0)
precision_fake = precision_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

recall_fake = recall_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

f1_fake = f1_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

# REAL = positive class (1)
precision_real = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall_real = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1_real = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[1, 0]
)

# rows:
# REAL
# FAKE
#
# columns:
# REAL
# FAKE

# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)

print("\nFAKE DETECTION")
print("-" * 40)

print(
    f"Precision: {precision_fake * 100:.2f}%"
)

print(
    f"Recall   : {recall_fake * 100:.2f}%"
)

print(
    f"F1 Score : {f1_fake * 100:.2f}%"
)

print("\nREAL DETECTION")
print("-" * 40)

print(
    f"Precision: {precision_real * 100:.2f}%"
)

print(
    f"Recall   : {recall_real * 100:.2f}%"
)

print(
    f"F1 Score : {f1_real * 100:.2f}%"
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("""
                  PREDICTED
                  REAL    FAKE
ACTUAL REAL
ACTUAL FAKE
""")

print(
    f"ACTUAL REAL       {cm[0][0]:4d}    {cm[0][1]:4d}"
)

print(
    f"ACTUAL FAKE       {cm[1][0]:4d}    {cm[1][1]:4d}"
)

# =========================================================
# PROBABILITY ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)

real_p = p_real[:len(real_images)]
fake_p = p_real[len(real_images):]

print("\nREAL images:")

print(
    f"Average REAL probability : "
    f"{np.mean(real_p) * 100:.2f}%"
)

print(
    f"Minimum REAL probability : "
    f"{np.min(real_p) * 100:.2f}%"
)

print(
    f"Maximum REAL probability : "
    f"{np.max(real_p) * 100:.2f}%"
)

print("\nFAKE images:")

print(
    f"Average REAL probability : "
    f"{np.mean(fake_p) * 100:.2f}%"
)

print(
    f"Minimum REAL probability : "
    f"{np.min(fake_p) * 100:.2f}%"
)

print(
    f"Maximum REAL probability : "
    f"{np.max(fake_p) * 100:.2f}%"
)

# =========================================================
# END
# =========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)