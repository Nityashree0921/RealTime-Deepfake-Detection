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

MODEL_PATH = "models/deepfake_face_model_v5.keras"

REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

IMG_SIZE = 224

# Since training labels are:
# fake = 0
# real = 1
#
# Model output = probability of REAL

THRESHOLD = 0.50


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

def load_images(folder, label):

    images = []
    labels = []
    names = []

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
        # Model already contains Rescaling layer.
        # Therefore DON'T preprocess here.

        images.append(
            img.astype(np.float32)
        )

        labels.append(label)
        names.append(filename)

    return images, labels, names


print("\nLoading REAL images...")

real_images, real_labels, real_names = load_images(
    REAL_DIR,
    1
)

print("REAL:", len(real_images))


print("\nLoading FAKE images...")

fake_images, fake_labels, fake_names = load_images(
    FAKE_DIR,
    0
)

print("FAKE:", len(fake_images))


# =========================================================
# COMBINE
# =========================================================

X = np.array(
    real_images + fake_images,
    dtype=np.float32
)

y_true = np.array(
    real_labels + fake_labels,
    dtype=int
)

names = real_names + fake_names


print("\nDataset:")
print("REAL:", len(real_images))
print("FAKE:", len(fake_images))
print("TOTAL:", len(X))

print("\nInput range:")
print("MIN:", X.min())
print("MAX:", X.max())


# =========================================================
# PREDICT
# =========================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

# Output = probability of REAL
real_probability = model.predict(
    X,
    batch_size=16,
    verbose=1
).flatten()


# =========================================================
# CLASSIFICATION
# =========================================================

predicted_labels = (
    real_probability >= THRESHOLD
).astype(int)


# 0 = FAKE
# 1 = REAL


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_true,
    predicted_labels
)

precision = precision_score(
    y_true,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    y_true,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predicted_labels,
    zero_division=0
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    predicted_labels
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nThreshold : {THRESHOLD:.2f}")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n                PREDICTED")
print("                FAKE    REAL")
print(
    f"ACTUAL FAKE    {cm[0,0]:5d}   {cm[0,1]:5d}"
)

print(
    f"ACTUAL REAL    {cm[1,0]:5d}   {cm[1,1]:5d}"
)


# =========================================================
# PROBABILITY ANALYSIS
# =========================================================

real_probs = real_probability[y_true == 1]
fake_probs = real_probability[y_true == 0]

print("\n" + "=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)

print("\nREAL IMAGES")
print(
    f"Average REAL probability : "
    f"{np.mean(real_probs) * 100:.2f}%"
)

print(
    f"Minimum REAL probability : "
    f"{np.min(real_probs) * 100:.2f}%"
)

print(
    f"Maximum REAL probability : "
    f"{np.max(real_probs) * 100:.2f}%"
)


print("\nFAKE IMAGES")
print(
    f"Average REAL probability : "
    f"{np.mean(fake_probs) * 100:.2f}%"
)

print(
    f"Minimum REAL probability : "
    f"{np.min(fake_probs) * 100:.2f}%"
)

print(
    f"Maximum REAL probability : "
    f"{np.max(fake_probs) * 100:.2f}%"
)


# =========================================================
# SAMPLE PREDICTIONS
# =========================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(min(10, len(X))):

    actual = "REAL" if y_true[i] == 1 else "FAKE"
    predicted = (
        "REAL"
        if predicted_labels[i] == 1
        else "FAKE"
    )

    print(
        f"{names[i]:30s} "
        f"ACTUAL={actual:5s} "
        f"PREDICTED={predicted:5s} "
        f"REAL={real_probability[i]*100:.2f}%"
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)