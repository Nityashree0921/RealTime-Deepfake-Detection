import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_PATH = "models/deepfake_face_model_v2.keras"

REAL_DIR = "face_frames/real"
FAKE_DIR = "face_frames/fake"

IMG_SIZE = 224

print("=" * 60)
print("EVALUATING FACE MODEL V2")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")
print("Input:", model.input_shape)
print("Output:", model.output_shape)

images = []
labels = []


def load_dataset(folder, label):

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(f"{folder}: {len(files)} images")

    for filename in files:

        path = os.path.join(folder, filename)

        img = cv2.imread(path)

        if img is None:
            print("Could not read:", path)
            continue

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        img = img.astype(np.float32)

        images.append(img)
        labels.append(label)


# REAL = 0
# FAKE = 1

load_dataset(REAL_DIR, 0)
load_dataset(FAKE_DIR, 1)


X = np.array(images)
y_true = np.array(labels)

print()
print("Dataset:", X.shape)
print("REAL:", np.sum(y_true == 0))
print("FAKE:", np.sum(y_true == 1))


# =========================================================
# PREDICTION
# =========================================================

print()
print("=" * 60)
print("RUNNING PREDICTIONS")
print("=" * 60)

predictions = model.predict(
    X,
    batch_size=16,
    verbose=1
).flatten()


# =========================================================
# TEST MULTIPLE THRESHOLDS
# =========================================================

print()
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

for threshold in [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]:

    y_pred = (
        predictions >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={accuracy:.3f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )


# =========================================================
# DEFAULT 0.5
# =========================================================

threshold = 0.5

y_pred = (
    predictions >= threshold
).astype(int)


print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )
)


# =========================================================
# RAW PREDICTION DISTRIBUTION
# =========================================================

real_predictions = predictions[
    y_true == 0
]

fake_predictions = predictions[
    y_true == 1
]


print()
print("=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print()
print("REAL images:")
print(
    "min:",
    np.min(real_predictions)
)

print(
    "max:",
    np.max(real_predictions)
)

print(
    "mean:",
    np.mean(real_predictions)
)

print(
    "median:",
    np.median(real_predictions)
)


print()
print("FAKE images:")
print(
    "min:",
    np.min(fake_predictions)
)

print(
    "max:",
    np.max(fake_predictions)
)

print(
    "mean:",
    np.mean(fake_predictions)
)

print(
    "median:",
    np.median(fake_predictions)
)