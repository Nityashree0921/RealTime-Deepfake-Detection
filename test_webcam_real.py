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
BATCH_SIZE = 16

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 70)
print("DEEPFAKE MODEL - REAL + FAKE EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print("\nModel input :", model.input_shape)
print("Model output:", model.output_shape)

# =========================================================
# LOAD IMAGES
# =========================================================

def load_images(folder, label):

    images = []
    labels = []
    filenames = []

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(
        f"\n{folder}: {len(files)} images"
    )

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        img = cv2.imread(path)

        if img is None:
            print("Could not read:", filename)
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
        # Model v5 already contains:
        # Rescaling(scale=1/127.5, offset=-1)
        #
        # Therefore we give the model
        # normal 0-255 float32 images.

        img = img.astype(
            np.float32
        )

        images.append(img)
        labels.append(label)
        filenames.append(filename)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=int),
        filenames
    )


# =========================================================
# LOAD REAL
# =========================================================

print("\n" + "=" * 70)
print("LOADING REAL IMAGES")
print("=" * 70)

X_real, y_real, real_files = load_images(
    REAL_DIR,
    0
)

# =========================================================
# LOAD FAKE
# =========================================================

print("\n" + "=" * 70)
print("LOADING FAKE IMAGES")
print("=" * 70)

X_fake, y_fake, fake_files = load_images(
    FAKE_DIR,
    1
)

# =========================================================
# COMBINE
# =========================================================

X = np.concatenate(
    [X_real, X_fake],
    axis=0
)

y = np.concatenate(
    [y_real, y_fake],
    axis=0
)

filenames = (
    real_files +
    fake_files
)

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nREAL images : {len(X_real)}")
print(f"FAKE images : {len(X_fake)}")
print(f"TOTAL       : {len(X)}")

print("\nInput shape:", X.shape)

print(
    "Input minimum:",
    X.min()
)

print(
    "Input maximum:",
    X.max()
)

# =========================================================
# PREDICTIONS
# =========================================================

print("\n" + "=" * 70)
print("RUNNING MODEL PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    X,
    batch_size=BATCH_SIZE,
    verbose=1
).flatten()

# =========================================================
# BASIC PROBABILITY ANALYSIS
# =========================================================

real_predictions = predictions[y == 0]
fake_predictions = predictions[y == 1]

print("\n" + "=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)

print("\nREAL IMAGES")
print(
    f"Average FAKE probability : "
    f"{np.mean(real_predictions) * 100:.2f}%"
)

print(
    f"Minimum FAKE probability : "
    f"{np.min(real_predictions) * 100:.2f}%"
)

print(
    f"Maximum FAKE probability : "
    f"{np.max(real_predictions) * 100:.2f}%"
)

print(
    f"Median FAKE probability  : "
    f"{np.median(real_predictions) * 100:.2f}%"
)

print("\nFAKE IMAGES")

print(
    f"Average FAKE probability : "
    f"{np.mean(fake_predictions) * 100:.2f}%"
)

print(
    f"Minimum FAKE probability : "
    f"{np.min(fake_predictions) * 100:.2f}%"
)

print(
    f"Maximum FAKE probability : "
    f"{np.max(fake_predictions) * 100:.2f}%"
)

print(
    f"Median FAKE probability  : "
    f"{np.median(fake_predictions) * 100:.2f}%"
)

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    "\nThreshold | Accuracy | Precision | Recall | F1 | "
    "REAL->FAKE | FAKE->REAL"
)

print("-" * 85)

results = []

for threshold in THRESHOLDS:

    predicted = (
        predictions >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predicted
    )

    precision = precision_score(
        y,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        y,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predicted,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y,
        predicted,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    real_to_fake = (
        fp / (tn + fp) * 100
        if (tn + fp) > 0
        else 0
    )

    fake_to_real = (
        fn / (fn + tp) * 100
        if (fn + tp) > 0
        else 0
    )

    print(
        f"{threshold:9.2f} | "
        f"{accuracy * 100:8.2f}% | "
        f"{precision * 100:9.2f}% | "
        f"{recall * 100:6.2f}% | "
        f"{f1 * 100:6.2f}% | "
        f"{real_to_fake:9.2f}% | "
        f"{fake_to_real:9.2f}%"
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

# =========================================================
# BEST THRESHOLD
# =========================================================

best = max(
    results,
    key=lambda x: x["f1"]
)

print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"\nBest threshold by F1: "
    f"{best['threshold']:.2f}"
)

print(
    f"Accuracy : "
    f"{best['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best['precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{best['recall'] * 100:.2f}%"
)

print(
    f"F1 Score : "
    f"{best['f1'] * 100:.2f}%"
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\n                  PREDICTED"
)

print(
    "                REAL    FAKE"
)

print(
    f"ACTUAL REAL    "
    f"{best['tn']:5d}   "
    f"{best['fp']:5d}"
)

print(
    f"ACTUAL FAKE    "
    f"{best['fn']:5d}   "
    f"{best['tp']:5d}"
)

# =========================================================
# FINAL CLASS COUNTS
# =========================================================

predicted_best = (
    predictions >= best["threshold"]
).astype(int)

real_detected = np.sum(
    (y == 0) &
    (predicted_best == 0)
)

real_false_fake = np.sum(
    (y == 0) &
    (predicted_best == 1)
)

fake_detected = np.sum(
    (y == 1) &
    (predicted_best == 1)
)

fake_missed = np.sum(
    (y == 1) &
    (predicted_best == 0)
)

print("\n" + "=" * 70)
print("FINAL DETECTION RESULTS")
print("=" * 70)

print(
    f"\nREAL correctly detected : "
    f"{real_detected}/{len(X_real)}"
)

print(
    f"REAL incorrectly FAKE   : "
    f"{real_false_fake}/{len(X_real)}"
)

print(
    f"\nFAKE correctly detected : "
    f"{fake_detected}/{len(X_fake)}"
)

print(
    f"FAKE incorrectly REAL   : "
    f"{fake_missed}/{len(X_fake)}"
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)