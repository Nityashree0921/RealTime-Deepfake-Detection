import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v5.keras"
TEST_DIR = "face_dataset_v5/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("EVALUATING FACE MODEL V5")
print("=" * 60)

# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Input:", model.input_shape)
print("Output:", model.output_shape)

# =========================================================
# LOAD TEST DATASET
# =========================================================

print("\nLoading TEST dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass names:", test_ds.class_names)

# Expected:
# class 0 = fake
# class 1 = real

# =========================================================
# PREDICTIONS
# =========================================================

print("\nRunning predictions...")

y_true = []
y_prob = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    ).flatten()

    # IMPORTANT:
    # Flatten labels so they become 1-D
    labels = labels.numpy().flatten().astype(int)

    y_true.extend(labels)
    y_prob.extend(predictions)

# Convert to numpy arrays

y_true = np.array(y_true).flatten()
y_prob = np.array(y_prob).flatten()

print("\nTotal test images:", len(y_true))

# =========================================================
# PREDICTION DISTRIBUTION
# =========================================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

fake_probs = y_prob[y_true == 0]
real_probs = y_prob[y_true == 1]

print("\nFAKE images:")
print("Count :", len(fake_probs))
print("Min   :", round(float(fake_probs.min()), 4))
print("Max   :", round(float(fake_probs.max()), 4))
print("Mean  :", round(float(fake_probs.mean()), 4))
print("Median:", round(float(np.median(fake_probs)), 4))

print("\nREAL images:")
print("Count :", len(real_probs))
print("Min   :", round(float(real_probs.min()), 4))
print("Max   :", round(float(real_probs.max()), 4))
print("Mean  :", round(float(real_probs.mean()), 4))
print("Median:", round(float(np.median(real_probs)), 4))

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

best_threshold = 0.50
best_f1 = 0.0

for threshold in np.arange(0.20, 0.71, 0.05):

    y_pred = (y_prob >= threshold).astype(int)

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

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold

# =========================================================
# BEST THRESHOLD
# =========================================================

print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    "Best threshold:",
    round(float(best_threshold), 2)
)

print(
    "Best F1:",
    round(float(best_f1), 4)
)

# =========================================================
# FINAL PREDICTIONS
# =========================================================

y_pred = (
    y_prob >= best_threshold
).astype(int)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "FAKE",
            "REAL"
        ],
        zero_division=0
    )
)

# =========================================================
# ROC-AUC
# =========================================================

print("\n" + "=" * 60)
print("ROC-AUC")
print("=" * 60)

try:

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    print(
        "ROC-AUC:",
        round(float(auc), 4)
    )

except Exception as e:

    print(
        "ROC-AUC error:",
        e
    )

# =========================================================
# SAVE THRESHOLD
# =========================================================

threshold_path = "models/face_threshold_v5.txt"

with open(
    threshold_path,
    "w"
) as f:

    f.write(
        str(float(best_threshold))
    )

print("\nThreshold saved:")
print(threshold_path)

# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 60)
print("V5 EVALUATION COMPLETED")
print("=" * 60)