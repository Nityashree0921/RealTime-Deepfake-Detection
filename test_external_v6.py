import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================================================
# SETTINGS & ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(description="External Dataset Evaluation for V6 Model")
parser.add_argument("--data_dir", type=str, default="external_test", help="Path to external test directory with real/ and fake/ subfolders")
parser.add_argument("--model_path", type=str, default="models/deepfake_face_model_v6.keras", help="Path to trained V6 model")
parser.add_argument("--threshold_path", type=str, default="models/face_threshold_v6.txt", help="Path to threshold file")
args = parser.parse_args()

print("=" * 70)
print("V6 EXTERNAL DATASET EVALUATION")
print("=" * 70)

# =========================================================
# DIRECTORY & MODEL CHECK
# =========================================================

external_dir = args.data_dir
real_dir = os.path.join(external_dir, "real")
fake_dir = os.path.join(external_dir, "fake")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

threshold = 0.50
if os.path.exists(args.threshold_path):
    with open(args.threshold_path, "r") as f:
        threshold = float(f.read().strip())

if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model not found: {args.model_path}")

model = tf.keras.models.load_model(args.model_path)
print("V6 Model Loaded Successfully!")
print(f"Classification Threshold : {threshold:.2f}")
print(f"External Dataset Path    : {os.path.abspath(external_dir)}")

# Check for files
real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Discovered REAL images   : {len(real_files)}")
print(f"Discovered FAKE images   : {len(fake_files)}")

if len(real_files) == 0 and len(fake_files) == 0:
    print("\n[NOTE] The 'external_test/' directory is currently empty.")
    print("To test external data, please place external test images into:")
    print(f"  - REAL: {os.path.abspath(real_dir)}")
    print(f"  - FAKE: {os.path.abspath(fake_dir)}")
    print("Script setup is verified and ready for external data.")
    exit(0)

# =========================================================
# LOAD DATASET VIA KERAS
# =========================================================

dataset = tf.keras.utils.image_dataset_from_directory(
    external_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=(224, 224),
    batch_size=16,
    shuffle=False
)

y_true = []
y_prob = []

for imgs, labels in dataset:
    preds = model.predict(imgs, verbose=0).flatten()
    y_true.extend(labels.numpy().astype(int))
    y_prob.extend(preds)

y_true = np.array(y_true).flatten()
y_prob = np.array(y_prob).flatten()

y_pred = (y_prob >= threshold).astype(int)

# =========================================================
# COMPUTE METRICS
# =========================================================

accuracy = accuracy_score(y_true, y_pred)
unique_labels = np.unique(y_true)

has_both_classes = len(unique_labels) > 1

if has_both_classes:
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
else:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = float('nan')
    cm = None

print("\n" + "=" * 70)
print("EXTERNAL EVALUATION RESULTS")
print("=" * 70)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1:.4f}")
if has_both_classes:
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nCONFUSION MATRIX:")
    print("                 PREDICTED")
    print("                 FAKE    REAL")
    print(f"ACTUAL FAKE      {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"ACTUAL REAL      {cm[1][0]:4d}    {cm[1][1]:4d}")

# Probabilities
if len(real_files) > 0:
    real_probs = y_prob[y_true == 1]
    print(f"\nREAL Images - Average P(REAL): {np.mean(real_probs)*100:.2f}% (Min: {np.min(real_probs)*100:.2f}%, Max: {np.max(real_probs)*100:.2f}%)")

if len(fake_files) > 0:
    fake_probs = y_prob[y_true == 0]
    print(f"FAKE Images - Average P(REAL): {np.mean(fake_probs)*100:.2f}% (P(FAKE)={100-np.mean(fake_probs)*100:.2f}%)")

print("=" * 70)
