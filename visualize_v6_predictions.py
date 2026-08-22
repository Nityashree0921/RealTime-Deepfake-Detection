import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v6.keras"
THRESHOLD_FILE = "models/face_threshold_v6.txt"
TEST_DIR = "face_dataset_v6/test"
OUTPUT_IMAGE = "v6_prediction_examples.jpg"

IMG_SIZE = 224

print("=" * 70)
print("VISUALIZING V6 MODEL PREDICTIONS")
print("=" * 70)

# Load Threshold
threshold = 0.50
if os.path.exists(THRESHOLD_FILE):
    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())

print(f"Using Threshold: {threshold:.2f}")

# Load Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def load_and_predict_folder(folder_path, actual_label_str, actual_label_idx):
    records = []
    if not os.path.exists(folder_path):
        return records
    
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for f in files:
        p = os.path.join(folder_path, f)
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        
        # Batch dimension, float32, no duplicate division
        inp = np.expand_dims(img_resized.astype(np.float32), axis=0)
        p_real = float(model.predict(inp, verbose=0)[0][0])
        p_fake = 1.0 - p_real
        
        pred_label_idx = 1 if p_real >= threshold else 0
        pred_label_str = "REAL" if pred_label_idx == 1 else "FAKE"
        is_correct = (pred_label_idx == actual_label_idx)
        
        records.append({
            "filename": f,
            "path": p,
            "image": img_rgb,
            "actual_label": actual_label_str,
            "pred_label": pred_label_str,
            "p_real": p_real,
            "p_fake": p_fake,
            "is_correct": is_correct
        })
    return records

print("Scanning test set for evaluation cases...")
real_test_records = load_and_predict_folder(os.path.join(TEST_DIR, "real"), "REAL", 1)
fake_test_records = load_and_predict_folder(os.path.join(TEST_DIR, "fake"), "FAKE", 0)

all_records = real_test_records + fake_test_records

# If some category is empty in test split, also check val split for representative visualization
if not any(r["actual_label"] == "REAL" and not r["is_correct"] for r in all_records):
    val_real_records = load_and_predict_folder("face_dataset_v6/val/real", "REAL", 1)
    all_records.extend(val_real_records)

if not any(r["actual_label"] == "FAKE" and r["is_correct"] for r in all_records):
    val_fake_records = load_and_predict_folder("face_dataset_v6/val/fake", "FAKE", 0)
    all_records.extend(val_fake_records)

# Find 4 target categories:
# 1. Correct REAL
# 2. Incorrect REAL (False FAKE)
# 3. Correct FAKE
# 4. Incorrect FAKE (False REAL)

correct_real = next((r for r in all_records if r["actual_label"] == "REAL" and r["is_correct"]), None)
incorrect_real = next((r for r in all_records if r["actual_label"] == "REAL" and not r["is_correct"]), None)
correct_fake = next((r for r in all_records if r["actual_label"] == "FAKE" and r["is_correct"]), None)
incorrect_fake = next((r for r in all_records if r["actual_label"] == "FAKE" and not r["is_correct"]), None)

samples = [
    ("Correct REAL (True Real)", correct_real),
    ("Incorrect REAL (False Fake)", incorrect_real),
    ("Correct FAKE (True Fake)", correct_fake),
    ("Incorrect FAKE (False Real)", incorrect_fake)
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, (title, sample) in enumerate(samples):
    ax = axes[i]
    if sample is not None:
        ax.imshow(sample["image"])
        status_color = "darkgreen" if sample["is_correct"] else "crimson"
        ax.set_title(
            f"{title}\n"
            f"File: {sample['filename']}\n"
            f"Actual: {sample['actual_label']} | Pred: {sample['pred_label']}\n"
            f"P(REAL): {sample['p_real']*100:.1f}% | P(FAKE): {sample['p_fake']*100:.1f}%",
            fontsize=10,
            fontweight="bold",
            color=status_color
        )
    else:
        ax.text(0.5, 0.5, f"No sample found for:\n{title}", ha="center", va="center", fontsize=12)
    ax.axis("off")

plt.suptitle(f"V6 Deepfake Model Prediction Examples (Threshold = {threshold:.2f})", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Prediction examples visualization saved to: {OUTPUT_IMAGE}")
print("=" * 70)
