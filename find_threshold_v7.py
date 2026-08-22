import os
import json
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

MODEL_PATH = "models/deepfake_face_model_v7.keras"
VAL_DIR = "face_dataset_v7/val"
OUTPUT_JSON = "models/v7_threshold.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("=" * 70)
print("V7 OPERATING THRESHOLD CALIBRATION (VALIDATION DATA ONLY)")
print("=" * 70)

# Load Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded V7 Model:", MODEL_PATH)

# Load Validation Set
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Validation class names: {val_ds.class_names}")

y_true = []
y_prob = []

for imgs, labels in val_ds:
    preds = model.predict(imgs, verbose=0).flatten()
    y_true.extend(labels.numpy().astype(int))
    y_prob.extend(preds)

y_true = np.array(y_true).flatten()
y_prob = np.array(y_prob).flatten()

total_samples = len(y_true)
total_real = int(np.sum(y_true == 1))
total_fake = int(np.sum(y_true == 0))

print(f"\nValidation Samples: {total_samples} (REAL={total_real}, FAKE={total_fake})")

# =========================================================
# THRESHOLD SWEEP (0.10 to 0.90)
# =========================================================

thresholds = np.arange(0.10, 0.91, 0.05)
sweep_results = []

print("\n" + "-" * 75)
print(f"{'Thresh':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<12} | {'Bal. Acc':<10} | {'F1 Score':<10}")
print("-" * 75)

best_threshold = 0.50
best_balanced_acc = -1.0
best_metrics = {}

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity = TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Balanced Accuracy = (Sensitivity + Specificity) / 2
    bal_acc = (rec + spec) / 2.0
    
    record = {
        "threshold": round(float(t), 2),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }
    sweep_results.append(record)
    
    print(f"{t:<8.2f} | {acc*100:<9.2f}% | {prec*100:<9.2f}% | {rec*100:<9.2f}% | {spec*100:<11.2f}% | {bal_acc*100:<9.2f}% | {f1:<10.4f}")
    
    # Selection criteria: highest balanced accuracy, tie-break on F1
    if bal_acc > best_balanced_acc or (bal_acc == best_balanced_acc and f1 > best_metrics.get("f1", 0)):
        best_balanced_acc = bal_acc
        best_threshold = round(float(t), 2)
        best_metrics = record

print("-" * 75)
print(f"\nOptimal Operating Threshold Selected: {best_threshold:.2f}")
print(f"Validation Metrics at Threshold {best_threshold:.2f}:")
print(f"  Accuracy          : {best_metrics['accuracy']*100:.2f}%")
print(f"  Precision (REAL)  : {best_metrics['precision']*100:.2f}%")
print(f"  Recall (REAL)     : {best_metrics['recall']*100:.2f}%")
print(f"  Specificity (FAKE): {best_metrics['specificity']*100:.2f}%")
print(f"  Balanced Accuracy : {best_metrics['balanced_accuracy']*100:.2f}%")
print(f"  F1 Score          : {best_metrics['f1']:.4f}")

# Save JSON configuration
threshold_config = {
    "model_path": MODEL_PATH,
    "validation_directory": VAL_DIR,
    "optimal_threshold": best_threshold,
    "selection_criterion": "highest_balanced_accuracy_on_validation",
    "validation_metrics": best_metrics,
    "full_sweep_table": sweep_results
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(threshold_config, f, indent=4)

print(f"\nOperating threshold configuration saved to: {OUTPUT_JSON}")
print("=" * 70)
