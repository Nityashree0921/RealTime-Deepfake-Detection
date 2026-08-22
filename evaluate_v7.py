import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v7.keras"
THRESHOLD_JSON = "models/v7_threshold.json"
TEST_DIR = "face_dataset_v7/test"
REPORTS_DIR = "reports"
REPORT_TXT = os.path.join(REPORTS_DIR, "v7_evaluation_report.txt")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("EVALUATING V7 MODEL ON UNSEEN TEST SPLIT")
print("=" * 70)

# =========================================================
# 1. LOAD MODEL & VALIDATION-CALIBRATED THRESHOLD
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded V7 Model:", MODEL_PATH)

threshold = 0.50
if os.path.exists(THRESHOLD_JSON):
    with open(THRESHOLD_JSON, "r") as f:
        config = json.load(f)
        threshold = float(config.get("optimal_threshold", 0.50))
    print(f"Loaded Frozen Operating Threshold from '{THRESHOLD_JSON}': {threshold:.2f}")
else:
    print(f"Threshold config not found. Defaulting to: {threshold:.2f}")

# =========================================================
# 2. LOAD UNSEEN TEST DATASET
# =========================================================

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test Class Mapping: {test_ds.class_names} (0 = FAKE, 1 = REAL)")

y_true = []
y_prob = []

for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0).flatten()
    y_true.extend(labels.numpy().astype(int))
    y_prob.extend(preds)

y_true = np.array(y_true).flatten()
y_prob = np.array(y_prob).flatten()

total_samples = len(y_true)
total_real = int(np.sum(y_true == 1))
total_fake = int(np.sum(y_true == 0))

print(f"Total Unseen Test Frames: {total_samples} (REAL={total_real}, FAKE={total_fake})")

# =========================================================
# 3. COMPUTE METRICS AT FROZEN THRESHOLD
# =========================================================

y_pred = (y_prob >= threshold).astype(int)

acc = accuracy_score(y_true, y_pred)
prec_real = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
rec_real = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1_real = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

prec_fake = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
rec_fake = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
f1_fake = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
auc = roc_auc_score(y_true, y_prob)

# Confusion Matrix:
# rows = ACTUAL [FAKE (0), REAL (1)]
# cols = PREDICTED [FAKE (0), REAL (1)]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Sensitivity & Specificity
sensitivity = rec_real  # TP / (TP + FN)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TN / (TN + FP)
fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0          # FP / (TN + FP)
fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0          # FN / (TP + FN)

# Probability distribution statistics
real_probs = y_prob[y_true == 1]
fake_probs = y_prob[y_true == 0]

print("\n" + "=" * 70)
print("FINAL TEST METRICS (FROZEN VALIDATION THRESHOLD = " + f"{threshold:.2f})")
print("=" * 70)
print(f"Accuracy                  : {acc * 100:.2f}%")
print(f"ROC-AUC                   : {auc:.4f}")
print(f"Sensitivity (Recall REAL) : {sensitivity * 100:.2f}%")
print(f"Specificity (Recall FAKE) : {specificity * 100:.2f}%")
print(f"False Positive Rate (FPR) : {fpr * 100:.2f}%")
print(f"False Negative Rate (FNR) : {fnr * 100:.2f}%")
print(f"Macro F1 Score            : {macro_f1:.4f}")

print("\nREAL Faces Breakdown:")
print(f"  Precision : {prec_real * 100:.2f}%")
print(f"  Recall    : {rec_real * 100:.2f}%")
print(f"  F1 Score  : {f1_real:.4f}")
print(f"  Correctly Detected: {tp} / {total_real} ({tp/total_real*100:.2f}%)")

print("\nFAKE Faces Breakdown:")
print(f"  Precision : {prec_fake * 100:.2f}%")
print(f"  Recall    : {rec_fake * 100:.2f}%")
print(f"  F1 Score  : {f1_fake:.4f}")
print(f"  Correctly Detected: {tn} / {total_fake} ({tn/total_fake*100:.2f}%)")

print("\n" + "=" * 70)
print("CONFUSION MATRIX:")
print("=" * 70)
print("                 PREDICTED")
print("                 FAKE    REAL")
print(f"ACTUAL FAKE      {tn:4d}    {fp:4d}   (Total: {total_fake})")
print(f"ACTUAL REAL      {fn:4d}    {tp:4d}   (Total: {total_real})")

print("\n" + "=" * 70)
print("PROBABILITY DISTRIBUTIONS P(REAL):")
print("=" * 70)
print("REAL Images (P(REAL)):")
print(f"  Mean = {np.mean(real_probs)*100:.2f}% | Median = {np.median(real_probs)*100:.2f}% | Std = {np.std(real_probs)*100:.2f}% | Range = [{np.min(real_probs)*100:.2f}%, {np.max(real_probs)*100:.2f}%]")
print("FAKE Images (P(REAL)):")
print(f"  Mean = {np.mean(fake_probs)*100:.2f}% | Median = {np.median(fake_probs)*100:.2f}% | Std = {np.std(fake_probs)*100:.2f}% | Range = [{np.min(fake_probs)*100:.2f}%, {np.max(fake_probs)*100:.2f}%]")

# =========================================================
# 4. GENERATE PLOTS
# =========================================================

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'V7 Confusion Matrix (Threshold={threshold:.2f})', fontsize=12, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['FAKE', 'REAL'], fontsize=11)
plt.yticks(tick_marks, ['FAKE', 'REAL'], fontsize=11)
plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
plt.ylabel('Actual Label', fontsize=11, fontweight='bold')

thresh_color = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh_color else "black",
                 fontsize=14, fontweight='bold')

plt.tight_layout()
cm_plot_path = os.path.join(REPORTS_DIR, "v7_confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"\nGenerated: {cm_plot_path}")

# ROC Curve Plot
fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2.5, label=f'V7 ROC Curve (AUC = {auc:0.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
plt.title('V7 Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
roc_plot_path = os.path.join(REPORTS_DIR, "v7_roc_curve.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"Generated: {roc_plot_path}")

# Save text report
report_text = f"""================================================================================
V7 UNSEEN TEST EVALUATION REPORT — REAL-TIME DEEPFAKE DETECTION
================================================================================
Model Path                : {MODEL_PATH}
Test Directory            : {TEST_DIR}
Frozen Operating Threshold: {threshold:.2f} (Calibrated on validation split)
Total Unseen Test Frames  : {total_samples} (REAL={total_real}, FAKE={total_fake})

OVERALL TEST METRICS:
--------------------------------------------------------------------------------
Accuracy                  : {acc * 100:.2f}%
ROC-AUC                   : {auc:.4f}
Sensitivity (Recall REAL) : {sensitivity * 100:.2f}%
Specificity (Recall FAKE) : {specificity * 100:.2f}%
False Positive Rate (FPR) : {fpr * 100:.2f}%
False Negative Rate (FNR) : {fnr * 100:.2f}%
Macro F1 Score            : {macro_f1:.4f}

CLASS BREAKDOWN:
--------------------------------------------------------------------------------
REAL Faces (Class 1):
  Precision               : {prec_real * 100:.2f}%
  Recall                  : {rec_real * 100:.2f}%
  F1 Score                : {f1_real:.4f}
  Correctly Classified    : {tp} / {total_real} ({tp/total_real*100:.2f}%)
  Misclassified as FAKE   : {fn} / {total_real} ({fn/total_real*100:.2f}%)

FAKE Faces (Class 0):
  Precision               : {prec_fake * 100:.2f}%
  Recall                  : {rec_fake * 100:.2f}%
  F1 Score                : {f1_fake:.4f}
  Correctly Classified    : {tn} / {total_fake} ({tn/total_fake*100:.2f}%)
  Misclassified as REAL   : {fp} / {total_fake} ({fp/total_fake*100:.2f}%)

CONFUSION MATRIX:
--------------------------------------------------------------------------------
                 PREDICTED
                 FAKE    REAL
ACTUAL FAKE      {tn:4d}    {fp:4d}
ACTUAL REAL      {fn:4d}    {tp:4d}

PROBABILITY DISTRIBUTIONS:
--------------------------------------------------------------------------------
REAL Images (P(REAL)):
  Mean = {np.mean(real_probs)*100:.2f}% | Median = {np.median(real_probs)*100:.2f}% | Std = {np.std(real_probs)*100:.2f}%
  Min = {np.min(real_probs)*100:.2f}% | Max = {np.max(real_probs)*100:.2f}%

FAKE Images (P(REAL)):
  Mean = {np.mean(fake_probs)*100:.2f}% | Median = {np.median(fake_probs)*100:.2f}% | Std = {np.std(fake_probs)*100:.2f}%
  Min = {np.min(fake_probs)*100:.2f}% | Max = {np.max(fake_probs)*100:.2f}%
================================================================================
"""

with open(REPORT_TXT, "w") as f:
    f.write(report_text)

print(f"Saved evaluation report to: {REPORT_TXT}")
print("=" * 70)
