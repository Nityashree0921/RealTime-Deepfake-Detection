import os
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
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v6.keras"
VAL_DIR = "face_dataset_v6/val"
TEST_DIR = "face_dataset_v6/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

REPORT_FILE = "v6_evaluation_report.txt"
THRESHOLD_FILE = "models/face_threshold_v6.txt"

print("=" * 70)
print("EVALUATING FACE DEEPFAKE MODEL V6")
print("=" * 70)

# =========================================================
# 1. LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully from:", MODEL_PATH)

# =========================================================
# 2. LOAD DATASETS
# =========================================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

def extract_labels_and_predictions(dataset):
    y_true = []
    y_prob = []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0).flatten()
        y_true.extend(labels.numpy().astype(int))
        y_prob.extend(preds)
    return np.array(y_true).flatten(), np.array(y_prob).flatten()

print("\nExtracting validation predictions for threshold optimization...")
val_true, val_prob = extract_labels_and_predictions(val_ds)

print("Extracting unseen test set predictions...")
test_true, test_prob = extract_labels_and_predictions(test_ds)

# =========================================================
# 3. THRESHOLD OPTIMIZATION USING VALIDATION DATA ONLY
# =========================================================

print("\n" + "=" * 70)
print("VALIDATION SET THRESHOLD SEARCH (P(REAL) >= Threshold -> REAL)")
print("=" * 70)

candidate_thresholds = np.arange(0.20, 0.75, 0.05)
best_val_threshold = 0.50
best_val_f1 = -1.0
val_threshold_records = []

for t in candidate_thresholds:
    val_pred = (val_prob >= t).astype(int)
    acc = accuracy_score(val_true, val_pred)
    prec = precision_score(val_true, val_pred, zero_division=0)
    rec = recall_score(val_true, val_pred, zero_division=0)
    f1 = f1_score(val_true, val_pred, zero_division=0)
    
    val_threshold_records.append((t, acc, prec, rec, f1))
    print(f"  Threshold {t:0.2f} | Acc: {acc*100:5.2f}% | Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | F1: {f1:0.4f}")
    
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_val_threshold = t

# If tie or extreme, pick balanced threshold around max F1
print(f"\nSelected Validation Threshold: {best_val_threshold:0.2f} (Val F1: {best_val_f1:0.4f})")

with open(THRESHOLD_FILE, "w") as f:
    f.write(f"{best_val_threshold:0.2f}\n")
print(f"Saved threshold to: {THRESHOLD_FILE}")

# =========================================================
# 4. FINAL EVALUATION ON UNSEEN TEST SET
# =========================================================

print("\n" + "=" * 70)
print(f"FINAL UNSEEN TEST EVALUATION (Threshold = {best_val_threshold:0.2f})")
print("=" * 70)

test_pred = (test_prob >= best_val_threshold).astype(int)

test_acc = accuracy_score(test_true, test_pred)
test_prec_macro = precision_score(test_true, test_pred, average="macro", zero_division=0)
test_rec_macro = recall_score(test_true, test_pred, average="macro", zero_division=0)
test_f1_macro = f1_score(test_true, test_pred, average="macro", zero_division=0)

test_prec_fake = precision_score(test_true, test_pred, pos_label=0, zero_division=0)
test_rec_fake = recall_score(test_true, test_pred, pos_label=0, zero_division=0)
test_f1_fake = f1_score(test_true, test_pred, pos_label=0, zero_division=0)

test_prec_real = precision_score(test_true, test_pred, pos_label=1, zero_division=0)
test_rec_real = recall_score(test_true, test_pred, pos_label=1, zero_division=0)
test_f1_real = f1_score(test_true, test_pred, pos_label=1, zero_division=0)

test_auc = roc_auc_score(test_true, test_prob)

# Confusion Matrix:
# rows = ACTUAL [FAKE (0), REAL (1)]
# cols = PREDICTED [FAKE (0), REAL (1)]
cm = confusion_matrix(test_true, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Breakdown
fake_correct = tn
fake_as_real = fp
real_correct = tp
real_as_fake = fn

total_fake = tn + fp
total_real = fn + tp

print(f"Test Accuracy : {test_acc * 100:0.2f}%")
print(f"Test ROC-AUC  : {test_auc:0.4f}")
print(f"\nREAL Detection Metrics (Class 1):")
print(f"  Precision : {test_prec_real * 100:0.2f}%")
print(f"  Recall    : {test_rec_real * 100:0.2f}%")
print(f"  F1 Score  : {test_f1_real:0.4f}")

print(f"\nFAKE Detection Metrics (Class 0):")
print(f"  Precision : {test_prec_fake * 100:0.2f}%")
print(f"  Recall    : {test_rec_fake * 100:0.2f}%")
print(f"  F1 Score  : {test_f1_fake:0.4f}")

print("\n" + "=" * 70)
print("CONFUSION MATRIX:")
print("=" * 70)
print("                 PREDICTED")
print("                 FAKE    REAL")
print(f"ACTUAL FAKE      {fake_correct:4d}    {fake_as_real:4d}   (Total: {total_fake})")
print(f"ACTUAL REAL      {real_as_fake:4d}    {real_correct:4d}   (Total: {total_real})")

print(f"\nREAL Correctly Detected         : {real_correct}/{total_real} ({real_correct/total_real*100:0.2f}%)")
print(f"REAL Incorrectly classified FAKE: {real_as_fake}/{total_real} ({real_as_fake/total_real*100:0.2f}%)")
print(f"FAKE Correctly Detected         : {fake_correct}/{total_fake} ({fake_correct/total_fake*100:0.2f}%)")
print(f"FAKE Incorrectly classified REAL: {fake_as_real}/{total_fake} ({fake_as_real/total_fake*100:0.2f}%)")

# =========================================================
# 5. GENERATE PLOTS
# =========================================================

# Plot 1: Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'V6 Confusion Matrix (Threshold={best_val_threshold:0.2f})', fontsize=12, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['FAKE', 'REAL'], fontsize=11)
plt.yticks(tick_marks, ['FAKE', 'REAL'], fontsize=11)
plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
plt.ylabel('Actual Label', fontsize=11, fontweight='bold')

# Text annotations in cells
thresh_color = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh_color else "black",
                 fontsize=14, fontweight='bold')

plt.tight_layout()
cm_plot_path = "v6_confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"\nGenerated: {cm_plot_path}")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(test_true, test_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'V6 ROC curve (AUC = {test_auc:0.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
plt.title('V6 Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
roc_plot_path = "v6_roc_curve.png"
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"Generated: {roc_plot_path}")

# Plot 3: Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(test_true, test_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, color='green', lw=2.5, label='V6 Precision-Recall curve')
plt.xlabel('Recall (True Positive Rate)', fontsize=11, fontweight='bold')
plt.ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
plt.title('V6 Precision-Recall Curve', fontsize=12, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
pr_plot_path = "v6_precision_recall_curve.png"
plt.savefig(pr_plot_path, dpi=300)
plt.close()
print(f"Generated: {pr_plot_path}")

# =========================================================
# 6. WRITE EVALUATION TEXT REPORT
# =========================================================

report_content = f"""================================================================================
V6 MODEL EVALUATION REPORT — REAL-TIME DEEPFAKE DETECTION
================================================================================
Model File                : {MODEL_PATH}
Test Dataset Directory    : {TEST_DIR}
Total Unseen Test Frames  : {len(test_true)} (REAL={total_real}, FAKE={total_fake})
Validation-Selected Thresh: {best_val_threshold:0.2f} (Selected on {VAL_DIR} without test leakage)

OVERALL TEST METRICS:
--------------------------------------------------------------------------------
Accuracy                  : {test_acc * 100:0.2f}%
ROC-AUC                   : {test_auc:0.4f}
Macro Precision           : {test_prec_macro * 100:0.2f}%
Macro Recall              : {test_rec_macro * 100:0.2f}%
Macro F1 Score            : {test_f1_macro:0.4f}

DETAILED CLASS BREAKDOWN:
--------------------------------------------------------------------------------
REAL Faces (Class 1):
  Precision               : {test_prec_real * 100:0.2f}%
  Recall                  : {test_rec_real * 100:0.2f}%
  F1 Score                : {test_f1_real:0.4f}
  Correctly Detected      : {real_correct} / {total_real} ({real_correct/total_real*100:0.2f}%)
  Misclassified as FAKE   : {real_as_fake} / {total_real} ({real_as_fake/total_real*100:0.2f}%)

FAKE Faces (Class 0):
  Precision               : {test_prec_fake * 100:0.2f}%
  Recall                  : {test_rec_fake * 100:0.2f}%
  F1 Score                : {test_f1_fake:0.4f}
  Correctly Detected      : {fake_correct} / {total_fake} ({fake_correct/total_fake*100:0.2f}%)
  Misclassified as REAL   : {fake_as_real} / {total_fake} ({fake_as_real/total_fake*100:0.2f}%)

CONFUSION MATRIX:
--------------------------------------------------------------------------------
                 PREDICTED
                 FAKE    REAL
ACTUAL FAKE      {fake_correct:4d}    {fake_as_real:4d}
ACTUAL REAL      {real_as_fake:4d}    {real_correct:4d}

PREDICTION PROBABILITY DISTRIBUTION (TEST SET):
--------------------------------------------------------------------------------
REAL Images (P(REAL)):
  Mean P(REAL)            : {np.mean(test_prob[test_true == 1])*100:0.2f}%
  Median P(REAL)          : {np.median(test_prob[test_true == 1])*100:0.2f}%
  Min / Max P(REAL)       : {np.min(test_prob[test_true == 1])*100:0.2f}% / {np.max(test_prob[test_true == 1])*100:0.2f}%

FAKE Images (P(REAL)):
  Mean P(REAL)            : {np.mean(test_prob[test_true == 0])*100:0.2f}%
  Median P(REAL)          : {np.median(test_prob[test_true == 0])*100:0.2f}%
  Min / Max P(REAL)       : {np.min(test_prob[test_true == 0])*100:0.2f}% / {np.max(test_prob[test_true == 0])*100:0.2f}%

VALIDATION THRESHOLD SWEEP TABLE:
--------------------------------------------------------------------------------
Threshold | Accuracy | Precision | Recall  | F1 Score
"""

for t, acc, prec, rec, f1 in val_threshold_records:
    report_content += f"{t:9.2f} | {acc*100:7.2f}% | {prec*100:8.2f}% | {rec*100:6.2f}% | {f1:0.4f}\n"

report_content += "================================================================================\n"

with open(REPORT_FILE, "w") as f:
    f.write(report_content)

print(f"\nSaved comprehensive text report: {REPORT_FILE}")
print("=" * 70)
print("V6 EVALUATION COMPLETE")
print("=" * 70)
