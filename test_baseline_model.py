import os
import cv2
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
# PATHS AND SETTINGS
# =========================================================
MODELS_TO_TEST = {
    "V6": {
        "model_path": "models/deepfake_face_model_v6.keras",
        "threshold": 0.35,
        "test_dir": "face_dataset_v6/test"
    },
    "V7": {
        "model_path": "models/deepfake_face_model_v7.keras",
        "threshold": 0.50, # default, will load if exists
        "test_dir": "face_dataset_v7/test"
    }
}
IMG_SIZE = 224

import json
if os.path.exists("models/v7_threshold.json"):
    try:
        with open("models/v7_threshold.json", "r") as f:
            config = json.load(f)
            MODELS_TO_TEST["V7"]["threshold"] = float(config.get("optimal_threshold", 0.50))
    except Exception:
        pass

for version, cfg in MODELS_TO_TEST.items():
    model_path = cfg["model_path"]
    threshold = cfg["threshold"]
    test_dir = cfg["test_dir"]
    
    if not os.path.exists(model_path):
        print(f"Model {version} at {model_path} not found. Skipping.")
        continue
        
    print("=" * 80)
    print(f"EVALUATING MODEL {version} (Path: {model_path})")
    print(f"Decision Threshold: {threshold:.2f}")
    print("=" * 80)

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.\n")

    # Helper for prediction
    def predict_image(filepath):
        img = cv2.imread(filepath)
        if img is None:
            return None, None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized.astype("float32"), axis=0)
        p_real = float(model.predict(img_array, verbose=0)[0][0])
        
        if p_real >= threshold:
            pred_label = "REAL"
            confidence = p_real * 100.0
        else:
            pred_label = "FAKE"
            confidence = (1.0 - p_real) * 100.0
            
        return p_real, pred_label, confidence

    # Test 5 Real and 5 Fake
    real_dir = os.path.join(test_dir, "real")
    fake_dir = os.path.join(test_dir, "fake")
    real_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:5]
    fake_files = sorted([f for f in os.listdir(fake_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:5]

    print("--- 5 SAMPLE REAL IMAGES ---")
    for f in real_files:
        path = os.path.join(real_dir, f)
        p_real, pred, conf = predict_image(path)
        if p_real is not None:
            print(f"{f} -> P(REAL): {p_real:.4f} ({p_real*100:.2f}%) | Pred: {pred} | Conf: {conf:.2f}%")

    print("\n--- 5 SAMPLE FAKE IMAGES ---")
    for f in fake_files:
        path = os.path.join(fake_dir, f)
        p_real, pred, conf = predict_image(path)
        if p_real is not None:
            print(f"{f} -> P(REAL): {p_real:.4f} ({p_real*100:.2f}%) | Pred: {pred} | Conf: {conf:.2f}%")

    # Full set
    all_images = []
    all_labels = []
    
    def load_all_from_dir(folder, label):
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for f in files:
            path = os.path.join(folder, f)
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            all_images.append(img_resized.astype("float32"))
            all_labels.append(label)

    load_all_from_dir(real_dir, 1)
    load_all_from_dir(fake_dir, 0)

    X = np.array(all_images)
    y_true = np.array(all_labels)

    p_reals = model.predict(X, batch_size=16, verbose=0).flatten()
    y_pred = (p_reals >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, p_reals)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\nFULL EVALUATION FOR {version}:")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("Confusion Matrix:")
    print("                 PREDICTED")
    print("                 FAKE    REAL")
    print(f"ACTUAL FAKE      {tn:4d}    {fp:4d}   (Total: {tn+fp})")
    print(f"ACTUAL REAL      {fn:4d}    {tp:4d}   (Total: {fn+tp})\n\n")

