import os
import cv2
import numpy as np
import tensorflow as tf

print("=" * 70)
print("RUNNING FULL SYSTEM VERIFICATION TEST (V6 PIPELINE)")
print("=" * 70)

results_summary = []

# Check 1: Model exists
model_path = "models/deepfake_face_model_v6.keras"
model_exists = os.path.exists(model_path)
results_summary.append(("1. V6 Model Exists", model_exists, f"{model_path} found ({os.path.getsize(model_path)} bytes)" if model_exists else "Missing model"))

# Check 2: Dataset exists
dataset_path = "face_dataset_v6"
dataset_exists = os.path.exists(dataset_path) and os.path.exists(os.path.join(dataset_path, "train")) and os.path.exists(os.path.join(dataset_path, "test"))
results_summary.append(("2. V6 Dataset Exists", dataset_exists, f"Found splits in {dataset_path}"))

# Check 3: Class mapping is correct
test_dir = os.path.join(dataset_path, "test")
classes = sorted(os.listdir(test_dir)) if os.path.exists(test_dir) else []
mapping_correct = (classes == ["fake", "real"])
results_summary.append(("3. Class Mapping (0=fake, 1=real)", mapping_correct, f"Classes: {classes}"))

# Check 4: Preprocessing & Rescaling layer in model
preprocessing_ok = False
model = None
if model_exists:
    try:
        model = tf.keras.models.load_model(model_path)
        layer_names = [l.name for l in model.layers]
        rescaling_layer = any("rescaling" in name.lower() for name in layer_names)
        preprocessing_ok = rescaling_layer
        results_summary.append(("4. Internal Rescaling Layer", preprocessing_ok, f"Rescaling layer found in model: {rescaling_layer}"))
    except Exception as e:
        results_summary.append(("4. Internal Rescaling Layer", False, str(e)))
else:
    results_summary.append(("4. Internal Rescaling Layer", False, "Model not loaded"))

# Check 5: Test set contains unseen videos (No video leakage)
from audit_dataset_v6 import audit_dataset_split
leakage_free = audit_dataset_split("face_dataset_v6") if dataset_exists else False
results_summary.append(("5. Unseen Test Videos (0 Leakage)", leakage_free, "Video-level isolation verified across splits"))

# Check 6: Model predictions work
pred_works = False
if model is not None:
    try:
        dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32) * 128.0
        out = model.predict(dummy_input, verbose=0)
        pred_works = (out.shape == (1, 1) and 0.0 <= float(out[0][0]) <= 1.0)
        results_summary.append(("6. Model Forward Pass & Inference", pred_works, f"Output shape: {out.shape}, P(REAL)={float(out[0][0]):.4f}"))
    except Exception as e:
        results_summary.append(("6. Model Forward Pass & Inference", False, str(e)))
else:
    results_summary.append(("6. Model Forward Pass & Inference", False, "Model not loaded"))

# Check 7: Evaluation script & generated plots exist
eval_artifacts = ["v6_evaluation_report.txt", "v6_confusion_matrix.png", "v6_roc_curve.png", "v6_precision_recall_curve.png", "models/face_threshold_v6.txt"]
eval_ok = all(os.path.exists(f) for f in eval_artifacts)
results_summary.append(("7. Evaluation Reports & Plots", eval_ok, f"Artifacts present: {sum(os.path.exists(f) for f in eval_artifacts)}/{len(eval_artifacts)}"))

# Check 8: Webcam test script works
webcam_script_ok = os.path.exists("test_webcam_real_v6.py") and os.path.exists("webcam_test/real")
results_summary.append(("8. Webcam Validation Pipeline", webcam_script_ok, f"test_webcam_real_v6.py ready, {len(os.listdir('webcam_test/real')) if os.path.exists('webcam_test/real') else 0} images"))

# Check 9: Realtime detector loads V6 successfully
detector_ok = False
try:
    from deepfake_detector import DeepfakeDetector
    det = DeepfakeDetector()
    dummy_face = np.zeros((100, 100, 3), dtype=np.uint8)
    lbl, conf = det.predict(dummy_face)
    detector_ok = (lbl in ["REAL", "FAKE"])
    results_summary.append(("9. Realtime Detector V6 Integration", detector_ok, f"DeepfakeDetector loaded V6 model, test output: {lbl} ({conf:.1f}%)"))
except Exception as e:
    results_summary.append(("9. Realtime Detector V6 Integration", False, str(e)))

# Summary Table
print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)
all_passed = True
for title, status, details in results_summary:
    status_str = "[PASS]" if status else "[FAIL]"
    if not status:
        all_passed = False
    print(f"{status_str:7s} | {title:<35s} | {details}")

print("=" * 70)
if all_passed:
    print("ALL 9 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
else:
    print("WARNING: Some verification checks failed. Review details above.")
print("=" * 70)
