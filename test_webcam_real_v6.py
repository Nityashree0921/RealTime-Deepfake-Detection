import os
import cv2
import numpy as np
import tensorflow as tf

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v6.keras"
THRESHOLD_FILE = "models/face_threshold_v6.txt"
WEBCAM_DIR = "webcam_test/real"
IMG_SIZE = 224

print("=" * 70)
print("V6 REAL WEBCAM IMAGE VALIDATION TEST")
print("=" * 70)

# =========================================================
# LOAD THRESHOLD & MODEL
# =========================================================

threshold = 0.50
if os.path.exists(THRESHOLD_FILE):
    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())

print(f"Using Validation-Selected Threshold: {threshold:.2f}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("V6 Model loaded successfully!")

# =========================================================
# LOAD & PREPROCESS WEBCAM IMAGES
# =========================================================

if not os.path.exists(WEBCAM_DIR):
    raise FileNotFoundError(f"Webcam test directory not found: {WEBCAM_DIR}")

files = [f for f in os.listdir(WEBCAM_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Found {len(files)} webcam test images in: {WEBCAM_DIR}")

images = []
valid_files = []

for filename in sorted(files):
    filepath = os.path.join(WEBCAM_DIR, filename)
    img = cv2.imread(filepath)
    if img is None:
        continue
    
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # IMPORTANT: Model contains internal Rescaling(scale=1/127.5, offset=-1).
    # Pass float32 pixels in [0, 255] without duplicate scaling.
    img = img.astype(np.float32)
    
    images.append(img)
    valid_files.append(filename)

X = np.array(images, dtype=np.float32)
print(f"Batch shape: {X.shape}, Min: {X.min()}, Max: {X.max()}")

# =========================================================
# RUN INFERENCE
# =========================================================

print("\nRunning inference...")
# Model output = P(REAL)
p_real = model.predict(X, batch_size=16, verbose=1).flatten()

# Binary classification: P(REAL) >= threshold -> REAL (1), else FAKE (0)
predicted_labels = (p_real >= threshold).astype(int)

total_images = len(p_real)
pred_real = int(np.sum(predicted_labels == 1))
pred_fake = int(np.sum(predicted_labels == 0))

real_detection_rate = (pred_real / total_images) * 100.0 if total_images > 0 else 0.0
false_fake_rate = (pred_fake / total_images) * 100.0 if total_images > 0 else 0.0

avg_real_prob = float(np.mean(p_real)) * 100.0
min_real_prob = float(np.min(p_real)) * 100.0
max_real_prob = float(np.max(p_real)) * 100.0
median_real_prob = float(np.median(p_real)) * 100.0

# =========================================================
# RESULTS & METRICS
# =========================================================

print("\n" + "=" * 70)
print("WEBCAM VALIDATION RESULTS (REAL IMAGES ONLY)")
print("=" * 70)
print(f"Total Images Evaluated   : {total_images}")
print(f"Correctly Predicted REAL : {pred_real}")
print(f"Incorrectly Pred. FAKE   : {pred_fake}")
print("-" * 70)
print(f"REAL Detection Rate      : {real_detection_rate:0.2f}%")
print(f"False FAKE Rate (Error)  : {false_fake_rate:0.2f}%")
print("-" * 70)
print(f"Average P(REAL)          : {avg_real_prob:0.2f}%")
print(f"Median P(REAL)           : {median_real_prob:0.2f}%")
print(f"Minimum P(REAL)          : {min_real_prob:0.2f}%")
print(f"Maximum P(REAL)          : {max_real_prob:0.2f}%")
print("=" * 70)
