import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

from face_detector import detect_faces

# =========================================================
# SETTINGS & CONFIGURATION
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v6.keras"
THRESHOLD_FILE = "models/face_threshold_v6.txt"

IMG_SIZE = 224
SMOOTHING_FRAMES = 10  # Low latency temporal moving average buffer

# =========================================================
# LOAD MODEL & THRESHOLD
# =========================================================

print("=" * 60)
print("REAL-TIME FACE DEEPFAKE DETECTION (V6 PIPELINE)")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: Face model not found at: {MODEL_PATH}")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print("V6 Face model loaded successfully!")

# Load validation-calibrated threshold
threshold = 0.50
if os.path.exists(THRESHOLD_FILE):
    try:
        with open(THRESHOLD_FILE, "r") as f:
            threshold = float(f.read().strip())
    except Exception:
        threshold = 0.50

print(f"Loaded Model: {MODEL_PATH}")
print(f"Decision Threshold: {threshold:.2f} (P(REAL) >= {threshold:.2f} -> REAL)")

# =========================================================
# CAMERA INITIALIZATION
# =========================================================

print("\nStarting webcam stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be accessed. Please verify camera connection.")
    exit(1)

print("Webcam started successfully.")
print("Press 'q' in the camera window to exit.")

# Temporal smoothing history queue (stores P(REAL))
prediction_history = deque(maxlen=SMOOTHING_FRAMES)

# =========================================================
# REAL-TIME INFERENCE LOOP
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam.")
        break

    # Horizontal mirror for natural webcam user experience
    frame = cv2.flip(frame, 1)

    # Detect faces using Caffe SSD DNN
    faces = detect_faces(frame)

    if len(faces) == 0:
        prediction_history.clear()
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )
    else:
        # Focus on largest face in frame
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Crop face
        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size > 0:
            # BGR -> RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

            # Float32 array in range [0, 255] (Model contains internal Rescaling layer)
            face_array = np.array(face_resized, dtype=np.float32)
            face_tensor = np.expand_dims(face_array, axis=0)

            # Predict P(REAL)
            p_real_raw = float(model.predict(face_tensor, verbose=0)[0][0])
            prediction_history.append(p_real_raw)

            # Temporal smoothing via moving average
            avg_p_real = float(np.mean(prediction_history))
            avg_p_fake = 1.0 - avg_p_real

            # Classification
            if avg_p_real >= threshold:
                label = "REAL"
                confidence = avg_p_real * 100.0
                box_color = (0, 255, 0)  # Green
            else:
                label = "FAKE"
                confidence = avg_p_fake * 100.0
                box_color = (0, 0, 255)  # Red

            # Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

            # Label & Confidence text
            result_text = f"{label} ({confidence:.1f}%)"
            cv2.putText(
                frame,
                result_text,
                (x, max(y - 12, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                box_color,
                2
            )

            # Subtitle with detailed probability breakdown
            prob_text = f"P(REAL): {avg_p_real*100:.1f}% | P(FAKE): {avg_p_fake*100:.1f}%"
            cv2.putText(
                frame,
                prob_text,
                (x, y + h + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                box_color,
                2
            )

    # UI Information Overlay
    cv2.putText(
        frame,
        "REAL-TIME DEEPFAKE DETECTION (V6)",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Backbone: MobileNetV2 | Thresh: {threshold:.2f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1
    )

    # Display video frame
    cv2.imshow("RealTime Deepfake Detector V6", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nWebcam session ended.")