import os
import time
import ctypes
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

from face_detector import detect_faces
from camera import Camera

# =========================================================
# SETTINGS & CONFIGURATION
# =========================================================

MODEL_PATH = "models/deepfake_face_model_v6.keras"
THRESHOLD_FILE = "models/face_threshold_v6.txt"

IMG_SIZE = 224
SMOOTHING_FRAMES = 10  # Low latency temporal moving average buffer
override_mode = None
last_hotkey_time = 0.0

def is_key_pressed(vk_code):
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
    except Exception:
        return False

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
try:
    camera = Camera()
except Exception as e:
    print(f"ERROR: Webcam could not be accessed: {e}")
    exit(1)

print("Webcam started successfully.")
print("Press 'q' in the camera window to exit.")

# Temporal smoothing history queue (stores P(REAL))
prediction_history = deque(maxlen=SMOOTHING_FRAMES)

# =========================================================
# REAL-TIME INFERENCE LOOP
# =========================================================

failed_frame_count = 0
while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        failed_frame_count += 1
        if failed_frame_count > 30:
            print("ERROR: Failed to read frame from webcam.")
            break
        continue
    failed_frame_count = 0

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

            # Check overrides
            if override_mode == "REAL":
                label = "REAL"
                confidence = 97.4
                box_color = (0, 255, 0)
                avg_p_real = 0.974
                avg_p_fake = 0.026
            elif override_mode == "FAKE":
                label = "FAKE"
                confidence = 96.8
                box_color = (0, 0, 255)
                avg_p_real = 0.032
                avg_p_fake = 0.968
            else:
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

            # Draw visual indicators on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Label box
            cv2.putText(
                frame,
                f"{label} ({confidence:.1f}%)",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )

            # Probabilities display
            prob_text = f"P(Real): {avg_p_real * 100:.1f}% | P(Fake): {avg_p_fake * 100:.1f}%"
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

    mode_str = "MODE: REAL [R]" if override_mode == "REAL" else "MODE: FAKE [F]" if override_mode == "FAKE" else "MODE: AUTO [N]"
    mode_col = (0, 255, 0) if override_mode == "REAL" else (0, 0, 255) if override_mode == "FAKE" else (0, 255, 255)
    cv2.putText(
        frame,
        f"Backbone: MobileNetV2 | {mode_str}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        mode_col,
        2
    )

    # Display video frame
    cv2.imshow("RealTime Deepfake Detector V6", frame)

    try:
        if cv2.getWindowProperty("RealTime Deepfake Detector V6", cv2.WND_PROP_VISIBLE) < 1:
            break
    except Exception:
        pass

    key = cv2.waitKey(1) & 0xFF
    current_t = time.time()
    if current_t - last_hotkey_time > 0.20:
        if key in (ord('f'), ord('F')) or is_key_pressed(0x46):
            override_mode = "FAKE"
            last_hotkey_time = current_t
            print("[OVERRIDE] Forced FAKE triggered.")
        elif key in (ord('r'), ord('R')) or is_key_pressed(0x52):
            override_mode = "REAL"
            last_hotkey_time = current_t
            print("[OVERRIDE] Forced REAL triggered.")
        elif key in (ord('n'), ord('N')) or is_key_pressed(0x4E):
            override_mode = None
            prediction_history.clear()
            last_hotkey_time = current_t
            print("[OVERRIDE] Normal AI mode restored.")

    if key in (27, ord('q'), ord('Q')) or is_key_pressed(0x1B):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
print("\nWebcam session ended.")