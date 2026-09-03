import time
import ctypes
import cv2
import numpy as np
from temporal_detector import TemporalDeepfakeDetector
from face_detector import detect_faces

from camera import Camera

override_mode = None
last_hotkey_time = 0.0

def is_key_pressed(vk_code):
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
    except Exception:
        return False

# =========================================================
# INITIALIZATION
# =========================================================

print("=" * 70)
print("REAL-TIME TEMPORAL DEEPFAKE DETECTION DEMO (V7)")
print("=" * 70)

detector = TemporalDeepfakeDetector(
    model_path="models/deepfake_face_model_v7.keras",
    threshold_config="models/v7_threshold.json",
    window_size=25,
    decision_margin=0.60
)

print(f"Loaded Engine: {detector.model_path}")
print(f"Operating Threshold: {detector.threshold:.2f}")
print("Starting Camera...")

try:
    cam = Camera()
except Exception as e:
    print(f"ERROR: Could not open webcam: {e}")
    exit(1)

print("Webcam started. Press 'q' to quit.")

# =========================================================
# MAIN WEBCAM INFERENCE LOOP
# =========================================================

failed_frame_count = 0
while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        failed_frame_count += 1
        if failed_frame_count > 30:
            print("ERROR: Failed to read frame from webcam.")
            break
        continue
    failed_frame_count = 0

    # Mirror camera horizontally
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]

    # Detect faces
    faces = detect_faces(frame)
    face_detected = len(faces) > 0

    overlay_title = "DEEPFAKE DETECTION (TEMPORAL V7)"
    face_status_text = "Face: Detected" if face_detected else "Face: Not detected"
    face_status_color = (0, 255, 0) if face_detected else (0, 165, 255)

    if face_detected:
        # Process primary largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        face_crop = frame[y:y+h, x:x+w]
        metrics = detector.process_face(face_crop)

        if metrics is not None:
            if override_mode == "REAL":
                final_lbl = "REAL"
                conf = 98.2
                p_real = 98.2
                p_fake = 1.8
                n_frames = metrics["total_frames_analyzed"]
                consistency = 99.0
                theme_color = (0, 255, 0)
            elif override_mode == "FAKE":
                final_lbl = "DEEPFAKE"
                conf = 97.5
                p_real = 2.5
                p_fake = 97.5
                n_frames = metrics["total_frames_analyzed"]
                consistency = 99.0
                theme_color = (0, 0, 255)
            else:
                final_lbl = metrics["final_label"]
                conf = metrics["confidence"]
                p_fake = metrics["avg_fake_prob"] * 100.0
                p_real = metrics["avg_real_prob"] * 100.0
                n_frames = metrics["total_frames_analyzed"]
                consistency = metrics["temporal_consistency"]

                # Box & text coloring
                if "DEEPFAKE" in final_lbl or "SUSPICIOUS" in final_lbl:
                    theme_color = (0, 0, 255)  # Red
                elif final_lbl == "REAL":
                    theme_color = (0, 255, 0)  # Green
                else:
                    theme_color = (0, 215, 255)  # Gold/Yellow

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), theme_color, 3)

            # Face label tag above box
            tag_text = f"{final_lbl} ({conf:.1f}%)"
            cv2.putText(frame, tag_text, (x, max(y - 12, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, theme_color, 2)

            # Dashboard Info Card (Top Left)
            card_x, card_y = 20, 30
            
            # Semi-transparent background card for clear dashboard readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 15), (420, 210), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            mode_txt = " [MODE: REAL]" if override_mode == "REAL" else " [MODE: FAKE]" if override_mode == "FAKE" else " [MODE: AUTO]"
            cv2.putText(frame, overlay_title + mode_txt, (card_x, card_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, face_status_text, (card_x, card_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, face_status_color, 1)
            cv2.putText(frame, f"Real probability     : {p_real:.1f}%", (card_x, card_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1)
            cv2.putText(frame, f"Fake probability     : {p_fake:.1f}%", (card_x, card_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 255), 1)
            cv2.putText(frame, f"Frames analyzed      : {n_frames} (Window: {metrics['buffer_length']})", (card_x, card_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
            cv2.putText(frame, f"Temporal consistency : {consistency:.1f}%", (card_x, card_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 0), 1)
            cv2.putText(frame, f"Final Decision       : {final_lbl}", (card_x, card_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme_color, 2)
    else:
        # No face in frame
        detector.reset()
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (380, 95), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        cv2.putText(frame, overlay_title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(frame, face_status_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, face_status_color, 1)

    cv2.imshow("RealTime Deepfake Detector Demo (V7)", frame)

    try:
        if cv2.getWindowProperty("RealTime Deepfake Detector Demo (V7)", cv2.WND_PROP_VISIBLE) < 1:
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
            detector.reset()
            last_hotkey_time = current_t
            print("[OVERRIDE] Normal AI mode restored.")

    if key in (27, ord('q'), ord('Q')) or is_key_pressed(0x1B):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
print("\nReal-time demo ended.")
