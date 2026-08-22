import cv2
import numpy as np
from temporal_detector import TemporalDeepfakeDetector
from face_detector import detect_faces

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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit(1)

print("Webcam started. Press 'q' to quit.")

# =========================================================
# MAIN WEBCAM INFERENCE LOOP
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam.")
        break

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

            cv2.putText(frame, overlay_title, (card_x, card_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nReal-time demo ended.")
