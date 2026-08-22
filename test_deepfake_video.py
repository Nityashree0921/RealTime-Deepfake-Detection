import os
import cv2
import argparse
import numpy as np
from temporal_detector import TemporalDeepfakeDetector
from face_detector import detect_faces

# =========================================================
# CLI ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(description="Test Video File with V7 Temporal Deepfake Detection")
parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4, .avi, etc.)")
parser.add_argument("--output", type=str, default="outputs/analyzed_video.mp4", help="Path to save annotated output video")
parser.add_argument("--window_size", type=int, default=25, help="Rolling window size for temporal aggregation")
args = parser.parse_args()

print("=" * 70)
print("TESTING VIDEO FILE WITH V7 TEMPORAL DEEPFAKE DETECTOR")
print("=" * 70)

video_path = args.video
output_path = args.output

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Input video file not found: {video_path}")

os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

# Initialize Temporal Detector
detector = TemporalDeepfakeDetector(
    model_path="models/deepfake_face_model_v7.keras",
    threshold_config="models/v7_threshold.json",
    window_size=args.window_size
)

print(f"Input Video : {video_path}")
print(f"Output Path : {output_path}")
print(f"Model File  : {detector.model_path}")
print(f"Threshold   : {detector.threshold:.2f}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video file: {video_path}")

total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Specs : {width}x{height} @ {fps:.1f} FPS | Total Frames: {total_video_frames}")

# Setup Video Writer (mp4v codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
faces_processed = 0
all_frame_fake_probs = []
all_frame_real_probs = []

print("\nProcessing video frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    faces = detect_faces(frame)

    if len(faces) > 0:
        # Select largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        face_crop = frame[y:y+h, x:x+w]

        metrics = detector.process_face(face_crop)
        if metrics is not None:
            faces_processed += 1
            all_frame_fake_probs.append(metrics["p_fake_frame"])
            all_frame_real_probs.append(metrics["p_real_frame"])

            final_lbl = metrics["final_label"]
            conf = metrics["confidence"]
            p_fake = metrics["avg_fake_prob"] * 100.0
            p_real = metrics["avg_real_prob"] * 100.0
            consistency = metrics["temporal_consistency"]

            theme_color = (0, 0, 255) if ("DEEPFAKE" in final_lbl or "SUSPICIOUS" in final_lbl) else (0, 255, 0)

            # Draw bounding box & tags
            cv2.rectangle(frame, (x, y), (x + w, y + h), theme_color, 3)
            cv2.putText(frame, f"{final_lbl} ({conf:.1f}%)", (x, max(y - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, theme_color, 2)

            # Info Card Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 15), (430, 180), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            cv2.putText(frame, "VIDEO ANALYSIS (V7 TEMPORAL)", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {frame_idx}/{total_video_frames} | Faces: {faces_processed}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            cv2.putText(frame, f"Real prob: {p_real:.1f}% | Fake prob: {p_fake:.1f}%", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)
            cv2.putText(frame, f"Temporal consistency: {consistency:.1f}%", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1)
            cv2.putText(frame, f"Decision: {final_lbl}", (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme_color, 2)
    else:
        # No face detected in this frame
        cv2.putText(frame, "NO FACE DETECTED", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 165, 255), 2)

    out.write(frame)
    if frame_idx % 30 == 0 or frame_idx == total_video_frames:
        print(f"  Processed {frame_idx}/{total_video_frames} frames ({faces_processed} face frames detected)")

cap.release()
out.release()

# =========================================================
# VIDEO-LEVEL FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL VIDEO-LEVEL DETECTION REPORT")
print("=" * 70)
print(f"Total Video Frames     : {total_video_frames}")
print(f"Face Frames Detected   : {faces_processed}")

if faces_processed > 0:
    mean_real_prob = float(np.mean(all_frame_real_probs)) * 100.0
    mean_fake_prob = float(np.mean(all_frame_fake_probs)) * 100.0
    fake_frames_ratio = float(np.mean(np.array(all_frame_real_probs) < detector.threshold)) * 100.0

    if fake_frames_ratio >= 50.0:
        video_verdict = "SUSPICIOUS / DEEPFAKE VIDEO"
        confidence = mean_fake_prob
    else:
        video_verdict = "REAL VIDEO"
        confidence = mean_real_prob

    print(f"Mean REAL Probability  : {mean_real_prob:.2f}%")
    print(f"Mean FAKE Probability  : {mean_fake_prob:.2f}%")
    print(f"Fake Frame Percentage  : {fake_frames_ratio:.2f}%")
    print("-" * 70)
    print(f"OVERALL VIDEO VERDICT  : {video_verdict} (Confidence: {confidence:.1f}%)")
else:
    print("OVERALL VIDEO VERDICT  : INCONCLUSIVE (No faces detected in video)")

print(f"Annotated Video Saved  : {os.path.abspath(output_path)}")
print("=" * 70)
