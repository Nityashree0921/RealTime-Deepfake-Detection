"""
Diagnostic Script to evaluate video detection predictions on sample fake videos
"""

import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector

detector = DeepfakeDetector()

test_videos = [
    "dataset/fake/vs1.mp4",
    "dataset/fake/vs2.mp4",
    "GenConViT/sample_prediction_data/0017_fake.mp4.mp4"
]

for vid_path in test_videos:
    if not os.path.exists(vid_path):
        print(f"File not found: {vid_path}")
        continue

    print("\n" + "=" * 70)
    print(f"TESTING VIDEO: {vid_path}")
    print("=" * 70)

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    predictions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 5 == 0:
            h, w = frame.shape[:2]
            faces = detect_faces(frame)
            if len(faces) > 0:
                # Largest face
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                padding = int(0.20 * max(fw, fh))
                x1 = max(0, fx - padding)
                y1 = max(0, fy - padding)
                x2 = min(w, fx + fw + padding)
                y2 = min(h, fy + fh + padding)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    lbl, conf, p_real = detector.predict(face_crop, return_raw=True)
                    predictions.append((frame_idx, p_real, lbl, conf))
                    print(f"  Frame {frame_idx:4d} | P(REAL): {p_real*100:5.2f}% | Label: {lbl:9s} | Conf: {conf:5.2f}%")

    cap.release()

    if len(predictions) > 0:
        p_reals = [p[1] for p in predictions]
        fake_frames = [p for p in predictions if p[2] == "FAKE"]
        real_frames = [p for p in predictions if p[2] == "REAL"]
        uncertain_frames = [p for p in predictions if p[2] == "UNCERTAIN"]

        mean_p_real = np.mean(p_reals)
        median_p_real = np.median(p_reals)
        min_p_real = np.min(p_reals)
        fake_ratio = len(fake_frames) / len(predictions)

        print("\nVIDEO SUMMARY:")
        print(f"  Total Sampled Face Frames : {len(predictions)}")
        print(f"  FAKE Frame Count          : {len(fake_frames)} ({fake_ratio*100:.1f}%)")
        print(f"  REAL Frame Count          : {len(real_frames)} ({len(real_frames)/len(predictions)*100:.1f}%)")
        print(f"  UNCERTAIN Frame Count     : {len(uncertain_frames)}")
        print(f"  Mean P(REAL)              : {mean_p_real*100:.2f}%")
        print(f"  Median P(REAL)            : {median_p_real*100:.2f}%")
        print(f"  Min P(REAL)               : {min_p_real*100:.2f}%")
        
        # Overall Calibrated
        overall_lbl, overall_conf = detector.calibrate_probability(mean_p_real)
        print(f"  Simple Mean Verdict       : {overall_lbl} ({overall_conf:.2f}%)")
