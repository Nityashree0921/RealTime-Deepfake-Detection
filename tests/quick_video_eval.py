"""
Quick evaluation of video prediction across test videos
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
    ("dataset/fake/vs1.mp4", "FAKE"),
    ("dataset/fake/vs2.mp4", "FAKE"),
    ("GenConViT/sample_prediction_data/0017_fake.mp4.mp4", "FAKE"),
    ("GenConViT/sample_prediction_data/0048_fake.mp4.mp4", "FAKE"),
    ("GenConViT/sample_prediction_data/sample_1.mp4", "UNKNOWN")
]

for v_path, true_lbl in test_videos:
    if not os.path.exists(v_path):
        continue

    print("\n" + "=" * 65, flush=True)
    print(f"VIDEO: {v_path} (Expected: {true_lbl})", flush=True)
    print("=" * 65, flush=True)

    cap = cv2.VideoCapture(v_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample up to 20 frames evenly across the video
    sample_indices = np.linspace(0, total_frames - 1, min(20, total_frames), dtype=int)
    
    frame_preds = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        faces = detect_faces(frame)
        if len(faces) == 0:
            continue

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        padding = int(0.20 * max(fw, fh))
        h, w = frame.shape[:2]
        x1 = max(0, fx - padding)
        y1 = max(0, fy - padding)
        x2 = min(w, fx + fw + padding)
        y2 = min(h, fy + fh + padding)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            lbl, conf, p_real = detector.predict(face_crop, return_raw=True)
            frame_preds.append((idx, p_real, lbl, conf))
            print(f"  Frame {idx:4d} | P(REAL): {p_real*100:5.2f}% | Label: {lbl:9s} | Conf: {conf:5.2f}%", flush=True)

    cap.release()

    if len(frame_preds) > 0:
        p_reals = [p[1] for p in frame_preds]
        fake_count = sum(1 for p in frame_preds if p[2] == "FAKE")
        real_count = sum(1 for p in frame_preds if p[2] == "REAL")
        unc_count = sum(1 for p in frame_preds if p[2] == "UNCERTAIN")
        
        mean_p = np.mean(p_reals)
        min_p = np.min(p_reals)
        p25 = np.percentile(p_reals, 25)

        print("-" * 65, flush=True)
        print(f"  Sampled Frames: {len(frame_preds)} | FAKE Frames: {fake_count} | REAL Frames: {real_count} | UNCERTAIN: {unc_count}", flush=True)
        print(f"  Mean P(REAL): {mean_p*100:.2f}% | P25: {p25*100:.2f}% | Min: {min_p*100:.2f}%", flush=True)
