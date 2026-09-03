"""
Test padding impact on deepfake detection
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

det = DeepfakeDetector()

test_items = [
    ("sample_fake.jpg", "image"),
    ("sample_real.jpg", "image"),
    ("GenConViT/sample_prediction_data/0017_fake.mp4.mp4", "video"),
    ("GenConViT/sample_prediction_data/0048_fake.mp4.mp4", "video"),
    ("dataset/fake/vs1.mp4", "video")
]

paddings = [0.0, 0.05, 0.10, 0.15, 0.20]

for item, item_type in test_items:
    if not os.path.exists(item):
        continue
    print("\n" + "=" * 65, flush=True)
    print(f"ITEM: {item}", flush=True)
    print("=" * 65, flush=True)

    if item_type == "image":
        img = cv2.imread(item)
        faces = detect_faces(img)
        if len(faces) == 0:
            continue
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        h, w = img.shape[:2]
        for pad in paddings:
            p = int(pad * max(fw, fh))
            crop = img[max(0, fy-p):min(h, fy+fh+p), max(0, fx-p):min(w, fx+fw+p)]
            lbl, conf, p_real = det.predict(crop, return_raw=True)
            print(f"  Padding {pad*100:4.1f}% -> P(REAL): {p_real*100:5.2f}% | Label: {lbl:9s} | Conf: {conf:5.2f}%", flush=True)

    elif item_type == "video":
        cap = cv2.VideoCapture(item)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        faces = detect_faces(frame)
        if len(faces) == 0:
            continue
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        h, w = frame.shape[:2]
        for pad in paddings:
            p = int(pad * max(fw, fh))
            crop = frame[max(0, fy-p):min(h, fy+fh+p), max(0, fx-p):min(w, fx+fw+p)]
            lbl, conf, p_real = det.predict(crop, return_raw=True)
            print(f"  Padding {pad*100:4.1f}% -> P(REAL): {p_real*100:5.2f}% | Label: {lbl:9s} | Conf: {conf:5.2f}%", flush=True)
