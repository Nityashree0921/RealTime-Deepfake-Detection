"""
Inspect vs1.mp4 face crops
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
cap = cv2.VideoCapture("dataset/fake/vs1.mp4")
ret, frame = cap.read()
if ret:
    faces = detect_faces(frame)
    print("Frame shape:", frame.shape)
    print("Detected faces in vs1.mp4 frame 0:", faces)
    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        crop = frame[fy:fy+fh, fx:fx+fw]
        lbl, conf, p_real = det.predict(crop, return_raw=True)
        print(f"Zero padding crop: P(REAL) = {p_real*100:.2f}% | Label = {lbl}")
        
        # Test 20% padding crop
        p = int(0.20 * max(fw, fh))
        crop_p = frame[max(0, fy-p):min(frame.shape[0], fy+fh+p), max(0, fx-p):min(frame.shape[1], fx+fw+p)]
        lbl_p, conf_p, p_real_p = det.predict(crop_p, return_raw=True)
        print(f"20% padding crop: P(REAL) = {p_real_p*100:.2f}% | Label = {lbl_p}")
cap.release()
