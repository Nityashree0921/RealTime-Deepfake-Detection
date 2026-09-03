"""
Benchmark all face models on sample fake and real videos
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from face_detector import detect_faces

models_to_test = [
    "models/deepfake_face_model_v7.keras",
    "models/deepfake_face_model_v6.keras",
    "models/deepfake_face_model_v5.keras",
    "models/deepfake_face_model_v4.keras",
    "models/deepfake_face_model.keras",
    "models/deepfake_model.keras"
]

test_videos = [
    ("dataset/fake/vs1.mp4", "FAKE"),
    ("dataset/fake/vs2.mp4", "FAKE"),
    ("GenConViT/sample_prediction_data/0017_fake.mp4.mp4", "FAKE"),
    ("GenConViT/sample_prediction_data/0048_fake.mp4.mp4", "FAKE")
]

# Check existing videos
existing_models = [m for m in models_to_test if os.path.exists(m)]
existing_videos = [v for v in test_videos if os.path.exists(v[0])]

print(f"Found {len(existing_models)} models and {len(existing_videos)} test videos.")

for m_path in existing_models:
    print("\n" + "=" * 70)
    print(f"TESTING MODEL: {m_path}")
    print("=" * 70)
    
    try:
        model = tf.keras.models.load_model(m_path)
    except Exception as e:
        print(f"Could not load {m_path}: {e}")
        continue

    for v_path, true_label in existing_videos:
        cap = cv2.VideoCapture(v_path)
        frame_idx = 0
        p_reals = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 5 == 0:
                h, w = frame.shape[:2]
                faces = detect_faces(frame)
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    padding = int(0.20 * max(fw, fh))
                    x1 = max(0, fx - padding)
                    y1 = max(0, fy - padding)
                    x2 = min(w, fx + fw + padding)
                    y2 = min(h, fy + fh + padding)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (224, 224))
                        face_array = np.expand_dims(face_resized.astype("float32"), axis=0)
                        
                        # In v6/v7 models, internal rescaling is present. For older models, test both:
                        p_val = float(model.predict(face_array, verbose=0)[0][0])
                        p_reals.append(p_val)

        cap.release()
        if len(p_reals) > 0:
            mean_p = np.mean(p_reals)
            min_p = np.min(p_reals)
            fake_count = sum(1 for p in p_reals if p <= 0.50)
            print(f"  {os.path.basename(v_path):28s} (True: {true_label}) -> Mean P(REAL): {mean_p*100:5.2f}% | Min P: {min_p*100:5.2f}% | Fake Frames: {fake_count}/{len(p_reals)} ({fake_count/len(p_reals)*100:.1f}%)")
