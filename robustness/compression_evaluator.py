"""
Real-World Video Compression Robustness Evaluation Engine
Evaluates deepfake detection retention across controlled video compression and degradation levels.
"""

import os
import cv2
import numpy as np


class CompressionRobustnessEvaluator:
    def __init__(self, model, detector_fn):
        self.model = model
        self.detector_fn = detector_fn

    def create_compressed_variant(self, frame, quality_level="medium"):
        """
        Simulates realistic H.264/JPEG compression degradation on a video frame.
        Levels:
            - 'original': 100% quality (no degradation)
            - 'mild'    : JPEG Q=75 (Mild compression, CRF ~23)
            - 'moderate': JPEG Q=45 + light chroma subsampling (Moderate compression, CRF ~32)
            - 'heavy'   : JPEG Q=20 + 2x downsampling (Heavy social media compression, CRF ~42)
        """
        if quality_level == "original":
            return frame.copy()
        
        elif quality_level == "mild":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, enc = cv2.imencode(".jpg", frame, encode_param)
            return cv2.imdecode(enc, cv2.IMREAD_COLOR)

        elif quality_level == "moderate":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45]
            _, enc = cv2.imencode(".jpg", frame, encode_param)
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            # Add subtle compression blur
            return cv2.GaussianBlur(dec, (3, 3), 0.5)

        elif quality_level == "heavy":
            h, w = frame.shape[:2]
            # Downsample then upsample to simulate bitrate compression & blockiness
            small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
            _, enc = cv2.imencode(".jpg", small, encode_param)
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            return cv2.resize(dec, (w, h), interpolation=cv2.INTER_LINEAR)

        return frame.copy()

    def evaluate_video_robustness(self, video_path, num_samples=15):
        """
        Runs comprehensive robustness evaluation across 4 compression levels on the input video.
        """
        if not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None

        sample_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
        
        levels = ["original", "mild", "moderate", "heavy"]
        results = {lvl: {"p_reals": [], "fakes": 0, "total_faces": 0} for lvl in levels}

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            for lvl in levels:
                comp_frame = self.create_compressed_variant(frame, lvl)
                faces = self.detector_fn(comp_frame)
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    padding = int(0.12 * max(fw, fh))
                    h, w = comp_frame.shape[:2]
                    crop = comp_frame[max(0, fy - padding):min(h, fy + fh + padding), max(0, fx - padding):min(w, fx + fw + padding)]
                    if crop.size > 0:
                        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        face_res = cv2.resize(face_rgb, (224, 224))
                        face_arr = np.expand_dims(face_res.astype("float32"), axis=0)
                        try:
                            p_real = float(self.model(face_arr, training=False)[0][0])
                        except Exception:
                            p_real = float(self.model.predict(face_arr, verbose=0)[0][0])

                        results[lvl]["p_reals"].append(p_real)
                        results[lvl]["total_faces"] += 1
                        if p_real <= 0.50:
                            results[lvl]["fakes"] += 1

        cap.release()

        # Format report table
        summary_table = []
        for lvl in levels:
            p_list = results[lvl]["p_reals"]
            n = len(p_list)
            if n > 0:
                mean_p = float(np.mean(p_list))
                fake_rt = results[lvl]["fakes"] / n
                retention = fake_rt if fake_rt > 0 else (1.0 - mean_p)
                summary_table.append({
                    "level": lvl.capitalize(),
                    "faces_analyzed": n,
                    "mean_p_real": mean_p,
                    "fake_detection_rate": fake_rt * 100.0,
                    "robustness_score": float(np.clip(retention * 100.0, 0, 100))
                })

        return summary_table
