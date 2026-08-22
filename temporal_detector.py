import os
import json
import numpy as np
import tensorflow as tf
from collections import deque

from face_detector import detect_faces

class TemporalDeepfakeDetector:
    """
    Temporal Deepfake Detection Engine with a rolling aggregation window.
    Prevents single-frame classification flickering and computes temporal consistency.
    """
    def __init__(
        self,
        model_path="models/deepfake_face_model_v7.keras",
        threshold_config="models/v7_threshold.json",
        window_size=25,
        decision_margin=0.60
    ):
        self.window_size = window_size
        self.decision_margin = decision_margin
        
        # Load Model
        if not os.path.exists(model_path):
            if os.path.exists("models/deepfake_face_model_v6.keras"):
                model_path = "models/deepfake_face_model_v6.keras"
            elif os.path.exists("models/deepfake_face_model_v5.keras"):
                model_path = "models/deepfake_face_model_v5.keras"
                
        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path
        
        # Load Operating Threshold
        self.threshold = 0.50
        if os.path.exists(threshold_config):
            try:
                with open(threshold_config, "r") as f:
                    cfg = json.load(f)
                    self.threshold = float(cfg.get("optimal_threshold", 0.50))
            except Exception:
                self.threshold = 0.50
                
        # History queue storing raw P(REAL) probabilities in [0.0, 1.0]
        self.history = deque(maxlen=self.window_size)
        self.total_frames_analyzed = 0

    def reset(self):
        """Reset temporal history buffer."""
        self.history.clear()
        self.total_frames_analyzed = 0

    def process_face(self, face_crop):
        """
        Process a single cropped face image (BGR uint8).
        Returns raw frame prediction and rolling temporal metrics.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # Preprocessing: BGR -> RGB, 224x224, float32 without duplicate division
        import cv2
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) if len(face_crop.shape) == 3 and face_crop.shape[2] == 3 else face_crop
        face_resized = cv2.resize(face_rgb, (224, 224))
        tensor = np.expand_dims(face_resized.astype("float32"), axis=0)

        # Model output = P(REAL)
        p_real_frame = float(self.model.predict(tensor, verbose=0)[0][0])
        p_fake_frame = 1.0 - p_real_frame

        self.history.append(p_real_frame)
        self.total_frames_analyzed += 1

        # Temporal Rolling Aggregations
        p_real_array = np.array(self.history)
        p_fake_array = 1.0 - p_real_array

        avg_real_prob = float(np.mean(p_real_array))
        avg_fake_prob = float(np.mean(p_fake_array))
        median_fake_prob = float(np.median(p_fake_array))
        
        # Fraction of frames in window classified as fake
        fake_frame_flags = (p_real_array < self.threshold)
        fake_frame_percentage = float(np.mean(fake_frame_flags)) * 100.0

        # Temporal consistency: agreement ratio with the dominant decision
        if avg_fake_prob >= (1.0 - self.threshold):
            temporal_consistency = float(np.mean(fake_frame_flags)) * 100.0
            preliminary_label = "FAKE"
        else:
            temporal_consistency = float(np.mean(~fake_frame_flags)) * 100.0
            preliminary_label = "REAL"

        # Final Decision with rolling multi-frame requirement (no single-frame snap decisions)
        if len(self.history) >= min(10, self.window_size // 2):
            if fake_frame_percentage >= (self.decision_margin * 100.0):
                final_label = "SUSPICIOUS / POSSIBLE DEEPFAKE"
                confidence = avg_fake_prob * 100.0
            elif (100.0 - fake_frame_percentage) >= (self.decision_margin * 100.0):
                final_label = "REAL"
                confidence = avg_real_prob * 100.0
            else:
                final_label = "UNCERTAIN / MIXED SIGNALS"
                confidence = max(avg_real_prob, avg_fake_prob) * 100.0
        else:
            final_label = "ANALYZING TEMPORAL BUFFER..."
            confidence = max(avg_real_prob, avg_fake_prob) * 100.0

        return {
            "p_real_frame": p_real_frame,
            "p_fake_frame": p_fake_frame,
            "avg_real_prob": avg_real_prob,
            "avg_fake_prob": avg_fake_prob,
            "median_fake_prob": median_fake_prob,
            "fake_frame_percentage": fake_frame_percentage,
            "temporal_consistency": temporal_consistency,
            "buffer_length": len(self.history),
            "total_frames_analyzed": self.total_frames_analyzed,
            "final_label": final_label,
            "confidence": confidence,
            "operating_threshold": self.threshold
        }
