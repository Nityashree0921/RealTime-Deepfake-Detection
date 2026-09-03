"""
Principled Uncertainty Estimation Module for Deepfake Detection
Implements Monte Carlo Dropout (MC-Dropout) stochastic forward passes,
predictive variance, and Shannon predictive entropy.
"""

import numpy as np
import tensorflow as tf
import cv2


class UncertaintyEstimator:
    def __init__(self, model, num_passes=10):
        self.model = model
        self.num_passes = num_passes

    def estimate_uncertainty(self, face_bgr):
        """
        Executes Monte Carlo Dropout inference on face input.
        Returns:
            dict containing:
                mean_probability: float [0..1],
                variance: float,
                std_dev: float,
                entropy: float [0..1],
                uncertainty_level: 'LOW' | 'MEDIUM' | 'HIGH',
                risk_level: 'LOW' | 'ELEVATED' | 'CRITICAL',
                mc_samples: list of floats
        """
        if face_bgr is None or face_bgr.size == 0:
            return {
                "mean_probability": 0.50,
                "variance": 0.25,
                "std_dev": 0.50,
                "entropy": 1.0,
                "uncertainty_level": "HIGH",
                "risk_level": "CRITICAL",
                "mc_samples": []
            }

        # Preprocess
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB) if len(face_bgr.shape) == 3 and face_bgr.shape[2] == 3 else face_bgr
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = np.expand_dims(face_resized.astype("float32"), axis=0)

        # Stochastic forward passes with active dropout
        samples = []
        for _ in range(self.num_passes):
            try:
                # training=True enables dropout layers during inference
                p = float(self.model(face_array, training=True)[0][0])
            except Exception:
                p = float(self.model.predict(face_array, verbose=0)[0][0])
            samples.append(p)

        samples_arr = np.array(samples)
        mean_p = float(np.mean(samples_arr))
        variance = float(np.var(samples_arr))
        std_dev = float(np.std(samples_arr))

        # Shannon Entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
        eps = 1e-7
        p_safe = np.clip(mean_p, eps, 1.0 - eps)
        entropy = float(- (p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe)))
        entropy = max(0.0, min(1.0, entropy))

        # Uncertainty Classification
        # High epistemic uncertainty: high variance across stochastic passes or high entropy near decision boundary
        if std_dev >= 0.08 or (0.42 <= mean_p <= 0.58):
            uncertainty_level = "HIGH"
        elif std_dev >= 0.04 or (0.35 <= mean_p <= 0.65):
            uncertainty_level = "MEDIUM"
        else:
            uncertainty_level = "LOW"

        # Security Risk Assessment
        if mean_p <= 0.40 and uncertainty_level == "LOW":
            risk_level = "CRITICAL (High-Confidence Deepfake)"
        elif mean_p <= 0.50:
            risk_level = "ELEVATED (Potential Manipulation)"
        elif uncertainty_level == "HIGH":
            risk_level = "ELEVATED (High Predictive Ambiguity)"
        else:
            risk_level = "LOW (Authentic Stream)"

        return {
            "mean_probability": mean_p,
            "variance": variance,
            "std_dev": std_dev,
            "entropy": entropy,
            "uncertainty_level": uncertainty_level,
            "risk_level": risk_level,
            "mc_samples": samples
        }
