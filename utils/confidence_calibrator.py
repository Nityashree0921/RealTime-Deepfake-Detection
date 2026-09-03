"""
Confidence Calibration Module for Deepfake Detection
Implements Temperature Scaling, Expected Calibration Error (ECE), and Brier Score evaluation.
"""

import numpy as np


class ConfidenceCalibrator:
    def __init__(self, temperature=1.18):
        self.temperature = temperature

    def calibrate(self, raw_probability):
        """
        Applies temperature scaling to raw probability in [0..1].
        """
        eps = 1e-7
        p = np.clip(raw_probability, eps, 1.0 - eps)
        # Logit transformation
        logit = np.log(p / (1.0 - p))
        # Scaled logit
        scaled_logit = logit / self.temperature
        # Sigmoid recovery
        p_calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(p_calibrated)

    @staticmethod
    def compute_ece(probabilities, true_labels, num_bins=10):
        """
        Computes Expected Calibration Error (ECE) across M probability bins.
        """
        probs = np.array(probabilities)
        labels = np.array(true_labels)
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        bin_data = []

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])
                abs_diff = np.abs(accuracy_in_bin - avg_confidence_in_bin)
                ece += abs_diff * prop_in_bin
                bin_data.append({
                    "bin": f"{bin_lower:.1f}-{bin_upper:.1f}",
                    "count": int(np.sum(in_bin)),
                    "confidence": float(avg_confidence_in_bin),
                    "accuracy": float(accuracy_in_bin),
                    "gap": float(abs_diff)
                })

        return float(ece), bin_data

    @staticmethod
    def compute_brier_score(probabilities, true_labels):
        """
        Computes Brier Score: Mean squared error of probabilities vs true binary labels.
        """
        probs = np.array(probabilities)
        labels = np.array(true_labels)
        return float(np.mean((probs - labels) ** 2))
