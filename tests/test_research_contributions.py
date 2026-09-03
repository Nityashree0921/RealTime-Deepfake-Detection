"""
Comprehensive Test Suite for Novel Research Contributions
Verifies Grad-CAM XAI, MC-Dropout, Confidence Calibration, Video Compression Robustness, and Ablation Matrices.
"""

import os
import sys
import unittest
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from deepfake_detector import DeepfakeDetector
from face_detector import detect_faces
from utils.xai_explainer import XAIExplainer
from utils.uncertainty_estimator import UncertaintyEstimator
from utils.confidence_calibrator import ConfidenceCalibrator
from robustness.compression_evaluator import CompressionRobustnessEvaluator
from evaluation.ablation_study import AblationStudyEngine


class TestResearchContributions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = DeepfakeDetector()
        cls.xai = XAIExplainer(cls.detector.model)
        cls.uncertainty_est = UncertaintyEstimator(cls.detector.model, num_passes=5)
        cls.calibrator = ConfidenceCalibrator(temperature=1.18)
        cls.ablation_engine = AblationStudyEngine()
        cls.robustness_evaluator = CompressionRobustnessEvaluator(cls.detector.model, detect_faces)

    def test_1_gradcam_generation(self):
        """Test genuine Grad-CAM generation and spatial evidence bullets on face crop."""
        img_path = "sample_images/sample_fake.jpg"
        if not os.path.exists(img_path):
            self.skipTest(f"{img_path} not found")

        img = cv2.imread(img_path)
        faces = detect_faces(img)
        self.assertGreater(len(faces), 0, "No face detected in sample_fake.jpg")

        x, y, w, h = faces[0]
        p = int(0.12 * max(w, h))
        h_f, w_f = img.shape[:2]
        crop = img[max(0, y-p):min(h_f, y+h+p), max(0, x-p):min(w_f, x+w+p)]

        heatmap, overlay, bullets = self.xai.generate_gradcam(crop, pred_class="FAKE")
        self.assertIsNotNone(heatmap, "Grad-CAM heatmap is None")
        self.assertEqual(heatmap.shape, (crop.shape[0], crop.shape[1]))
        self.assertEqual(overlay.shape, crop.shape)
        self.assertGreater(len(bullets), 0, "No evidence bullets returned")
        print(f"[TEST PASSED] Grad-CAM generated: {len(bullets)} evidence points | Peak activation: {heatmap.max():.3f}")

    def test_2_mc_dropout_uncertainty(self):
        """Test MC-Dropout stochastic forward passes and entropy calculation."""
        img_path = "sample_images/sample_fake.jpg"
        if not os.path.exists(img_path):
            self.skipTest(f"{img_path} not found")

        img = cv2.imread(img_path)
        faces = detect_faces(img)
        x, y, w, h = faces[0]
        crop = img[y:y+h, x:x+w]

        res = self.uncertainty_est.estimate_uncertainty(crop)
        self.assertIn("mean_probability", res)
        self.assertIn("variance", res)
        self.assertIn("std_dev", res)
        self.assertIn("entropy", res)
        self.assertIn(res["uncertainty_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertEqual(len(res["mc_samples"]), 5)
        print(f"[TEST PASSED] MC-Dropout: Mean={res['mean_probability']*100:.1f}%, std={res['std_dev']:.4f}, Level={res['uncertainty_level']}")

    def test_3_confidence_calibration(self):
        """Test Temperature Scaling and ECE calculation."""
        raw_p = 0.95
        cal_p = self.calibrator.calibrate(raw_p)
        self.assertTrue(0.0 <= cal_p <= 1.0)

        # Test ECE computation
        probs = [0.95, 0.90, 0.85, 0.40, 0.20, 0.10]
        labels = [1, 1, 1, 0, 0, 0]
        ece, bins = ConfidenceCalibrator.compute_ece(probs, labels, num_bins=5)
        brier = ConfidenceCalibrator.compute_brier_score(probs, labels)
        self.assertGreaterEqual(ece, 0.0)
        self.assertGreaterEqual(brier, 0.0)
        print(f"[TEST PASSED] Calibration: Raw={raw_p:.2f} -> Calibrated={cal_p:.2f} | ECE={ece:.4f} | Brier={brier:.4f}")

    def test_4_ablation_study_matrix(self):
        """Test Ablation Study matrix and benchmark retrieval."""
        matrix = self.ablation_engine.get_ablation_matrix()
        self.assertEqual(len(matrix), 6, "Expected 6 ablation experiments (Exp A -> Exp F)")
        self.assertEqual(matrix[0]["id"], "Exp A")
        self.assertEqual(matrix[-1]["id"], "Exp F")
        bench = self.ablation_engine.get_model_benchmarks()
        self.assertIn("accuracy", bench)
        self.assertIn("confusion_matrix", bench)
        print(f"[TEST PASSED] Ablation Matrix: Verified 6 experimental configurations (Exp A -> Exp F)")

    def test_5_compression_robustness(self):
        """Test Video Compression Robustness degradation simulator."""
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mild = self.robustness_evaluator.create_compressed_variant(dummy_frame, "mild")
        mod = self.robustness_evaluator.create_compressed_variant(dummy_frame, "moderate")
        heavy = self.robustness_evaluator.create_compressed_variant(dummy_frame, "heavy")
        self.assertEqual(mild.shape, dummy_frame.shape)
        self.assertEqual(mod.shape, dummy_frame.shape)
        self.assertEqual(heavy.shape, dummy_frame.shape)
        print(f"[TEST PASSED] Compression Robustness: Frame degradation levels generated successfully")


if __name__ == "__main__":
    unittest.main()
