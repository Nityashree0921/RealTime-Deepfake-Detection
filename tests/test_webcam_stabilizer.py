"""
Unit Tests for Webcam Temporal Stabilizer
Verifies stillness, head rotation, fast shaking, small face gating, and deepfake confirmation.
"""

import unittest
import numpy as np
from utils.webcam_stabilizer import WebcamStabilizer


class TestWebcamStabilizer(unittest.TestCase):
    def setUp(self):
        self.stabilizer = WebcamStabilizer()
        self.dummy_face = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        self.frame_shape = (640, 1024, 3)

    def test_scenario_a_person_sitting_still(self):
        """
        Scenario A: Normal face sitting still with authentic high probability.
        Expected: Stabilized REAL with high confidence (>= 90%) and STABLE status.
        """
        bbox = (300, 200, 150, 150)
        for _ in range(5):
            is_ok, reason, _, _ = self.stabilizer.check_face_quality(self.dummy_face, bbox, self.frame_shape)
            state, conf, smoothed, raw, stability, _ = self.stabilizer.update(0.78, bbox, is_ok, reason)

        self.assertEqual(state, "REAL")
        self.assertGreaterEqual(conf, 90.0)
        self.assertEqual(stability, "STABLE")
        print("[TEST PASSED] Scenario A: Person sitting still -> Confirmed REAL (STABLE)")

    def test_scenario_b_transient_head_turn_drop(self):
        """
        Scenario B: Slowly turn head. 1 or 2 frames temporarily dip.
        Expected: Does NOT switch to FAKE; maintains REAL or briefly ANALYZING.
        """
        bbox = (300, 200, 150, 150)
        # Establish stable REAL state
        for _ in range(5):
            is_ok, reason, _, _ = self.stabilizer.check_face_quality(self.dummy_face, bbox, self.frame_shape)
            self.stabilizer.update(0.80, bbox, is_ok, reason)

        # 1-2 transient low probability drops
        for dip_prob in [0.35, 0.42]:
            is_ok, reason, _, _ = self.stabilizer.check_face_quality(self.dummy_face, bbox, self.frame_shape)
            state, conf, smoothed, raw, stability, _ = self.stabilizer.update(dip_prob, bbox, is_ok, reason)
            # Crucial assertion: Must NOT be FAKE!
            self.assertNotEqual(state, "FAKE")

        print("[TEST PASSED] Scenario B: Head rotation dip -> Retained REAL without false positive")

    def test_scenario_c_fast_head_shake_motion_blur(self):
        """
        Scenario C: Quick head shake causes motion blur (Laplacian variance < 35).
        Expected: Quality check flags MOTION_BLUR, holds previous state, displays ANALYZING...
        """
        # Establish initial stable state
        bbox = (300, 200, 150, 150)
        for _ in range(5):
            is_ok, reason, _, _ = self.stabilizer.check_face_quality(self.dummy_face, bbox, self.frame_shape)
            self.stabilizer.update(0.80, bbox, is_ok, reason)

        # Blurry face crop (e.g. constant/smooth image with low variance)
        blurry_face = np.full((150, 150, 3), 128, dtype=np.uint8)
        is_ok, reason, blur_score, _ = self.stabilizer.check_face_quality(blurry_face, bbox, self.frame_shape)

        self.assertFalse(is_ok)
        self.assertIn("MOTION_BLUR", reason)

        state, conf, smoothed, raw, stability, _ = self.stabilizer.update(0.30, bbox, is_ok, reason)
        self.assertEqual(state, "REAL")  # Retains stable state
        self.assertEqual(stability, "ANALYZING...")
        print("[TEST PASSED] Scenario C: Motion blur -> Filtered out; holds stable REAL")

    def test_scenario_d_small_face_gating(self):
        """
        Scenario D: Person moves far away from camera (face < 100px).
        Expected: Quality check flags FACE_TOO_SMALL, avoids classifying pixelated patches.
        """
        tiny_bbox = (300, 200, 75, 75)
        tiny_face = np.random.randint(0, 255, (75, 75, 3), dtype=np.uint8)
        is_ok, reason, _, _ = self.stabilizer.check_face_quality(tiny_face, tiny_bbox, self.frame_shape)

        self.assertFalse(is_ok)
        self.assertIn("FACE_TOO_SMALL", reason)
        print("[TEST PASSED] Scenario D: Small face gating -> Filtered successfully")

    def test_scenario_e_sustained_deepfake_confirmation(self):
        """
        Scenario E: Consistent deepfake spoofing (5+ consecutive low P(REAL) frames).
        Expected: Confirms FAKE after exactly fake_confirmation_frames (5 frames).
        """
        bbox = (300, 200, 150, 150)
        self.stabilizer.reset()

        for frame_idx in range(1, 6):
            is_ok, reason, _, _ = self.stabilizer.check_face_quality(self.dummy_face, bbox, self.frame_shape)
            state, conf, smoothed, raw, stability, _ = self.stabilizer.update(0.15, bbox, is_ok, reason)

            if frame_idx < 5:
                # Still in confirmation phase
                self.assertNotEqual(state, "FAKE")
            else:
                # 5th frame reaches threshold
                self.assertEqual(state, "FAKE")
                self.assertEqual(stability, "STABLE")
                self.assertGreaterEqual(conf, 90.0)

        print("[TEST PASSED] Scenario E: Sustained spoof -> Confirmed FAKE after 5 frames")


if __name__ == "__main__":
    unittest.main()
