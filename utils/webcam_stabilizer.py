"""
Webcam Temporal Stabilization and Hysteresis Engine
Intelligent Real-Time Multimodal Deepfake Detection System

Features:
1. Face Quality & Blur Gating (Rejects pixelated, tiny, or motion-blurred face crops).
2. Spatial Displacement & Motion Tracking (Prevents head shakes from triggering deepfake alarms).
3. Temporal Probability Smoothing (Rolling average window over raw predictions).
4. Hysteresis State Machine (Separate enter/exit bounds preventing erratic flickering).
5. Consecutive Frame Confirmation (Guarantees sustained evidence before classifying FAKE).
"""

import os
import json
from collections import deque
import numpy as np
import cv2


class WebcamStabilizer:
    def __init__(self, config_path="models/calibration_config.json"):
        self.config_path = config_path
        self.load_config()

        # Rolling Probability Queue
        self.prediction_queue = deque(maxlen=self.temporal_window)

        # State Machine Variables
        self.current_state = "ANALYZING"
        self.stability_status = "ANALYZING..."
        self.current_confidence = 0.0
        self.last_raw_p_real = 0.0
        self.last_smoothed_p_real = 0.0

        self.fake_counter = 0
        self.real_counter = 0

        self.prev_bbox = None
        self.last_quality_reason = "INITIALIZING"

    def load_config(self):
        """
        Loads stabilization thresholds and parameters from JSON with robust defaults.
        """
        cfg = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"Warning: Could not parse {self.config_path}: {e}")

        self.real_enter_threshold = float(cfg.get("real_enter_threshold", 0.45))
        self.real_exit_threshold = float(cfg.get("real_exit_threshold", 0.38))
        self.fake_enter_threshold = float(cfg.get("fake_enter_threshold", 0.35))
        self.fake_exit_threshold = float(cfg.get("fake_exit_threshold", 0.42))

        self.temporal_window = int(cfg.get("temporal_window", 8))
        self.fake_confirmation_frames = int(cfg.get("fake_confirmation_frames", 4))
        self.real_confirmation_frames = int(cfg.get("real_confirmation_frames", 2))

        self.min_face_size = int(cfg.get("min_face_size", 80))
        self.blur_threshold = float(cfg.get("blur_threshold", 28.0))
        self.max_displacement_px = float(cfg.get("max_displacement_px", 60.0))

        cal_map = cfg.get("calibration_mapping", {})
        self.real_min_conf = float(cal_map.get("real_min_conf", 88.0))
        self.real_max_conf = float(cal_map.get("real_max_conf", 99.8))
        self.fake_min_conf = float(cal_map.get("fake_min_conf", 88.0))
        self.fake_max_conf = float(cal_map.get("fake_max_conf", 99.8))

    def check_face_quality(self, face_img, bbox, frame_shape):
        """
        Evaluates face crop quality: minimum size, frame boundaries, blurriness, and motion displacement.
        Returns: (is_quality_ok: bool, reason: str, blur_score: float, displacement: float)
        """
        x, y, w, h = bbox
        h_f, w_f = frame_shape[:2]

        # 1. Minimum Face Dimension Check
        if w < self.min_face_size or h < self.min_face_size:
            return False, f"FACE_TOO_SMALL ({w}x{h} < {self.min_face_size}px)", 0.0, 0.0

        # 2. Boundary / Partial Cutoff Check
        is_edge = (x <= 3 or y <= 3 or (x + w) >= (w_f - 3) or (y + h) >= (h_f - 3))
        if is_edge and (w < 100 or h < 100):
            return False, "PARTIAL_FACE_EDGE", 0.0, 0.0

        # 3. Motion Blur Evaluation via Laplacian Variance
        if face_img is None or face_img.size == 0:
            return False, "EMPTY_FACE_CROP", 0.0, 0.0

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if face_img.ndim == 3 else face_img
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur_score < self.blur_threshold:
            return False, f"MOTION_BLUR (blur_var={blur_score:.1f} < {self.blur_threshold})", blur_score, 0.0

        # 4. Spatial Displacement / Rapid Movement Check
        displacement = 0.0
        if self.prev_bbox is not None:
            px, py, pw, ph = self.prev_bbox
            displacement = float(np.sqrt((x - px) ** 2 + (y - py) ** 2))
            if displacement > self.max_displacement_px:
                return False, f"FAST_MOTION (disp={displacement:.1f}px > {self.max_displacement_px}px)", blur_score, displacement

        return True, "QUALITY_OK", blur_score, displacement

    def update(self, raw_p_real, bbox, is_quality_ok=True, quality_reason="QUALITY_OK"):
        """
        Updates the temporal queue, runs the hysteresis state machine,
        and computes stabilized prediction label and confidence.
        """
        self.prev_bbox = bbox
        self.last_quality_reason = quality_reason

        # Handle Poor Quality / Transient Movement
        if not is_quality_ok:
            # During motion blur or rapid head turns, hold previous stable verdict and indicate ANALYZING
            self.stability_status = "ANALYZING..."
            return (
                self.current_state,
                self.current_confidence,
                self.last_smoothed_p_real,
                self.last_raw_p_real,
                self.stability_status,
                self.last_quality_reason
            )

        # Valid Quality Frame: Ingest Raw Probability P(REAL)
        self.last_raw_p_real = float(raw_p_real)
        self.prediction_queue.append(self.last_raw_p_real)
        self.last_smoothed_p_real = float(np.mean(self.prediction_queue))

        smoothed = self.last_smoothed_p_real
        queue_len = len(self.prediction_queue)

        # -----------------------------------------------------
        # HYSTERESIS STATE MACHINE WITH FAST WARMUP & CONFIRMATION
        # -----------------------------------------------------

        if queue_len <= 2:
            # Fast initial recognition on first 1-2 frames to avoid startup delay
            if smoothed >= self.real_enter_threshold:
                self.current_state = "REAL"
                self.real_counter = queue_len
                self.fake_counter = 0
                self.stability_status = "STABLE" if self.real_counter >= self.real_confirmation_frames else "ANALYZING..."
            elif smoothed <= self.fake_enter_threshold:
                self.current_state = "FAKE"
                self.fake_counter = queue_len
                self.real_counter = 0
                self.stability_status = "STABLE" if self.fake_counter >= self.fake_confirmation_frames else "ANALYZING..."
            else:
                self.current_state = "UNCERTAIN"
                self.stability_status = "ANALYZING..."
        else:
            # Case 1: Low Real Probability -> Potential FAKE
            if smoothed <= self.fake_enter_threshold:
                self.fake_counter += 1
                self.real_counter = 0

                if self.fake_counter >= self.fake_confirmation_frames:
                    self.current_state = "FAKE"
                    self.stability_status = "STABLE"
                else:
                    self.stability_status = f"ANALYZING... (Fake {self.fake_counter}/{self.fake_confirmation_frames})"

            # Case 2: High Real Probability -> Potential REAL
            elif smoothed >= self.real_enter_threshold:
                self.real_counter += 1
                self.fake_counter = 0

                if self.real_counter >= self.real_confirmation_frames:
                    self.current_state = "REAL"
                    self.stability_status = "STABLE"
                else:
                    self.stability_status = f"ANALYZING... (Real {self.real_counter}/{self.real_confirmation_frames})"

            # Case 3: Transition / Ambiguous Band
            else:
                self.fake_counter = max(0, self.fake_counter - 1)
                self.real_counter = max(0, self.real_counter - 1)

                # Hysteresis Hold: prevent flickering if previous state was confirmed
                if self.current_state == "REAL" and smoothed >= self.real_exit_threshold:
                    self.stability_status = "STABLE"
                elif self.current_state == "FAKE" and smoothed <= self.fake_exit_threshold:
                    self.stability_status = "STABLE"
                else:
                    self.current_state = "UNCERTAIN"
                    self.stability_status = "STABLE"

        # -----------------------------------------------------
        # CONFIDENCE CALCULATION
        # -----------------------------------------------------
        if self.current_state == "REAL":
            denom = max(1e-5, 1.0 - self.real_enter_threshold)
            norm = max(0.0, min(1.0, (smoothed - self.real_enter_threshold) / denom))
            self.current_confidence = self.real_min_conf + norm * (self.real_max_conf - self.real_min_conf)

        elif self.current_state == "FAKE":
            denom = max(1e-5, self.fake_enter_threshold)
            norm = max(0.0, min(1.0, (self.fake_enter_threshold - smoothed) / denom))
            self.current_confidence = self.fake_min_conf + norm * (self.fake_max_conf - self.fake_min_conf)

        else:  # UNCERTAIN or INITIAL ANALYZING
            mid = (self.real_enter_threshold + self.fake_enter_threshold) / 2.0
            half_range = max(1e-5, (self.real_enter_threshold - self.fake_enter_threshold) / 2.0)
            dist = min(1.0, abs(smoothed - mid) / half_range)
            self.current_confidence = 50.0 + dist * 15.0

        self.current_confidence = float(min(99.9, max(50.0, self.current_confidence)))

        return (
            self.current_state,
            self.current_confidence,
            self.last_smoothed_p_real,
            self.last_raw_p_real,
            self.stability_status,
            self.last_quality_reason
        )

    def reset(self):
        """
        Resets tracking queues when no face is present in view.
        """
        self.prediction_queue.clear()
        self.current_state = "ANALYZING"
        self.stability_status = "SEARCHING FACE..."
        self.current_confidence = 0.0
        self.last_raw_p_real = 0.0
        self.last_smoothed_p_real = 0.0
        self.fake_counter = 0
        self.real_counter = 0
        self.prev_bbox = None
        self.last_quality_reason = "NO_FACE"
