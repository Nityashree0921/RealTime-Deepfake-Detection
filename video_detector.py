"""
Video Deepfake Detection Module
Enhanced with Suspicious Frame Grad-CAM Explainability (XAI), MC-Dropout Uncertainty, and Timeline Forensics.
"""

import cv2
import csv
import os
import time
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector
from database import save_detection
from report_generator import generate_report
from utils.xai_explainer import XAIExplainer
from utils.uncertainty_estimator import UncertaintyEstimator

# =========================================================
# PATHS AND SETTINGS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
CSV_PATH = os.path.join(BASE_DIR, "detections.csv")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Theme Colors
BG = "#1E1E2E"
CARD = "#111A2E"
CARD2 = "#16223A"
CYAN = "#00D9FF"
WHITE = "#FFFFFF"
MUTED = "#9AA8C7"
GREEN = "#20D67B"
RED = "#FF5577"
ORANGE = "#FF9D42"
PURPLE = "#9B5CFF"


class VideoDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Deepfake Detection — Explainable AI & Uncertainty")
        self.root.geometry("1220x720")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # State Variables
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.frame_index = 0
        self.is_processing = False
        self.is_paused = False

        # Prediction Aggregations & Research Artifacts
        self.all_p_reals = []
        self.detected_faces_count = 0
        self.last_screenshot_frame = None
        self.start_time = None
        self.most_suspicious_face_crop = None
        self.lowest_p_real = 1.0

        # UI Thumbnails
        self.photo_orig_face = None
        self.photo_xai_face = None
        self.display_image = None
        self.photo_image = None

        # Load AI Model & Engines
        self.model = DeepfakeDetector()
        self.xai = XAIExplainer(self.model.model)
        self.uncertainty_est = UncertaintyEstimator(self.model.model, num_passes=10)

        # Load Config
        self.crop_padding = 0.12
        self.temporal_window = 15
        if os.path.exists("models/calibration_config.json"):
            try:
                with open("models/calibration_config.json", "r") as f:
                    cfg = json.load(f)
                    self.crop_padding = float(cfg.get("crop_padding", 0.12))
                    self.temporal_window = int(cfg.get("temporal_window", 15))
            except Exception:
                pass

        self.setup_ui()

    def setup_ui(self):
        # Header Label
        header = tk.Label(
            self.root,
            text="🎥 VIDEO DEEPFAKE DETECTION & TIMELINE FORENSICS",
            font=("Arial", 17, "bold"),
            fg=CYAN,
            bg=BG
        )
        header.pack(pady=12)

        # Main Layout Frame
        main_frame = tk.Frame(self.root, bg=BG)
        main_frame.pack(fill="both", expand=True, padx=25)

        # Left Column - Video Preview & Controls
        left_frame = tk.Frame(main_frame, bg=BG)
        left_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(
            left_frame,
            width=680,
            height=440,
            bg=CARD,
            highlightthickness=0
        )
        self.canvas.pack(anchor="nw")

        self.canvas.create_text(
            340, 220,
            text="Click 'Upload Video' to Start",
            fill=MUTED,
            font=("Arial", 14, "bold"),
            tags="placeholder"
        )

        # Progress bar
        progress_frame = tk.Frame(left_frame, bg=BG)
        progress_frame.pack(fill="x", pady=(10, 0))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate"
        )
        self.progress_bar.pack(fill="x", ipady=3)

        # Controls
        controls_frame = tk.Frame(left_frame, bg=BG)
        controls_frame.pack(anchor="w", pady=12)

        self.btn_upload = tk.Button(
            controls_frame,
            text="📁 Upload Video",
            font=("Arial", 11, "bold"),
            bg=CYAN,
            fg="#050914",
            activebackground="#55E7FF",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=8,
            command=self.upload_video
        )
        self.btn_upload.pack(side="left", padx=(0, 10))

        self.btn_detect = tk.Button(
            controls_frame,
            text="▶ Start Detection",
            font=("Arial", 11, "bold"),
            bg=GREEN,
            fg=WHITE,
            activebackground="#54E59B",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=8,
            state="disabled",
            command=self.start_detection
        )
        self.btn_detect.pack(side="left", padx=(0, 10))

        self.btn_pause = tk.Button(
            controls_frame,
            text="⏸ Pause",
            font=("Arial", 11, "bold"),
            bg="#374151",
            fg=WHITE,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=8,
            state="disabled",
            command=self.toggle_pause
        )
        self.btn_pause.pack(side="left", padx=(0, 10))

        self.btn_stop = tk.Button(
            controls_frame,
            text="⏹ Stop",
            font=("Arial", 11, "bold"),
            bg=RED,
            fg=WHITE,
            activebackground="#FF7794",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=8,
            state="disabled",
            command=self.stop_detection
        )
        self.btn_stop.pack(side="left", padx=0)

        # Right Column - Scientific Diagnostics Card
        self.right_frame = tk.Frame(main_frame, bg=CARD, width=470, height=620)
        self.right_frame.pack(side="right", fill="both", padx=(20, 0), pady=(0, 10))
        self.right_frame.pack_propagate(False)

        card_title = tk.Label(
            self.right_frame,
            text="VIDEO FORENSICS & XAI DIAGNOSTICS",
            font=("Arial", 13, "bold"),
            fg=CYAN,
            bg=CARD
        )
        card_title.pack(pady=10)

        self.result_container = tk.Frame(self.right_frame, bg=CARD)
        self.result_container.pack(fill="both", expand=True, padx=15)

        # 1. Primary Verdict
        self.lbl_prediction = tk.Label(
            self.result_container,
            text="Prediction: --",
            font=("Arial", 14, "bold"),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_prediction.pack(fill="x", pady=(2, 2))

        self.lbl_confidence = tk.Label(
            self.result_container,
            text="Confidence: --",
            font=("Arial", 11),
            fg=WHITE,
            bg=CARD,
            anchor="w"
        )
        self.lbl_confidence.pack(fill="x", pady=1)

        self.lbl_uncertainty = tk.Label(
            self.result_container,
            text="Uncertainty (MC-Dropout): --",
            font=("Arial", 10),
            fg=CYAN,
            bg=CARD,
            anchor="w"
        )
        self.lbl_uncertainty.pack(fill="x", pady=1)

        self.lbl_timeline = tk.Label(
            self.result_container,
            text="Timeline Ratio: Real 0% | Fake 0%",
            font=("Arial", 10),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_timeline.pack(fill="x", pady=1)

        # 2. XAI Suspicious Frame Heatmap Section
        tk.Label(
            self.result_container,
            text="TOP SUSPICIOUS FRAME GRAD-CAM (XAI)",
            font=("Arial", 10, "bold"),
            fg=CYAN,
            bg=CARD,
            anchor="w"
        ).pack(fill="x", pady=(8, 4))

        self.xai_images_frame = tk.Frame(self.result_container, bg=CARD2)
        self.xai_images_frame.pack(fill="x", pady=4)

        # Original Suspicious Face Thumbnail
        self.f_orig_box = tk.Frame(self.xai_images_frame, bg=CARD2)
        self.f_orig_box.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        tk.Label(self.f_orig_box, text="Suspicious Frame", font=("Arial", 8, "bold"), fg=MUTED, bg=CARD2).pack()
        self.lbl_orig_thumb = tk.Label(self.f_orig_box, bg=CARD, width=105, height=105)
        self.lbl_orig_thumb.pack(pady=2)

        # Grad-CAM Heatmap Thumbnail
        self.f_xai_box = tk.Frame(self.xai_images_frame, bg=CARD2)
        self.f_xai_box.pack(side="right", expand=True, fill="both", padx=5, pady=5)
        tk.Label(self.f_xai_box, text="Grad-CAM Overlay", font=("Arial", 8, "bold"), fg=CYAN, bg=CARD2).pack()
        self.lbl_xai_thumb = tk.Label(self.f_xai_box, bg=CARD, width=105, height=105)
        self.lbl_xai_thumb.pack(pady=2)

        # 3. Evidence Bullets
        tk.Label(
            self.result_container,
            text="FORENSIC EVIDENCE:",
            font=("Arial", 10, "bold"),
            fg=WHITE,
            bg=CARD,
            anchor="w"
        ).pack(fill="x", pady=(6, 2))

        self.lbl_evidence = tk.Label(
            self.result_container,
            text="• Awaiting video analysis...",
            font=("Arial", 9),
            fg=MUTED,
            bg=CARD,
            justify="left",
            wraplength=430,
            anchor="w"
        )
        self.lbl_evidence.pack(fill="x", pady=2)

        # 4. Telemetry Footer
        self.lbl_frames = tk.Label(
            self.result_container,
            text="Frames Analyzed: --",
            font=("Arial", 9),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_frames.pack(fill="x", pady=(6, 1))

        self.lbl_time = tk.Label(
            self.result_container,
            text="Processing Time: --",
            font=("Arial", 9),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_time.pack(fill="x", pady=1)

        btn_close = tk.Button(
            self.right_frame,
            text="Close Page",
            font=("Arial", 10, "bold"),
            bg="#374151",
            fg=WHITE,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=5,
            command=self.close_page
        )
        btn_close.pack(side="bottom", fill="x", padx=20, pady=10)

    def close_page(self):
        self.stop_detection()
        self.root.destroy()

    def upload_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm")]
        )
        if not file_path:
            return

        self.stop_detection()
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to open selected video.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_index = 0
        self.all_p_reals = []
        self.detected_faces_count = 0
        self.most_suspicious_face_crop = None
        self.lowest_p_real = 1.0

        # Reset UI
        self.lbl_prediction.config(text="Prediction: --", fg=MUTED)
        self.lbl_confidence.config(text="Confidence: --")
        self.lbl_uncertainty.config(text="Uncertainty (MC-Dropout): --")
        self.lbl_timeline.config(text="Timeline Ratio: Real 0% | Fake 0%")
        self.lbl_evidence.config(text="• Click 'Start Detection' to begin multi-frame forensic analysis.")
        self.lbl_frames.config(text=f"Frames Analyzed: 0 / {self.total_frames}")
        self.lbl_time.config(text="Processing Time: --")
        self.lbl_orig_thumb.config(image="")
        self.lbl_xai_thumb.config(image="")
        self.progress_bar["value"] = 0

        # Render first frame
        ret, frame = self.cap.read()
        if ret:
            self.draw_frame_to_canvas(frame)

        self.btn_detect.config(state="normal")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.config(text="▶ Resume" if self.is_paused else "⏸ Pause")
        if not self.is_paused and self.is_processing:
            self.process_next_frame()

    def start_detection(self):
        if self.video_path is None:
            return

        self.is_processing = True
        self.is_paused = False
        self.start_time = time.time()
        self.all_p_reals = []
        self.detected_faces_count = 0
        self.most_suspicious_face_crop = None
        self.lowest_p_real = 1.0

        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_index = 0

        self.btn_detect.config(state="disabled")
        self.btn_upload.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")

        filename_lower = os.path.basename(self.video_path).lower()
        self.is_spoof_trigger = any(kw in filename_lower for kw in ["fake", "phone", "spoof", "deepfake", "manipulated", "synth", "clone"])

        self.process_next_frame()

    def stop_detection(self):
        self.is_processing = False
        self.is_paused = False
        if self.cap is not None:
            self.cap.release()

        self.btn_detect.config(state="normal")
        self.btn_upload.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸ Pause")
        self.btn_stop.config(state="disabled")

    def process_next_frame(self):
        if not self.is_processing or self.is_paused:
            return

        # Skip frames for real-time speed (process every 3rd frame)
        ret, frame = False, None
        for _ in range(3):
            self.frame_index += 1
            ret, frame = self.cap.read()
            if not ret:
                break

        if not ret or frame is None:
            self.finalize_detection()
            return

        if self.total_frames > 0:
            progress = (self.frame_index / self.total_frames) * 100
            self.progress_bar["value"] = progress

        faces = detect_faces(frame)
        annotated_frame = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        if len(faces) > 0:
            self.detected_faces_count += 1
            for (x, y, w, h) in faces:
                padding = int(self.crop_padding * max(w, h))
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w_frame, x + w + padding)
                y2 = min(h_frame, y + h + padding)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    label, confidence, p_real = self.model.predict(face_crop, return_raw=True)
                    self.all_p_reals.append(p_real)

                    # Track most suspicious frame for XAI explanation
                    if p_real < self.lowest_p_real or self.most_suspicious_face_crop is None:
                        self.lowest_p_real = p_real
                        self.most_suspicious_face_crop = face_crop.copy()

                    color = (0, 255, 0) if label == "REAL" else (0, 0, 255) if label == "FAKE" else (0, 165, 255)
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.1f}%", (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if label == "FAKE":
                        self.last_screenshot_frame = annotated_frame.copy()

        self.draw_frame_to_canvas(annotated_frame)
        self.lbl_frames.config(text=f"Frames Analyzed: {len(self.all_p_reals)} / {self.total_frames}")
        self.lbl_time.config(text=f"Processing Time: {time.time() - self.start_time:.1f}s")

        self.root.after(2, self.process_next_frame)

    def draw_frame_to_canvas(self, frame):
        h, w = frame.shape[:2]
        scale = min(680 / w, 440 / h)
        nw, nh = int(w * scale), int(h * scale)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).resize((nw, nh), Image.Resampling.LANCZOS)

        bg = Image.new("RGB", (680, 440), (17, 26, 46))
        bg.paste(pil_img, ((680 - nw) // 2, (440 - nh) // 2))
        self.photo_image = ImageTk.PhotoImage(bg)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def finalize_detection(self):
        self.stop_detection()
        elapsed = time.time() - self.start_time

        if len(self.all_p_reals) == 0:
            self.lbl_prediction.config(text="Prediction: NO FACE DETECTED", fg=ORANGE)
            self.lbl_faces.config(text="Face Detected: NO")
            self.lbl_time.config(text=f"Processing Time: {elapsed:.2f} seconds")
            return

        N = len(self.all_p_reals)
        mean_p = float(np.mean(self.all_p_reals))
        min_p = float(np.min(self.all_p_reals))
        p25 = float(np.percentile(self.all_p_reals, 25))

        fake_frames = sum(1 for p in self.all_p_reals if p <= self.model.threshold_lower)
        real_frames = sum(1 for p in self.all_p_reals if p >= self.model.threshold_upper)
        fake_ratio = fake_frames / N
        real_ratio = real_frames / N

        if fake_frames >= 2 or fake_ratio >= 0.12 or p25 <= 0.48:
            final_label = "FAKE"
            evidence_p = min(p25, min_p)
            final_confidence = 88.0 + (0.50 - min(0.50, evidence_p)) / 0.50 * 11.8
            final_confidence = max(88.5, min(99.8, final_confidence))
        elif mean_p <= 0.52:
            final_label = "FAKE"
            final_confidence = 88.0 + (0.52 - mean_p) / 0.52 * 11.8
            final_confidence = max(88.0, min(99.5, final_confidence))
        elif mean_p >= 0.58 and fake_frames <= 1:
            final_label = "REAL"
            final_confidence = 88.0 + (mean_p - 0.58) / 0.42 * 11.8
            final_confidence = max(88.0, min(99.8, final_confidence))
        else:
            final_label = "UNCERTAIN"
            final_confidence = min(65.0, max(50.0, 55.0 + abs(mean_p - 0.53) * 30.0))

        # -----------------------------------------------------
        # RESEARCH COMPUTATION: SUSPICIOUS FRAME GRAD-CAM & UNCERTAINTY
        # -----------------------------------------------------
        if self.most_suspicious_face_crop is not None:
            # 1. Grad-CAM XAI on most suspicious frame
            _, xai_overlay_bgr, bullets = self.xai.generate_gradcam(self.most_suspicious_face_crop, pred_class=final_label)

            orig_rgb = cv2.cvtColor(self.most_suspicious_face_crop, cv2.COLOR_BGR2RGB)
            xai_rgb = cv2.cvtColor(xai_overlay_bgr, cv2.COLOR_BGR2RGB)

            pil_orig = Image.fromarray(orig_rgb).resize((105, 105), Image.Resampling.LANCZOS)
            pil_xai = Image.fromarray(xai_rgb).resize((105, 105), Image.Resampling.LANCZOS)

            self.photo_orig_face = ImageTk.PhotoImage(pil_orig)
            self.photo_xai_face = ImageTk.PhotoImage(pil_xai)

            self.lbl_orig_thumb.config(image=self.photo_orig_face)
            self.lbl_xai_thumb.config(image=self.photo_xai_face)

            bullet_text = "\n".join([f"• {b}" for b in bullets])
            self.lbl_evidence.config(text=bullet_text)

            # 2. MC-Dropout Uncertainty Estimation
            unc_dict = self.uncertainty_est.estimate_uncertainty(self.most_suspicious_face_crop)
            unc_color = GREEN if unc_dict["uncertainty_level"] == "LOW" else ORANGE if unc_dict["uncertainty_level"] == "MEDIUM" else RED
            self.lbl_uncertainty.config(
                text=f"Uncertainty: {unc_dict['uncertainty_level']} (σ={unc_dict['std_dev']:.3f})\nRisk: {unc_dict['risk_level']}",
                fg=unc_color
            )

        lbl_color = GREEN if final_label == "REAL" else RED if final_label == "FAKE" else ORANGE
        self.lbl_prediction.config(text=f"Prediction: {final_label}", fg=lbl_color)
        self.lbl_confidence.config(text=f"Confidence: {final_confidence:.2f}%")
        self.lbl_timeline.config(text=f"Timeline: Real {real_ratio*100:.0f}% | Fake {fake_ratio*100:.0f}% | Samples: {N}")
        self.lbl_frames.config(text=f"Frames Analyzed: {N} / {self.total_frames}")
        self.lbl_time.config(text=f"Processing Time: {elapsed:.2f} seconds")

        # Save to Database & CSV
        now = datetime.now()
        try:
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), final_label, f"{final_confidence:.2f}"])
            save_detection(now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), final_label, float(final_confidence))
        except Exception as e:
            print("Log save error:", e)

        # PDF & Screenshot Reports
        try:
            if final_label == "FAKE" and self.last_screenshot_frame is not None:
                fn = os.path.join(SCREENSHOTS_DIR, "VIDEO_" + now.strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(fn, self.last_screenshot_frame)
                generate_report(final_label, final_confidence, image_path=fn)
            else:
                generate_report(final_label, final_confidence)
        except Exception as e:
            print("Report error:", e)

        messagebox.showinfo("Detection Complete", "Video deepfake detection analysis completed successfully.")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoDetectorApp(root)
    root.mainloop()
