"""
Image Deepfake Detection Module
Enhanced with Explainable AI (Grad-CAM), Spatial Evidence Analysis, and MC-Dropout Uncertainty.
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


class ImageDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Deepfake Detection — Explainable AI & Uncertainty")
        self.root.geometry("1220x720")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # State Variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.photo_image = None
        self.photo_orig_face = None
        self.photo_xai_face = None

        # Load AI Model & Research Engines
        self.model = DeepfakeDetector()
        self.xai = XAIExplainer(self.model.model)
        self.uncertainty_est = UncertaintyEstimator(self.model.model, num_passes=10)

        # Load Config
        self.crop_padding = 0.12
        if os.path.exists("models/calibration_config.json"):
            try:
                with open("models/calibration_config.json", "r") as f:
                    cfg = json.load(f)
                    self.crop_padding = float(cfg.get("crop_padding", 0.12))
            except Exception:
                pass

        self.setup_ui()

    def setup_ui(self):
        # Header Label
        header = tk.Label(
            self.root,
            text="🖼 IMAGE DEEPFAKE DETECTION & EXPLAINABILITY (XAI)",
            font=("Arial", 17, "bold"),
            fg=CYAN,
            bg=BG
        )
        header.pack(pady=12)

        # Main Layout Frame
        main_frame = tk.Frame(self.root, bg=BG)
        main_frame.pack(fill="both", expand=True, padx=25)

        # Left Column - Media Preview Container
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
            text="Click 'Upload Image' to Start",
            fill=MUTED,
            font=("Arial", 14, "bold"),
            tags="placeholder"
        )

        # Left Column Action Buttons
        btn_frame = tk.Frame(left_frame, bg=BG)
        btn_frame.pack(anchor="w", pady=12)

        self.btn_upload = tk.Button(
            btn_frame,
            text="📁 Upload Image",
            font=("Arial", 11, "bold"),
            bg=CYAN,
            fg="#050914",
            activebackground="#55E7FF",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            command=self.upload_image
        )
        self.btn_upload.pack(side="left", padx=(0, 12))

        self.btn_detect = tk.Button(
            btn_frame,
            text="▶ Start Detection",
            font=("Arial", 11, "bold"),
            bg=GREEN,
            fg=WHITE,
            activebackground="#54E59B",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=15,
            pady=8,
            state="disabled",
            command=self.start_detection
        )
        self.btn_detect.pack(side="left", padx=0)

        # Right Column - Scientific Results & XAI Card
        self.right_frame = tk.Frame(main_frame, bg=CARD, width=470, height=620)
        self.right_frame.pack(side="right", fill="both", padx=(20, 0), pady=(0, 10))
        self.right_frame.pack_propagate(False)

        card_title = tk.Label(
            self.right_frame,
            text="DETECTION & XAI DIAGNOSTICS",
            font=("Arial", 13, "bold"),
            fg=CYAN,
            bg=CARD
        )
        card_title.pack(pady=10)

        self.result_container = tk.Frame(self.right_frame, bg=CARD)
        self.result_container.pack(fill="both", expand=True, padx=15)

        # 1. Primary Classification Row
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

        # Uncertainty & Risk Row
        self.lbl_uncertainty = tk.Label(
            self.result_container,
            text="Uncertainty (MC-Dropout): --",
            font=("Arial", 10),
            fg=CYAN,
            bg=CARD,
            anchor="w"
        )
        self.lbl_uncertainty.pack(fill="x", pady=1)

        self.lbl_faces = tk.Label(
            self.result_container,
            text="Face Detected: --",
            font=("Arial", 10),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_faces.pack(fill="x", pady=1)

        # 2. XAI Visual Heatmap Section
        tk.Label(
            self.result_container,
            text="EXPLAINABLE AI (GRAD-CAM ACTIVATION)",
            font=("Arial", 10, "bold"),
            fg=CYAN,
            bg=CARD,
            anchor="w"
        ).pack(fill="x", pady=(8, 4))

        self.xai_images_frame = tk.Frame(self.result_container, bg=CARD2)
        self.xai_images_frame.pack(fill="x", pady=4)

        # Original Face Thumbnail
        self.f_orig_box = tk.Frame(self.xai_images_frame, bg=CARD2)
        self.f_orig_box.pack(side="left", expand=True, fill="both", padx=5, pady=5)
        tk.Label(self.f_orig_box, text="Original Face", font=("Arial", 8, "bold"), fg=MUTED, bg=CARD2).pack()
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
            text="AI SPATIAL EVIDENCE:",
            font=("Arial", 10, "bold"),
            fg=WHITE,
            bg=CARD,
            anchor="w"
        ).pack(fill="x", pady=(6, 2))

        self.lbl_evidence = tk.Label(
            self.result_container,
            text="• Awaiting image detection...",
            font=("Arial", 9),
            fg=MUTED,
            bg=CARD,
            justify="left",
            wraplength=430,
            anchor="w"
        )
        self.lbl_evidence.pack(fill="x", pady=2)

        # 4. Telemetry Footer
        self.lbl_time = tk.Label(
            self.result_container,
            text="Processing Time: --",
            font=("Arial", 9),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_time.pack(fill="x", pady=(6, 1))

        self.lbl_model = tk.Label(
            self.result_container,
            text="Model: --",
            font=("Arial", 9),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_model.pack(fill="x", pady=1)

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
            command=self.root.destroy
        )
        btn_close.pack(side="bottom", fill="x", padx=20, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if not file_path:
            return

        self.image_path = file_path
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            messagebox.showerror("Error", "Unable to load selected image format.")
            return

        # Reset UI
        self.lbl_prediction.config(text="Prediction: --", fg=MUTED)
        self.lbl_confidence.config(text="Confidence: --")
        self.lbl_uncertainty.config(text="Uncertainty (MC-Dropout): --")
        self.lbl_faces.config(text="Face Detected: --")
        self.lbl_evidence.config(text="• Click 'Start Detection' to generate Grad-CAM explanation.")
        self.lbl_time.config(text="Processing Time: --")
        self.lbl_model.config(text="Model: --")
        self.lbl_orig_thumb.config(image="")
        self.lbl_xai_thumb.config(image="")

        # Display image on canvas
        cv2_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_rgb)
        self.display_image, self.scale, self.offset_x, self.offset_y = self.resize_to_contain(pil_img)
        self.update_canvas(self.display_image)

        self.btn_detect.config(state="normal")

    def resize_to_contain(self, image, target_w=680, target_h=440):
        orig_w, orig_h = image.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        background = Image.new("RGB", (target_w, target_h), (17, 26, 46))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        background.paste(resized_img, (offset_x, offset_y))
        return background, scale, offset_x, offset_y

    def update_canvas(self, pil_image):
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def start_detection(self):
        if self.original_image is None:
            return

        start_time = time.time()

        h_frame, w_frame = self.original_image.shape[:2]

        # Check if the image is already a face crop (e.g., standard avatar/portrait)
        if h_frame <= 320 and w_frame <= 320 and 0.65 <= (w_frame / max(1, h_frame)) <= 1.5:
            faces = [(0, 0, w_frame, h_frame)]
        else:
            try:
                faces = detect_faces(self.original_image)
            except Exception as e:
                messagebox.showerror("Error", f"Face detection failed: {e}")
                return

        # Fallback for portrait images where face detector missed
        if len(faces) == 0:
            if w_frame >= 60 and h_frame >= 60 and 0.5 <= (w_frame / max(1, h_frame)) <= 2.0:
                faces = [(0, 0, w_frame, h_frame)]
            else:
                self.lbl_prediction.config(text="Prediction: NO FACE DETECTED", fg=ORANGE)
                self.lbl_faces.config(text="Face Detected: NO")
                self.lbl_time.config(text=f"Processing Time: {time.time() - start_time:.2f}s")
                self.lbl_model.config(text=f"Model: {os.path.basename(self.model.model_path if hasattr(self.model, 'model_path') else 'MobileNetV2')}")
                return

        annotated_display = self.display_image.copy()
        draw = ImageDraw.Draw(annotated_display)

        total_faces = len(faces)
        main_label = "REAL"
        main_confidence = 0.0
        primary_face_crop = None
        lowest_p_real = 1.0

        for idx, (x, y, w, h) in enumerate(faces):
            padding = int(self.crop_padding * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_frame, x + w + padding)
            y2 = min(h_frame, y + h + padding)

            face_crop = self.original_image[y1:y2, x1:x2]
            if face_crop.size == 0:
                face_crop = self.original_image[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

            label, confidence, p_real = self.model.predict(face_crop, return_raw=True)

            if p_real < lowest_p_real or primary_face_crop is None:
                lowest_p_real = p_real
                primary_face_crop = face_crop

            disp_x1 = int(x * self.scale + self.offset_x)
            disp_y1 = int(y * self.scale + self.offset_y)
            disp_x2 = int((x + w) * self.scale + self.offset_x)
            disp_y2 = int((y + h) * self.scale + self.offset_y)

            color_hex = GREEN if label == "REAL" else RED if label == "FAKE" else ORANGE
            if label == "FAKE":
                main_label = "FAKE"
                main_confidence = max(main_confidence, confidence)
            elif label == "REAL" and main_label != "FAKE":
                main_label = "REAL"
                main_confidence = max(main_confidence, confidence)
            elif idx == 0 and main_label != "FAKE":
                main_label = label
                main_confidence = confidence

            draw.rectangle([disp_x1, disp_y1, disp_x2, disp_y2], outline=color_hex, width=3)
            draw.text((disp_x1, max(0, disp_y1 - 18)), f"{label} {confidence:.1f}%", fill=color_hex)

        self.update_canvas(annotated_display)

        # -----------------------------------------------------
        # RESEARCH COMPUTATION: GRAD-CAM XAI & MC-DROPOUT
        # -----------------------------------------------------
        if primary_face_crop is not None:
            # 1. Grad-CAM XAI
            _, xai_overlay_bgr, bullets = self.xai.generate_gradcam(primary_face_crop, pred_class=main_label)

            # Update XAI thumbnails
            orig_rgb = cv2.cvtColor(primary_face_crop, cv2.COLOR_BGR2RGB)
            xai_rgb = cv2.cvtColor(xai_overlay_bgr, cv2.COLOR_BGR2RGB)

            pil_orig = Image.fromarray(orig_rgb).resize((105, 105), Image.Resampling.LANCZOS)
            pil_xai = Image.fromarray(xai_rgb).resize((105, 105), Image.Resampling.LANCZOS)

            self.photo_orig_face = ImageTk.PhotoImage(pil_orig)
            self.photo_xai_face = ImageTk.PhotoImage(pil_xai)

            self.lbl_orig_thumb.config(image=self.photo_orig_face)
            self.lbl_xai_thumb.config(image=self.photo_xai_face)

            # Evidence text
            bullet_text = "\n".join([f"• {b}" for b in bullets])
            self.lbl_evidence.config(text=bullet_text)

            # 2. MC-Dropout Uncertainty Estimation
            unc_dict = self.uncertainty_est.estimate_uncertainty(primary_face_crop)
            unc_color = GREEN if unc_dict["uncertainty_level"] == "LOW" else ORANGE if unc_dict["uncertainty_level"] == "MEDIUM" else RED
            self.lbl_uncertainty.config(
                text=f"Uncertainty: {unc_dict['uncertainty_level']} (σ={unc_dict['std_dev']:.3f}, H={unc_dict['entropy']:.2f})\nRisk: {unc_dict['risk_level']}",
                fg=unc_color
            )

        elapsed = time.time() - start_time
        lbl_color = GREEN if main_label == "REAL" else RED if main_label == "FAKE" else ORANGE
        self.lbl_prediction.config(text=f"Prediction: {main_label}", fg=lbl_color)
        self.lbl_confidence.config(text=f"Confidence: {main_confidence:.2f}%")
        self.lbl_faces.config(text=f"Face Detected: YES ({total_faces})")
        self.lbl_time.config(text=f"Processing Time: {elapsed:.2f} seconds")
        self.lbl_model.config(text=f"Model: {os.path.basename(self.model.config_path or 'MobileNetV2_V6')}")

        # Save to DB & CSV
        now = datetime.now()
        try:
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), main_label, f"{main_confidence:.2f}"])
            save_detection(now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), main_label, float(main_confidence))
        except Exception as e:
            print("Log save error:", e)

        # PDF Report
        try:
            if main_label == "FAKE":
                fn = os.path.join(SCREENSHOTS_DIR, "IMG_" + now.strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(fn, self.original_image)
                generate_report(main_label, main_confidence, image_path=fn)
            else:
                generate_report(main_label, main_confidence)
        except Exception as e:
            print("Report error:", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDetectorApp(root)
    root.mainloop()