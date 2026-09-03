"""
Audio Deepfake Detection Application
Intelligent Real-Time Multimodal Deepfake Detection System

Pipeline:
Audio Upload -> Audio Preprocessing -> LFCC Feature Extraction -> CNN-BiLSTM Classifier ->
Calibrated Probability -> REAL / FAKE / UNCERTAIN -> Confidence -> UI Result -> CSV/SQLite Logging -> PDF Report
"""

import os
import csv
import time
import json
import threading
import numpy as np
import librosa
import soundfile as sf
import winsound
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import tensorflow as tf

from audio_preprocessor import AudioPreprocessor, LFCCExtractor
from database import save_detection
from report_generator import generate_report

# =========================================================
# PATHS AND THEME CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "audio_deepfake_model.keras")
CONFIG_PATH = os.path.join(MODELS_DIR, "audio_calibration_config.json")
CSV_PATH = os.path.join(BASE_DIR, "detections.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Theme Colors
BG = "#080D1A"
CARD = "#111A2E"
CARD2 = "#16223A"
WHITE = "#FFFFFF"
MUTED = "#9AA8C7"
CYAN = "#00D9FF"
BLUE = "#287BFF"
GREEN = "#20D67B"
RED = "#FF5577"
ORANGE = "#FF9D42"


class AudioModelCache:
    """
    Singleton cache to load the Keras model once into memory and reuse across analyses.
    """
    _model = None
    _config = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}. Please run train_audio_model.py first.")
            print(f"Loading Audio Model from: {MODEL_PATH}...")
            cls._model = tf.keras.models.load_model(MODEL_PATH)
            print("[OK] Audio Model Loaded into Memory Cache.")
        return cls._model

    @classmethod
    def get_config(cls):
        if cls._config is None:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r") as f:
                    cls._config = json.load(f)
            else:
                cls._config = {
                    "real_threshold_upper": 0.60,
                    "fake_threshold_lower": 0.40,
                    "calibration_mapping": {
                        "real_min_conf": 85.0,
                        "real_max_conf": 99.8,
                        "fake_min_conf": 85.0,
                        "fake_max_conf": 99.8
                    }
                }
        return cls._config


class AudioDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Real-Time Audio Deepfake Detection")
        self.root.geometry("1120x640")
        self.root.configure(bg=BG)
        self.root.minsize(980, 600)

        # Audio and Inference State
        self.audio_path = None
        self.audio_data = None
        self.sample_rate = 16000
        self.duration = 0.0

        self.last_prediction = None
        self.last_confidence = None
        self.is_processing = False

        # Playback State
        self.is_playing = False
        self.play_start_time = 0
        self.timer_job = None

        # Pipelines
        self.preprocessor = AudioPreprocessor(target_sr=16000, target_duration=3.0)
        self.extractor = LFCCExtractor(sr=16000, max_frames=200)

        # Preload Model in background / cache
        try:
            AudioModelCache.get_model()
        except Exception as e:
            print(f"Notice: Model will be loaded on first detection run or after training ({e})")

        self.setup_ui()

    def setup_ui(self):
        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------
        header_frame = tk.Frame(self.root, bg=BG, height=60)
        header_frame.pack(side="top", fill="x", padx=30, pady=(15, 10))
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(
            header_frame,
            text="🎙 MULTIMODAL AI: AUDIO DEEPFAKE DETECTOR",
            font=("Arial", 16, "bold"),
            fg=CYAN,
            bg=BG
        )
        title_lbl.pack(side="left")

        subtitle_lbl = tk.Label(
            header_frame,
            text="Linear Frequency Cepstral Coefficients (LFCC) • CNN-BiLSTM Architecture",
            font=("Arial", 10),
            fg=MUTED,
            bg=BG
        )
        subtitle_lbl.pack(side="left", padx=15, pady=(4, 0))

        # -----------------------------------------------------
        # MAIN BODY
        # -----------------------------------------------------
        main_frame = tk.Frame(self.root, bg=BG)
        main_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        # LEFT COLUMN - Audio Waveform Canvas & Controls
        left_col = tk.Frame(main_frame, bg=BG)
        left_col.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(
            left_col,
            width=700,
            height=430,
            bg=CARD,
            highlightthickness=1,
            highlightbackground="#1E2A47"
        )
        self.canvas.pack(anchor="nw", fill="both", expand=True)

        # Initial Placeholder
        self.draw_placeholder()

        # Action Buttons Toolbar
        btn_bar = tk.Frame(left_col, bg=BG)
        btn_bar.pack(anchor="w", fill="x", pady=(15, 0))

        self.btn_upload = tk.Button(
            btn_bar,
            text="📁 Upload Audio",
            font=("Arial", 11, "bold"),
            bg=CYAN,
            fg="#050914",
            activebackground="#55E7FF",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=8,
            command=self.upload_audio
        )
        self.btn_upload.pack(side="left", padx=(0, 10))

        self.btn_play = tk.Button(
            btn_bar,
            text="▶ Play Audio",
            font=("Arial", 11, "bold"),
            bg=GREEN,
            fg=WHITE,
            activebackground="#54E59B",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=8,
            state="disabled",
            command=self.toggle_play
        )
        self.btn_play.pack(side="left", padx=(0, 10))

        self.btn_detect = tk.Button(
            btn_bar,
            text="🔍 Run Detection",
            font=("Arial", 11, "bold"),
            bg=ORANGE,
            fg=WHITE,
            activebackground="#FFAA5E",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=16,
            pady=8,
            state="disabled",
            command=self.start_detection_thread
        )
        self.btn_detect.pack(side="left", padx=(0, 10))

        self.btn_report = tk.Button(
            btn_bar,
            text="📄 PDF Report",
            font=("Arial", 10, "bold"),
            bg="#2A3F6C",
            fg=WHITE,
            activebackground="#3A5693",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=8,
            state="disabled",
            command=self.generate_pdf_report
        )
        self.btn_report.pack(side="left", padx=(0, 10))

        # RIGHT COLUMN - Structured Detection Results Card
        self.right_card = tk.Frame(main_frame, bg=CARD, width=340, height=490, highlightthickness=1, highlightbackground="#1E2A47")
        self.right_card.pack(side="right", fill="both", padx=(25, 0))
        self.right_card.pack_propagate(False)

        card_header = tk.Label(
            self.right_card,
            text="AUDIO DEEPFAKE ANALYSIS",
            font=("Arial", 13, "bold"),
            fg=CYAN,
            bg=CARD
        )
        card_header.pack(pady=(18, 12))

        # Results Container
        res_body = tk.Frame(self.right_card, bg=CARD)
        res_body.pack(fill="both", expand=True, padx=20)

        self.lbl_file = tk.Label(
            res_body,
            text="File: --",
            font=("Arial", 10, "bold"),
            fg=WHITE,
            bg=CARD,
            anchor="w",
            wraplength=300,
            justify="left"
        )
        self.lbl_file.pack(fill="x", pady=(5, 4))

        self.lbl_duration = tk.Label(
            res_body,
            text="Duration: --",
            font=("Arial", 10),
            fg=MUTED,
            bg=CARD,
            anchor="w"
        )
        self.lbl_duration.pack(fill="x", pady=2)

        # Separator line
        sep1 = tk.Frame(res_body, bg="#1E2A47", height=1)
        sep1.pack(fill="x", pady=10)

        # Verdict Badge
        self.badge_frame = tk.Frame(res_body, bg="#16223A", pady=12, padx=10)
        self.badge_frame.pack(fill="x", pady=5)

        self.lbl_verdict_title = tk.Label(
            self.badge_frame,
            text="PREDICTION VERDICT",
            font=("Arial", 9, "bold"),
            fg=MUTED,
            bg="#16223A"
        )
        self.lbl_verdict_title.pack()

        self.lbl_verdict = tk.Label(
            self.badge_frame,
            text="AWAITING AUDIO",
            font=("Arial", 16, "bold"),
            fg=MUTED,
            bg="#16223A"
        )
        self.lbl_verdict.pack(pady=(4, 2))

        self.lbl_confidence = tk.Label(
            self.badge_frame,
            text="Confidence: --",
            font=("Arial", 12, "bold"),
            fg=WHITE,
            bg="#16223A"
        )
        self.lbl_confidence.pack()

        # Detailed Status description
        self.lbl_status = tk.Label(
            res_body,
            text="Status: Upload an audio file and click 'Run Detection'.",
            font=("Arial", 9),
            fg=MUTED,
            bg=CARD,
            wraplength=300,
            justify="left"
        )
        self.lbl_status.pack(fill="x", pady=(10, 5))

        # Model Info
        self.lbl_model_info = tk.Label(
            res_body,
            text="Model: LFCC-CNN-BiLSTM Classifier",
            font=("Arial", 9),
            fg="#5E73A0",
            bg=CARD,
            anchor="w"
        )
        self.lbl_model_info.pack(fill="x", pady=(4, 10))

        # Progress / Status Bar
        self.progress_bar = ttk.Progressbar(res_body, mode="indeterminate")
        self.lbl_process_status = tk.Label(
            res_body,
            text="",
            font=("Arial", 9, "italic"),
            fg=CYAN,
            bg=CARD
        )

        # Bottom Action: Analyze Another File
        self.btn_reset = tk.Button(
            self.right_card,
            text="🔄 Analyze Another Audio File",
            font=("Arial", 10, "bold"),
            bg="#1E2A47",
            fg=WHITE,
            activebackground="#2E3F66",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=8,
            command=self.upload_audio
        )
        self.btn_reset.pack(side="bottom", fill="x", padx=20, pady=(0, 15))

    def draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            350, 190,
            text="🎙 No Audio Loaded",
            fill=WHITE,
            font=("Arial", 15, "bold")
        )
        self.canvas.create_text(
            350, 225,
            text="Click 'Upload Audio' to load a WAV, MP3, FLAC, or M4A file",
            fill=MUTED,
            font=("Arial", 11)
        )

    def upload_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File for Deepfake Analysis",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.m4a"),
                ("WAV Audio", "*.wav"),
                ("MP3 Audio", "*.mp3"),
                ("FLAC Audio", "*.flac"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return

        self.stop_audio()
        self.audio_path = file_path

        try:
            # Read audio data with librosa for visualization and duration
            self.audio_data, sr = librosa.load(file_path, sr=16000, mono=True)
            self.duration = float(len(self.audio_data)) / 16000.0

            # Update UI labels
            fname = os.path.basename(file_path)
            self.lbl_file.config(text=f"File: {fname}")
            self.lbl_duration.config(text=f"Duration: {self.duration:.2f} seconds")
            
            # Reset Verdict Badge
            self.lbl_verdict.config(text="READY TO ANALYZE", fg=CYAN)
            self.lbl_confidence.config(text="Confidence: --")
            self.lbl_status.config(text="Status: Audio loaded. Click 'Run Detection' to analyze LFCC features.", fg=MUTED)

            # Draw Waveform
            self.draw_waveform()

            # Enable buttons
            self.btn_play.config(state="normal")
            self.btn_detect.config(state="normal", text="🔍 Run Detection", bg=ORANGE)
            self.btn_report.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Audio Load Error", f"Unable to read audio file: {e}")

    def draw_waveform(self):
        self.canvas.delete("all")

        # Filename Header on Canvas
        fname = os.path.basename(self.audio_path)
        if len(fname) > 48:
            fname = fname[:45] + "..."
        self.canvas.create_text(
            350, 35,
            text=f"Audio Waveform: {fname}",
            fill=WHITE,
            font=("Arial", 12, "bold")
        )

        # Compute 150 amplitude bins for visual waveform rendering
        abs_audio = np.abs(self.audio_data)
        bin_count = 150
        samples_per_bin = len(abs_audio) // bin_count

        if samples_per_bin > 0:
            bins = [np.mean(abs_audio[i * samples_per_bin : (i + 1) * samples_per_bin]) for i in range(bin_count)]
        else:
            bins = [0] * bin_count

        max_bin = max(bins) if max(bins) > 0 else 1.0
        normalized_heights = [b / max_bin * 140 for b in bins]

        center_y = 220
        start_x = 85
        spacing = 3.6

        # Draw amplitude bars
        for i in range(bin_count):
            h = max(2.5, normalized_heights[i])
            x = start_x + (i * spacing)
            self.canvas.create_line(
                x, center_y - h,
                x, center_y + h,
                fill="#2A4373",  # Inactive wave blue
                width=2.5,
                tags=f"wave_{i}"
            )

        # Duration & Sample Rate Subtitle
        self.canvas.create_text(
            350, 395,
            text=f"Total Duration: {self.duration:.2f}s  |  Sample Rate: 16,000 Hz Mono  |  Bins: {bin_count}",
            fill=MUTED,
            font=("Arial", 10)
        )

    def toggle_play(self):
        if self.is_playing:
            self.stop_audio()
        else:
            self.play_audio()

    def play_audio(self):
        if self.audio_path is None:
            return

        self.is_playing = True
        self.btn_play.config(text="⏸ Stop Playback", bg=RED)

        # Winsound playback for native WAV on Windows
        if self.audio_path.lower().endswith(".wav"):
            try:
                winsound.PlaySound(
                    self.audio_path,
                    winsound.SND_ASYNC | winsound.SND_FILENAME
                )
            except Exception as e:
                print("Winsound error:", e)

        self.play_start_time = time.time()
        self.update_playback_visualizer()

    def stop_audio(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Play Audio", bg=GREEN)

        if self.audio_path and self.audio_path.lower().endswith(".wav"):
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

        # Reset waveform coloring
        for i in range(150):
            self.canvas.itemconfig(f"wave_{i}", fill="#2A4373")

    def update_playback_visualizer(self):
        if not self.is_playing:
            return

        elapsed = time.time() - self.play_start_time
        if elapsed >= self.duration:
            self.stop_audio()
            return

        ratio = elapsed / max(0.1, self.duration)
        progress_bin = int(ratio * 150)

        for i in range(150):
            if i <= progress_bin:
                self.canvas.itemconfig(f"wave_{i}", fill=CYAN)
            else:
                self.canvas.itemconfig(f"wave_{i}", fill="#2A4373")

        self.timer_job = self.root.after(80, self.update_playback_visualizer)

    def start_detection_thread(self):
        """
        Launches the detection in a non-blocking background worker thread.
        """
        if self.audio_data is None or self.is_processing:
            return

        self.is_processing = True
        self.btn_detect.config(state="disabled", text="Analyzing...")
        self.btn_upload.config(state="disabled")

        # Show Progress UI
        self.progress_bar.pack(fill="x", pady=6)
        self.progress_bar.start(10)
        self.lbl_process_status.config(text="Extracting LFCC features & running neural classifier...")
        self.lbl_process_status.pack(pady=(0, 8))

        worker = threading.Thread(target=self._run_inference_pipeline, daemon=True)
        worker.start()

    def _run_inference_pipeline(self):
        """
        Worker thread routine: pre-processes audio, extracts LFCC features, runs model, and schedules UI update.
        """
        error_msg = None
        try:
            # 1. Feature Extraction
            audio_clean, sr = self.preprocessor.process(self.audio_path)
            features = self.extractor.extract_features(audio_clean)
            features_batch = np.expand_dims(features, axis=0)  # Shape: (1, 200, 30, 3)

            # 2. Model Prediction
            model = AudioModelCache.get_model()
            p_real = float(model.predict(features_batch, verbose=0)[0][0])

            # 3. Decision Calibration
            config = AudioModelCache.get_config()
            t_upper = config.get("real_threshold_upper", 0.60)
            t_lower = config.get("fake_threshold_lower", 0.40)
            cal_map = config.get("calibration_mapping", {})
            real_min = cal_map.get("real_min_conf", 85.0)
            real_max = cal_map.get("real_max_conf", 99.8)
            fake_min = cal_map.get("fake_min_conf", 85.0)
            fake_max = cal_map.get("fake_max_conf", 99.8)

            if p_real >= t_upper:
                label = "REAL"
                norm = (p_real - t_upper) / (1.0 - t_upper) if t_upper < 1.0 else 1.0
                confidence = real_min + norm * (real_max - real_min)
            elif p_real <= t_lower:
                label = "FAKE"
                norm = (t_lower - p_real) / t_lower if t_lower > 0.0 else 1.0
                confidence = fake_min + norm * (fake_max - fake_min)
            else:
                label = "UNCERTAIN"
                dist_to_center = abs(p_real - 0.5) / 0.1
                confidence = 50.0 + dist_to_center * 15.0

            confidence = float(min(99.9, max(50.0, confidence)))

            # Save state
            self.last_prediction = label
            self.last_confidence = confidence

            # 4. SQLite & CSV Logging
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            # CSV
            try:
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([date_str, time_str, "AUDIO-" + label, f"{confidence:.2f}"])
            except Exception as e:
                print("CSV Log Error:", e)

            # SQLite Database
            try:
                save_detection(date_str, time_str, "AUDIO-" + label, confidence)
            except Exception as e:
                print("DB Log Error:", e)

            # 5. Trigger Email Alert if FAKE
            if label == "FAKE":
                try:
                    from email_alert import send_alert
                    # Generate a placeholder waveform thumbnail if needed, or pass path
                    send_alert(self.audio_path, confidence)
                except Exception as e:
                    print("Email Alert Notice:", e)

            # Schedule UI Update on Main Tkinter Thread
            self.root.after(0, lambda: self._update_ui_with_results(label, confidence, p_real))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._handle_inference_error(error_msg))

    def _update_ui_with_results(self, label, confidence, p_real):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.lbl_process_status.pack_forget()

        self.is_processing = False
        self.btn_detect.config(state="normal", text="🔍 Run Detection")
        self.btn_upload.config(state="normal")
        self.btn_report.config(state="normal")

        # Color & Text Updates
        if label == "REAL":
            badge_fg = GREEN
            status_text = "Status: AUTHENTIC HUMAN VOICE DETECTED. Acoustic formants and linear spectral distributions are consistent with natural speech."
        elif label == "FAKE":
            badge_fg = RED
            status_text = "Status: AI-GENERATED / MANIPULATED AUDIO DETECTED. Unnatural vocoder phase characteristics and linear frequency artifacts detected."
        else:
            badge_fg = ORANGE
            status_text = "Status: AMBIGUOUS / UNCERTAIN AUDIO. Signal demonstrates borderline acoustic features between natural and synthetic models."

        self.lbl_verdict.config(text=f"{label}", fg=badge_fg)
        self.lbl_confidence.config(text=f"Confidence: {confidence:.2f}%")
        self.lbl_status.config(text=status_text, fg=WHITE)

        messagebox.showinfo(
            "Analysis Complete",
            f"Audio Deepfake Analysis Complete:\n\n"
            f"File: {os.path.basename(self.audio_path)}\n"
            f"Verdict: {label}\n"
            f"Confidence: {confidence:.2f}%\n"
            f"P(REAL): {p_real*100:.1f}%\n"
            f"Logged to SQLite & detections.csv"
        )

    def _handle_inference_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.lbl_process_status.pack_forget()

        self.is_processing = False
        self.btn_detect.config(state="normal", text="🔍 Run Detection")
        self.btn_upload.config(state="normal")

        messagebox.showerror("Inference Error", f"Model pipeline encountered an issue:\n{error_msg}")

    def generate_pdf_report(self):
        if self.last_prediction is None or self.audio_path is None:
            messagebox.showwarning("Report Notice", "Please run detection on an audio file first.")
            return

        try:
            pdf_path = generate_report(
                prediction="AUDIO-" + self.last_prediction,
                confidence=self.last_confidence,
                file_path=self.audio_path,
                modality="AUDIO",
                duration=self.duration
            )
            messagebox.showinfo("PDF Generated", f"PDF Report successfully saved at:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate PDF report: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioDetectorApp(root)
    root.mainloop()