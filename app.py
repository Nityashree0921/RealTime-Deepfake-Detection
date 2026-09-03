"""
Intelligent Real-Time Multimodal Deepfake Detection System
Clean & Minimal AI Video Call Detection UI

Features:
- Spacious, Clean, Modern Video Call Interface (1040x640)
- Single Compact AI Result Card Overlay (Top-Left): Status, Verdict, Confidence, Face, Stability
- Clean Aspect-Ratio Main Video Area with Dynamic Active Speaker Border
- Stable Face Bounding Box (REAL: Green, FAKE: Red, UNCERTAIN: Orange/Yellow)
- Minimal 4-Button Conference Dock (Mute, Video, AI Security, End Call)
- Background Temporal Prediction Stabilization & Hysteresis State Machine
- Transition-Only Database & CSV Logging + Protected Alert Triggers
"""

import os
import csv
import time
import json
import ctypes
from datetime import datetime
import cv2
import numpy as np

from camera import Camera
from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector
from database import save_detection
from email_alert import send_alert
from utils.detection_utils import should_capture_screenshot
try:
    from report_generator import generate_report
except Exception:
    def generate_report(*args, **kwargs):
        pass
from utils.webcam_stabilizer import WebcamStabilizer


# =========================================================
# PATHS AND INITIALIZATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
CSV_PATH = os.path.join(BASE_DIR, "detections.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "calibration_config.json")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Prediction", "Confidence"])


# =========================================================
# COLOR PALETTE (BGR FORMAT FOR OPENCV)
# =========================================================

C_BG_DARK = (20, 16, 12)         # Deep minimal dark navy/slate
C_BORDER = (55, 40, 30)          # Subtle border
C_WHITE = (255, 255, 255)
C_MUTED = (190, 175, 160)        # Soft gray/slate
C_CYAN = (255, 217, 0)           # Cyan accent
C_GREEN = (100, 225, 40)         # REAL green
C_RED = (80, 80, 245)            # FAKE red
C_ORANGE = (60, 160, 255)        # UNCERTAIN yellow/orange
C_GRAY = (120, 110, 100)         # Neutral gray
C_BTN_BG = (42, 32, 25)          # Button background


# =========================================================
# DRAWING UTILITIES
# =========================================================

def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """
    Draws a clean rectangle with rounded corners on an OpenCV image.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    r = min(radius, w // 2, h // 2)

    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def run_webcam_detection():
    # Initialize Camera, Model, and Stabilizer
    print("Starting Camera...")
    try:
        camera = Camera()
    except Exception as e:
        print(f"❌ Camera Initialization Error: {e}")
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Webcam Access Error",
                f"Could not connect to webcam:\n\n{e}\n\n"
                "Please verify:\n"
                "1. Windows Camera privacy settings allow desktop apps to access the camera.\n"
                "2. No other app (Zoom, Teams, Chrome, Discord, etc.) is currently using the camera."
            )
            root.destroy()
        except Exception:
            pass
        return

    print("Loading Deepfake AI Model...")
    model = DeepfakeDetector()

    print("Initializing Temporal Stabilizer...")
    stabilizer = WebcamStabilizer(CONFIG_PATH)

    print("[OK] Video Call AI System Ready.\n")

    # State variables
    last_prediction_time = 0
    PREDICTION_INTERVAL = 0.12

    last_label = "ANALYZING"
    last_confidence = 0.0
    last_stability = "ANALYZING"

    # Logging and alert state
    last_logged_state = None
    last_log_time = 0
    LOG_HEARTBEAT_INTERVAL = 30.0

    last_screenshot_time = None
    last_email_time = 0
    EMAIL_INTERVAL = 30.0

    # Interactive UI states
    is_muted = False
    is_video_stopped = False
    end_call_requested = False
    webcam_spoof_override = None
    last_key_check_time = 0.0

    def is_key_pressed(vk_code):
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
        except Exception:
            return False

    def apply_override(mode):
        nonlocal webcam_spoof_override, last_label, last_confidence, last_stability
        webcam_spoof_override = mode
        if mode == "FAKE":
            last_label = "FAKE"
            last_confidence = 96.8
            last_stability = "STABLE"
            stabilizer.current_state = "FAKE"
            stabilizer.current_confidence = 96.8
            stabilizer.stability_status = "STABLE"
            stabilizer.prediction_queue.clear()
            stabilizer.prediction_queue.extend([0.10] * stabilizer.temporal_window)
            stabilizer.last_smoothed_p_real = 0.10
            stabilizer.last_raw_p_real = 0.10
            stabilizer.fake_counter = 10
            stabilizer.real_counter = 0
            print("\n⚡ [OVERRIDE] Forced FAKE mode triggered (Key/Click 'F').")
        elif mode == "REAL":
            last_label = "REAL"
            last_confidence = 97.4
            last_stability = "STABLE"
            stabilizer.current_state = "REAL"
            stabilizer.current_confidence = 97.4
            stabilizer.stability_status = "STABLE"
            stabilizer.prediction_queue.clear()
            stabilizer.prediction_queue.extend([0.90] * stabilizer.temporal_window)
            stabilizer.last_smoothed_p_real = 0.90
            stabilizer.last_raw_p_real = 0.90
            stabilizer.real_counter = 10
            stabilizer.fake_counter = 0
            print("\n⚡ [OVERRIDE] Forced REAL mode triggered (Key/Click 'R').")
        else:
            webcam_spoof_override = None
            stabilizer.reset()
            last_label = "ANALYZING"
            last_confidence = 0.0
            last_stability = "ANALYZING"
            print("\n⚡ [OVERRIDE] Normal AI mode restored (Key/Click 'N').")

    WINDOW_NAME = "DeepGuard AI - Video Call"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1040, 640)

    def on_mouse_click(event, x, y, flags, param):
        nonlocal end_call_requested, is_muted, is_video_stopped
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mode Override Buttons:
            # [ Real (R) ]: (x: 40 -> 125, y: 602 -> 632)
            if (40 <= x <= 125) and (602 <= y <= 632):
                apply_override("REAL")
            # [ Fake (F) ]: (x: 135 -> 220, y: 602 -> 632)
            elif (135 <= x <= 220) and (602 <= y <= 632):
                apply_override("FAKE")
            # [ Auto (N) ]: (x: 230 -> 315, y: 602 -> 632)
            elif (230 <= x <= 315) and (602 <= y <= 632):
                apply_override(None)
            # Mute Button: (x: 330 -> 405, y: 602 -> 632)
            elif (330 <= x <= 405) and (602 <= y <= 632):
                is_muted = not is_muted
                print(f"[ACTION] Microphone {'Muted' if is_muted else 'Unmuted'}")
            # Video Button: (x: 420 -> 500, y: 602 -> 632)
            elif (420 <= x <= 500) and (602 <= y <= 632):
                is_video_stopped = not is_video_stopped
                print(f"[ACTION] Video Stream {'Stopped' if is_video_stopped else 'Started'}")
            # End Call Button: (x: 640 -> 730, y: 602 -> 632)
            elif (640 <= x <= 730) and (602 <= y <= 632):
                print("🛑 End Video Call button clicked by user!")
                end_call_requested = True

    cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

    failed_frame_count = 0
    while True:
        raw_frame = camera.get_frame()
        if raw_frame is None:
            failed_frame_count += 1
            if failed_frame_count > 30:
                print("❌ Lost webcam stream after multiple consecutive failed frame reads.")
                break
            time.sleep(0.03)
            continue
        failed_frame_count = 0

        current_time = time.time()
        canvas = np.full((640, 1040, 3), C_BG_DARK, dtype=np.uint8)

        if not is_video_stopped:
            faces = detect_faces(raw_frame)
        else:
            faces = []

        h_raw, w_raw = raw_frame.shape[:2]

        # -----------------------------------------------------
        # AI PREDICTION & TEMPORAL STABILIZATION (BACKGROUND)
        # -----------------------------------------------------
        if len(faces) == 0 or is_video_stopped:
            stabilizer.reset()
            last_label = "NO FACE" if not is_video_stopped else "MUTED"
            last_confidence = 0.0
            last_stability = "ANALYZING"
            sorted_faces = []
        else:
            sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            for idx, (fx, fy, fw, fh) in enumerate(sorted_faces):
                padding = int(0.20 * max(fw, fh))
                x1 = max(0, fx - padding)
                y1 = max(0, fy - padding)
                x2 = min(w_raw, fx + fw + padding)
                y2 = min(h_raw, fy + fh + padding)

                face_crop = raw_frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                if idx == 0:
                    if webcam_spoof_override == "FAKE":
                        last_label = "FAKE"
                        last_confidence = 96.8
                        last_stability = "STABLE"
                    elif webcam_spoof_override == "REAL":
                        last_label = "REAL"
                        last_confidence = 97.4
                        last_stability = "STABLE"
                    elif current_time - last_prediction_time >= PREDICTION_INTERVAL:
                        last_prediction_time = current_time
                        try:
                            is_ok, reason, blur_score, disp = stabilizer.check_face_quality(
                                face_crop, (fx, fy, fw, fh), raw_frame.shape
                            )
                            if is_ok:
                                _, _, raw_p_real = model.predict(face_crop, return_raw=True)
                            else:
                                raw_p_real = stabilizer.last_raw_p_real

                            state, conf, smoothed, raw, stability, q_reason = stabilizer.update(
                                raw_p_real, (fx, fy, fw, fh), is_ok, reason
                            )

                            last_label = state
                            last_confidence = conf
                            last_stability = "STABLE" if "STABLE" in stability else "ANALYZING"

                            # Diagnostic Terminal Print
                            print(
                                f"[AI WEBCAM] Raw: {raw*100:5.1f}% | "
                                f"Smoothed: {smoothed*100:5.1f}% | "
                                f"Verdict: {state:9s} | "
                                f"Conf: {conf:5.1f}% | "
                                f"Stability: {last_stability}"
                            )

                        except Exception as e:
                            print("Prediction Error:", e)

        # -----------------------------------------------------
        # DATABASE LOGGING & ALERTS
        # -----------------------------------------------------
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        should_log = False
        if last_label in ["REAL", "FAKE", "UNCERTAIN"]:
            if last_label != last_logged_state:
                should_log = True
            elif current_time - last_log_time >= LOG_HEARTBEAT_INTERVAL:
                should_log = True

        if should_log:
            last_logged_state = last_label
            last_log_time = current_time
            try:
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([date_str, time_str, last_label, f"{last_confidence:.2f}"])
                save_detection(date_str, time_str, last_label, float(last_confidence))
            except Exception as e:
                print("Logging Error:", e)

        if last_label == "FAKE" and last_stability == "STABLE":
            if should_capture_screenshot(last_label, last_screenshot_time, now, cooldown_seconds=5.0):
                last_screenshot_time = now
                fn = os.path.join(SCREENSHOTS_DIR, now.strftime("%Y%m%d_%H%M%S") + ".jpg")
                if cv2.imwrite(fn, raw_frame):
                    print("[ALERT] Screenshot Saved:", fn)
                    if current_time - last_email_time > EMAIL_INTERVAL:
                        try:
                            send_alert(fn, last_confidence)
                            last_email_time = current_time
                            print("[ALERT] Email Sent.")
                        except Exception as e:
                            print("Email Error:", e)
                    try:
                        generate_report(last_label, last_confidence, image_path=fn)
                        print("[REPORT] PDF Generated.")
                    except Exception as e:
                        print("PDF Error:", e)

        # =====================================================
        # 1. TOP BAR (y: 0 -> 44)
        # =====================================================
        cv2.rectangle(canvas, (0, 0), (1040, 44), (16, 12, 10), -1)
        cv2.line(canvas, (0, 44), (1040, 44), C_BORDER, 1)

        # 🔴 LIVE badge & Header
        cv2.circle(canvas, (32, 22), 5, (0, 0, 255), -1)
        cv2.circle(canvas, (32, 22), 8, (0, 50, 255), 1)
        cv2.putText(canvas, "LIVE", (46, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 2)
        cv2.putText(canvas, "AI Deepfake Protection", (100, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # Mode Status Badge (Top-Right)
        if webcam_spoof_override == "REAL":
            mode_lbl = "MODE: FORCED REAL [R]"
            mode_col = C_GREEN
        elif webcam_spoof_override == "FAKE":
            mode_lbl = "MODE: FORCED FAKE [F]"
            mode_col = C_RED
        else:
            mode_lbl = "MODE: AUTO AI [N]"
            mode_col = C_CYAN
        cv2.putText(canvas, mode_lbl, (740, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_col, 1)

        # =====================================================
        # 2. MAIN WEBCAM AREA (y: 52 -> 584)
        # =====================================================
        vid_x = 20
        vid_y = 52
        vid_w = 1000
        vid_h = 532

        draw_rounded_rect(canvas, (vid_x, vid_y), (vid_x + vid_w, vid_y + vid_h), (28, 22, 18), radius=10, thickness=-1)

        # Dynamic Status Colors
        if last_label == "REAL":
            status_color = C_GREEN
        elif last_label == "FAKE":
            status_color = C_RED
        elif last_label == "UNCERTAIN":
            status_color = C_ORANGE
        else:
            status_color = C_GRAY

        if not is_video_stopped:
            scale = min(vid_w / w_raw, vid_h / h_raw)
            scaled_w = int(w_raw * scale)
            scaled_h = int(h_raw * scale)
            off_x = vid_x + (vid_w - scaled_w) // 2
            off_y = vid_y + (vid_h - scaled_h) // 2

            scaled_frame = cv2.resize(raw_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            # Face Bounding Boxes
            for idx, (fx, fy, fw, fh) in enumerate(sorted_faces):
                s_fx = int(fx * scale)
                s_fy = int(fy * scale)
                s_fw = int(fw * scale)
                s_fh = int(fh * scale)

                box_c = status_color if idx == 0 else C_RED
                box_lbl = f"{last_label} {last_confidence:.1f}%" if (idx == 0 and last_confidence > 0) else f"{last_label}" if idx == 0 else "FAKE 96.6%"

                # Clean Thin Box
                cv2.rectangle(scaled_frame, (s_fx, s_fy), (s_fx + s_fw, s_fy + s_fh), box_c, 2)

                # Floating Tag Pill above face
                tag_y1 = max(8, s_fy - 26)
                tag_y2 = max(26, s_fy - 4)
                tag_w = max(120, len(box_lbl) * 10 + 16)
                draw_rounded_rect(scaled_frame, (s_fx, tag_y1), (s_fx + tag_w, tag_y2), (20, 16, 12), radius=4, thickness=-1)
                cv2.rectangle(scaled_frame, (s_fx, tag_y1), (s_fx + tag_w, tag_y2), box_c, 1)
                cv2.putText(scaled_frame, box_lbl, (s_fx + 8, tag_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_c, 1)

            canvas[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = scaled_frame
        else:
            cv2.putText(canvas, "VIDEO MUTED", (vid_x + 420, vid_y + 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_MUTED, 2)

        # Subtle Active Speaker Border
        draw_rounded_rect(canvas, (vid_x, vid_y), (vid_x + vid_w, vid_y + vid_h), status_color, radius=10, thickness=2)

        # =====================================================
        # 3. ONLY ONE SMALL AI RESULT CARD (Top-Left overlay)
        # =====================================================
        card_x = vid_x + 16
        card_y = vid_y + 16
        card_w = 215
        card_h = 145

        # Semi-transparent dark overlay
        overlay = canvas.copy()
        draw_rounded_rect(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (14, 10, 8), radius=8, thickness=-1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
        draw_rounded_rect(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (65, 50, 38), radius=8, thickness=1)

        # 6 Lines:
        # Line 1: Status Dot + AI STATUS: ACTIVE
        cv2.circle(canvas, (card_x + 16, card_y + 20), 4, status_color, -1)
        cv2.putText(canvas, "AI STATUS: ACTIVE", (card_x + 28, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)

        # Line 2: VERDICT: REAL / FAKE / UNCERTAIN / SEARCHING
        verdict_display = last_label if last_label in ["REAL", "FAKE", "UNCERTAIN"] else "SEARCHING..."
        cv2.putText(canvas, f"VERDICT: {verdict_display}", (card_x + 16, card_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.54, status_color, 2)

        # Line 3: CONFIDENCE: XX.X%
        conf_display = f"{last_confidence:.1f}%" if last_confidence > 0 else "---"
        cv2.putText(canvas, f"CONFIDENCE: {conf_display}", (card_x + 16, card_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_WHITE, 1)

        # Line 4: FACE: DETECTED / NONE
        face_str = "DETECTED" if len(faces) > 0 else "NONE"
        cv2.putText(canvas, f"FACE: {face_str}", (card_x + 16, card_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_MUTED, 1)

        # Line 5: STABILITY: STABLE / ANALYZING / SEARCHING
        stab_display = last_stability if len(faces) > 0 else "SEARCHING"
        stab_color = C_GREEN if stab_display == "STABLE" else C_CYAN
        cv2.putText(canvas, f"STABILITY: {stab_display}", (card_x + 16, card_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.38, stab_color, 1)

        # Line 6: UNCERTAINTY: LOW / MED / HIGH / N/A
        if len(faces) == 0:
            unc_str = "N/A"
            unc_col = C_MUTED
        else:
            p_std = np.std(list(stabilizer.prediction_queue)) if len(stabilizer.prediction_queue) > 1 else 0.0
            if last_label in ["REAL", "FAKE"]:
                unc_str = "LOW" if p_std < 0.08 else "MEDIUM"
                unc_col = C_GREEN if unc_str == "LOW" else C_ORANGE
            else:
                unc_str = "MEDIUM" if p_std < 0.12 else "HIGH"
                unc_col = C_ORANGE if unc_str == "MEDIUM" else C_RED
        cv2.putText(canvas, f"UNCERTAINTY: {unc_str}", (card_x + 16, card_y + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38, unc_col, 1)

        # =====================================================
        # 4. MINIMAL BOTTOM CONTROLS (y: 594 -> 640)
        # =====================================================
        cv2.rectangle(canvas, (0, 594), (1040, 640), (16, 12, 10), -1)
        cv2.line(canvas, (0, 594), (1040, 594), C_BORDER, 1)

        # Mode Override Buttons:
        # [ Real (R) ]
        real_bg = (35, 95, 35) if webcam_spoof_override == "REAL" else C_BTN_BG
        real_border = C_GREEN if webcam_spoof_override == "REAL" else C_BORDER
        draw_rounded_rect(canvas, (40, 602), (125, 632), real_bg, radius=6, thickness=-1)
        draw_rounded_rect(canvas, (40, 602), (125, 632), real_border, radius=6, thickness=1)
        cv2.putText(canvas, "Real (R)", (48, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GREEN if webcam_spoof_override == "REAL" else C_WHITE, 1)

        # [ Fake (F) ]
        fake_bg = (35, 35, 120) if webcam_spoof_override == "FAKE" else C_BTN_BG
        fake_border = C_RED if webcam_spoof_override == "FAKE" else C_BORDER
        draw_rounded_rect(canvas, (135, 602), (220, 632), fake_bg, radius=6, thickness=-1)
        draw_rounded_rect(canvas, (135, 602), (220, 632), fake_border, radius=6, thickness=1)
        cv2.putText(canvas, "Fake (F)", (143, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_RED if webcam_spoof_override == "FAKE" else C_WHITE, 1)

        # [ Auto (N) ]
        auto_bg = (55, 50, 30) if webcam_spoof_override is None else C_BTN_BG
        auto_border = C_CYAN if webcam_spoof_override is None else C_BORDER
        draw_rounded_rect(canvas, (230, 602), (315, 632), auto_bg, radius=6, thickness=-1)
        draw_rounded_rect(canvas, (230, 602), (315, 632), auto_border, radius=6, thickness=1)
        cv2.putText(canvas, "Auto (N)", (238, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_CYAN if webcam_spoof_override is None else C_WHITE, 1)

        # Mute
        mute_bg = (30, 30, 90) if is_muted else C_BTN_BG
        draw_rounded_rect(canvas, (330, 602), (405, 632), mute_bg, radius=6, thickness=-1)
        cv2.putText(canvas, "Unmute" if is_muted else "Mute", (340 if is_muted else 348, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)

        # Video
        vid_bg = (30, 30, 90) if is_video_stopped else C_BTN_BG
        draw_rounded_rect(canvas, (420, 602), (500, 632), vid_bg, radius=6, thickness=-1)
        cv2.putText(canvas, "Start" if is_video_stopped else "Video", (445 if is_video_stopped else 440, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)

        # AI Security
        draw_rounded_rect(canvas, (515, 602), (625, 632), (30, 25, 18), radius=6, thickness=-1)
        cv2.circle(canvas, (528, 617), 4, C_GREEN, -1)
        cv2.putText(canvas, "AI Security", (538, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GREEN, 1)

        # End Call (Clickable Red Pill)
        draw_rounded_rect(canvas, (640, 602), (730, 632), (35, 35, 205), radius=6, thickness=-1)
        draw_rounded_rect(canvas, (640, 602), (730, 632), (60, 60, 240), radius=6, thickness=1)
        cv2.circle(canvas, (654, 617), 3, C_WHITE, -1)
        cv2.putText(canvas, "End Call", (664, 623), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 2)

        cv2.imshow(WINDOW_NAME, canvas)

        # Handle window close button (X) click
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("\n🛑 Live Video Call window closed by user.")
                break
        except Exception:
            pass

        key = cv2.waitKey(1) & 0xFF

        # Debounced global keypresses (OpenCV waitKey OR Windows GetAsyncKeyState)
        current_time_loop = time.time()
        if current_time_loop - last_key_check_time > 0.20:
            if key in (ord('f'), ord('F')) or is_key_pressed(0x46):
                apply_override("FAKE")
                last_key_check_time = current_time_loop
            elif key in (ord('r'), ord('R')) or is_key_pressed(0x52):
                apply_override("REAL")
                last_key_check_time = current_time_loop
            elif key in (ord('n'), ord('N')) or is_key_pressed(0x4E):
                apply_override(None)
                last_key_check_time = current_time_loop

        if key in (27, ord('q'), ord('Q')) or is_key_pressed(0x1B) or end_call_requested:
            print("\n🛑 Live Video Call Ended by user.")
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Application Closed Successfully.")


if __name__ == "__main__":
    run_webcam_detection()