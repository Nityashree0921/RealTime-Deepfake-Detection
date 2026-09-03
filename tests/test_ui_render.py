"""
Minimal Clean UI Render Test
Renders the simplified, single-card modern video call detection UI.
"""

import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Canvas Dimensions: 1040 x 640
W, H = 1040, 640
canvas = np.full((H, W, 3), (20, 16, 12), dtype=np.uint8)

# Colors
C_BORDER = (55, 40, 30)
C_WHITE = (255, 255, 255)
C_MUTED = (190, 175, 160)
C_CYAN = (255, 217, 0)
C_GREEN = (100, 225, 40)
C_RED = (80, 80, 245)
C_ORANGE = (60, 160, 255)
C_GRAY = (120, 110, 100)
C_BTN_BG = (42, 32, 25)

def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
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

# 1. TOP BAR (y: 0 -> 44)
cv2.rectangle(canvas, (0, 0), (W, 44), (16, 12, 10), -1)
cv2.line(canvas, (0, 44), (W, 44), C_BORDER, 1)

# LIVE badge
cv2.circle(canvas, (32, 22), 5, (0, 0, 255), -1)
cv2.circle(canvas, (32, 22), 8, (0, 50, 255), 1)
cv2.putText(canvas, "LIVE", (46, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 2)
cv2.putText(canvas, "AI Deepfake Protection", (100, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

# 2. MAIN WEBCAM AREA (Occupies almost whole screen)
vid_x, vid_y, vid_w, vid_h = 20, 52, 1000, 532
draw_rounded_rect(canvas, (vid_x, vid_y), (vid_x + vid_w, vid_y + vid_h), (28, 22, 18), radius=10, thickness=-1)

# Mock webcam video background
mock_video = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
mock_video[:] = (35, 28, 24)
# Speaker silhouette
cv2.circle(mock_video, (vid_w // 2, vid_h // 2 - 20), 85, (60, 50, 42), -1)
cv2.ellipse(mock_video, (vid_w // 2, vid_h // 2 + 130), (145, 80), 0, 0, 180, (60, 50, 42), -1)

# Face box
fx, fy, fw, fh = vid_w // 2 - 75, vid_h // 2 - 95, 150, 150
cv2.rectangle(mock_video, (fx, fy), (fx + fw, fy + fh), C_GREEN, 2)
# Clean tag above box
tag_y1 = max(8, fy - 26)
tag_y2 = max(26, fy - 4)
draw_rounded_rect(mock_video, (fx, tag_y1), (fx + 115, tag_y2), (20, 16, 12), radius=4, thickness=-1)
cv2.rectangle(mock_video, (fx, tag_y1), (fx + 115, tag_y2), C_GREEN, 1)
cv2.putText(mock_video, "REAL 95.3%", (fx + 8, tag_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GREEN, 1)

canvas[vid_y:vid_y + vid_h, vid_x:vid_x + vid_w] = mock_video

# Subtle Active Speaker Border
draw_rounded_rect(canvas, (vid_x, vid_y), (vid_x + vid_w, vid_y + vid_h), C_GREEN, radius=10, thickness=2)

# 3. ONLY ONE SMALL AI RESULT CARD (Top-Left overlay)
card_x, card_y, card_w, card_h = vid_x + 16, vid_y + 16, 215, 130
overlay = canvas.copy()
draw_rounded_rect(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (14, 10, 8), radius=8, thickness=-1)
cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
draw_rounded_rect(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (65, 50, 38), radius=8, thickness=1)

# 5 Card Lines:
# Line 1: Status Dot + AI STATUS: ACTIVE
cv2.circle(canvas, (card_x + 16, card_y + 20), 4, C_GREEN, -1)
cv2.putText(canvas, "AI STATUS: ACTIVE", (card_x + 28, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)

# Line 2: VERDICT: REAL
cv2.putText(canvas, "VERDICT: REAL", (card_x + 16, card_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_GREEN, 2)

# Line 3: CONFIDENCE: 95.3%
cv2.putText(canvas, "CONFIDENCE: 95.3%", (card_x + 16, card_y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)

# Line 4: FACE: DETECTED
cv2.putText(canvas, "FACE: DETECTED", (card_x + 16, card_y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1)

# Line 5: STABILITY: STABLE
cv2.putText(canvas, "STABILITY: STABLE", (card_x + 16, card_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GREEN, 1)

# 4. BOTTOM CONTROLS (Clean, minimal 4 buttons)
cv2.rectangle(canvas, (0, 594), (W, H), (16, 12, 10), -1)
cv2.line(canvas, (0, 594), (W, 594), C_BORDER, 1)

# 4 Centered Buttons: Mute, Video, AI Security, End Call
# Mute
draw_rounded_rect(canvas, (320, 602), (395, 632), C_BTN_BG, radius=6, thickness=-1)
cv2.putText(canvas, "Mute", (338, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)

# Video
draw_rounded_rect(canvas, (410, 602), (490, 632), C_BTN_BG, radius=6, thickness=-1)
cv2.putText(canvas, "Video", (430, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)

# AI Security
draw_rounded_rect(canvas, (505, 602), (625, 632), (30, 25, 18), radius=6, thickness=-1)
cv2.circle(canvas, (520, 617), 4, C_GREEN, -1)
cv2.putText(canvas, "AI Security", (532, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GREEN, 1)

# End Call
draw_rounded_rect(canvas, (640, 602), (730, 632), (35, 35, 205), radius=6, thickness=-1)
draw_rounded_rect(canvas, (640, 602), (730, 632), (60, 60, 240), radius=6, thickness=1)
cv2.circle(canvas, (654, 617), 3, C_WHITE, -1)
cv2.putText(canvas, "End Call", (664, 623), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 2)

out_path = "reports/webcam_ui_preview.png"
os.makedirs("reports", exist_ok=True)
cv2.imwrite(out_path, canvas)
print(f"[OK] Minimal UI Preview saved to: {out_path}")
