import cv2
import csv
import os
import time
from datetime import datetime

from camera import Camera
from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector
from database import save_detection
from email_alert import send_alert
from report_generator import generate_report
from utils.detection_utils import should_capture_screenshot

# ---------------------------------------
# Create folders/files
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
CSV_PATH = os.path.join(BASE_DIR, "detections.csv")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Prediction", "Confidence"])

# ---------------------------------------
# Initialize
# ---------------------------------------

try:
    camera = Camera()
    model = DeepfakeDetector()
except Exception as e:
    print("Initialization Error:", e)
    raise

last_saved = ""
prev_time = 0
last_screenshot_time = None

# Email Control
last_email_time = 0
EMAIL_INTERVAL = 30  # seconds

print("=" * 50)
print("      AI DEEPFAKE DETECTION SYSTEM")
print("=" * 50)
print("Press ESC to Exit\n")

# ---------------------------------------
# Main Loop
# ---------------------------------------

while True:

    frame = camera.get_frame()

    if frame is None:
        break

    # ---------------- FPS ----------------

    current = time.time()

    fps = 0
    if prev_time != 0:
        fps = 1 / (current - prev_time)

    prev_time = current

    # -------------------------------------

    faces = detect_faces(frame)

    if len(faces) == 0:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        label, confidence = model.predict(face)

        color = (0, 255, 0)

        if label == "FAKE":
            color = (0, 0, 255)

        # Rectangle

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        # Prediction

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # Confidence Bar

        bar_width = int((confidence / 100) * 250)

        cv2.rectangle(frame, (20, 430), (270, 455), (70, 70, 70), -1)

        cv2.rectangle(
            frame,
            (20, 430),
            (20 + bar_width, 455),
            color,
            -1,
        )

        cv2.putText(
            frame,
            f"Confidence : {confidence:.1f}%",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        now = datetime.now()

        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        if current_time != last_saved:

            last_saved = current_time

            # Save CSV

            with open(CSV_PATH, "a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        label,
                        f"{confidence:.2f}",
                    ]
                )

            # Save Database

            save_detection(
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                label,
                float(confidence),
            )

            # Save Screenshot + Email + PDF Report

            report_path = None
            if label == "FAKE":
                capture_now = should_capture_screenshot(
                    label,
                    last_screenshot_time,
                    now,
                    cooldown_seconds=2.0,
                )

                if capture_now:
                    last_screenshot_time = now

                    filename = os.path.join(
                        SCREENSHOTS_DIR,
                        now.strftime("%Y%m%d_%H%M%S") + ".jpg",
                    )

                    success = cv2.imwrite(filename, frame)

                    if success:
                        print("Screenshot Saved :", filename)
                        report_path = filename

                        current_email_time = time.time()

                        if (
                            current_email_time - last_email_time
                            > EMAIL_INTERVAL
                        ):
                            try:
                                send_alert(filename, confidence)
                                last_email_time = current_email_time
                                print("Email Sent Successfully")
                            except Exception as e:
                                print("Email Error :", e)
                    else:
                        print("Screenshot Save Failed :", filename)

            if not report_path:
                report_path = None

            generate_report(label, confidence, image_path=report_path)

    # ---------------- Dashboard ----------------

    now = datetime.now()

    cv2.putText(
        frame,
        now.strftime("%d-%m-%Y"),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        now.strftime("%H:%M:%S"),
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "AI STATUS : ACTIVE",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "AI DEEPFAKE DETECTOR",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
    )

    cv2.imshow("AI Deepfake Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# ---------------------------------------

camera.release()

cv2.destroyAllWindows()

print("Application Closed Successfully.")