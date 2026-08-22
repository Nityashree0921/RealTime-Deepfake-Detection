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


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCREENSHOTS_DIR = os.path.join(
    BASE_DIR,
    "screenshots"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "detections.csv"
)

os.makedirs(
    SCREENSHOTS_DIR,
    exist_ok=True
)


# =========================================================
# CSV INITIALIZATION
# =========================================================

if not os.path.exists(CSV_PATH):

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Date",
                "Time",
                "Prediction",
                "Confidence"
            ]
        )


# =========================================================
# INITIALIZE CAMERA + MODEL
# =========================================================

try:

    print("Starting camera...")

    camera = Camera()

    print("Loading AI model...")

    model = DeepfakeDetector()

    print("System ready.")

except Exception as e:

    print(
        "Initialization Error:",
        e
    )

    raise


# =========================================================
# VARIABLES
# =========================================================

prev_time = time.time()

fps = 0

last_prediction_time = 0

# AI prediction interval
# Smaller = more frequent predictions
# Larger = smoother performance

PREDICTION_INTERVAL = 0.5

last_label = "ANALYZING"

last_confidence = 0

last_face = None

last_saved = ""

last_screenshot_time = None

last_email_time = 0

EMAIL_INTERVAL = 30


# =========================================================
# START MESSAGE
# =========================================================

print("=" * 60)

print(
    "        AI DEEPFAKE DETECTION SYSTEM"
)

print("=" * 60)

print(
    "Press ESC to exit"
)

print()


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # -----------------------------------------------------
    # GET CAMERA FRAME
    # -----------------------------------------------------

    frame = camera.get_frame()

    if frame is None:
        break


    # -----------------------------------------------------
    # FPS
    # -----------------------------------------------------

    current_time = time.time()

    elapsed = current_time - prev_time

    if elapsed > 0:

        fps = 1 / elapsed

    prev_time = current_time


    # -----------------------------------------------------
    # FACE DETECTION
    # -----------------------------------------------------

    faces = detect_faces(frame)


    # -----------------------------------------------------
    # NO FACE
    # -----------------------------------------------------

    if len(faces) == 0:

        last_label = "NO FACE"

        last_confidence = 0

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # -----------------------------------------------------
    # FACE FOUND
    # -----------------------------------------------------

    for (x, y, w, h) in faces:

        face = frame[
            y:y+h,
            x:x+w
        ]

        if face.size == 0:
            continue


        # -------------------------------------------------
        # AI PREDICTION ONLY EVERY 0.5 SECOND
        # -------------------------------------------------

        if (
            current_time - last_prediction_time
            >= PREDICTION_INTERVAL
        ):

            last_prediction_time = current_time

            try:

                label, confidence = model.predict(face)

                last_label = label

                last_confidence = confidence

                last_face = (
                    x,
                    y,
                    w,
                    h
                )

            except Exception as e:

                print(
                    "Prediction error:",
                    e
                )


        # -------------------------------------------------
        # COLOR
        # -------------------------------------------------

        if last_label == "FAKE":

            color = (
                0,
                0,
                255
            )

        elif last_label == "REAL":

            color = (
                0,
                255,
                0
            )

        else:

            color = (
                0,
                255,
                255
            )


        # -------------------------------------------------
        # FACE BOX
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )


        # -------------------------------------------------
        # PREDICTION TEXT
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"{last_label} {last_confidence:.2f}%",
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        # -------------------------------------------------
        # CONFIDENCE BAR
        # -------------------------------------------------

        bar_width = int(
            (last_confidence / 100) * 250
        )

        bar_width = max(
            0,
            min(
                bar_width,
                250
            )
        )


        cv2.rectangle(
            frame,
            (20, 430),
            (270, 455),
            (70, 70, 70),
            -1
        )


        cv2.rectangle(
            frame,
            (20, 430),
            (20 + bar_width, 455),
            color,
            -1
        )


        cv2.putText(
            frame,
            f"Confidence : {last_confidence:.1f}%",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # =================================================
        # SAVE RESULTS
        # =================================================

        now = datetime.now()

        current_second = now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        if current_second != last_saved:

            last_saved = current_second


            # -------------------------------------------------
            # CSV
            # -------------------------------------------------

            with open(
                CSV_PATH,
                "a",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        last_label,
                        f"{last_confidence:.2f}"
                    ]
                )


            # -------------------------------------------------
            # DATABASE
            # -------------------------------------------------

            try:

                save_detection(
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    last_label,
                    float(last_confidence)
                )

            except Exception as e:

                print(
                    "Database error:",
                    e
                )


            # =================================================
            # FAKE DETECTION
            # =================================================

            if last_label == "FAKE":

                capture_now = (
                    should_capture_screenshot(
                        last_label,
                        last_screenshot_time,
                        now,
                        cooldown_seconds=5.0
                    )
                )


                if capture_now:

                    last_screenshot_time = now


                    # -------------------------------------------------
                    # SCREENSHOT
                    # -------------------------------------------------

                    filename = os.path.join(
                        SCREENSHOTS_DIR,
                        now.strftime(
                            "%Y%m%d_%H%M%S"
                        ) + ".jpg"
                    )


                    success = cv2.imwrite(
                        filename,
                        frame
                    )


                    if success:

                        print(
                            "Screenshot Saved:",
                            filename
                        )


                        # =================================================
                        # EMAIL
                        # =================================================

                        email_time = time.time()


                        if (
                            email_time
                            - last_email_time
                            > EMAIL_INTERVAL
                        ):

                            try:

                                send_alert(
                                    filename,
                                    last_confidence
                                )

                                last_email_time = email_time

                                print(
                                    "Email Alert Sent"
                                )

                            except Exception as e:

                                print(
                                    "Email Error:",
                                    e
                                )


                        # =================================================
                        # PDF REPORT
                        # =================================================

                        try:

                            generate_report(
                                last_label,
                                last_confidence,
                                image_path=filename
                            )

                            print(
                                "PDF Report Generated"
                            )

                        except Exception as e:

                            print(
                                "PDF Error:",
                                e
                            )


    # =========================================================
    # TOP INFORMATION
    # =========================================================

    now = datetime.now()


    cv2.putText(
        frame,
        now.strftime("%d-%m-%Y"),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        now.strftime("%H:%M:%S"),
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "AI STATUS : ACTIVE",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "AI DEEPFAKE DETECTOR",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )


    # =========================================================
    # DISPLAY
    # =========================================================

    cv2.imshow(
        "AI Deepfake Detection",
        frame
    )


    # =========================================================
    # ESC TO EXIT
    # =========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()

print(
    "Application Closed Successfully."
)