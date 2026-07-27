
import cv2
import csv
import os
from datetime import datetime
from tkinter import Tk, filedialog, messagebox

from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector
from database import save_detection
from report_generator import generate_report

# -------------------------------------
# Create folders/files
# -------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
CSV_PATH = os.path.join(BASE_DIR, "detections.csv")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Prediction", "Confidence"])

# -------------------------------------
# Load Model
# -------------------------------------

model = DeepfakeDetector()

# -------------------------------------
# Select Video
# -------------------------------------

root = Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select Video",
    filetypes=[
        ("Video Files", "*.mp4 *.avi *.mov *.mkv")
    ]
)

if video_path == "":
    print("No video selected.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video.")
    exit()

frame_count = 0
last_saved = ""

print("Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Process every 5th frame
    if frame_count % 5 != 0:
        continue

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        label, confidence = model.predict(face)

        color = (0,255,0)

        if label == "FAKE":
            color = (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}%",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        now = datetime.now()

        current = now.strftime("%Y-%m-%d %H:%M:%S")

        if current != last_saved:

            last_saved = current

            with open(CSV_PATH, "a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    label,
                    f"{confidence:.2f}"
                ])

            save_detection(
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                label,
                float(confidence)
            )

            if label == "FAKE":

                filename = os.path.join(
                    SCREENSHOTS_DIR,
                    "VIDEO_" + now.strftime("%Y%m%d_%H%M%S") + ".jpg"
                )

                cv2.imwrite(filename, frame)
                print("Screenshot Saved:", filename)
                generate_report(label, confidence, image_path=filename)
            else:
                generate_report(label, confidence)

    cv2.imshow("Video Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()

messagebox.showinfo(
    "Completed",
    "Video Detection Finished Successfully."
)
