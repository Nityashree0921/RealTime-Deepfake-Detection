import cv2
import csv
import os
from datetime import datetime
from tkinter import Tk, filedialog, messagebox

from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector
from database import save_detection
from report_generator import generate_report

# -----------------------------
# Create folders/files
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
CSV_PATH = os.path.join(BASE_DIR, "detections.csv")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Prediction", "Confidence"])

# -----------------------------
# Load AI Model
# -----------------------------

model = DeepfakeDetector()

# -----------------------------
# Select Image
# -----------------------------

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp")
    ]
)

if image_path == "":
    print("No image selected.")
    exit()

frame = cv2.imread(image_path)

if frame is None:
    print("Cannot open image.")
    exit()

faces = detect_faces(frame)

if len(faces) == 0:
    messagebox.showinfo("Result", "No face detected.")
    exit()

# -----------------------------
# Detect Faces
# -----------------------------

for (x, y, w, h) in faces:

    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        continue

    label, confidence = model.predict(face)

    color = (0, 255, 0)

    if label == "FAKE":
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (x, y),
        (x+w, y+h),
        color,
        2
    )

    cv2.putText(
        frame,
        f"{label} {confidence:.2f}%",
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    now = datetime.now()

    # Save CSV

    with open(CSV_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            label,
            f"{confidence:.2f}"
        ])

    # Save Database

    save_detection(
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        label,
        float(confidence)
    )

    # Save Screenshot

    if label == "FAKE":

        filename = os.path.join(
            SCREENSHOTS_DIR,
            "IMG_" + now.strftime("%Y%m%d_%H%M%S") + ".jpg"
        )

        cv2.imwrite(filename, frame)
        print("Screenshot Saved:", filename)
        generate_report(label, confidence, image_path=filename)
    else:
        generate_report(label, confidence)

# -----------------------------
# Display Result
# -----------------------------

cv2.imshow("Image Deepfake Detection", frame)

cv2.waitKey(0)

cv2.destroyAllWindows()

messagebox.showinfo(
    "Detection Complete",
    "Image analysis completed successfully."
)