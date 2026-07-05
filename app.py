import cv2
import csv
import os
from datetime import datetime

from database import save_detection
from camera import Camera
from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector

# Create folders/files if they don't exist
os.makedirs("screenshots", exist_ok=True)

if not os.path.exists("detections.csv"):
    with open("detections.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Prediction", "Confidence"])

camera = Camera()
model = DeepfakeDetector()

last_saved = ""

while True:

    frame = camera.get_frame()

    if frame is None:
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        label, confidence = model.predict(face)

        # Choose box color
        color = (0, 255, 0)
        if label == "FAKE":
            color = (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display prediction
        cv2.putText(
            frame,
            f"{label} {confidence:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # Save only one record per second
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        if current_time != last_saved:

            last_saved = current_time

            # Save to CSV
            with open("detections.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    label,
                    f"{confidence:.2f}"
                ])

            # Save to SQLite Database
            save_detection(
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                label,
                float(confidence)
            )

            # Save screenshot if fake
            if label == "FAKE":
                filename = f"screenshots/{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)

    cv2.imshow("Deepfake Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()