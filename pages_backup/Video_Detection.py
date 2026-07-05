import streamlit as st
import cv2
import tempfile
import os
from datetime import datetime

from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector
from database import save_detection

st.header("🎥 Video Deepfake Detection")

detector = DeepfakeDetector()

uploaded_video = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_window = st.empty()

    os.makedirs("screenshots", exist_ok=True)

    last_saved = ""

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            label, confidence = detector.predict(face)

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
                f"{label} {confidence:.1f}%",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            now = datetime.now()
            current = now.strftime("%Y-%m-%d %H:%M:%S")

            if current != last_saved:

                last_saved = current

                save_detection(
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    label,
                    float(confidence)
                )

                if label == "FAKE":

                    filename = f"screenshots/{now.strftime('%Y%m%d_%H%M%S')}.jpg"

                    cv2.imwrite(filename, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_window.image(frame, width="stretch")

    cap.release()

    os.unlink(tfile.name)

    st.success("✅ Video Processing Completed")