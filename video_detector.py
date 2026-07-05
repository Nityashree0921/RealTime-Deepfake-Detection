import streamlit as st
import cv2
import tempfile

from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector


st.title("🎥 Deepfake Detection from Video")

detector = DeepfakeDetector()

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    real = 0
    fake = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            label, confidence = detector.predict(face)

            color = (0,255,0)

            if label == "FAKE":
                color = (0,0,255)
                fake += 1
            else:
                real += 1

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            cv2.putText(
                frame,
                f"{label} {confidence:.1f}%",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        stframe.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

    cap.release()

    st.success("Video Analysis Completed")

    st.write("### Results")

    st.write(f"✅ REAL Frames : {real}")

    st.write(f"❌ FAKE Frames : {fake}")

    if fake > real:
        st.error("Overall Result : FAKE VIDEO")
    else:
        st.success("Overall Result : REAL VIDEO")