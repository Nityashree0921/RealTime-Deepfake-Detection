import streamlit as st
import cv2
import numpy as np

from face_detector import detect_faces
from deepfake_detector import DeepfakeDetector

st.title("🖼️ Deepfake Detection from Image")

detector = DeepfakeDetector()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, 1)

    faces = detect_faces(image)

    for (x, y, w, h) in faces:

        face = image[y:y+h, x:x+w]

        label, confidence = detector.predict(face)

        color = (0,255,0)

        if label == "FAKE":
            color = (0,0,255)

        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)

        cv2.putText(
            image,
            f"{label} {confidence:.1f}%",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )