import streamlit as st
import subprocess

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Deepfake Detection System")

page = st.sidebar.selectbox(
    "Navigation",
    (
        "🏠 Home",
        "📷 Live Detection",
        "🖼 Image Detection",
        "🎥 Video Detection",
        "📊 Dashboard",
        "ℹ About"
    )
)

# ---------------- HOME ----------------

if page == "🏠 Home":

    st.header("Welcome")

    st.write("""
This application detects Deepfake Images and Videos using Artificial Intelligence.

### Features

✅ Live Webcam Detection

✅ Image Detection

✅ Video Detection

✅ Dashboard

✅ SQLite Database

✅ Fake Screenshot Saving
""")

    c1, c2, c3 = st.columns(3)

    c1.metric("Model", "MobileNetV2")

    c2.metric("Framework", "TensorFlow")

    c3.metric("UI", "Streamlit")

# ---------------- LIVE ----------------

elif page == "📷 Live Detection":

    st.header("📷 Live Webcam Detection")

    st.info("Click below to open webcam detection.")

    if st.button("Start Detection"):

        subprocess.Popen(["python", "app.py"])

        st.success("Webcam Started")

# ---------------- IMAGE ----------------

elif page == "🖼 Image Detection":

    exec(open("image_detector.py").read())

# ---------------- VIDEO ----------------

elif page == "🎥 Video Detection":

    exec(open("video_detector.py").read())

# ---------------- DASHBOARD ----------------

elif page == "📊 Dashboard":

    exec(open("dashboard.py").read())

# ---------------- ABOUT ----------------

elif page == "ℹ About":

    st.header("About")

    st.write("""
## Deepfake Detection System

Final Year Project

### Developed Using

- Python
- TensorFlow
- OpenCV
- Streamlit
- SQLite
- MobileNetV2

### Features

✔ Live Detection

✔ Image Detection

✔ Video Detection

✔ Dashboard

✔ Detection History

✔ Screenshot Saving

✔ Confidence Score

""")