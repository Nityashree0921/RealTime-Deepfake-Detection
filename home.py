import streamlit as st

st.set_page_config(page_title="Deepfake Detection System", layout="wide")

st.title("🎥 Deepfake Detection System")

st.markdown("---")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Project Overview")

    st.write("""
This project detects AI-generated fake faces from

✅ Live Webcam

✅ Images

✅ Videos

using Deep Learning (MobileNetV2).

The system also stores detection history,
creates reports,
and provides a dashboard for monitoring.
""")

with col2:

    st.info("""
Technology

• Python

• TensorFlow

• OpenCV

• Streamlit

• MobileNetV2
""")

st.markdown("---")

st.subheader("Features")

st.success("✔ Live Webcam Detection")

st.success("✔ Image Upload Detection")

st.success("✔ Video Upload Detection")

st.success("✔ AI Deepfake Model")

st.success("✔ Dashboard")

st.success("✔ Detection History")

st.success("✔ Screenshot Saving")

st.success("✔ Report Generation")