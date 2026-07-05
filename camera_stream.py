import av
import cv2
from streamlit_webrtc import VideoProcessorBase

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Face detection will be added here later

        return av.VideoFrame.from_ndarray(img, format="bgr24")