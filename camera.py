import sys
import time
import cv2


def open_capture(preferred_index=0, width=640, height=480, fps=30, max_attempts=3):
    """
    Attempts to open a working cv2.VideoCapture object by testing backends and indices.
    On Windows, cv2.CAP_DSHOW (DirectShow) is prioritized for high speed and reliability.
    Includes multi-attempt retry to allow OS camera hardware mutexes to release cleanly.
    Returns:
        (cap, index, backend_name) if successful, or (None, None, None) if no camera is found.
    """
    if sys.platform.startswith("win"):
        backends = [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("DEFAULT", None),
            ("CAP_MSMF", cv2.CAP_MSMF),
        ]
    else:
        backends = [
            ("DEFAULT", None),
            ("CAP_V4L2", getattr(cv2, "CAP_V4L2", None)),
        ]

    # Prioritize preferred_index, then search other common camera indices
    indices = [preferred_index] + [i for i in [0, 1, 2, 3] if i != preferred_index]

    for attempt in range(max_attempts):
        for idx in indices:
            for b_name, b_val in backends:
                if b_val is None and b_name != "DEFAULT":
                    continue
                try:
                    if b_val is not None:
                        cap = cv2.VideoCapture(idx, b_val)
                    else:
                        cap = cv2.VideoCapture(idx)

                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        cap.set(cv2.CAP_PROP_FPS, fps)

                        # Verify camera actually returns valid image frames
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"✅ Camera initialized successfully on index {idx} using {b_name} backend ({frame.shape[1]}x{frame.shape[0]}).")
                            return cap, idx, b_name
                        else:
                            cap.release()
                except Exception:
                    pass

        # Brief pause between attempts in case the camera mutex was momentarily locked by another process
        if attempt < max_attempts - 1:
            time.sleep(0.35)

    return None, None, None


class Camera:
    """
    Robust Camera wrapper for Deepfake Detection UI & live inference.
    Handles device discovery, Windows DirectShow backends, and automatic retry on frame drops.
    """

    def __init__(self, preferred_index=0, width=640, height=480, fps=30):
        self.cap, self.index, self.backend = open_capture(
            preferred_index=preferred_index,
            width=width,
            height=height,
            fps=fps,
        )

        if self.cap is None:
            raise RuntimeError(
                "Cannot access the webcam. Please check:\n"
                "1. Your camera is physically connected / integrated webcam is enabled.\n"
                "2. Windows Camera privacy settings (Settings -> Privacy & security -> Camera -> 'Let desktop apps access your camera' is ON).\n"
                "3. No other application (Zoom, Microsoft Teams, Google Meet, OBS, Discord, or Browser) is using the webcam."
            )

    def get_frame(self, max_retries=3):
        """
        Reads a frame from the camera with retry logic to avoid quitting on transient frame drops.
        Returns:
            numpy.ndarray frame (BGR) or None if all retries fail.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        for _ in range(max_retries):
            success, frame = self.cap.read()
            if success and frame is not None and frame.size > 0:
                return frame
            time.sleep(0.01)

        return None

    def read(self):
        """
        OpenCV VideoCapture compatible read() method.
        Returns (success: bool, frame: np.ndarray).
        """
        frame = self.get_frame()
        if frame is not None:
            return True, frame
        return False, None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def isOpened(self):
        return self.is_opened()

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()