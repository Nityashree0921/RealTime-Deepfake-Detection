import os
import cv2
import numpy as np
import tensorflow as tf

class DeepfakeDetector:
    def __init__(self, model_path="models/deepfake_face_model_v6.keras", threshold_path="models/face_threshold_v6.txt"):
        # Fallback to older model if v6 is not present
        if not os.path.exists(model_path):
            if os.path.exists("models/deepfake_face_model_v5.keras"):
                model_path = "models/deepfake_face_model_v5.keras"
            elif os.path.exists("models/deepfake_model.keras"):
                model_path = "models/deepfake_model.keras"

        self.model = tf.keras.models.load_model(model_path)
        
        # Load threshold
        self.threshold = 0.50
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path, "r") as f:
                    self.threshold = float(f.read().strip())
            except Exception:
                self.threshold = 0.50
                
        print(f" Deepfake AI Model Loaded: {model_path} (Threshold: {self.threshold:.2f})")

    def predict(self, face):
        if face is None or face.size == 0:
            return "NO FACE", 0.0

        # Preprocessing: RGB, Resize 224x224, Float32 (no duplicate / 255.0 because model contains internal Rescaling)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) if len(face.shape) == 3 and face.shape[2] == 3 else face
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = np.expand_dims(face_resized.astype("float32"), axis=0)

        # Model output = P(REAL) in [0, 1]
        p_real = float(self.model.predict(face_array, verbose=0)[0][0])
        
        if p_real >= self.threshold:
            label = "REAL"
            confidence = p_real * 100.0
        else:
            label = "FAKE"
            confidence = (1.0 - p_real) * 100.0

        return label, confidence