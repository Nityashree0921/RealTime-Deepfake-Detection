import numpy as np
import tensorflow as tf
import cv2

class DeepfakeDetector:

    def __init__(self):
        self.model = tf.keras.models.load_model("models/deepfake_model.keras")
        print("✅ Deepfake AI Model Loaded")

    def predict(self, face):

        if face.size == 0:
            return "NO FACE", 0

        face = cv2.resize(face, (224, 224))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = self.model.predict(face, verbose=0)[0][0]
        print("Prediction:", prediction)

        if prediction >= 0.5:
            label = "REAL"
            confidence = prediction * 100
        else:
            label = "FAKE"
            confidence = (1 - prediction) * 100

        return label, confidence