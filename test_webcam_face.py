import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/deepfake_face_model_v2.keras"
IMG_SIZE = 224

print("=" * 60)
print("WEBCAM FACE MODEL TEST")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded!")
print("Press SPACE to capture")
print("Press Q to quit")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera cannot be opened")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = TEST",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            face = frame[
                y:y+h,
                x:x+w
            ]

            if face.size == 0:
                continue

            # BGR -> RGB
            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # Resize
            face_resized = cv2.resize(
                face_rgb,
                (IMG_SIZE, IMG_SIZE)
            )

            # IMPORTANT:
            # Do NOT divide by 255
            face_array = np.array(
                face_resized,
                dtype=np.float32
            )

            face_array = np.expand_dims(
                face_array,
                axis=0
            )

            prediction = model.predict(
                face_array,
                verbose=0
            )[0][0]

            print()
            print("=" * 60)
            print("WEBCAM FACE RESULT")
            print("=" * 60)
            print(
                f"Raw fake probability: {prediction:.4f}"
            )
            print(
                f"Fake probability: {prediction * 100:.2f}%"
            )
            print(
                f"Real probability: {(1-prediction)*100:.2f}%"
            )

            if prediction >= 0.5:
                print("Prediction: FAKE")
            else:
                print("Prediction: REAL")

            print("=" * 60)

    cv2.imshow(
        "Webcam Face Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()