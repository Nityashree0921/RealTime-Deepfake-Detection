import cv2
import os
import time

# =========================================================
# SETTINGS
# =========================================================

SAVE_DIR = "webcam_test/real"
MAX_IMAGES = 200

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("=" * 60)
print("REAL WEBCAM DATA COLLECTION")
print("=" * 60)
print()
print("Instructions:")
print("1. Look normally at the camera")
print("2. Move your head slowly")
print("3. Turn left and right")
print("4. Change facial expressions")
print("5. Move slightly closer/farther")
print()
print("Press S to save a face frame")
print("Press Q to quit")
print()
print("Images will be saved to:")
print(SAVE_DIR)

# =========================================================
# FACE DETECTOR
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

count = 0

# =========================================================
# CAMERA LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
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

    # -----------------------------------------------------
    # DISPLAY FACES
    # -----------------------------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Saved: {count}/{MAX_IMAGES}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "S = Save | Q = Quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Webcam Real Dataset",
        frame
    )

    # -----------------------------------------------------
    # KEYBOARD
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------------------------------
    # SAVE FACE
    # -----------------------------------------------------

    if key == ord("s"):

        if len(faces) == 0:

            print("No face detected. Try again.")

            continue

        # Use largest detected face
        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = face

        # Add a small margin around face
        margin = int(0.15 * max(w, h))

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)

        x2 = min(
            frame.shape[1],
            x + w + margin
        )

        y2 = min(
            frame.shape[0],
            y + h + margin
        )

        face_crop = frame[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            continue

        # Resize exactly like model input
        face_crop = cv2.resize(
            face_crop,
            (224, 224)
        )

        filename = os.path.join(
            SAVE_DIR,
            f"real_webcam_{count:04d}.jpg"
        )

        cv2.imwrite(
            filename,
            face_crop
        )

        count += 1

        print(
            f"Saved {count}/{MAX_IMAGES}: {filename}"
        )

        if count >= MAX_IMAGES:

            print()
            print("=" * 60)
            print("COLLECTION COMPLETED")
            print("=" * 60)

            break

    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    if key == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Camera stopped.")
print(f"Total images saved: {count}")