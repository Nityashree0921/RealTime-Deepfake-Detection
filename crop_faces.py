import os
import cv2

# =========================================================
# SETTINGS
# =========================================================

INPUT_REAL = "frames/real"
INPUT_FAKE = "frames/fake"

OUTPUT_REAL = "face_frames/real"
OUTPUT_FAKE = "face_frames/fake"

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


# =========================================================
# FACE DETECTOR
# =========================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================================================
# PROCESS FOLDER
# =========================================================

def process_folder(input_folder, output_folder, label):

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    print("Input frames:", len(files))

    saved = 0
    no_face = 0

    for i, filename in enumerate(files):

        input_path = os.path.join(
            input_folder,
            filename
        )

        image = cv2.imread(input_path)

        if image is None:
            print("Could not read:", filename)
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:

            no_face += 1

            continue

        # Select largest face
        face = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        x, y, w, h = face

        # Add a little padding around face
        padding = int(0.20 * max(w, h))

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(
            image.shape[1],
            x + w + padding
        )

        y2 = min(
            image.shape[0],
            y + h + padding
        )

        face_crop = image[
            y1:y2,
            x1:x2
        ]

        # Resize to model input size
        face_crop = cv2.resize(
            face_crop,
            (224, 224)
        )

        output_path = os.path.join(
            output_folder,
            filename
        )

        cv2.imwrite(
            output_path,
            face_crop
        )

        saved += 1

        if (i + 1) % 50 == 0:

            print(
                f"Processed {i + 1}/{len(files)} "
                f"| Faces saved: {saved}"
            )

    print("\nCompleted:", label)
    print("Faces saved:", saved)
    print("No face detected:", no_face)


# =========================================================
# RUN
# =========================================================

process_folder(
    INPUT_REAL,
    OUTPUT_REAL,
    "REAL"
)

process_folder(
    INPUT_FAKE,
    OUTPUT_FAKE,
    "FAKE"
)


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 60)
print("FACE DATASET CREATION COMPLETED")
print("=" * 60)

print(
    "Real face frames:",
    len(os.listdir(OUTPUT_REAL))
)

print(
    "Fake face frames:",
    len(os.listdir(OUTPUT_FAKE))
)