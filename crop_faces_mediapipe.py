import os
import cv2
import mediapipe as mp

# =========================================================
# SETTINGS
# =========================================================

INPUT_REAL = "frames/real"
INPUT_FAKE = "frames/fake"

OUTPUT_REAL = "face_frames_mp/real"
OUTPUT_FAKE = "face_frames_mp/fake"

IMG_SIZE = 224

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


# =========================================================
# MEDIAPIPE FACE DETECTOR
# =========================================================

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
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
            continue

        h, w = image.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = face_detector.process(rgb)

        if not results.detections:
            no_face += 1
            continue

        # -------------------------------------------------
        # Select largest detected face
        # -------------------------------------------------

        best_detection = None
        best_area = 0

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)

            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            area = box_w * box_h

            if area > best_area:
                best_area = area
                best_detection = (
                    x,
                    y,
                    box_w,
                    box_h
                )

        if best_detection is None:
            no_face += 1
            continue

        x, y, box_w, box_h = best_detection

        # -------------------------------------------------
        # Add padding
        # -------------------------------------------------

        padding_x = int(box_w * 0.25)
        padding_y = int(box_h * 0.35)

        x1 = max(
            0,
            x - padding_x
        )

        y1 = max(
            0,
            y - padding_y
        )

        x2 = min(
            w,
            x + box_w + padding_x
        )

        y2 = min(
            h,
            y + box_h + padding_y
        )

        # Make sure crop is valid
        if x2 <= x1 or y2 <= y1:
            no_face += 1
            continue

        face = image[
            y1:y2,
            x1:x2
        ]

        # Resize
        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        output_path = os.path.join(
            output_folder,
            filename
        )

        cv2.imwrite(
            output_path,
            face
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
# PROCESS REAL + FAKE
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
# CLOSE MEDIAPIPE
# =========================================================

face_detector.close()


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 60)
print("MEDIAPIPE FACE DATASET CREATED")
print("=" * 60)

print(
    "REAL:",
    len(os.listdir(OUTPUT_REAL))
)

print(
    "FAKE:",
    len(os.listdir(OUTPUT_FAKE))
)