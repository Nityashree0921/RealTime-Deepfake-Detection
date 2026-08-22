import cv2
import os

# ==========================================
# SETTINGS
# ==========================================

DATASET_DIR = "dataset"
OUTPUT_DIR = "frames"

# Extract 1 frame every N frames
FRAME_INTERVAL = 10

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

for label in ["real", "fake"]:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)


# ==========================================
# EXTRACT FRAMES
# ==========================================

def extract_from_folder(label):

    input_folder = os.path.join(DATASET_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)

    videos = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".mp4")
    ]

    print(f"\nProcessing {label.upper()} videos: {len(videos)}")

    total_frames = 0

    for video_index, video_name in enumerate(videos):

        video_path = os.path.join(input_folder, video_name)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Could not open:", video_name)
            continue

        frame_number = 0
        saved = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_number % FRAME_INTERVAL == 0:

                # Resize frame
                frame = cv2.resize(frame, (224, 224))

                filename = (
                    f"{label}_{video_index:03d}_"
                    f"{saved:04d}.jpg"
                )

                output_path = os.path.join(
                    output_folder,
                    filename
                )

                cv2.imwrite(output_path, frame)

                saved += 1
                total_frames += 1

            frame_number += 1

        cap.release()

        print(
            f"[{video_index + 1}/{len(videos)}] "
            f"{video_name} -> {saved} frames"
        )

    print(
        f"\n{label.upper()} TOTAL FRAMES: "
        f"{total_frames}"
    )


# ==========================================
# START
# ==========================================

print("=" * 50)
print("VIDEO FRAME EXTRACTION")
print("=" * 50)

extract_from_folder("real")
extract_from_folder("fake")

print("\nExtraction completed!")