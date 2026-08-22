import os
import cv2
import random

REAL_DIR = "frames/real"
FAKE_DIR = "frames/fake"

def check_folder(folder, name):

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print("=" * 60)
    print(name)
    print("=" * 60)
    print("Number of frames:", len(files))

    if len(files) == 0:
        return

    samples = random.sample(
        files,
        min(10, len(files))
    )

    for filename in samples:

        path = os.path.join(folder, filename)

        image = cv2.imread(path)

        if image is None:
            print("ERROR:", filename)
            continue

        h, w = image.shape[:2]

        print(
            filename,
            "->",
            w,
            "x",
            h
        )


check_folder(
    REAL_DIR,
    "REAL DATA"
)

check_folder(
    FAKE_DIR,
    "FAKE DATA"
)