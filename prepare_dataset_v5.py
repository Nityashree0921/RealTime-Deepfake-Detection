import os
import random
import shutil

# =========================================================
# SETTINGS
# =========================================================

SOURCE_REAL = "face_frames/real"
SOURCE_FAKE = "face_frames/fake"

OUTPUT = "face_dataset_v5"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# =========================================================
# CREATE DIRECTORIES
# =========================================================

for split in ["train", "val", "test"]:

    for label in ["real", "fake"]:

        os.makedirs(
            os.path.join(
                OUTPUT,
                split,
                label
            ),
            exist_ok=True
        )


# =========================================================
# GET VIDEO IDs
# =========================================================

def get_video_ids(folder):

    video_ids = set()

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        parts = filename.split("_")

        if len(parts) < 3:
            continue

        # Example:
        # real_024_0001.jpg
        #
        # video ID = 024

        video_id = parts[1]

        video_ids.add(video_id)

    return sorted(video_ids)


# =========================================================
# SPLIT VIDEOS
# =========================================================

def split_videos(video_ids):

    video_ids = list(video_ids)

    random.shuffle(video_ids)

    total = len(video_ids)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_ids = video_ids[:train_end]

    val_ids = video_ids[
        train_end:val_end
    ]

    test_ids = video_ids[
        val_end:
    ]

    return (
        train_ids,
        val_ids,
        test_ids
    )


# =========================================================
# COPY FRAMES
# =========================================================

def copy_frames(
    source_folder,
    label,
    split,
    video_ids
):

    video_ids = set(video_ids)

    files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    copied = 0

    destination = os.path.join(
        OUTPUT,
        split,
        label
    )

    for filename in files:

        parts = filename.split("_")

        if len(parts) < 3:
            continue

        video_id = parts[1]

        if video_id not in video_ids:
            continue

        source = os.path.join(
            source_folder,
            filename
        )

        target = os.path.join(
            destination,
            filename
        )

        shutil.copy2(
            source,
            target
        )

        copied += 1

    print(
        f"{label.upper():5s} "
        f"{split.upper():5s}: "
        f"{copied} frames"
    )


# =========================================================
# PROCESS REAL
# =========================================================

print("=" * 60)
print("PREPARING V5 DATASET")
print("=" * 60)

real_videos = get_video_ids(
    SOURCE_REAL
)

fake_videos = get_video_ids(
    SOURCE_FAKE
)

print()
print("REAL videos:", len(real_videos))
print("FAKE videos:", len(fake_videos))


# =========================================================
# SPLIT
# =========================================================

real_train, real_val, real_test = split_videos(
    real_videos
)

fake_train, fake_val, fake_test = split_videos(
    fake_videos
)


print()
print("VIDEO SPLIT")
print("-" * 60)

print(
    "REAL:",
    len(real_train),
    "train |",
    len(real_val),
    "val |",
    len(real_test),
    "test"
)

print(
    "FAKE:",
    len(fake_train),
    "train |",
    len(fake_val),
    "val |",
    len(fake_test),
    "test"
)


# =========================================================
# COPY
# =========================================================

print()
print("COPYING FRAMES")
print("-" * 60)


copy_frames(
    SOURCE_REAL,
    "real",
    "train",
    real_train
)

copy_frames(
    SOURCE_REAL,
    "real",
    "val",
    real_val
)

copy_frames(
    SOURCE_REAL,
    "real",
    "test",
    real_test
)


copy_frames(
    SOURCE_FAKE,
    "fake",
    "train",
    fake_train
)

copy_frames(
    SOURCE_FAKE,
    "fake",
    "val",
    fake_val
)

copy_frames(
    SOURCE_FAKE,
    "fake",
    "test",
    fake_test
)


print()
print("=" * 60)
print("V5 DATASET PREPARATION COMPLETED")
print("=" * 60)

print()
print("Dataset:")
print(OUTPUT)