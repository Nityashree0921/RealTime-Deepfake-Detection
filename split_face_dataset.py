import os
import shutil
import random

# =========================================================
# SETTINGS
# =========================================================

REAL_SRC = "face_frames/real"
FAKE_SRC = "face_frames/fake"

OUTPUT_DIR = "face_dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# =========================================================
# GET VIDEO ID
# =========================================================

def get_video_id(filename):
    """
    Example:
    real_024_0005.jpg -> 024
    fake_031_0010.jpg -> 031
    """

    parts = filename.split("_")

    if len(parts) < 3:
        return None

    return parts[1]


# =========================================================
# GROUP IMAGES BY VIDEO
# =========================================================

def group_by_video(folder):

    groups = {}

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for filename in files:

        video_id = get_video_id(filename)

        if video_id is None:
            continue

        if video_id not in groups:
            groups[video_id] = []

        groups[video_id].append(filename)

    return groups


# =========================================================
# SPLIT VIDEO IDS
# =========================================================

def split_video_ids(video_ids):

    video_ids = list(video_ids)

    random.shuffle(video_ids)

    total = len(video_ids)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train_ids = video_ids[:train_end]

    val_ids = video_ids[train_end:val_end]

    test_ids = video_ids[val_end:]

    return train_ids, val_ids, test_ids


# =========================================================
# COPY FILES
# =========================================================

def copy_split(
    groups,
    split_ids,
    label,
    split_name
):

    output_folder = os.path.join(
        OUTPUT_DIR,
        split_name,
        label
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    count = 0

    for video_id in split_ids:

        for filename in groups.get(
            video_id,
            []
        ):

            src = os.path.join(
                REAL_SRC if label == "real" else FAKE_SRC,
                filename
            )

            dst = os.path.join(
                output_folder,
                filename
            )

            shutil.copy2(
                src,
                dst
            )

            count += 1

    return count


# =========================================================
# MAIN
# =========================================================

print("=" * 60)
print("VIDEO-LEVEL FACE DATASET SPLIT")
print("=" * 60)


real_groups = group_by_video(
    REAL_SRC
)

fake_groups = group_by_video(
    FAKE_SRC
)


print()
print("REAL videos:", len(real_groups))
print("FAKE videos:", len(fake_groups))


# =========================================================
# SPLIT REAL
# =========================================================

real_train, real_val, real_test = split_video_ids(
    real_groups.keys()
)


# =========================================================
# SPLIT FAKE
# =========================================================

fake_train, fake_val, fake_test = split_video_ids(
    fake_groups.keys()
)


# =========================================================
# PRINT VIDEO SPLITS
# =========================================================

print()
print("=" * 60)
print("REAL VIDEO SPLIT")
print("=" * 60)

print("TRAIN:", real_train)
print("VAL  :", real_val)
print("TEST :", real_test)


print()
print("=" * 60)
print("FAKE VIDEO SPLIT")
print("=" * 60)

print("TRAIN:", fake_train)
print("VAL  :", fake_val)
print("TEST :", fake_test)


# =========================================================
# CREATE DATASET
# =========================================================

print()
print("=" * 60)
print("COPYING IMAGES")
print("=" * 60)


train_real = copy_split(
    real_groups,
    real_train,
    "real",
    "train"
)

val_real = copy_split(
    real_groups,
    real_val,
    "real",
    "val"
)

test_real = copy_split(
    real_groups,
    real_test,
    "real",
    "test"
)


train_fake = copy_split(
    fake_groups,
    fake_train,
    "fake",
    "train"
)

val_fake = copy_split(
    fake_groups,
    fake_val,
    "fake",
    "val"
)

test_fake = copy_split(
    fake_groups,
    fake_test,
    "fake",
    "test"
)


# =========================================================
# RESULTS
# =========================================================

print()
print("=" * 60)
print("DATASET CREATED")
print("=" * 60)

print()
print("TRAIN")
print("REAL:", train_real)
print("FAKE:", train_fake)

print()
print("VALIDATION")
print("REAL:", val_real)
print("FAKE:", val_fake)

print()
print("TEST")
print("REAL:", test_real)
print("FAKE:", test_fake)


print()
print("Dataset location:")
print(OUTPUT_DIR)

print()
print("=" * 60)
print("DONE")
print("=" * 60)