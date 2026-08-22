import os
import shutil
import random
from collections import defaultdict

# =========================================================
# SETTINGS
# =========================================================

SOURCE_REAL = "face_frames/real"
SOURCE_FAKE = "face_frames/fake"
OUTPUT_DIR = "face_dataset_v7"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

print("=" * 70)
print("PREPARING REPRODUCIBLE V7 DATASET (VIDEO-LEVEL PARTITIONING)")
print("=" * 70)

# =========================================================
# 1. CLEAN EXISTING OUTPUT (DO NOT TOUCH SOURCE)
# =========================================================

if os.path.exists(OUTPUT_DIR):
    print(f"Removing old '{OUTPUT_DIR}' directory...")
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# =========================================================
# 2. GROUP FRAMES BY VIDEO ID
# =========================================================

def build_video_frame_map(source_folder):
    video_map = defaultdict(list)
    files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for filename in files:
        parts = filename.split("_")
        if len(parts) >= 3:
            vid_id = parts[1]
            video_map[vid_id].append(filename)
    return video_map

real_map = build_video_frame_map(SOURCE_REAL)
fake_map = build_video_frame_map(SOURCE_FAKE)

print(f"\nDiscovered in Source:")
print(f"  REAL: {len(real_map)} unique videos | {sum(len(v) for v in real_map.values())} frames")
print(f"  FAKE: {len(fake_map)} unique videos | {sum(len(v) for v in fake_map.values())} frames")

# =========================================================
# 3. REPRODUCIBLE VIDEO-LEVEL SPLITTING
# =========================================================

def partition_video_ids(video_ids, seed=42):
    vids = sorted(list(video_ids))
    rng = random.Random(seed)
    rng.shuffle(vids)
    
    n_total = len(vids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_ids = sorted(vids[:n_train])
    val_ids = sorted(vids[n_train:n_train + n_val])
    test_ids = sorted(vids[n_train + n_val:])
    
    return train_ids, val_ids, test_ids

real_train, real_val, real_test = partition_video_ids(real_map.keys(), seed=SEED)
fake_train, fake_val, fake_test = partition_video_ids(fake_map.keys(), seed=SEED)

print("\n" + "=" * 70)
print("VIDEO ID PARTITION SUMMARY:")
print("=" * 70)
print(f"REAL Train IDs ({len(real_train)}): {real_train}")
print(f"REAL Val IDs   ({len(real_val)}): {real_val}")
print(f"REAL Test IDs  ({len(real_test)}): {real_test}")
print("-" * 70)
print(f"FAKE Train IDs ({len(fake_train)}): {fake_train}")
print(f"FAKE Val IDs   ({len(fake_val)}): {fake_val}")
print(f"FAKE Test IDs  ({len(fake_test)}): {fake_test}")

# =========================================================
# 4. COPY FRAMES TO V7 DATASET
# =========================================================

def copy_partition_frames(source_folder, video_map, video_ids, label, split):
    dest = os.path.join(OUTPUT_DIR, split, label)
    count = 0
    for vid in video_ids:
        for f in video_map[vid]:
            src_path = os.path.join(source_folder, f)
            dst_path = os.path.join(dest, f)
            shutil.copy2(src_path, dst_path)
            count += 1
    return count

r_train = copy_partition_frames(SOURCE_REAL, real_map, real_train, "real", "train")
r_val   = copy_partition_frames(SOURCE_REAL, real_map, real_val, "real", "val")
r_test  = copy_partition_frames(SOURCE_REAL, real_map, real_test, "real", "test")

f_train = copy_partition_frames(SOURCE_FAKE, fake_map, fake_train, "fake", "train")
f_val   = copy_partition_frames(SOURCE_FAKE, fake_map, fake_val, "fake", "val")
f_test  = copy_partition_frames(SOURCE_FAKE, fake_map, fake_test, "fake", "test")

print("\n" + "=" * 70)
print("V7 DATASET SPLIT COUNTS:")
print("=" * 70)
print(f"TRAIN : REAL={r_train:3d} frames ({len(real_train):2d} vids) | FAKE={f_train:3d} frames ({len(fake_train):2d} vids) | Total={r_train + f_train} frames")
print(f"VAL   : REAL={r_val:3d} frames ({len(real_val):2d} vids) | FAKE={f_val:3d} frames ({len(fake_val):2d} vids) | Total={r_val + f_val} frames")
print(f"TEST  : REAL={r_test:3d} frames ({len(real_test):2d} vids) | FAKE={f_test:3d} frames ({len(fake_test):2d} vids) | Total={r_test + f_test} frames")
print("-" * 70)
print(f"TOTAL : {r_train + r_val + r_test + f_train + f_val + f_test} frames across {len(real_map) + len(fake_map)} videos")
print("=" * 70)
print(f"V7 Dataset successfully created at: {os.path.abspath(OUTPUT_DIR)}")