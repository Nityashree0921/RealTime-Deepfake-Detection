import os
import random
import shutil
from collections import defaultdict

# =========================================================
# SETTINGS
# =========================================================

SOURCE_REAL = "face_frames/real"
SOURCE_FAKE = "face_frames/fake"

OUTPUT = "face_dataset_v6"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

print("=" * 70)
print("PREPARING CLEAN V6 DATASET (VIDEO-LEVEL SPLIT)")
print("=" * 70)

# =========================================================
# CREATE DIRECTORIES
# =========================================================

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        target_dir = os.path.join(OUTPUT, split, label)
        os.makedirs(target_dir, exist_ok=True)

# =========================================================
# EXTRACT VIDEO IDs AND CORRESPONDING FRAMES
# =========================================================

def get_video_frame_map(folder):
    video_map = defaultdict(list)
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    
    for filename in files:
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        video_id = parts[1]
        video_map[video_id].append(filename)
        
    return video_map

# =========================================================
# SPLIT VIDEOS (DETERMINISTIC)
# =========================================================

def split_video_ids(video_ids):
    video_ids = sorted(list(video_ids))
    # Shuffle with fixed seed for determinism
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(video_ids)
    
    total = len(video_ids)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_ids = sorted(video_ids[:train_end])
    val_ids = sorted(video_ids[train_end:val_end])
    test_ids = sorted(video_ids[val_end:])
    
    return train_ids, val_ids, test_ids

# =========================================================
# COPY FRAMES CLEANLY
# =========================================================

def copy_split_frames(source_folder, video_map, video_ids, label, split):
    dest_folder = os.path.join(OUTPUT, split, label)
    copied = 0
    
    for vid in video_ids:
        frames = video_map.get(vid, [])
        for f in frames:
            src = os.path.join(source_folder, f)
            dst = os.path.join(dest_folder, f)
            # Copy without overwrite error
            shutil.copy2(src, dst)
            copied += 1
            
    print(f"  {label.upper():4s} {split.upper():5s}: {len(video_ids):2d} videos | {copied:3d} frames")
    return copied

# =========================================================
# MAIN SPLIT EXECUTION
# =========================================================

real_map = get_video_frame_map(SOURCE_REAL)
fake_map = get_video_frame_map(SOURCE_FAKE)

print(f"\nDiscovered in {SOURCE_REAL}: {len(real_map)} unique REAL videos, {sum(len(v) for v in real_map.values())} frames")
print(f"Discovered in {SOURCE_FAKE}: {len(fake_map)} unique FAKE videos, {sum(len(v) for v in fake_map.values())} frames")

real_train, real_val, real_test = split_video_ids(real_map.keys())
fake_train, fake_val, fake_test = split_video_ids(fake_map.keys())

print("\n" + "=" * 70)
print("VIDEO PARTITIONING:")
print("=" * 70)
print(f"REAL Train IDs ({len(real_train)}): {real_train}")
print(f"REAL Val IDs   ({len(real_val)}): {real_val}")
print(f"REAL Test IDs  ({len(real_test)}): {real_test}")

print(f"\nFAKE Train IDs ({len(fake_train)}): {fake_train}")
print(f"FAKE Val IDs   ({len(fake_val)}): {fake_val}")
print(f"FAKE Test IDs  ({len(fake_test)}): {fake_test}")

print("\n" + "=" * 70)
print("COPYING FRAMES TO DATASET V6:")
print("=" * 70)

# Real
r_train_count = copy_split_frames(SOURCE_REAL, real_map, real_train, "real", "train")
r_val_count   = copy_split_frames(SOURCE_REAL, real_map, real_val, "real", "val")
r_test_count  = copy_split_frames(SOURCE_REAL, real_map, real_test, "real", "test")

# Fake
f_train_count = copy_split_frames(SOURCE_FAKE, fake_map, fake_train, "fake", "train")
f_val_count   = copy_split_frames(SOURCE_FAKE, fake_map, fake_val, "fake", "val")
f_test_count  = copy_split_frames(SOURCE_FAKE, fake_map, fake_test, "fake", "test")

print("\n" + "=" * 70)
print("V6 DATASET SUMMARY:")
print("=" * 70)
print(f"TRAIN: REAL={r_train_count}, FAKE={f_train_count} | Total={r_train_count + f_train_count}")
print(f"VAL  : REAL={r_val_count}, FAKE={f_val_count} | Total={r_val_count + f_val_count}")
print(f"TEST : REAL={r_test_count}, FAKE={f_test_count} | Total={r_test_count + f_test_count}")
print(f"ALL  : {r_train_count + r_val_count + r_test_count + f_train_count + f_val_count + f_test_count} frames across {len(real_map) + len(fake_map)} videos")
print("\nDataset ready at:", os.path.abspath(OUTPUT))
