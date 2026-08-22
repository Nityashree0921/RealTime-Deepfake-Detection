import os
import cv2
import random
import numpy as np
from collections import defaultdict
from PIL import Image

# =========================================================
# SETTINGS
# =========================================================

SOURCE_REAL = "face_frames/real"
SOURCE_FAKE = "face_frames/fake"
REPORT_PATH = "reports/dataset_audit.txt"
PREVIEW_PATH = "reports/dataset_preview.jpg"

os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("RUNNING COMPREHENSIVE DATASET AUDIT (V7 PIPELINE)")
print("=" * 70)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_video_id(filename):
    """
    Extract video ID from format: label_vidID_frameID.ext
    Example: real_024_0001.jpg -> 024
    """
    parts = filename.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"

def compute_dhash(image_path, hash_size=8):
    """
    Compute difference hash (dHash) for perceptual image duplicate detection.
    """
    try:
        img = Image.open(image_path).convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return None

def analyze_dataset_folder(folder_path, label_name):
    if not os.path.exists(folder_path):
        return {
            "exists": False,
            "total_files": 0,
            "corrupt_files": [],
            "dimensions": set(),
            "video_map": {},
            "duplicate_files": [],
            "hash_duplicates": []
        }
    
    files = sorted(os.listdir(folder_path))
    valid_files = []
    corrupt_files = []
    dimensions = set()
    video_map = defaultdict(list)
    hash_map = defaultdict(list)
    duplicate_names = []
    
    seen_names = set()
    for f in files:
        if f in seen_names:
            duplicate_names.append(f)
        seen_names.add(f)
        
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        full_path = os.path.join(folder_path, f)
        
        # Test readability & dimensions with OpenCV and PIL
        try:
            img = cv2.imread(full_path)
            if img is None:
                corrupt_files.append(f)
                continue
            h, w, c = img.shape
            dimensions.add((w, h, c))
            
            # Extract video ID
            vid_id = extract_video_id(f)
            video_map[vid_id].append(f)
            valid_files.append((f, full_path))
            
            # Perceptual hash
            d_hash = compute_dhash(full_path)
            if d_hash is not None:
                hash_map[d_hash].append(f)
                
        except Exception:
            corrupt_files.append(f)
            
    hash_duplicates = [group for group in hash_map.values() if len(group) > 1]
    
    return {
        "exists": True,
        "total_files": len(files),
        "valid_images": len(valid_files),
        "valid_list": valid_files,
        "corrupt_files": corrupt_files,
        "dimensions": dimensions,
        "video_map": video_map,
        "duplicate_names": duplicate_names,
        "hash_duplicates": hash_duplicates
    }

# =========================================================
# AUDIT EXECUTION
# =========================================================

real_stats = analyze_dataset_folder(SOURCE_REAL, "REAL")
fake_stats = analyze_dataset_folder(SOURCE_FAKE, "FAKE")

# Build Audit Report Text
lines = []
lines.append("=" * 75)
lines.append("REALTIME DEEPFAKE DETECTION — COMPREHENSIVE DATASET AUDIT REPORT")
lines.append("=" * 75)
lines.append("")
lines.append("1. DATASET TOTALS & INVENTORY:")
lines.append(f"  Source Folders      : {SOURCE_REAL} | {SOURCE_FAKE}")
lines.append(f"  REAL Total Frames   : {real_stats['valid_images']} valid images ({len(real_stats['corrupt_files'])} corrupt)")
lines.append(f"  FAKE Total Frames   : {fake_stats['valid_images']} valid images ({len(fake_stats['corrupt_files'])} corrupt)")
lines.append(f"  Total Dataset Frames: {real_stats['valid_images'] + fake_stats['valid_images']} face frames")
lines.append(f"  Class Balance       : REAL = {real_stats['valid_images'] / (real_stats['valid_images'] + fake_stats['valid_images']) * 100:.2f}% | FAKE = {fake_stats['valid_images'] / (real_stats['valid_images'] + fake_stats['valid_images']) * 100:.2f}%")
lines.append("")

lines.append("2. UNIQUE VIDEO ID ANALYSIS:")
lines.append(f"  Unique REAL Video IDs ({len(real_stats['video_map'])}): {sorted(real_stats['video_map'].keys())}")
lines.append(f"  Unique FAKE Video IDs ({len(fake_stats['video_map'])}): {sorted(fake_stats['video_map'].keys())}")
lines.append("")

real_counts = [len(v) for v in real_stats['video_map'].values()]
fake_counts = [len(v) for v in fake_stats['video_map'].values()]

lines.append("3. FRAMES-PER-VIDEO DISTRIBUTION:")
lines.append(f"  REAL Frames/Video   : Min = {min(real_counts)}, Max = {max(real_counts)}, Avg = {np.mean(real_counts):.2f}, Median = {np.median(real_counts):.1f}")
lines.append(f"  FAKE Frames/Video   : Min = {min(fake_counts)}, Max = {max(fake_counts)}, Avg = {np.mean(fake_counts):.2f}, Median = {np.median(fake_counts):.1f}")
lines.append("")

lines.append("4. IMAGE DIMENSIONS & CORRUPT CHECK:")
lines.append(f"  REAL Dimensions Found: {list(real_stats['dimensions'])}")
lines.append(f"  FAKE Dimensions Found: {list(fake_stats['dimensions'])}")
lines.append(f"  Corrupt/Unreadable   : {len(real_stats['corrupt_files']) + len(fake_stats['corrupt_files'])} files found")
lines.append("")

lines.append("5. DUPLICATE ANALYSIS:")
lines.append(f"  Duplicate Filenames  : REAL={len(real_stats['duplicate_names'])}, FAKE={len(fake_stats['duplicate_names'])}")
lines.append(f"  Perceptual Hash Exact Duplicates: REAL={len(real_stats['hash_duplicates'])} clusters, FAKE={len(fake_stats['hash_duplicates'])} clusters")
lines.append("")

lines.append("6. PER-VIDEO DETAILED BREAKDOWN:")
lines.append("  REAL VIDEOS:")
for vid, f_list in sorted(real_stats['video_map'].items()):
    lines.append(f"    Video ID '{vid}': {len(f_list)} frames (e.g. {f_list[0]})")
lines.append("  FAKE VIDEOS:")
for vid, f_list in sorted(fake_stats['video_map'].items()):
    lines.append(f"    Video ID '{vid}': {len(f_list)} frames (e.g. {f_list[0]})")

lines.append("=" * 75)

report_text = "\n".join(lines)
with open(REPORT_PATH, "w") as f:
    f.write(report_text)

print(report_text)
print(f"\nAudit Report Saved to: {REPORT_PATH}")

# =========================================================
# GENERATE CONTACT SHEET (PREVIEW)
# =========================================================

print("\nGenerating Dataset Preview Contact Sheet...")
random.seed(42)

# Pick 8 random REAL and 8 random FAKE
sample_real = random.sample(real_stats['valid_list'], min(8, len(real_stats['valid_list'])))
sample_fake = random.sample(fake_stats['valid_list'], min(8, len(fake_stats['valid_list'])))

grid_cols = 4
grid_rows = 4
cell_size = 200

contact_sheet = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)

for idx, (fname, fpath) in enumerate(sample_real + sample_fake):
    r = idx // grid_cols
    c = idx % grid_cols
    
    img = cv2.imread(fpath)
    if img is not None:
        img_resized = cv2.resize(img, (cell_size, cell_size))
        
        # Overlay label
        is_real = idx < len(sample_real)
        lbl_str = f"REAL: {fname[:14]}" if is_real else f"FAKE: {fname[:14]}"
        color = (0, 255, 0) if is_real else (0, 0, 255)
        
        # Border
        cv2.rectangle(img_resized, (0, 0), (cell_size - 1, cell_size - 1), color, 3)
        cv2.putText(img_resized, lbl_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        contact_sheet[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = img_resized

cv2.imwrite(PREVIEW_PATH, contact_sheet)
print(f"Dataset preview contact sheet saved to: {PREVIEW_PATH}")
print("=" * 70)
