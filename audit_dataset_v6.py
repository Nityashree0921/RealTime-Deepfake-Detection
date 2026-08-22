import os
from collections import defaultdict

def extract_video_id(filename):
    """
    Extract video ID from format: label_vidID_frameID.ext
    e.g. real_024_0001.jpg -> 024
    """
    parts = filename.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"

def analyze_folder(folder_path):
    if not os.path.exists(folder_path):
        return {}
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    vid_map = defaultdict(list)
    for f in files:
        vid_id = extract_video_id(f)
        vid_map[vid_id].append(f)
    return vid_map

def audit_dataset_source(source_dir="face_frames"):
    print("=" * 70)
    print(f"AUDITING DATASET SOURCE: {source_dir}")
    print("=" * 70)
    
    real_dir = os.path.join(source_dir, "real")
    fake_dir = os.path.join(source_dir, "fake")
    
    real_map = analyze_folder(real_dir)
    fake_map = analyze_folder(fake_dir)
    
    real_frames_count = sum(len(v) for v in real_map.values())
    fake_frames_count = sum(len(v) for v in fake_map.values())
    
    print(f"REAL videos : {len(real_map)}")
    print(f"FAKE videos : {len(fake_map)}")
    print(f"REAL frames : {real_frames_count}")
    print(f"FAKE frames : {fake_frames_count}")
    print(f"TOTAL frames: {real_frames_count + fake_frames_count}")
    
    print("\nREAL video IDs:", sorted(real_map.keys()))
    print("FAKE video IDs:", sorted(fake_map.keys()))
    
    print("\nFrames per REAL video:")
    for vid, frames in sorted(real_map.items()):
        print(f"  Video {vid}: {len(frames)} frames")
        
    print("\nFrames per FAKE video:")
    for vid, frames in sorted(fake_map.items()):
        print(f"  Video {vid}: {len(frames)} frames")

def audit_dataset_split(dataset_dir="face_dataset_v6"):
    print("\n" + "=" * 70)
    print(f"AUDITING DATASET SPLIT: {dataset_dir}")
    print("=" * 70)
    
    if not os.path.exists(dataset_dir):
        print(f"Split directory {dataset_dir} does not exist yet. Run prepare_dataset_v6.py first.")
        return False
    
    splits = ["train", "val", "test"]
    split_vids = {s: {"real": set(), "fake": set()} for s in splits}
    split_frames = {s: {"real": 0, "fake": 0} for s in splits}
    
    for split in splits:
        for label in ["real", "fake"]:
            folder = os.path.join(dataset_dir, split, label)
            vmap = analyze_folder(folder)
            split_vids[split][label] = set(vmap.keys())
            split_frames[split][label] = sum(len(v) for v in vmap.values())
            
    print("\nSPLIT SUMMARY:")
    for split in splits:
        r_vids = split_vids[split]["real"]
        f_vids = split_vids[split]["fake"]
        r_f = split_frames[split]["real"]
        f_f = split_frames[split]["fake"]
        print(f"\n{split.upper()} SPLIT:")
        print(f"  REAL: {len(r_vids)} videos ({r_f} frames) -> IDs: {sorted(r_vids)}")
        print(f"  FAKE: {len(f_vids)} videos ({f_f} frames) -> IDs: {sorted(f_vids)}")
        print(f"  Total: {len(r_vids) + len(f_vids)} videos, {r_f + f_f} frames")
        
    # Check for leakage
    leakage_detected = False
    print("\n" + "-" * 70)
    print("LEAKAGE VERIFICATION (No video ID in multiple splits):")
    print("-" * 70)
    
    for label in ["real", "fake"]:
        train_set = split_vids["train"][label]
        val_set = split_vids["val"][label]
        test_set = split_vids["test"][label]
        
        train_val = train_set.intersection(val_set)
        train_test = train_set.intersection(test_set)
        val_test = val_set.intersection(test_set)
        
        if train_val:
            print(f"LEAKAGE DETECTED in {label.upper()}: Train and Val share video IDs: {train_val}")
            leakage_detected = True
        if train_test:
            print(f"LEAKAGE DETECTED in {label.upper()}: Train and Test share video IDs: {train_test}")
            leakage_detected = True
        if val_test:
            print(f"LEAKAGE DETECTED in {label.upper()}: Val and Test share video IDs: {val_test}")
            leakage_detected = True
            
    if not leakage_detected:
        print("VERIFIED CLEAN: 0 video ID leakage across Train / Val / Test splits!")
    return not leakage_detected

if __name__ == "__main__":
    audit_dataset_source("face_frames")
    if os.path.exists("face_dataset_v6"):
        audit_dataset_split("face_dataset_v6")
    elif os.path.exists("face_dataset_v5"):
        audit_dataset_split("face_dataset_v5")
