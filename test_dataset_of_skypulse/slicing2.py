import io
import os
import zipfile
import numpy as np
import cv2

ZIP_PATH = r"C:\Users\sathv\OneDrive\Desktop\training data set\skypulse\FY-2H_NPY_9CH_Feb-May2026.zip".strip('"')

TILE_SIZE = 128
TRAIN_CLEAR = 'dataset_fy2h_train/clear'
TRAIN_CLOUDY = 'dataset_fy2h_train/cloudy'
TEST_CLEAR = 'test_dataset_of_skypulse/clear'
TEST_CLOUDY = 'test_dataset_of_skypulse/cloudy'

# Midday IST window (11:00 - 14:59)
MIN_HOUR = 11
MAX_HOUR = 14

# Strict chronological cutoff (YYYYMMDD): train < cutoff, unseen test >= cutoff
TEST_START_DATE = "20260421"

MAX_TRAIN_PER_CLASS = 1000  # 2,000 balanced training tiles
MAX_TEST_PER_CLASS = 500    # 1,000 balanced unseen test tiles

# Thresholds on standard 0-255 uint8 PNG scale (maps to 35 and 55 after ToTensor()*127)
CLEAR_MAX = 70
CLOUDY_MIN = 110

def reset_folder(folder):
    os.makedirs(folder, exist_ok=True)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

def extract_split(zf, members, clear_dir, cloudy_dir, max_per_class, label):
    clear_count = 0
    cloudy_count = 0

    for member in members:
        if clear_count >= max_per_class and cloudy_count >= max_per_class:
            break

        arr = np.load(io.BytesIO(zf.read(member)))
        vis = arr[:, :, 4]  # Channel 4: Visible

        if vis.max() <= 0.01:
            continue

        # Save as standard 0-255 uint8 so ToTensor()*127.0 yields true 0-127 INT8 range
        frame_u8 = np.round(np.clip(vis, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = frame_u8.shape

        for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                patch = frame_u8[y:y+TILE_SIZE, x:x+TILE_SIZE]
                avg_intensity = float(np.mean(patch))

                if avg_intensity < 6:
                    continue

                if avg_intensity > CLOUDY_MIN and cloudy_count < max_per_class:
                    cv2.imwrite(os.path.join(cloudy_dir, f"fy2h_{label}_cloudy_{cloudy_count:04d}.png"), patch)
                    cloudy_count += 1
                elif avg_intensity < CLEAR_MAX and clear_count < max_per_class:
                    cv2.imwrite(os.path.join(clear_dir, f"fy2h_{label}_clear_{clear_count:04d}.png"), patch)
                    clear_count += 1

    print(f"[{label.upper()}] Saved {clear_count} clear and {cloudy_count} cloudy tiles.")

def auto_slice_from_zip():
    for folder in (TRAIN_CLEAR, TRAIN_CLOUDY, TEST_CLEAR, TEST_CLOUDY):
        reset_folder(folder)
    print("Cleared train/test FY-2H folders.")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        train_members, test_members = [], []

        for m in sorted(zf.namelist()):
            if not m.lower().endswith('.npy'):
                continue
            stem = os.path.splitext(m.split('/')[-1])[0]
            try:
                date_str, time_str = stem.split('_')
                hour = int(time_str[:2])
                if MIN_HOUR <= hour <= MAX_HOUR:
                    if date_str < TEST_START_DATE:
                        train_members.append(m)
                    else:
                        test_members.append(m)
            except (IndexError, ValueError):
                continue

        # Spread sampling evenly across dates in each split
        train_selected = train_members[::max(1, len(train_members) // 250)]
        test_selected = test_members[::max(1, len(test_members) // 150)]

        print(f"Found {len(train_members)} train frames (Feb-Apr 20) & {len(test_members)} unseen test frames (Apr 21-May 20).")
        extract_split(zf, train_selected, TRAIN_CLEAR, TRAIN_CLOUDY, MAX_TRAIN_PER_CLASS, "train")
        extract_split(zf, test_selected, TEST_CLEAR, TEST_CLOUDY, MAX_TEST_PER_CLASS, "test")

if __name__ == "__main__":
    auto_slice_from_zip()