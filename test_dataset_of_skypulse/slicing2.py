import io
import os
import zipfile
import numpy as np
import cv2

# --- KEEP YOUR SAME .ZIP PATH HERE ---
ZIP_PATH = r"PASTE_YOUR_ZIP_PATH_HERE".strip('"')

# Configuration
TILE_SIZE = 128
OUTPUT_DIR_CLEAR = 'test_dataset_of_skypulse/clear'
OUTPUT_DIR_CLOUDY = 'test_dataset_of_skypulse/cloudy'

# True midday window (filenames are IST -> 11:00 AM to 2:59 PM IST)
MIN_HOUR = 11
MAX_HOUR = 14
MAX_PER_CLASS = 500  # Balanced 50/50 test set (1,000 tiles total)

# Thresholds on 0-127 scale (dead zone 35..55 discarded as ambiguous)
CLEAR_MAX = 35
CLOUDY_MIN = 55

def reset_folder(folder):
    os.makedirs(folder, exist_ok=True)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

def auto_slice_from_zip():
    reset_folder(OUTPUT_DIR_CLEAR)
    reset_folder(OUTPUT_DIR_CLOUDY)
    print("Cleared old test folders.")

    clear_count = 0
    cloudy_count = 0

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        npy_members = sorted(m for m in zf.namelist() if m.lower().endswith('.npy'))
        midday_members = []

        for m in npy_members:
            fname = m.split('/')[-1]
            stem = os.path.splitext(fname)[0]
            try:
                hour = int(stem.split('_')[1][:2])
                if MIN_HOUR <= hour <= MAX_HOUR:
                    midday_members.append(m)
            except (IndexError, ValueError):
                continue

        # Spread sampling across all 109 days instead of only early February
        step = max(1, len(midday_members) // 150)
        selected = midday_members[::step]
        print(f"Found {len(midday_members)} midday frames; scanning up to {len(selected)}...")

        printed_stats = False

        for member in selected:
            if clear_count >= MAX_PER_CLASS and cloudy_count >= MAX_PER_CLASS:
                break

            arr = np.load(io.BytesIO(zf.read(member)))
            vis = arr[:, :, 4]  # Channel 4: Visible

            if vis.max() <= 0.01:
                continue

            scaled_frame = np.round(np.clip(vis, 0.0, 1.0) * 127.0).astype(np.uint8)

            if not printed_stats:
                p25, p50, p75 = np.percentile(scaled_frame, [25, 50, 75])
                print(f"Midday VIS stats ({member.split('/')[-1]}) [0-127] -> "
                      f"min:{scaled_frame.min()}  p25:{p25:.1f}  p50:{p50:.1f}  "
                      f"p75:{p75:.1f}  max:{scaled_frame.max()}")
                printed_stats = True

            h, w = scaled_frame.shape
            for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
                for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                    patch = scaled_frame[y:y+TILE_SIZE, x:x+TILE_SIZE]
                    avg_intensity = float(np.mean(patch))

                    if avg_intensity < 3:
                        continue

                    if avg_intensity > CLOUDY_MIN and cloudy_count < MAX_PER_CLASS:
                        cv2.imwrite(
                            os.path.join(OUTPUT_DIR_CLOUDY, f"fy2h_cloudy_{cloudy_count:04d}.png"),
                            patch
                        )
                        cloudy_count += 1
                    elif avg_intensity < CLEAR_MAX and clear_count < MAX_PER_CLASS:
                        cv2.imwrite(
                            os.path.join(OUTPUT_DIR_CLEAR, f"fy2h_clear_{clear_count:04d}.png"),
                            patch
                        )
                        clear_count += 1

    print(f"\nDone! Saved {clear_count} clear tiles and {cloudy_count} cloudy tiles.")

if __name__ == "__main__":
    auto_slice_from_zip()