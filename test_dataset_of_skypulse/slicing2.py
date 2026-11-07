import cv2
import numpy as np
import os
import glob

# Configuration
TILE_SIZE = 128
INPUT_DIR = 'initial dataset/skypulse dataset'
OUTPUT_DIR_CLEAR = 'test_dataset_of_skypulse/clear'
OUTPUT_DIR_CLOUDY = 'test_dataset_of_skypulse/cloudy'

# Ensure directories exist
os.makedirs(OUTPUT_DIR_CLEAR, exist_ok=True)
os.makedirs(OUTPUT_DIR_CLOUDY, exist_ok=True)

def auto_slice_skypulse():
    clear_count = 0
    cloudy_count = 0
    
    # Grab any jpg or png in the sample dataset folder
    image_files = glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + glob.glob(os.path.join(INPUT_DIR, '*.png'))
    
    if not image_files:
        print(f"No images found in '{INPUT_DIR}'.")
        return
        
    for img_path in image_files:
        print(f"Slicing and Auto-sorting {os.path.basename(img_path)}...")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        h, w = img.shape
        
        # Slide a 128x128 window across the image
        for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                
                patch = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                
                # Scale pixels strictly to 0-127 for Vivado INT8
                scaled_patch = np.round((patch / 255.0) * 127.0).astype(np.uint8)
                
                # --- AUTOMATED INTENSITY SORTING ---
                # Calculate the average pixel brightness of the tile
                avg_intensity = np.mean(scaled_patch)
                
                # Thresholds (out of 127 max)
                if avg_intensity > 55:
                    # High brightness = Cloud
                    save_path = os.path.join(OUTPUT_DIR_CLOUDY, f"sky_cloudy_{cloudy_count:04d}.png")
                    cv2.imwrite(save_path, scaled_patch)
                    cloudy_count += 1
                elif avg_intensity < 35:
                    # Low brightness = Clear Terrain
                    save_path = os.path.join(OUTPUT_DIR_CLEAR, f"sky_clear_{clear_count:04d}.png")
                    cv2.imwrite(save_path, scaled_patch)
                    clear_count += 1
                
                # Tiles with averages between 35 and 55 are discarded as ambiguous
                
    print(f"\nAutomated extraction complete!")
    print(f"Sorted {clear_count} clear tiles and {cloudy_count} cloudy tiles into the test folders.")

if __name__ == "__main__":
    auto_slice_skypulse()