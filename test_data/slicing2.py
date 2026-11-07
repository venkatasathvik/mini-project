import cv2
import numpy as np
import os
import glob

# Configuration
TILE_SIZE = 128

# Using relative paths assuming execution from the root 'mini-project' folder
INPUT_DIR = 'sample dataset' # Put your raw SkyPulse .jpg or .png images here
OUTPUT_DIR = 'test_data/clear' # Default drop location for manual sorting

# Ensure directories exist
os.makedirs('test_data/clear', exist_ok=True)
os.makedirs('test_data/cloudy', exist_ok=True)

def slice_skypulse_telemetry():
    tile_count = 0
    
    # Grab any jpg or png in the sample dataset folder
    image_files = glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + glob.glob(os.path.join(INPUT_DIR, '*.png'))
    
    if not image_files:
        print(f"No images found in '{INPUT_DIR}'. Drop your raw SkyPulse images there.")
        return
        
    for img_path in image_files:
        print(f"Slicing {img_path}...")
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
                
                # Save to 'clear' folder by default
                save_path = os.path.join(OUTPUT_DIR, f"skypulse_{tile_count:04d}.png")
                cv2.imwrite(save_path, scaled_patch)
                tile_count += 1
                
    print(f"\nSuccessfully extracted {tile_count} INT8-scaled tiles into 'test_data/clear'.")
    print("ACTION REQUIRED: Manually review these tiles and move the cloudy ones into 'test_data/cloudy'.")

if __name__ == "__main__":
    slice_skypulse_telemetry()